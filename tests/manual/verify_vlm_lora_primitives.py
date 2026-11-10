#!/usr/bin/env python3

import sys

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.lora as lora
from openpi.models.lora_attention import LoRAMultiHeadDotProductAttention as LoRAMHA

OK = True


def chk(cond, msg):
    global OK
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    OK = OK and bool(cond)


def flat(v):
    return flax.traverse_util.flatten_dict(v, sep="/")


def main() -> int:
    rng = jax.random.key(0)
    x = jax.random.normal(jax.random.key(1), (2, 5, 64), dtype=jnp.float32)

    print("== [1] Einsum: legacy formula (random A/B, scaled output) ==")
    cfg_old = lora.LoRAConfig(rank=4, alpha=8.0)  # scale = 2.0
    m = lora.Einsum(shape=(64, 32), init_fn=nn.initializers.normal(0.02), lora_config=cfg_old)
    v = m.init(rng, "btd,dk->btk", x)
    p = flat(v["params"])
    y = m.apply(v, "btd,dk->btk", x)
    y_manual = (
        jnp.einsum("btd,dk->btk", x, p["w"])
        + jnp.einsum("btd,dk->btk", jnp.einsum("btd,dL->btL", x, p["lora_a"]), p["lora_b"]) * cfg_old.scaling_value
    )
    chk(bool((y == y_manual).all()), "Einsum output equals base + scale*(x@A@B) bit-for-bit")
    chk(bool((p["lora_b"] != 0).any()), "Legacy lora_b initialization is random, not all zeros")

    print("== [2] FFN: legacy unscaled behavior and optional scaling ==")
    ff_old = lora.FeedForward(features=64, hidden_dim=96, lora_config=lora.LoRAConfig(rank=4, alpha=8.0))
    v = ff_old.init(rng, x)
    pf = flat(v["params"])
    y_old = ff_old.apply(v, x)

    def ffn_manual(scale_ffn: float):
        g = (
            jnp.dot(x, pf["gating_einsum"][0])
            + jnp.dot(jnp.dot(x, pf["gating_einsum_lora_a"][0]), pf["gating_einsum_lora_b"][0]) * scale_ffn
        )
        u = (
            jnp.dot(x, pf["gating_einsum"][1])
            + jnp.dot(jnp.dot(x, pf["gating_einsum_lora_a"][1]), pf["gating_einsum_lora_b"][1]) * scale_ffn
        )
        h = nn.gelu(g) * u
        return jnp.dot(h, pf["linear"]) + jnp.dot(jnp.dot(h, pf["linear_lora_a"]), pf["linear_lora_b"]) * scale_ffn

    chk(
        bool((y_old == ffn_manual(1.0)).all()),
        "apply_ffn_scaling=False: unscaled FFN delta matches upstream bit-for-bit",
    )
    ff_new = lora.FeedForward(
        features=64, hidden_dim=96, lora_config=lora.LoRAConfig(rank=4, alpha=8.0, apply_ffn_scaling=True)
    )
    y_new = ff_new.apply(v, x)
    chk(bool((y_new == ffn_manual(2.0)).all()), "apply_ffn_scaling=True: FFN delta scaled by alpha/r=2.0")

    print("== [3] Separate A/B initialization and zero-B identity (Einsum/FFN) ==")
    cfg_zb = lora.LoRAConfig(
        rank=4,
        alpha=8.0,
        init_fn_a=nn.initializers.normal(0.01),
        init_fn_b=nn.initializers.zeros_init(),
        apply_ffn_scaling=True,
    )
    mz = lora.Einsum(shape=(64, 32), init_fn=nn.initializers.normal(0.02), lora_config=cfg_zb)
    vz = mz.init(rng, "btd,dk->btk", x)
    pz = flat(vz["params"])
    chk(bool((pz["lora_b"] == 0).all()) and bool((pz["lora_a"] != 0).any()), "Einsum: B initialized to zero, A random")
    y_base = jnp.einsum("btd,dk->btk", x, pz["w"])
    chk(bool((mz.apply(vz, "btd,dk->btk", x) == y_base).all()), "Einsum zero-B output matches base bit-for-bit")
    ffz = lora.FeedForward(features=64, hidden_dim=96, lora_config=cfg_zb)
    vfz = ffz.init(rng, x)
    pfz = flat(vfz["params"])
    chk(
        bool((pfz["gating_einsum_lora_b"] == 0).all()) and bool((pfz["linear_lora_b"] == 0).all()),
        "FFN: both lora_b groups are zero",
    )
    ff_ref = lora.FeedForward(features=64, hidden_dim=96)
    v_ref = {"params": {"gating_einsum": pfz["gating_einsum"], "linear": pfz["linear"]}}
    chk(bool((ffz.apply(vfz, x) == ff_ref.apply(v_ref, x)).all()), "FFN zero-B output matches base bit-for-bit")

    print("== [4] LoRAMHA modes with shared flax 0.10.2 nn.MHA weights ==")
    xa = jax.random.normal(jax.random.key(2), (2, 9, 1152), dtype=jnp.float32)
    ref = nn.MultiHeadDotProductAttention(
        num_heads=16, kernel_init=nn.initializers.xavier_uniform(), deterministic=True, dtype=jnp.float32
    )
    vref = ref.init(rng, xa, xa)
    yref = ref.apply(vref, xa, xa)
    m0 = LoRAMHA(
        num_heads=16, kernel_init=nn.initializers.xavier_uniform(), deterministic=True, dtype=jnp.float32, lora_rank=0
    )
    chk(
        jax.tree_util.tree_structure(m0.init(rng, xa, xa)) == jax.tree_util.tree_structure(vref),
        "rank=0 parameter tree matches nn.MHA",
    )
    chk(bool((m0.apply(vref, xa, xa) == yref).all()), "rank=0 shared-weight output is bit-exact")
    m8 = LoRAMHA(
        num_heads=16,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=True,
        dtype=jnp.float32,
        lora_rank=8,
        lora_scale=2.0,
    )
    f8 = flat(m8.init(jax.random.key(3), xa, xa)["params"])
    lk = sorted(k for k in f8 if "lora" in k)
    chk(
        len(lk) == 8
        and set(k.split("/")[0] for k in lk)
        == {
            "query_lora_a",
            "query_lora_b",
            "key_lora_a",
            "key_lora_b",
            "value_lora_a",
            "value_lora_b",
            "out_lora_a",
            "out_lora_b",
        },
        f"rank=8 has exactly 8 LoRA keys: {sorted(set(k.split('/')[0] for k in lk))}",
    )
    fref = flat(vref["params"])
    chk(set(f8) - set(lk) == set(fref), "rank=8 base keys match nn.MHA")
    for k in fref:
        f8[k] = fref[k]
    chk(all(bool((f8[k] == 0).all()) for k in lk if "_lora_b" in k), "rank=8 all lora_b parameters initialized to zero")
    v8 = {"params": flax.traverse_util.unflatten_dict(f8, sep="/")}
    chk(bool((m8.apply(v8, xa, xa) == yref).all()), "rank=8 zero-B shared-weight output is bit-exact")
    for k in lk:
        if "_lora_b" in k:
            f8[k] = jnp.full_like(f8[k], 0.01)
    y8n = m8.apply({"params": flax.traverse_util.unflatten_dict(f8, sep="/")}, xa, xa)
    d = float(jnp.abs(y8n - yref).max())
    chk(bool((y8n != yref).any()) and np.isfinite(d), f"rank=8 B!=0 changes the output, max|Δ|={d:.3e}")

    print("== [5] Zero-B Dense/Conv adapters used by SigLIP ==")
    z = jax.random.normal(jax.random.key(4), (2, 7, 48))
    a = nn.Dense(4, use_bias=False, kernel_init=nn.initializers.normal(0.01))
    b = nn.Dense(48, use_bias=False, kernel_init=nn.initializers.zeros_init())
    va, vb = a.init(rng, z), b.init(rng, a.apply(a.init(rng, z), z))
    chk(bool((b.apply(vb, a.apply(va, z)) == 0).all()), "Dense adapter: zero-B produces exactly zero delta")
    img = jax.random.normal(jax.random.key(5), (1, 28, 28, 3))
    ca = nn.Conv(
        4, (14, 14), strides=(14, 14), padding="VALID", use_bias=False, kernel_init=nn.initializers.normal(0.01)
    )
    cb = nn.Conv(16, (1, 1), use_bias=False, kernel_init=nn.initializers.zeros_init())
    vca = ca.init(rng, img)
    vcb = cb.init(rng, ca.apply(vca, img))
    chk(
        bool((cb.apply(vcb, ca.apply(vca, img)) == 0).all()),
        "Conv adapter (14x14 -> 1x1): zero-B produces exactly zero delta",
    )

    print("\n" + ("✅ verify_vlm_lora_primitives: All checks passed" if OK else "❌ Checks failed"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
