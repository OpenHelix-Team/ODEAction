#!/usr/bin/env python3

#   UV_NO_SYNC=1 uv run tests/manual/verify_vlm_lora_zero_init.py --level siglip
#   CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 UV_NO_SYNC=1 \
#     uv run tests/manual/verify_vlm_lora_zero_init.py --level full
import argparse
import dataclasses as _dc
import gc
import sys

import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0_config as pi0_config
import openpi.models.siglip as _siglip
import openpi.models.vlm_lora as vlm_lora

OK = True


def chk(cond, msg):
    global OK
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    OK = OK and bool(cond)


def _all_targets(rank=8, alpha=8.0):
    t = vlm_lora.LoRATargetConfig(enabled=True, rank=rank, alpha=alpha, zero_b=True)
    return vlm_lora.VlmLoRAConfig(
        enabled=True,
        language_attention=t,
        language_ffn=t,
        vision_attention=t,
        vision_mlp=t,
        vision_projector=t,
        vision_patch_embedding=t,
    )


def run_siglip() -> None:
    print("== [siglip] Zero-B equivalence for siglip.Module(B/16, scan=True) ==")
    img = jnp.ones((1, 224, 224, 3), dtype=jnp.float32) * 0.5

    def mk(vl):
        return _siglip.Module(
            num_classes=768, variant="B/16", pool_type="none", scan=True, dtype_mm="float32", vision_lora=vl
        )

    lora_mod = mk(_all_targets())
    v_l = lora_mod.init(jax.random.key(1), img, train=False)
    fl = flax.traverse_util.flatten_dict(v_l["params"], sep="/")
    lora_keys = sorted(k for k in fl if "lora" in k)
    zb = [k for k in lora_keys if "lora_b" in k]
    chk(
        len(lora_keys) > 0,
        f"Vision tower LoRA keys={len(lora_keys)} (MHA 8 + MLP 4 + head 2 + patch 2, expanded by shape)",
    )
    chk(all(bool((fl[k] == 0).all()) for k in zb), f"All {len(zb)} lora_b parameters are exactly zero")

    y_l, _ = jax.jit(lambda v, im: lora_mod.apply(v, im, train=False))(v_l, img)
    y_l = np.asarray(y_l)

    base_mod = mk(None)
    v_b = base_mod.init(jax.random.key(0), img, train=False)
    fb = flax.traverse_util.flatten_dict(v_b["params"], sep="/")
    chk(set(fl) - set(lora_keys) == set(fb), "Adapter tower base keys match the base tower")
    for k in fb:
        fb[k] = fl[k]
    v_b = {"params": flax.traverse_util.unflatten_dict(fb, sep="/")}
    y_b, _ = jax.jit(lambda v, im: base_mod.apply(v, im, train=False))(v_b, img)
    y_b = np.asarray(y_b)

    bit = bool((y_b == y_l).all())
    md = float(np.abs(y_b - y_l).max())
    chk(bit or md < 1e-6, f"Tower output (after head): bit-exact={bit}, max|Δ|={md:.3e}")

    for k in zb:
        fl[k] = jnp.full_like(fl[k], 0.01)
    y_n, _ = lora_mod.apply({"params": flax.traverse_util.unflatten_dict(fl, sep="/")}, img, train=False)
    chk(bool((np.asarray(y_n) != y_l).any()), "B!=0 changes the tower output")


def make_cfg(lora_on: bool) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        action_horizon=50,
        use_odeaction=True,
        odeaction_samples_per_segment=2,
        noise_schedule="uniform",
        odeaction_correction_gain=5.0,
        odeaction_neighborhood_scale=0.1,
        odeaction_use_correction=True,
        vlm_lora=(_all_targets() if lora_on else vlm_lora.VlmLoRAConfig()),
    )


def flat_state(model):
    return {"/".join(str(p) for p in path): v for path, v in nnx.state(model, nnx.Param).flat_state().items()}


@nnx.jit
def _img_fwd(model, im):
    y, _ = model.PaliGemma.img(im, train=False)
    return y


@nnx.jit
def _loss_fwd(model, rng, obs, act):
    return jnp.mean(model.compute_loss(rng, obs, act, train=False))


def run_full() -> None:
    print("== [full] Pi0(gemma_2b+So400m) zero-B equivalence with sequential construction (GPU) ==")
    img = jnp.ones((1, 224, 224, 3), dtype=jnp.float32) * 0.5
    lora_cfg = make_cfg(lora_on=True)
    obs, act = lora_cfg.fake_obs(1), lora_cfg.fake_act(1)

    obs = _dc.replace(obs, initial_action=obs.state)

    lora_m = lora_cfg.create(jax.random.key(1))
    fl = flat_state(lora_m)
    lora_keys = sorted(k for k in fl if "lora" in k)
    zb = [k for k in lora_keys if "lora_b" in k]
    chk(all(bool((fl[k].value == 0).all()) for k in zb), f"All {len(zb)} lora_b parameters are exactly zero")
    base_weights = {k: v.value for k, v in fl.items() if "lora" not in k}
    yl = np.asarray(_img_fwd(lora_m, img))
    ll = float(_loss_fwd(lora_m, jax.random.key(7), obs, act))
    del lora_m, fl
    gc.collect()

    base_m = make_cfg(lora_on=False).create(jax.random.key(0))
    fb = flat_state(base_m)
    chk(set(base_weights) == set(fb), f"Base parameter keys match ({len(lora_keys)} adapter keys added)")
    st = nnx.state(base_m, nnx.Param)
    fs = dict(st.flat_state().items())
    for path, v in fs.items():
        fs[path] = v.replace(base_weights["/".join(str(p) for p in path)])
    nnx.update(base_m, nnx.State.from_flat_path(fs))
    del base_weights, fs, st
    gc.collect()

    yb = np.asarray(_img_fwd(base_m, img))
    bit = bool((yb == yl).all())
    md = float(np.abs(yb.astype(np.float32) - yl.astype(np.float32)).max())
    chk(bit or md < 1e-6, f"Vision tower forward pass: bit-exact={bit}, max|Δ|={md:.3e}")

    lb = float(_loss_fwd(base_m, jax.random.key(7), obs, act))
    md2 = abs(lb - ll)
    chk(
        lb == ll or md2 < 1e-2,
        f"Full compute_loss: bit-exact={lb == ll}, |Δ|={md2:.3e}  (base={lb:.6f}, lora={ll:.6f})",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["siglip", "full"], default="siglip")
    args = ap.parse_args()
    if args.level == "siglip":
        run_siglip()
    else:
        run_full()
    print(
        "\n" + (f"✅ verify_vlm_lora_zero_init(--level {args.level}): All checks passed" if OK else "❌ Checks failed")
    )
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
