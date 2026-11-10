#!/usr/bin/env python3

import sys

import flax.nnx as nnx
import flax.traverse_util as tu
import jax
import numpy as np

import openpi.models.vlm_lora as vlm_lora
import openpi.training.config as _config

DEV = "odeaction_magic_lora"
BASELINES = ["pi05_magic_full", "odeaction_magic_full"]

OK = True


def chk(cond, msg):
    global OK
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    OK = OK and bool(cond)


def abstract_flat(train_config):
    m = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    return tu.flatten_dict(nnx.split(m)[1].to_pure_dict(), sep="/"), m


def main() -> int:
    print(f"== [1] {DEV}: Abstract parameter groups and counts ==")
    cfg = _config.get_config(DEV)
    flat, model = abstract_flat(cfg)
    groups = {}
    invalid = []
    for k, v in flat.items():
        g = vlm_lora.classify_param(k)
        groups.setdefault(g.value, [0, 0])
        groups[g.value][0] += 1
        groups[g.value][1] += int(np.prod(v.shape))
        if g == vlm_lora.ParamGroup.INVALID:
            invalid.append(k)
    for g, (n, p) in sorted(groups.items()):
        print(f"    {g:<18} leaves={n:>4}  params={p:>14,}")
    chk(not invalid, f"No INVALID classifications (unexpected paths: {invalid[:3]})")
    lora_keys = sorted(k for k in flat if "lora" in k)
    chk(
        len(lora_keys) > 0 and groups.get("vlm_lora", [0, 0])[0] == len(lora_keys),
        f"LoRA keys={len(lora_keys)}, all in the vlm_lora group",
    )
    print(f"    INFO  vlm_all_linear r32 LoRA parameter count = {groups.get('vlm_lora', [0, 0])[1]:,}")
    n_lang = sum(1 for k in lora_keys if k.startswith("PaliGemma/llm"))
    n_vis = sum(1 for k in lora_keys if k.startswith("PaliGemma/img"))
    chk(n_lang > 0 and n_vis > 0, f"Language LoRA keys={n_lang}, vision LoRA keys={n_vis}; both towers have adapters")
    chk(
        not any("_1/lora" in k or "_1_lora" in k for k in lora_keys if k.startswith("PaliGemma/llm")),
        "Action expert (expert1, *_1) has no LoRA adapters",
    )

    print("== [2] freeze_filter agrees with classify ==")
    frozen_by_filter = set(
        "/".join(str(p) for p in path) for path, _ in nnx.state(model, cfg.freeze_filter).flat_state().items()
    )
    frozen_by_classify = {k for k in flat if vlm_lora.classify_param(k) == vlm_lora.ParamGroup.FROZEN_VLM_BASE}
    chk(
        frozen_by_filter == frozen_by_classify,
        f"Filter matches FROZEN classification(filter={len(frozen_by_filter)}, classify={len(frozen_by_classify)})",
    )

    print("== [3] Optimizer labels agree with classify (disjoint trainable groups) ==")
    trainable = {k for k in flat if k not in frozen_by_classify}
    lab = {("vlm_lora" if "lora" in k else "action") for k in trainable}
    mism = [
        k
        for k in trainable
        if ("vlm_lora" if "lora" in k else "action")
        != ("vlm_lora" if vlm_lora.classify_param(k) == vlm_lora.ParamGroup.VLM_LORA else "action")
    ]
    chk(
        not mism and lab == {"vlm_lora", "action"},
        f"Labels agree with classify and both groups are non-empty (mismatches: {mism[:3]})",
    )

    print("== [4] Full fine-tuning configurations: no LoRA parameters or enabled adapters ==")
    for name in BASELINES:
        try:
            c = _config.get_config(name)
        except ValueError:
            print(f"    SKIP  {name} (configuration not found)")
            continue
        f, _ = abstract_flat(c)
        no_lora = not any("lora" in k for k in f)
        off = not c.model.vlm_lora.enabled
        chk(no_lora and off, f"{name}: no LoRA keys={no_lora}, vlm_lora.enabled=False={off}, leaves={len(f)}")

    print("\n" + ("✅ verify_vlm_lora_targets: All checks passed" if OK else "❌ Checks failed"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
