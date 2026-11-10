#!/usr/bin/env python3

# uv run tests/manual/verify_vlm_lora_ckpt_contract.py --config odeaction_magic_lora --ckpt /path/to/checkpoint

import argparse
import pathlib
import sys

import flax.nnx as nnx
import flax.traverse_util as tu
import jax
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config

OK = True


def chk(cond, msg):
    global OK
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    OK = OK and bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="Checkpoint step directory or its params directory")
    args = ap.parse_args()

    cfg = _config.get_config(args.config)
    m = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    expect = tu.flatten_dict(nnx.split(m)[1].to_pure_dict(), sep="/")
    expect_lora = {k: tuple(v.shape) for k, v in expect.items() if "lora" in k}
    print(
        f"config '{args.config}': expected LoRA keys={len(expect_lora)} from the abstract model tree, "
        f"vlm_lora.enabled={cfg.model.vlm_lora.enabled}"
    )

    p = pathlib.Path(args.ckpt)
    params_path = p if p.name == "params" else (p / "params")
    got = tu.flatten_dict(_model.restore_params(params_path, restore_type=np.ndarray), sep="/")
    got_lora = {k: tuple(np.shape(v)) for k, v in got.items() if "lora" in k}
    print(f"Checkpoint: parameter keys={len(got)}, LoRA keys={len(got_lora)}")

    if got_lora:
        print("== Mode A: LoRA checkpoint contract ==")
        missing = sorted(set(expect_lora) - set(got_lora))
        extra = sorted(set(got_lora) - set(expect_lora))
        chk(not missing, f"No missing LoRA keys (missing={len(missing)}: {missing[:3]})")
        chk(not extra, f"No extra LoRA keys (extra={len(extra)}: {extra[:3]})")
        bad = [k for k in expect_lora if k in got_lora and expect_lora[k] != got_lora[k]]
        chk(
            not bad,
            f"All LoRA shapes match (mismatches={len(bad)}: {[(k, expect_lora[k], got_lora[k]) for k in bad[:2]]})",
        )
        non_lora_missing = sorted(k for k in expect if "lora" not in k and k not in got)
        chk(not non_lora_missing, f"No missing base keys (missing={len(non_lora_missing)}: {non_lora_missing[:3]})")

        import json

        import openpi.models.vlm_lora as vlm_lora

        sidecar = next(
            (
                q
                for q in (
                    p / "vlm_lora_contract.json",
                    p.parent / "vlm_lora_contract.json",
                    params_path.parent / "vlm_lora_contract.json",
                    params_path.parent.parent / "vlm_lora_contract.json",
                )
                if q.exists()
            ),
            None,
        )
        if sidecar is not None:
            saved = json.loads(sidecar.read_text())
            chk(
                saved.get("vlm_lora") == vlm_lora.contract_dict(cfg.model.vlm_lora),
                f"Sidecar contract (alpha, rslora, topology) matches config: {sidecar}",
            )
        else:
            print("    WARN  vlm_lora_contract.json not found; alpha/rslora are not stored in parameter arrays")
    else:
        print("== Mode B: Base checkpoint initialization ==")
        missing = sorted(k for k in expect if k not in got)
        only_lora = all("lora" in k for k in missing)
        chk(
            set(missing) == set(expect_lora) and only_lora,
            f"Missing keys ({len(missing)}) are exactly the LoRA keys; weight_loader uses model initialization"
            f"(A=normal(0.01), B=zeros, zero adapter output at step 0)",
        )
        chk(cfg.model.vlm_lora.enabled or not expect_lora, "Configuration and expected LoRA keys agree")

    print("\n" + ("✅ verify_vlm_lora_ckpt_contract: All checks passed" if OK else "❌ Checks failed"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
