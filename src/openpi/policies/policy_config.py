import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
from openpi.policies.inference_config import ODEInferenceConfig
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    ode_config: ODEInferenceConfig | None = None,
) -> _policy.Policy:
    """Load a JAX checkpoint and apply ODE options before compiling the policy."""
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)

    import json as _json
    import pathlib as _pathlib

    import flax.nnx as _nnx
    import flax.traverse_util as _traverse_util
    import jax as _jax
    import numpy as _np

    _flat_ckpt = _traverse_util.flatten_dict(params, sep="/")
    _ckpt_lora = {k: tuple(_np.shape(v)) for k, v in _flat_ckpt.items() if "lora" in k}
    _cfg_expects_lora = (
        "lora" in str(getattr(train_config.model, "paligemma_variant", ""))
        or "lora" in str(getattr(train_config.model, "action_expert_variant", ""))
        or (getattr(train_config.model, "vlm_lora", None) is not None and train_config.model.vlm_lora.enabled)
    )
    if _ckpt_lora or _cfg_expects_lora:
        _abs_model = _nnx.eval_shape(train_config.model.create, _jax.random.key(0))
        _expected = {
            k: tuple(v.shape)
            for k, v in _traverse_util.flatten_dict(_nnx.split(_abs_model)[1].to_pure_dict(), sep="/").items()
            if "lora" in k
        }
        _extra = sorted(set(_ckpt_lora) - set(_expected))
        _missing = sorted(set(_expected) - set(_ckpt_lora))
        _bad = sorted(k for k in _expected if k in _ckpt_lora and _expected[k] != _ckpt_lora[k])
        if _extra or _missing or _bad:
            raise ValueError(
                f"[VLM-LoRA] Checkpoint LoRA parameters do not match config '{train_config.name}': "
                "loading stopped; "
                f"{len(_extra)} extra keys, examples {_extra[:3]}; "
                f"{len(_missing)} missing keys, examples {_missing[:3]}; "
                f"{len(_bad)} shape mismatches, examples {[(k, _expected[k], _ckpt_lora[k]) for k in _bad[:2]]}. "
                "Use the LoRA configuration from training. "
                "Check with tests/manual/verify_vlm_lora_ckpt_contract.py."
            )

        _cd = _pathlib.Path(str(checkpoint_dir))
        _sidecar = next(
            (p for p in (_cd / "vlm_lora_contract.json", _cd.parent / "vlm_lora_contract.json") if p.exists()),
            None,
        )
        if _sidecar is not None:
            import openpi.models.vlm_lora as _vlm_lora_mod

            _saved = _json.loads(_sidecar.read_text())
            _cur = _vlm_lora_mod.contract_dict(train_config.model.vlm_lora)
            if _saved.get("vlm_lora") != _cur:
                raise ValueError(
                    f"[VLM-LoRA] {_sidecar} does not match the configured LoRA contract "
                    f"(alpha, rslora, target topology). Saved={_saved.get('vlm_lora')}, configured={_cur}"
                )
        elif _ckpt_lora:
            logging.warning(
                "[VLM-LoRA] vlm_lora_contract.json not found. "
                "Parameter keys and shapes match; alpha and rslora are taken from the configuration. "
                "Use the configuration from training."
            )
    model = train_config.model.load(params)
    if ode_config is not None:
        ode_config.apply(model)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            transforms.NormalizeInitialAction(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
