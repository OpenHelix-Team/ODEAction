"""VLM adapter targets, frozen parameter groups, and checkpoint scaling metadata."""

from collections.abc import Callable
import dataclasses
import enum
import math

import flax.linen as _nn_init_only
import flax.nnx as nnx
import jax.tree_util as jtu

import openpi.models.lora as lora


@dataclasses.dataclass(frozen=True)
class LoRATargetConfig:
    """Rank, scaling, and initialization of one adapter target."""

    enabled: bool = False
    rank: int = 16
    alpha: float = 16.0
    rslora: bool = False

    zero_b: bool = True

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


@dataclasses.dataclass(frozen=True)
class VlmLoRAConfig:
    """Component-wise adapters for the language and vision backbones."""

    enabled: bool = False

    language_attention: LoRATargetConfig = dataclasses.field(default_factory=LoRATargetConfig)
    language_ffn: LoRATargetConfig = dataclasses.field(default_factory=LoRATargetConfig)

    vision_attention: LoRATargetConfig = dataclasses.field(default_factory=LoRATargetConfig)
    vision_mlp: LoRATargetConfig = dataclasses.field(default_factory=LoRATargetConfig)
    vision_projector: LoRATargetConfig = dataclasses.field(default_factory=LoRATargetConfig)

    vision_patch_embedding: LoRATargetConfig = dataclasses.field(default_factory=LoRATargetConfig)

    def any_vision_enabled(self) -> bool:
        return self.enabled and any(
            t.enabled
            for t in (self.vision_attention, self.vision_mlp, self.vision_projector, self.vision_patch_embedding)
        )

    def any_language_enabled(self) -> bool:
        return self.enabled and (self.language_attention.enabled or self.language_ffn.enabled)


def preset_vlm_all_linear(*, rank: int = 16, alpha: float = 16.0, rslora: bool = False) -> VlmLoRAConfig:
    """preset vlm all linear."""
    t = LoRATargetConfig(enabled=True, rank=rank, alpha=alpha, rslora=rslora, zero_b=True)
    return VlmLoRAConfig(
        enabled=True,
        language_attention=t,
        language_ffn=t,
        vision_attention=t,
        vision_mlp=t,
        vision_projector=t,
        vision_patch_embedding=LoRATargetConfig(enabled=False),
    )


def preset_language_all(*, rank: int = 16, alpha: float = 16.0, rslora: bool = False) -> VlmLoRAConfig:
    t = LoRATargetConfig(enabled=True, rank=rank, alpha=alpha, rslora=rslora, zero_b=True)
    return VlmLoRAConfig(enabled=True, language_attention=t, language_ffn=t)


def preset_vision_all_linear(*, rank: int = 16, alpha: float = 16.0, rslora: bool = False) -> VlmLoRAConfig:
    t = LoRATargetConfig(enabled=True, rank=rank, alpha=alpha, rslora=rslora, zero_b=True)
    return VlmLoRAConfig(enabled=True, vision_attention=t, vision_mlp=t, vision_projector=t)


LORA_A_INIT = _nn_init_only.initializers.normal(stddev=0.01)
LORA_B_ZERO_INIT = _nn_init_only.initializers.zeros_init()


def make_language_lora_configs(cfg: VlmLoRAConfig) -> dict[str, lora.LoRAConfig]:
    """make language lora configs."""
    out: dict[str, lora.LoRAConfig] = {}

    def _mk(t: LoRATargetConfig) -> lora.LoRAConfig:
        return lora.LoRAConfig(
            rank=t.rank,
            alpha=t.alpha,
            rslora=t.rslora,
            init_fn_a=LORA_A_INIT,
            init_fn_b=LORA_B_ZERO_INIT if t.zero_b else None,
            apply_ffn_scaling=True,
        )

    if cfg.language_attention.enabled:
        out["attn"] = _mk(cfg.language_attention)
    if cfg.language_ffn.enabled:
        out["ffn"] = _mk(cfg.language_ffn)
    return out


class ParamGroup(enum.Enum):
    FROZEN_VLM_BASE = "frozen_vlm_base"
    VLM_LORA = "vlm_lora"
    ACTION = "action"
    INVALID = "invalid"


_TOP_LEVEL_ACTION_PREFIXES = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "state_proj")


def classify_param(path: str) -> ParamGroup:
    """Assign a parameter path to VLM base, VLM adapters, or the action expert."""
    if "lora" in path:
        return ParamGroup.VLM_LORA
    parts = path.split("/")
    if path.startswith("PaliGemma/img"):
        return ParamGroup.FROZEN_VLM_BASE
    if path.startswith("PaliGemma/llm"):
        if any(p.endswith("_1") for p in parts):
            return ParamGroup.ACTION
        return ParamGroup.FROZEN_VLM_BASE
    if parts[0] in _TOP_LEVEL_ACTION_PREFIXES:
        return ParamGroup.ACTION
    return ParamGroup.INVALID


def _nnx_path_to_str(path) -> str:
    """nnx path to str."""
    return "/".join(str(p) for p in path)


def _jtu_path_to_str(path) -> str:
    """jtu path to str."""
    out = []
    for k in path:
        if hasattr(k, "key"):
            out.append(str(k.key))
        elif hasattr(k, "name"):
            out.append(str(k.name))
        elif hasattr(k, "idx"):
            out.append(str(k.idx))
        else:
            out.append(str(k))
    return "/".join(out)


@dataclasses.dataclass(frozen=True)
class FreezeVlmBaseFilter:
    """FreezeVlmBaseFilter."""

    def __call__(self, path: nnx.filterlib.PathParts, x) -> bool:
        return classify_param(_nnx_path_to_str(path)) == ParamGroup.FROZEN_VLM_BASE


def make_freeze_filter() -> nnx.filterlib.Filter:
    """Freeze the VLM base while training its adapters and the full action expert."""
    return nnx.All(nnx.Param, FreezeVlmBaseFilter())


def _label_for_path(path_str: str) -> str:
    """label for path."""
    g = classify_param(path_str)
    if g == ParamGroup.VLM_LORA:
        return "vlm_lora"
    if g == ParamGroup.ACTION:
        return "action"
    raise ValueError(
        f"[VLM-LoRA] Trainable parameter '{path_str}' was classified as {g.value}. "
        "The freeze filter does not match the parameter tree. "
        "Inspect the targets with tests/manual/verify_vlm_lora_targets.py."
    )


def optimizer_label_tree(tree):
    """optimizer label tree."""
    return jtu.tree_map_with_path(lambda p, v: _label_for_path(_jtu_path_to_str(p)), tree)


def group_leaves(tree, group_label: str):
    """group leaves."""
    return [v for p, v in jtu.tree_leaves_with_path(tree) if _label_for_path(_jtu_path_to_str(p)) == group_label]


def summarize_flat_params(flat: dict[str, "object"]) -> dict[str, tuple[int, int]]:
    """summarize flat params."""
    import numpy as np

    stats: dict[str, list[int]] = {}
    for k, v in flat.items():
        g = classify_param(k).value
        stats.setdefault(g, [0, 0])
        stats[g][0] += 1
        stats[g][1] += int(np.prod(v.shape))
    return {k: (v[0], v[1]) for k, v in stats.items()}


def lora_paths(flat_keys) -> set[str]:
    return {k for k in flat_keys if "lora" in k}


def validate_model_config(cfg: VlmLoRAConfig, paligemma_variant: str, action_expert_variant: str) -> None:
    """validate model config."""
    if not cfg.enabled:
        return
    targets = {
        "language_attention": cfg.language_attention,
        "language_ffn": cfg.language_ffn,
        "vision_attention": cfg.vision_attention,
        "vision_mlp": cfg.vision_mlp,
        "vision_projector": cfg.vision_projector,
        "vision_patch_embedding": cfg.vision_patch_embedding,
    }
    enabled = {n: t for n, t in targets.items() if t.enabled}
    if not enabled:
        raise ValueError("[VLM-LoRA] vlm_lora.enabled=True requires at least one enabled target.")
    for n, t in enabled.items():
        if t.rank < 1 or t.alpha <= 0:
            raise ValueError(
                f"[VLM-LoRA] Invalid target {n}: rank={t.rank} (must be >=1), alpha={t.alpha} (must be >0)."
            )
    for vname, v in (("paligemma_variant", paligemma_variant), ("action_expert_variant", action_expert_variant)):
        if "lora" in str(v):
            raise ValueError(
                f"[VLM-LoRA] Cannot combine vlm_lora with a legacy LoRA variant: {vname}={v!r}. "
                "Legacy variants use random/random initialization and unscaled FFN adapters; "
                "vlm_lora uses zero-B initialization and scaled adapters."
            )


ExpectedLabels = Callable[[object], object]


def contract_dict(cfg: VlmLoRAConfig) -> dict:
    """contract dict."""
    return dataclasses.asdict(cfg)


def contract_fingerprint(cfg: VlmLoRAConfig) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(contract_dict(cfg), sort_keys=True).encode()).hexdigest()
