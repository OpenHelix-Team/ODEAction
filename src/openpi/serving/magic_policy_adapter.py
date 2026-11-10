from __future__ import annotations

import dataclasses
import enum
import logging
import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)

_MODEL_GRIP_LEFT = 6
_MODEL_GRIP_RIGHT = 13

_WIRE_GRIP_RIGHT = 6
_WIRE_GRIP_LEFT = 13

_MODEL_JOINT_IDX = [i for i in range(14) if i not in (_MODEL_GRIP_LEFT, _MODEL_GRIP_RIGHT)]

_STATE_DIM = 26
_ACTION_DIM = 14
_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_NATIVE_CAM_MAP = {
    "observation/image_top": "cam_high",
    "observation/image_left": "cam_left_wrist",
    "observation/image_right": "cam_right_wrist",
}

_DISTINGUISH_EPS = 0.02


class InputSchema(str, enum.Enum):
    NATIVE = "NATIVE"  # observation/state + observation/image_top|left|right
    ROBODRIVER = "ROBODRIVER"  # state + images{cam_high,cam_left_wrist,cam_right_wrist}


class OutputActionOrder(str, enum.Enum):
    NATIVE = "NATIVE"
    LEFT_THEN_RIGHT = "LEFT_THEN_RIGHT"


class MagicObservationError(ValueError):
    """An observation does not satisfy the Cobot Magic interface."""


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
def _to_hwc_uint8(img: Any, name: str) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise MagicObservationError(f"{name}: expected a 3-dimensional image, got shape={arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise MagicObservationError(f"{name}: image contains non-finite values")
        arr = (255.0 * np.clip(arr, 0.0, 1.0)).astype(np.uint8)
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.shape[-1] != 3:
        raise MagicObservationError(f"{name}: expected 3 channels (HWC or CHW), got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _check_state(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.shape[0] != _STATE_DIM:
        raise MagicObservationError(f"state: expected {_STATE_DIM} dimensions, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise MagicObservationError(f"state: non-finite values at indices {np.where(~np.isfinite(arr))[0].tolist()}")
    return arr


def _check_prompt(prompt: Any) -> str:
    if not isinstance(prompt, (str, bytes)):
        raise MagicObservationError(f"prompt: must be a string, got {type(prompt).__name__}")
    text = prompt.decode() if isinstance(prompt, bytes) else prompt
    if not text.strip():
        raise MagicObservationError("prompt: must not be empty")
    return text


def _state26_to_model14(state26: np.ndarray) -> np.ndarray:
    """state26 to model14."""
    return np.concatenate([state26[13:20], state26[0:7]], axis=-1).astype(np.float32)


def _wire_to_model(actions_wire: np.ndarray) -> np.ndarray:
    """wire to model."""
    return np.concatenate([actions_wire[..., 7:14], actions_wire[..., 0:7]], axis=-1)


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class MagicPolicyAdapter(_base_policy.BasePolicy):
    """Adapt robot observations and maintain executed gripper commands between requests."""

    policy: _base_policy.BasePolicy
    is_odeaction: bool
    input_schema: InputSchema = InputSchema.ROBODRIVER
    output_action_order: OutputActionOrder = OutputActionOrder.LEFT_THEN_RIGHT
    action_horizon: int = 50
    replan_steps: int = 50
    supported_prompts: tuple = ()
    default_prompt: str | None = None
    jump_warn_rad: float = 0.5
    jump_reset_rad: float = 0.0
    idle_reset_sec: float = 300.0

    _prev_chunk: Any = dataclasses.field(default=None, init=False, repr=False)
    _gripper_override: Any = dataclasses.field(default=None, init=False, repr=False)
    _last_request_ts: Any = dataclasses.field(default=None, init=False, repr=False)
    _step_count: int = dataclasses.field(default=0, init=False, repr=False)
    _reset_count: int = dataclasses.field(default=0, init=False, repr=False)
    _last_reset_reason: str = dataclasses.field(default="init", init=False, repr=False)
    _last_diag: dict = dataclasses.field(default_factory=dict, init=False, repr=False)
    _warned_prompts: set = dataclasses.field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        if self.default_prompt is not None:
            self.default_prompt = _check_prompt(self.default_prompt)
        if not (1 <= self.replan_steps <= self.action_horizon):
            raise ValueError(f"replan_steps must be in [1, {self.action_horizon}], got {self.replan_steps}")

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def reset(self, reason: str = "manual") -> None:
        """reset."""
        had_state = self._prev_chunk is not None or self._gripper_override is not None
        self._prev_chunk = None
        self._gripper_override = None
        self._last_request_ts = None
        self._last_reset_reason = reason
        self._last_diag = {}
        self._step_count = 0
        self._reset_count += 1
        logger.info(
            "[MagicPolicyAdapter] Session reset (reason=%s, previous state=%s); next inference uses init_gripper_open",
            reason,
            "present" if had_state else "empty",
        )

    @property
    def last_gripper_cmd(self):
        """last gripper cmd."""
        g = self._resolve_gripper()
        return None if g is None else g.copy()

    def status(self) -> dict:
        """status."""
        g = self.last_gripper_cmd
        return {
            "is_odeaction": self.is_odeaction,
            "input_schema": self.input_schema.value,
            "output_action_order": self.output_action_order.value,
            "action_horizon": self.action_horizon,
            "replan_steps": self.replan_steps if self.is_odeaction else None,
            "steps_since_reset": self._step_count,
            "reset_count": self._reset_count,
            "last_reset_reason": self._last_reset_reason,
            "has_prev_chunk": self._prev_chunk is not None,
            "last_gripper_cmd": None if g is None else [float(x) for x in g],
            "last_diagnostics": self._last_diag,
        }

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def _extract_gripper_from_client_row(self, row: np.ndarray) -> np.ndarray:
        """extract gripper from client row."""
        if self.output_action_order is OutputActionOrder.LEFT_THEN_RIGHT:
            return np.array([row[6], row[13]], dtype=np.float32)
        return np.array([row[_WIRE_GRIP_LEFT], row[_WIRE_GRIP_RIGHT]], dtype=np.float32)

    @staticmethod
    def _gripper_from_native_row(row: np.ndarray) -> np.ndarray:
        """gripper from native row."""
        return np.array([row[_WIRE_GRIP_LEFT], row[_WIRE_GRIP_RIGHT]], dtype=np.float32)

    def _resolve_gripper(self):
        """resolve gripper."""
        if self._gripper_override is not None:
            return self._gripper_override
        if self._prev_chunk is not None:
            return self._gripper_from_native_row(self._prev_chunk[self.replan_steps - 1])
        return None

    def _ingest_client_gripper(self, obs: dict) -> None:
        row = obs.get("last_executed_action")
        if row is not None:
            arr = np.asarray(row, dtype=np.float32).reshape(-1)
            if arr.shape[0] != _ACTION_DIM or not np.all(np.isfinite(arr)):
                raise MagicObservationError(
                    f"last_executed_action: expected {_ACTION_DIM} finite values"
                    f" in server output order, got shape={np.asarray(row).shape}"
                )
            self._gripper_override = self._extract_gripper_from_client_row(arr)
            return
        lgc = obs.get("last_gripper_cmd")
        if lgc is not None:
            arr = np.asarray(lgc, dtype=np.float32).reshape(-1)
            if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
                raise MagicObservationError(
                    f"last_gripper_cmd: expected two finite values [left, right], got {np.asarray(lgc).tolist()}"
                )
            self._gripper_override = arr[:2].astype(np.float32)

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def _diagnose(self, state26: np.ndarray) -> dict:
        if self._prev_chunk is None:
            return {}
        state_j = _state26_to_model14(state26)[_MODEL_JOINT_IDX]
        chunk_j = _wire_to_model(self._prev_chunk)[:, _MODEL_JOINT_IDX]  # (H, 12)
        dists = np.linalg.norm(chunk_j - state_j[None, :], axis=-1)  # (H,)
        k_match = int(np.argmin(dists))
        k_declared = self.replan_steps - 1
        d_declared = float(dists[k_declared])
        distinguish = float(np.linalg.norm(chunk_j[k_match] - chunk_j[k_declared]))

        diag = {
            "matched_row": k_match,
            "declared_row": k_declared,
            "dist_at_declared_row": round(d_declared, 5),
            "dist_at_matched_row": round(float(dists[k_match]), 5),
            "row_distinguishability": round(distinguish, 5),
        }

        if k_match != k_declared and distinguish > _DISTINGUISH_EPS:
            diag["replan_mismatch"] = True
            logger.warning(
                "[MagicPolicyAdapter] Replan mismatch: replan_steps=%d (row %d), "
                "but measured joints best match row %d (distinguishability %.4f). "
                "Match the client execution length to --replan-steps.",
                self.replan_steps,
                k_declared,
                k_match,
                distinguish,
            )

        if self.jump_warn_rad > 0 and d_declared > self.jump_warn_rad:
            diag["jump"] = True
            if self.jump_reset_rad > 0 and d_declared > self.jump_reset_rad:
                logger.warning(
                    "[MagicPolicyAdapter] Joint displacement %.3f > %.3f; resetting session.",
                    d_declared,
                    self.jump_reset_rad,
                )
                self.reset(reason="jump")
                diag["jump_reset"] = True
            else:
                logger.warning(
                    "[MagicPolicyAdapter] Joint displacement %.3f > %.3f. For a new episode, "
                    "call HTTP /reset. Automatic reset is disabled.",
                    d_declared,
                    self.jump_warn_rad,
                )
        return diag

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def _build_inner_obs(self, obs: dict):
        if self.input_schema is InputSchema.NATIVE:
            if "observation/state" not in obs:
                raise MagicObservationError(
                    "NATIVE schema requires 'observation/state'. For RoboDriver observations, "
                    "start with --input-schema ROBODRIVER. Received keys: " + str(sorted(obs.keys()))
                )
            state = _check_state(obs["observation/state"])
            images = {}
            for src, dest in _NATIVE_CAM_MAP.items():
                if src not in obs:
                    raise MagicObservationError(f"NATIVE schema is missing camera '{src}'")
                images[dest] = _to_hwc_uint8(obs[src], src)
        else:
            if "state" not in obs:
                raise MagicObservationError(
                    "ROBODRIVER schema requires 'state'. For native observations, "
                    "start with --input-schema NATIVE. Received keys: " + str(sorted(obs.keys()))
                )
            state = _check_state(obs["state"])
            raw_images = obs.get("images")
            if not isinstance(raw_images, dict):
                raise MagicObservationError("ROBODRIVER schema requires an 'images' dictionary")
            images = {}
            for cam in _CAMS:
                if cam not in raw_images:
                    raise MagicObservationError(
                        f"ROBODRIVER schema is missing camera 'images/{cam}'; received {sorted(raw_images.keys())}"
                    )
                images[cam] = _to_hwc_uint8(raw_images[cam], f"images/{cam}")

        prompt = _check_prompt(obs.get("prompt", self.default_prompt))
        if self.supported_prompts and prompt not in self.supported_prompts and prompt not in self._warned_prompts:
            self._warned_prompts.add(prompt)
            logger.warning(
                "[MagicPolicyAdapter] Prompt %r is outside the example task set %s; continuing inference.",
                prompt,
                list(self.supported_prompts),
            )
        return {"state": state, "images": images, "prompt": prompt}, state

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def _check_actions(self, actions: Any) -> np.ndarray:
        arr = np.asarray(actions)
        if arr.ndim != 2 or arr.shape[1] != _ACTION_DIM:
            raise MagicObservationError(f"Invalid action shape: expected [H,{_ACTION_DIM}], got {arr.shape}")
        if arr.shape[0] != self.action_horizon:
            raise MagicObservationError(f"Invalid action horizon: expected {self.action_horizon}, got {arr.shape[0]}")
        arr = arr.astype(np.float32)
        if not np.all(np.isfinite(arr)):
            raise MagicObservationError(f"Model output contains {int((~np.isfinite(arr)).sum())} non-finite values")
        return arr

    @staticmethod
    def _swap_lr(actions: np.ndarray) -> np.ndarray:
        """swap lr."""
        return np.concatenate([actions[:, 7:14], actions[:, 0:7]], axis=-1)

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def infer(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            raise MagicObservationError(f"observation must be a dict, got {type(obs).__name__}")

        now = time.monotonic()

        if (
            self.idle_reset_sec > 0
            and self._last_request_ts is not None
            and now - self._last_request_ts > self.idle_reset_sec
        ):
            logger.info(
                "[MagicPolicyAdapter] Idle time %.1f seconds > %.1f seconds; starting a new episode.",
                now - self._last_request_ts,
                self.idle_reset_sec,
            )
            self.reset(reason="idle_timeout")

        if bool(obs.get("reset", False)):
            self.reset(reason="obs_field")

        inner, state26 = self._build_inner_obs(obs)

        diag = {}
        if self.is_odeaction:
            diag = self._diagnose(state26)
            self._ingest_client_gripper(obs)
            grip = self._resolve_gripper()
            if grip is not None:
                inner["last_gripper_cmd"] = grip.copy()

        result = self.policy.infer(inner)
        native_actions = self._check_actions(result["actions"])

        if self.is_odeaction:
            self._prev_chunk = native_actions.copy()
            self._gripper_override = None

        out_actions = (
            self._swap_lr(native_actions)
            if self.output_action_order is OutputActionOrder.LEFT_THEN_RIGHT
            else native_actions
        )

        self._step_count += 1
        self._last_request_ts = now
        self._last_diag = diag

        out = {k: v for k, v in result.items() if k != "actions"}
        out["actions"] = np.ascontiguousarray(out_actions, dtype=np.float32)
        magic_info = {
            "step": self._step_count,
            "action_order": self.output_action_order.value,
            "input_schema": self.input_schema.value,
            "is_odeaction": self.is_odeaction,
        }
        if self.is_odeaction:
            g = self._resolve_gripper()
            magic_info["replan_steps"] = self.replan_steps
            magic_info["last_gripper_cmd"] = None if g is None else [float(x) for x in g]
            if diag:
                magic_info["diagnostics"] = diag
        out["magic_server"] = magic_info
        return out
