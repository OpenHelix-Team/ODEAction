import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import magic_policy

_GRIP_LEFT = 6
_GRIP_RIGHT = 13


def make_magic_odeaction_example() -> dict:
    """make magic odeaction example."""
    H = 50
    return {
        "state": np.ones((26,), dtype=np.float32),
        "actions": np.random.rand(H + 1, 14).astype(np.float32),
        "images/cam_high": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "prompt": "put the spoon into the cup",
    }


@dataclasses.dataclass(frozen=True)
class MagicODEActionInputs(transforms.DataTransformFn):
    """Build absolute action targets and an initial action with executed gripper commands."""

    init_gripper_open: float = 1.05

    def __call__(self, data: dict) -> dict:
        state_model = np.asarray(magic_policy.magic_state_to_model(np.asarray(data["state"])), dtype=np.float32)

        base = magic_policy._get_camera(data, "cam_high")
        if base is None:
            raise KeyError("MagicODEActionInputs: 'images/cam_high' not found; got " + str(sorted(data.keys())))
        base_image = magic_policy._to_hwc_uint8(base)
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for dest, cam in (("left_wrist_0_rgb", "cam_left_wrist"), ("right_wrist_0_rgb", "cam_right_wrist")):
            img = magic_policy._get_camera(data, cam)
            if img is not None:
                images[dest] = magic_policy._to_hwc_uint8(img)
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {"state": state_model, "image": images, "image_mask": image_masks}

        prev_grip = None
        if "actions" in data:
            acts = np.asarray(data["actions"], dtype=np.float32)
            if acts.ndim != 2 or acts.shape[0] < 2:
                raise ValueError(
                    f"MagicODEActionInputs training expects actions=[H+1,14] (at least 2 frames), got {acts.shape}. "
                    "Set DataConfig.extra_sequence_windows=(('action', -1, action_horizon),)."
                )
            prev_raw = acts[0]  # action[t-1]
            chunk_raw = acts[1:]  # action[t..t+H-1]  [H,14]
            inputs["actions"] = np.asarray(magic_policy.magic_action_to_model(chunk_raw), dtype=np.float32)
            prev_model = np.asarray(magic_policy.magic_action_to_model(prev_raw), dtype=np.float32)  # [14]
            prev_grip = np.array([prev_model[_GRIP_LEFT], prev_model[_GRIP_RIGHT]], dtype=np.float32)
        else:
            lgc = data.get("last_gripper_cmd")
            if lgc is not None:
                lgc = np.asarray(lgc, dtype=np.float32).reshape(-1)
                if lgc.shape[0] >= 2:
                    prev_grip = lgc[:2]
            if prev_grip is None:
                prev_grip = np.array([self.init_gripper_open, self.init_gripper_open], dtype=np.float32)

        sf = state_model.copy()
        sf[_GRIP_LEFT] = prev_grip[0]
        sf[_GRIP_RIGHT] = prev_grip[1]
        inputs["initial_action"] = sf

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MagicODEActionOutputs(transforms.DataTransformFn):
    """Return physical action dimensions in the native right-then-left order."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": magic_policy.magic_action_to_wire(actions)}
