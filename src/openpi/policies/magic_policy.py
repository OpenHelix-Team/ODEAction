"""Data transforms for the AgileX Cobot Magic dual-arm robot (Pi0 / Pi0.5).

MODEL layout (LEFT-then-RIGHT; grippers at model idx 6 and 13):
    [ left j1..6 , left gripper , right j1..6 , right gripper ]
Raw dataset is right-then-left absolute; 12 joints -> delta (mask make_bool_mask(6,-1,6,-1)),
2 grippers remain absolute. Cameras: top->base_0_rgb,
left->left_wrist_0_rgb, right->right_wrist_0_rgb.

Post-repack keys are '/'-flattened: images/cam_high, images/cam_left_wrist, images/cam_right_wrist,
plus state, actions, prompt. A nested data['images'] dict (deployment adapter form) is also accepted.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def magic_state_to_model(state: np.ndarray) -> np.ndarray:
    """26-dim follower state -> 14-dim model state, left-then-right (drops the 12 EE-pose dims)."""
    s = np.asarray(state)
    right = s[..., 0:7]
    left = s[..., 13:20]
    return np.concatenate([left, right], axis=-1)


def magic_action_to_model(actions: np.ndarray) -> np.ndarray:
    """Raw action [right7, left7] -> model action [left7, right7]."""
    a = np.asarray(actions)
    right = a[..., 0:7]
    left = a[..., 7:14]
    return np.concatenate([left, right], axis=-1)


def magic_action_to_wire(actions: np.ndarray) -> np.ndarray:
    """Model action [left7, right7] -> native wire action [right7, left7]."""
    a = np.asarray(actions)
    left = a[..., 0:7]
    right = a[..., 7:14]
    return np.concatenate([right, left], axis=-1)


MAGIC_DELTA_MASK = transforms.make_bool_mask(6, -1, 6, -1)  # (T*6, F, T*6, F)


def _to_hwc_uint8(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


def _get_camera(data: dict, cam: str):
    """Fetch a camera accepting BOTH flat 'images/<cam>' and nested data['images'][<cam>]."""
    flat_key = f"images/{cam}"
    if flat_key in data:
        return data[flat_key]
    imgs = data.get("images")
    if isinstance(imgs, dict) and cam in imgs:
        return imgs[cam]
    return None


def make_magic_example() -> dict:
    return {
        "state": np.ones((26,), dtype=np.float32),
        "images/cam_high": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "prompt": "put the spoon into the cup",
    }


@dataclasses.dataclass(frozen=True)
class MagicInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = magic_state_to_model(data["state"])

        base = _get_camera(data, "cam_high")
        if base is None:
            raise KeyError(
                "MagicInputs: top camera not found; expected 'images/cam_high' "
                "(or data['images']['cam_high']). Got keys: " + str(sorted(data.keys()))
            )
        base_image = _to_hwc_uint8(base)
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for dest, cam in (("left_wrist_0_rgb", "cam_left_wrist"), ("right_wrist_0_rgb", "cam_right_wrist")):
            img = _get_camera(data, cam)
            if img is not None:
                images[dest] = _to_hwc_uint8(img)
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }
        if "actions" in data:
            inputs["actions"] = magic_action_to_model(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MagicOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": magic_action_to_wire(actions)}
