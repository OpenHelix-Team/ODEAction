import numpy as np
import pytest

from openpi.policies import magic_odeaction_policy
from openpi.policies import magic_policy

MODEL_GRIP_LEFT, MODEL_GRIP_RIGHT = 6, 13

WIRE_GRIP_RIGHT, WIRE_GRIP_LEFT = 6, 13


def _state26():
    """state26."""
    return np.arange(26, dtype=np.float32)


def _img(c=3, h=8, w=12, chw=True, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (c, h, w) if chw else (h, w, c), dtype=np.uint8)


def test_state_to_model_takes_joints_and_reorders_left_first():
    m = magic_policy.magic_state_to_model(_state26())
    assert m.shape == (14,)

    np.testing.assert_array_equal(m[0:7], np.arange(13, 20, dtype=np.float32))

    np.testing.assert_array_equal(m[7:14], np.arange(0, 7, dtype=np.float32))

    assert not np.isin(np.arange(7, 13), m).any()
    assert not np.isin(np.arange(20, 26), m).any()


def test_action_to_model_swaps_right_left():
    raw = np.arange(14, dtype=np.float32)
    m = magic_policy.magic_action_to_model(raw)
    np.testing.assert_array_equal(m[0:7], raw[7:14])
    np.testing.assert_array_equal(m[7:14], raw[0:7])


def test_action_to_wire_is_inverse_of_to_model():
    raw = np.arange(14, dtype=np.float32)
    back = magic_policy.magic_action_to_wire(magic_policy.magic_action_to_model(raw))
    np.testing.assert_array_equal(back, raw), "to_model and to_wire must be inverse operations"


def test_action_maps_work_on_batched_chunks():
    raw = np.tile(np.arange(14, dtype=np.float32), (50, 1))
    m = magic_policy.magic_action_to_model(raw)
    assert m.shape == (50, 14)
    np.testing.assert_array_equal(magic_policy.magic_action_to_wire(m), raw)


def test_gripper_index_positions():
    """test gripper index positions."""
    raw = np.zeros(14, dtype=np.float32)
    raw[WIRE_GRIP_RIGHT] = 0.31
    raw[WIRE_GRIP_LEFT] = 0.73
    m = magic_policy.magic_action_to_model(raw)
    assert m[MODEL_GRIP_LEFT] == pytest.approx(0.73), "Model index 6 must be the left gripper"
    assert m[MODEL_GRIP_RIGHT] == pytest.approx(0.31), "Model index 13 must be the right gripper"
    w = magic_policy.magic_action_to_wire(m)
    assert w[WIRE_GRIP_RIGHT] == pytest.approx(0.31)
    assert w[WIRE_GRIP_LEFT] == pytest.approx(0.73)


def test_delta_mask_is_12_joints_true_2_grippers_false():
    mask = np.asarray(magic_policy.MAGIC_DELTA_MASK)
    assert mask.shape == (14,)
    assert mask.sum() == 12, "The delta mask must include all 12 joint dimensions"
    assert not mask[MODEL_GRIP_LEFT] and not mask[MODEL_GRIP_RIGHT], (
        "Both gripper dimensions must remain absolute (False in the delta mask)"
    )
    assert mask[0:6].all() and mask[7:13].all()


# ------------------------------------------------------------- MagicInputs --
def test_inputs_accepts_flat_camera_keys():
    out = magic_policy.MagicInputs()(
        {
            "state": _state26(),
            "images/cam_high": _img(seed=1),
            "images/cam_left_wrist": _img(seed=2),
            "images/cam_right_wrist": _img(seed=3),
            "prompt": "p",
        }
    )
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image"]["base_0_rgb"].shape == (8, 12, 3), "CHW images must be converted to HWC"
    assert all(bool(v) for v in out["image_mask"].values())
    assert out["state"].shape == (14,) and out["prompt"] == "p"


def test_inputs_accepts_nested_images_dict_deployment_form():
    out = magic_policy.MagicInputs()(
        {
            "state": _state26(),
            "images": {"cam_high": _img(seed=1), "cam_left_wrist": _img(seed=2), "cam_right_wrist": _img(seed=3)},
            "prompt": "p",
        }
    )
    assert out["image"]["base_0_rgb"].shape == (8, 12, 3)


def test_inputs_missing_top_camera_raises():
    with pytest.raises(KeyError, match="cam_high"):
        magic_policy.MagicInputs()({"state": _state26(), "prompt": "p"})


def test_inputs_missing_wrist_camera_zero_fills_and_masks_false():
    out = magic_policy.MagicInputs()(
        {
            "state": _state26(),
            "images": {"cam_high": _img(seed=1)},
            "prompt": "p",
        }
    )
    assert not bool(out["image_mask"]["left_wrist_0_rgb"])
    assert not bool(out["image_mask"]["right_wrist_0_rgb"])
    np.testing.assert_array_equal(out["image"]["left_wrist_0_rgb"], np.zeros_like(out["image"]["base_0_rgb"]))


def test_inputs_hwc_image_passes_through():
    out = magic_policy.MagicInputs()(
        {
            "state": _state26(),
            "images": {"cam_high": _img(chw=False, seed=1)},
            "prompt": "p",
        }
    )
    assert out["image"]["base_0_rgb"].shape == (8, 12, 3)


def test_inputs_converts_actions_only_when_present():
    base = {"state": _state26(), "images": {"cam_high": _img()}, "prompt": "p"}
    assert "actions" not in magic_policy.MagicInputs()(dict(base))
    raw = np.arange(14, dtype=np.float32)
    out = magic_policy.MagicInputs()({**base, "actions": raw})
    np.testing.assert_array_equal(out["actions"], magic_policy.magic_action_to_model(raw))


# ------------------------------------------------------------ MagicOutputs --
def test_outputs_slices_to_14_and_converts_to_wire():
    padded = np.zeros((50, 32), dtype=np.float32)
    padded[:, :14] = np.arange(14, dtype=np.float32)
    padded[:, 14:] = 999.0
    out = magic_policy.MagicOutputs()({"actions": padded})["actions"]
    assert out.shape == (50, 14)
    assert not (out == 999.0).any(), "Padding dimensions must not appear in the output"
    np.testing.assert_array_equal(
        out, np.tile(magic_policy.magic_action_to_wire(np.arange(14, dtype=np.float32)), (50, 1))
    )


def test_outputs_order_is_native_right_then_left():
    """test outputs order is native right then left."""
    model = np.zeros((1, 32), dtype=np.float32)
    model[0, MODEL_GRIP_LEFT] = 0.73
    model[0, MODEL_GRIP_RIGHT] = 0.31
    w = magic_policy.MagicOutputs()({"actions": model})["actions"][0]
    assert w[WIRE_GRIP_RIGHT] == pytest.approx(0.31)
    assert w[WIRE_GRIP_LEFT] == pytest.approx(0.73)


def _odeaction_base():
    return {
        "state": _state26(),
        "images": {"cam_high": _img(seed=1), "cam_left_wrist": _img(seed=2), "cam_right_wrist": _img(seed=3)},
        "prompt": "p",
    }


def test_odeaction_training_branch_uses_prev_action_gripper():
    """test odeaction training branch uses prev action gripper."""
    H = 5
    acts = np.zeros((H + 1, 14), dtype=np.float32)
    acts[0, WIRE_GRIP_LEFT] = 0.77
    acts[0, WIRE_GRIP_RIGHT] = 0.33
    out = magic_odeaction_policy.MagicODEActionInputs()({**_odeaction_base(), "actions": acts})
    sf = out["initial_action"]
    assert sf.shape == (14,)
    assert sf[MODEL_GRIP_LEFT] == pytest.approx(0.77)
    assert sf[MODEL_GRIP_RIGHT] == pytest.approx(0.33)
    assert out["actions"].shape == (H, 14), "Target actions must be actions[1:] with horizon H"


def test_odeaction_training_branch_rejects_too_short_actions():
    with pytest.raises(Exception):
        magic_odeaction_policy.MagicODEActionInputs()(
            {**_odeaction_base(), "actions": np.zeros((1, 14), dtype=np.float32)}
        )


def test_odeaction_inference_branch_uses_last_gripper_cmd():
    out = magic_odeaction_policy.MagicODEActionInputs()(
        {**_odeaction_base(), "last_gripper_cmd": np.array([0.77, 0.33], dtype=np.float32)}
    )
    sf = out["initial_action"]
    assert sf[MODEL_GRIP_LEFT] == pytest.approx(0.77), "last_gripper_cmd must use [left, right] order"
    assert sf[MODEL_GRIP_RIGHT] == pytest.approx(0.33)
    assert "actions" not in out, "The inference input transform must not produce actions"


def test_odeaction_inference_fallback_to_init_gripper_open():
    sf = magic_odeaction_policy.MagicODEActionInputs()(_odeaction_base())["initial_action"]
    assert sf[MODEL_GRIP_LEFT] == pytest.approx(1.05)
    assert sf[MODEL_GRIP_RIGHT] == pytest.approx(1.05)


def test_odeaction_init_gripper_open_is_configurable():
    sf = magic_odeaction_policy.MagicODEActionInputs(init_gripper_open=0.5)(_odeaction_base())["initial_action"]
    assert sf[MODEL_GRIP_LEFT] == pytest.approx(0.5)


def test_odeaction_initial_action_joints_come_from_measured_state():
    """test odeaction state for flow joints come from measured state."""
    data = {**_odeaction_base(), "last_gripper_cmd": np.array([0.77, 0.33], dtype=np.float32)}
    sf = magic_odeaction_policy.MagicODEActionInputs()(data)["initial_action"]
    state_model = magic_policy.magic_state_to_model(data["state"])
    joints = [i for i in range(14) if i not in (MODEL_GRIP_LEFT, MODEL_GRIP_RIGHT)]
    np.testing.assert_allclose(sf[joints], state_model[joints], atol=1e-6)


def test_odeaction_state_is_still_the_plain_measured_state():
    """test odeaction state is still the plain measured state."""
    data = {**_odeaction_base(), "last_gripper_cmd": np.array([0.77, 0.33], dtype=np.float32)}
    out = magic_odeaction_policy.MagicODEActionInputs()(data)
    np.testing.assert_allclose(out["state"], magic_policy.magic_state_to_model(data["state"]), atol=1e-6)
    assert out["state"][MODEL_GRIP_LEFT] != pytest.approx(0.77), (
        "Gripper commands must update initial_action without overwriting measured state"
    )


def test_odeaction_outputs_order_is_native_right_then_left():
    model = np.zeros((1, 32), dtype=np.float32)
    model[0, MODEL_GRIP_LEFT] = 0.73
    model[0, MODEL_GRIP_RIGHT] = 0.31
    w = magic_odeaction_policy.MagicODEActionOutputs()({"actions": model})["actions"][0]
    assert w.shape == (14,)
    assert w[WIRE_GRIP_RIGHT] == pytest.approx(0.31)
    assert w[WIRE_GRIP_LEFT] == pytest.approx(0.73)


def test_odeaction_and_pi05_outputs_use_same_wire_order():
    """test odeaction and pi05 outputs use same wire order."""
    model = np.arange(32, dtype=np.float32)[None, :]
    a = magic_policy.MagicOutputs()({"actions": model.copy()})["actions"]
    b = magic_odeaction_policy.MagicODEActionOutputs()({"actions": model.copy()})["actions"]
    np.testing.assert_array_equal(a, b)
