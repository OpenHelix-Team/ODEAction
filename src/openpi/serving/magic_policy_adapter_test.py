import numpy as np
import pytest

from openpi.policies import magic_odeaction_policy
from openpi.policies import magic_policy
from openpi.serving.magic_policy_adapter import InputSchema
from openpi.serving.magic_policy_adapter import MagicObservationError
from openpi.serving.magic_policy_adapter import MagicPolicyAdapter
from openpi.serving.magic_policy_adapter import OutputActionOrder

H = 50
PROMPT = "Agilex_Cobot_Magic_Put_the_bananas_on_the_plate_1"


class StubPolicy:
    """StubPolicy."""

    def __init__(self, fill=None):
        self.seen = []
        self._fill = fill if fill is not None else np.tile(np.arange(14, dtype=np.float32), (H, 1))

    def set_fill(self, fill):
        self._fill = fill

    def infer(self, obs):
        self.seen.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()})
        return {"actions": self._fill.copy(), "policy_timing": {"infer_ms": 1.0}}

    def reset(self):
        pass


def _img_chw(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(3, 48, 64), dtype=np.uint8)


def _obs_rd(state=None, **extra):
    obs = {
        "state": np.arange(26, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32),
        "images": {"cam_high": _img_chw(1), "cam_left_wrist": _img_chw(2), "cam_right_wrist": _img_chw(3)},
        "prompt": PROMPT,
    }
    obs.update(extra)
    return obs


def _obs_native(**extra):
    obs = {
        "observation/state": np.arange(26, dtype=np.float32),
        "observation/image_top": np.transpose(_img_chw(1), (1, 2, 0)),
        "observation/image_left": np.transpose(_img_chw(2), (1, 2, 0)),
        "observation/image_right": np.transpose(_img_chw(3), (1, 2, 0)),
        "prompt": PROMPT,
    }
    obs.update(extra)
    return obs


def _ad(stub, *, is_odeaction, schema=InputSchema.ROBODRIVER, order=OutputActionOrder.NATIVE, **kw):
    kw.setdefault("idle_reset_sec", 0.0)
    return MagicPolicyAdapter(
        policy=stub, is_odeaction=is_odeaction, input_schema=schema, output_action_order=order, action_horizon=H, **kw
    )


def _grip_chunk(left, right, h=H):
    """grip chunk."""
    fill = np.zeros((h, 14), dtype=np.float32)
    fill[:, 6] = right
    fill[:, 13] = left
    return fill


# ------------------------------------------------------------------ schema --
def test_robodriver_schema_converts_chw_to_hwc():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=False)
    ad.infer(_obs_rd())
    inner = stub.seen[-1]
    assert set(inner["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    for img in inner["images"].values():
        assert img.shape == (48, 64, 3) and img.dtype == np.uint8
    assert inner["state"].shape == (26,) and inner["prompt"] == PROMPT


def test_native_schema_maps_camera_names():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=False, schema=InputSchema.NATIVE)
    ad.infer(_obs_native())
    assert set(stub.seen[-1]["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}


def test_wrong_schema_gives_actionable_error():
    with pytest.raises(MagicObservationError, match="ROBODRIVER"):
        _ad(StubPolicy(), is_odeaction=False, schema=InputSchema.NATIVE).infer(_obs_rd())
    with pytest.raises(MagicObservationError, match="NATIVE"):
        _ad(StubPolicy(), is_odeaction=False, schema=InputSchema.ROBODRIVER).infer(_obs_native())


@pytest.mark.parametrize(
    "bad,msg",
    [
        ({"state": np.arange(20, dtype=np.float32)}, "26 dimensions"),
        ({"state": np.full(26, np.nan, dtype=np.float32)}, "non-finite"),
        ({"prompt": ""}, "must not be empty"),
        ({"prompt": 123}, "must be a string"),
    ],
)
def test_bad_observation_rejected(bad, msg):
    with pytest.raises(MagicObservationError, match=msg):
        _ad(StubPolicy(), is_odeaction=False).infer(_obs_rd(**bad))


def test_missing_camera_rejected():
    obs = _obs_rd()
    del obs["images"]["cam_right_wrist"]
    with pytest.raises(MagicObservationError, match="cam_right_wrist"):
        _ad(StubPolicy(), is_odeaction=False).infer(obs)


# ------------------------------------------------------------ action order --
def test_native_order_is_passthrough():
    out = _ad(StubPolicy(), is_odeaction=False, order=OutputActionOrder.NATIVE).infer(_obs_rd())["actions"]
    np.testing.assert_array_equal(out, np.tile(np.arange(14, dtype=np.float32), (H, 1)))


def test_left_then_right_swaps_exactly_once():
    out = _ad(StubPolicy(), is_odeaction=False, order=OutputActionOrder.LEFT_THEN_RIGHT).infer(_obs_rd())["actions"]
    expect = np.tile(np.array([7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6], dtype=np.float32), (H, 1))
    np.testing.assert_array_equal(out, expect)
    np.testing.assert_array_equal(
        np.concatenate([out[:, 7:14], out[:, 0:7]], axis=-1), np.tile(np.arange(14, dtype=np.float32), (H, 1))
    )


def test_bad_model_output_rejected():
    bad = np.tile(np.arange(14, dtype=np.float32), (H, 1))
    bad[3, 5] = np.inf
    with pytest.raises(MagicObservationError, match="non-finite"):
        _ad(StubPolicy(fill=bad), is_odeaction=False).infer(_obs_rd())
    with pytest.raises(MagicObservationError, match="Invalid action shape"):
        _ad(StubPolicy(fill=np.zeros((H, 16), dtype=np.float32)), is_odeaction=False).infer(_obs_rd())


def test_replan_steps_out_of_range_rejected():
    with pytest.raises(ValueError, match="replan_steps"):
        _ad(StubPolicy(), is_odeaction=True, replan_steps=0)
    with pytest.raises(ValueError, match="replan_steps"):
        _ad(StubPolicy(), is_odeaction=True, replan_steps=H + 1)


def test_gripper_index_mapping_from_wire():
    """test gripper index mapping from wire."""
    ad = _ad(StubPolicy(fill=_grip_chunk(left=0.11, right=0.99)), is_odeaction=True, replan_steps=1)
    ad.infer(_obs_rd())
    np.testing.assert_allclose(ad.last_gripper_cmd, [0.11, 0.99], atol=1e-6)


def test_gripper_memory_independent_of_lr_swap():
    fill = _grip_chunk(left=0.11, right=0.99)
    a = _ad(StubPolicy(fill=fill), is_odeaction=True, replan_steps=1, order=OutputActionOrder.NATIVE)
    b = _ad(StubPolicy(fill=fill), is_odeaction=True, replan_steps=1, order=OutputActionOrder.LEFT_THEN_RIGHT)
    a.infer(_obs_rd())
    b.infer(_obs_rd())
    np.testing.assert_array_equal(a.last_gripper_cmd, b.last_gripper_cmd)


@pytest.mark.parametrize("k", [1, 5, 30, 50])
def test_replan_steps_selects_row_k_minus_1(k):
    """test replan steps selects row k minus 1."""
    fill = np.zeros((H, 14), dtype=np.float32)
    fill[:, 13] = np.arange(H, dtype=np.float32)
    fill[:, 6] = 100 + np.arange(H, dtype=np.float32)
    ad = _ad(StubPolicy(fill=fill), is_odeaction=True, replan_steps=k)
    ad.infer(_obs_rd())
    np.testing.assert_allclose(ad.last_gripper_cmd, [k - 1, 100 + k - 1], atol=1e-6)


def test_first_call_has_no_memory_then_feeds_back():
    stub = StubPolicy(fill=_grip_chunk(left=0.24, right=0.42))
    ad = _ad(stub, is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    assert "last_gripper_cmd" not in stub.seen[0], (
        "The first request must use init_gripper_open without injecting gripper history"
    )
    ad.infer(_obs_rd())
    np.testing.assert_allclose(stub.seen[1]["last_gripper_cmd"], [0.24, 0.42], atol=1e-6)


def test_client_last_executed_action_wins_native_order():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=True, replan_steps=30, order=OutputActionOrder.NATIVE)
    ad.infer(_obs_rd())
    row = np.zeros(14, dtype=np.float32)
    row[13] = 0.7
    row[6] = 0.8
    ad.infer(_obs_rd(last_executed_action=row))
    np.testing.assert_allclose(stub.seen[1]["last_gripper_cmd"], [0.7, 0.8], atol=1e-6)


def test_client_last_executed_action_wins_lr_order():
    """test client last executed action wins lr order."""
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=True, replan_steps=30, order=OutputActionOrder.LEFT_THEN_RIGHT)
    ad.infer(_obs_rd())
    row = np.zeros(14, dtype=np.float32)
    row[6] = 0.7
    row[13] = 0.8
    ad.infer(_obs_rd(last_executed_action=row))
    np.testing.assert_allclose(stub.seen[1]["last_gripper_cmd"], [0.7, 0.8], atol=1e-6)


def test_client_last_gripper_cmd_wins():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd(last_gripper_cmd=np.array([0.33, 0.44], dtype=np.float32)))
    np.testing.assert_allclose(stub.seen[1]["last_gripper_cmd"], [0.33, 0.44], atol=1e-6)


def test_client_override_is_consumed_once():
    """test client override is consumed once."""
    stub = StubPolicy(fill=_grip_chunk(left=0.24, right=0.42))
    ad = _ad(stub, is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd(last_gripper_cmd=np.array([0.9, 0.9], dtype=np.float32)))
    ad.infer(_obs_rd())
    np.testing.assert_allclose(stub.seen[2]["last_gripper_cmd"], [0.24, 0.42], atol=1e-6)


@pytest.mark.parametrize(
    "bad,msg",
    [
        ({"last_gripper_cmd": np.array([np.nan, 1.0], dtype=np.float32)}, "last_gripper_cmd"),
        ({"last_executed_action": np.zeros(10, dtype=np.float32)}, "last_executed_action"),
        ({"last_executed_action": np.full(14, np.inf, dtype=np.float32)}, "last_executed_action"),
    ],
)
def test_bad_client_feedback_rejected(bad, msg):
    ad = _ad(StubPolicy(), is_odeaction=True, replan_steps=30)
    with pytest.raises(MagicObservationError, match=msg):
        ad.infer(_obs_rd(**bad))


def test_reset_clears_memory():
    ad = _ad(StubPolicy(), is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    assert ad.last_gripper_cmd is not None
    ad.reset(reason="http_endpoint")
    assert ad.last_gripper_cmd is None
    assert ad.status()["last_reset_reason"] == "http_endpoint"


def test_obs_reset_field_clears_memory():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd(reset=True))
    assert "last_gripper_cmd" not in stub.seen[1]
    assert ad.status()["steps_since_reset"] == 1
    assert ad.status()["reset_count"] == 1


def test_idle_timeout_resets():
    import time as _t

    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=True, replan_steps=30, idle_reset_sec=0.05)
    ad.infer(_obs_rd())
    _t.sleep(0.12)
    ad.infer(_obs_rd())
    assert "last_gripper_cmd" not in stub.seen[1], "An idle timeout must reset the session"
    assert ad.status()["last_reset_reason"] == "idle_timeout"
    assert ad.status()["steps_since_reset"] == 1
    assert ad.status()["reset_count"] == 1


def test_status_fields():
    ad = _ad(StubPolicy(), is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    st = ad.status()
    assert st["is_odeaction"] and st["replan_steps"] == 30 and st["steps_since_reset"] == 1
    assert st["has_prev_chunk"] and st["last_gripper_cmd"] is not None


def _state_from_model14(model14):
    """state from model14."""
    s = np.zeros(26, dtype=np.float32)
    s[13:20] = model14[0:7]
    s[0:7] = model14[7:14]
    return s


def test_diagnosis_matches_declared_row_when_client_agrees():
    """test diagnosis matches declared row when client agrees."""
    rng = np.random.default_rng(0)
    chunk_model = np.cumsum(rng.normal(0, 0.05, size=(H, 14)), axis=0).astype(np.float32)
    chunk_wire = np.concatenate([chunk_model[:, 7:14], chunk_model[:, 0:7]], axis=-1)
    K = 30
    ad = _ad(StubPolicy(fill=chunk_wire), is_odeaction=True, replan_steps=K)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd(state=_state_from_model14(chunk_model[K - 1])))
    d = ad.status()["last_diagnostics"]
    assert d["matched_row"] == K - 1 == d["declared_row"]
    assert "replan_mismatch" not in d


def test_diagnosis_flags_replan_mismatch():
    """test diagnosis flags replan mismatch."""
    rng = np.random.default_rng(1)
    chunk_model = np.cumsum(rng.normal(0, 0.05, size=(H, 14)), axis=0).astype(np.float32)
    chunk_wire = np.concatenate([chunk_model[:, 7:14], chunk_model[:, 0:7]], axis=-1)
    ad = _ad(StubPolicy(fill=chunk_wire), is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd(state=_state_from_model14(chunk_model[19])))
    d = ad.status()["last_diagnostics"]
    assert d["matched_row"] == 19 and d["declared_row"] == 29
    assert d.get("replan_mismatch") is True


def test_diagnosis_silent_when_rows_indistinguishable():
    """test diagnosis silent when rows indistinguishable."""
    chunk_wire = np.zeros((H, 14), dtype=np.float32)
    ad = _ad(StubPolicy(fill=chunk_wire), is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd(state=np.zeros(26, dtype=np.float32)))
    assert "replan_mismatch" not in ad.status()["last_diagnostics"]


def test_jump_warns_but_does_not_reset_by_default():
    ad = _ad(
        StubPolicy(fill=np.zeros((H, 14), dtype=np.float32)),
        is_odeaction=True,
        replan_steps=30,
        jump_warn_rad=0.5,
        jump_reset_rad=0.0,
    )
    ad.infer(_obs_rd(state=np.zeros(26, dtype=np.float32)))
    big = np.zeros(26, dtype=np.float32)
    big[13:19] = 3.0
    ad.infer(_obs_rd(state=big))
    d = ad.status()["last_diagnostics"]
    assert d.get("jump") is True
    assert "jump_reset" not in d, "The default behavior must warn without resetting the session"
    assert ad.last_gripper_cmd is not None


def test_jump_resets_when_enabled():
    stub = StubPolicy(fill=np.zeros((H, 14), dtype=np.float32))
    ad = _ad(stub, is_odeaction=True, replan_steps=30, jump_warn_rad=0.5, jump_reset_rad=0.5)
    ad.infer(_obs_rd(state=np.zeros(26, dtype=np.float32)))
    big = np.zeros(26, dtype=np.float32)
    big[13:19] = 3.0
    ad.infer(_obs_rd(state=big))
    assert "last_gripper_cmd" not in stub.seen[1], "A large displacement should reset the session when enabled"
    assert ad.status()["steps_since_reset"] == 1
    assert ad.status()["reset_count"] == 1


# --------------------------------------------------------------- pi0.5 --
def test_pi05_never_uses_session_state():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=False)
    for _ in range(3):
        ad.infer(_obs_rd())
    assert ad.last_gripper_cmd is None
    for seen in stub.seen:
        assert "last_gripper_cmd" not in seen
    assert "last_gripper_cmd" not in ad.status() or ad.status()["last_gripper_cmd"] is None


def test_closed_loop_feedback_lands_in_initial_action():
    """test closed loop feedback lands in state for flow."""
    stub = StubPolicy(fill=_grip_chunk(left=0.77, right=0.33))
    ad = _ad(stub, is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd())
    inner = stub.seen[1]
    out = magic_odeaction_policy.MagicODEActionInputs()(
        {
            "state": inner["state"],
            "images": inner["images"],
            "prompt": inner["prompt"],
            "last_gripper_cmd": inner["last_gripper_cmd"],
        }
    )
    sf = out["initial_action"]
    assert sf.shape == (14,)
    assert sf[6] == pytest.approx(0.77), "The left gripper in a(0) must match the previous left gripper command"
    assert sf[13] == pytest.approx(0.33), "The right gripper in a(0) must match the previous right gripper command"
    state_model = magic_policy.magic_state_to_model(inner["state"])
    joints = [i for i in range(14) if i not in (6, 13)]
    np.testing.assert_allclose(sf[joints], state_model[joints], atol=1e-6)


def test_first_call_falls_back_to_init_gripper_open():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=True, replan_steps=30)
    ad.infer(_obs_rd())
    inner = stub.seen[0]
    assert "last_gripper_cmd" not in inner
    sf = magic_odeaction_policy.MagicODEActionInputs()(
        {
            "state": inner["state"],
            "images": inner["images"],
            "prompt": inner["prompt"],
        }
    )["initial_action"]
    assert sf[6] == pytest.approx(1.05) and sf[13] == pytest.approx(1.05)


@pytest.mark.parametrize("is_odeaction", [False, True])
@pytest.mark.parametrize("schema", list(InputSchema))
def test_default_prompt_is_resolved_before_policy_call(is_odeaction, schema):
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=is_odeaction, schema=schema, default_prompt=PROMPT)
    obs = _obs_rd() if schema is InputSchema.ROBODRIVER else _obs_native()
    del obs["prompt"]
    ad.infer(obs)
    assert stub.seen[-1]["prompt"] == PROMPT
    obs["prompt"] = "Explicit task"
    ad.infer(obs)
    assert stub.seen[-1]["prompt"] == "Explicit task"
    for invalid in (None, "", "   ", 123):
        obs["prompt"] = invalid
        with pytest.raises(MagicObservationError):
            ad.infer(obs)
    assert len(stub.seen) == 2


def test_missing_prompt_without_default_is_rejected():
    stub = StubPolicy()
    ad = _ad(stub, is_odeaction=False)
    obs = _obs_rd()
    del obs["prompt"]
    with pytest.raises(MagicObservationError, match="prompt"):
        ad.infer(obs)
    assert not stub.seen


@pytest.mark.parametrize("prompt", ["", "   ", 123])
def test_invalid_default_prompt_is_rejected(prompt):
    with pytest.raises(MagicObservationError, match="prompt"):
        _ad(StubPolicy(), is_odeaction=False, default_prompt=prompt)


@pytest.mark.parametrize("is_odeaction", [False, True])
def test_reset_restarts_request_count_and_preserves_reset_count(is_odeaction):
    ad = _ad(StubPolicy(), is_odeaction=is_odeaction)
    ad.infer(_obs_rd())
    ad.infer(_obs_rd())
    assert ad.status()["steps_since_reset"] == 2
    for expected_resets in (1, 2):
        ad.reset()
        assert ad.status()["steps_since_reset"] == 0
        assert ad.status()["reset_count"] == expected_resets
        with pytest.raises(MagicObservationError):
            ad.infer(_obs_rd(prompt=""))
        assert ad.status()["steps_since_reset"] == 0
        assert ad.infer(_obs_rd())["magic_server"]["step"] == 1
