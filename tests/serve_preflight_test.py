"""Validate the public service's checkpoint assets and inference metadata."""

import argparse
import json
import shutil

import pytest

from scripts import serve_magic_policy as server


def test_preflight_without_tokenizer_or_model_download(tmp_path):
    checkpoint = tmp_path / "odeaction_magic_full" / "run" / "9999"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").touch()
    (checkpoint / "params/_METADATA").touch()
    assets = checkpoint / "assets/magic_4tasks_v1"
    assets.mkdir(parents=True)
    shutil.copyfile("assets/odeaction_magic/magic_4tasks_v1/norm_stats.json", assets / "norm_stats.json")
    args = argparse.Namespace(
        config="odeaction_magic_full",
        checkpoint=str(checkpoint),
        deployment_id="test",
        default_prompt=None,
        ode_solver="rk4",
        integration_steps=2,
        replan_steps=20,
        allow_config_ckpt_mismatch=False,
        allow_missing_ckpt_metadata=False,
        input_schema="ROBODRIVER",
        output_action_order="LEFT_THEN_RIGHT",
        no_reset_on_connect=False,
        idle_reset_sec=300.0,
        jump_warn_rad=0.5,
        jump_reset_rad=0.0,
    )
    config, data, resolved = server.preflight(args)
    assert resolved == checkpoint.resolve()
    metadata = server.build_metadata(args, config, data.asset_id, True, 50)
    assert str(checkpoint) not in json.dumps(metadata)
    assert "checkpoint_dir" not in metadata["deployment"]
    assert "units" not in metadata["action_schema"]
    assert metadata["observation_schema"]["prompt_required"]
    assert metadata["ode_solver"] == "rk4"
    assert metadata["integration_steps"] == 2
    assert metadata["gripper_state_policy"]["row_used"] == 19
    args.replan_steps = 0
    with pytest.raises(ValueError, match="replan"):
        server.preflight(args)


def test_preflight_rejects_empty_default_prompt():
    args = argparse.Namespace(ode_solver="euler", integration_steps=1, default_prompt="   ")
    with pytest.raises(ValueError, match="Default prompt"):
        server.preflight(args)
