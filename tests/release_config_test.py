"""Public training recipes and the corresponding data transformations."""

from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.policies.magic_odeaction_policy import MagicODEActionInputs
from openpi.training import config


def test_four_recipes_and_shared_statistics():
    expected = {
        "pi05_magic_full": (10000, False, False),
        "pi05_magic_lora": (10000, False, True),
        "odeaction_magic_full": (10000, True, False),
        "odeaction_magic_lora": (10000, True, True),
    }
    assert set(config._CONFIGS_DICT) == set(expected)
    for name, (steps, odeaction, lora) in expected.items():
        cfg = config.get_config(name)
        assert cfg.num_train_steps == steps
        assert cfg.model.use_odeaction == odeaction
        assert cfg.model.vlm_lora.enabled == lora
        assert cfg.batch_size == 128 and cfg.fsdp_devices == 8
        base = cfg.data.create_base_config(cfg.assets_dirs, cfg.model)
        assert base.repo_id == "magic_4tasks_v1"
        assert base.prompt_from_task
        assert len(base.norm_stats["actions"].mean) == 14
    assert len(list(Path("assets").rglob("norm_stats.json"))) == 2


def test_initial_action_uses_previous_gripper_and_action_statistics():
    data = {
        "state": np.zeros(26, np.float32),
        "images/cam_high": np.zeros((8, 8, 3), np.uint8),
        "actions": np.zeros((3, 14), np.float32),
    }
    # The dataset stores right then left. Model order is left then right.
    data["actions"][0, 6] = 0.3
    data["actions"][0, 13] = 0.7
    inputs = MagicODEActionInputs()(data)
    np.testing.assert_allclose(inputs["initial_action"][[6, 13]], [0.7, 0.3])
    assert inputs["actions"].shape == (2, 14)
    stats = {
        "actions": transforms.NormStats(
            mean=np.ones(14),
            std=np.full(14, 2.0),
            q01=np.zeros(14),
            q99=np.full(14, 2.0),
        )
    }
    initial = inputs["initial_action"].copy()
    transformed = transforms.NormalizeInitialAction(stats, use_quantiles=True)(inputs)
    np.testing.assert_allclose(transformed["initial_action"], initial - 1, atol=1e-5)
    padded = transforms.PadStatesAndActions(32)(transformed)
    assert padded["initial_action"].shape == (32,)
    np.testing.assert_array_equal(padded["initial_action"][14:], 0)
