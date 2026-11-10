"""Check inference options at the checkpoint-to-policy boundary."""

from types import SimpleNamespace

import pytest

from openpi import transforms
from openpi.policies import policy_config
from openpi.policies.inference_config import ODEInferenceConfig


@pytest.mark.parametrize("solver,steps", [("euler", 1), ("rk4", 3)])
def test_ode_options_are_applied_before_policy_compilation(tmp_path, monkeypatch, solver, steps):
    model = SimpleNamespace(use_odeaction=True, ode_solver="euler", integration_steps=1)
    model_config = SimpleNamespace(load=lambda params: model)
    data_config = SimpleNamespace(
        asset_id="test",
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(),
        use_quantile_norm=True,
    )
    config = SimpleNamespace(
        model=model_config,
        data=SimpleNamespace(create=lambda *args: data_config),
        assets_dirs=tmp_path,
        policy_metadata={},
        name="test",
    )
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    captured = []

    def construct_policy(loaded, **kwargs):
        captured.append((loaded.ode_solver, loaded.integration_steps))
        return loaded

    monkeypatch.setattr(policy_config._policy, "Policy", construct_policy)
    loaded = policy_config.create_trained_policy(
        config, tmp_path, norm_stats={}, ode_config=ODEInferenceConfig(solver, steps)
    )
    assert loaded is model
    assert captured == [(solver, steps)]
