"""Action-field supervision, numerical rollout, and runtime configuration checks."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0
from openpi.policies.inference_config import ODEInferenceConfig


def _field_model():
    def llm(tokens, **kwargs):
        return (None, tokens[1]), None

    return SimpleNamespace(
        use_odeaction=True,
        action_horizon=4,
        action_dim=2,
        ode_solver="euler",
        integration_steps=1,
        _require_initial_action=lambda obs: obs.initial_action,
        embed_prefix=lambda obs: (jnp.zeros((1, 1, 2)), jnp.ones((1, 1), bool), jnp.zeros(1, bool)),
        embed_suffix=lambda obs, a, tau: (a, jnp.ones((1, 1), bool), jnp.ones(1, bool), None),
        PaliGemma=SimpleNamespace(llm=llm),
        action_out_proj=lambda x: x,
    )


@pytest.mark.parametrize("solver", ["euler", "rk4"])
@pytest.mark.parametrize("substeps", [1, 3])
def test_rollout_integrates_the_field(monkeypatch, solver, substeps):
    # F(a,tau)=a has solution a(tau)=exp(tau), starting from a(0)=1.
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs)
    model = _field_model()
    ODEInferenceConfig(solver, substeps).apply(model)
    obs = SimpleNamespace(state=jnp.zeros((1, 2)), initial_action=jnp.ones((1, 2)))
    actions = Pi0._sample_actions_odeaction(model, jax.random.key(0), obs)
    assert actions.shape == (1, 4, 2)
    dt = 1 / (4 * substeps)
    factor = 1 + dt if solver == "euler" else 1 + dt + dt**2 / 2 + dt**3 / 6 + dt**4 / 24
    expected = factor ** (np.arange(1, 5) * substeps)
    np.testing.assert_allclose(actions[0, :, 0], expected, rtol=2e-6)
    if solver == "rk4":
        np.testing.assert_allclose(actions[0, :, 0], np.exp(np.arange(1, 5) / 4), rtol=4e-5)


def test_field_loss_supervises_valid_action_dimensions(monkeypatch):
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs)
    model = _field_model()
    model.odeaction_samples_per_segment = 2
    model.odeaction_loss_valid_dims = 2
    model.odeaction_use_correction = True
    model.odeaction_correction_gain = 5.0
    model.odeaction_neighborhood_scale = 0.0
    model._neighborhood_scale_at = lambda tau: Pi0._neighborhood_scale_at(model, tau)
    model._sample_evolution_times = lambda rng, b, h: Pi0._sample_evolution_times(model, rng, b, h)
    model._embed_field_queries = lambda a, tau: (a, jnp.ones(a.shape[:2], bool), None)
    model.action_out_proj = lambda x: jnp.zeros_like(x)
    obs = SimpleNamespace(initial_action=jnp.zeros((1, 4)))
    actions = jnp.array([[[1.0, 1.0, 100.0, 100.0], [2.0, 2.0, 200.0, 200.0]]])
    loss = Pi0._compute_loss_odeaction(model, jax.random.key(3), obs, actions)
    assert loss.shape == (1, 4)
    np.testing.assert_allclose(loss, 4.0)


@pytest.mark.parametrize("solver,steps", [("bad", 1), ("euler", 0), ("rk4", -1), ("euler", 1.5)])
def test_invalid_ode_options(solver, steps):
    with pytest.raises(ValueError):
        ODEInferenceConfig(solver, steps)


def test_ode_options_do_not_apply_to_baseline():
    with pytest.raises(ValueError, match="ODEAction"):
        ODEInferenceConfig().apply(SimpleNamespace(use_odeaction=False))
