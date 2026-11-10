"""Trace each public model's loss and inference without allocating full weights."""

import jax

from openpi.policies.inference_config import ODEInferenceConfig
from openpi.training import config


def main():
    names = ("pi05_magic_full", "pi05_magic_lora", "odeaction_magic_full", "odeaction_magic_lora")
    for name in names:
        cfg = config.get_config(name)
        for solver in ("euler", "rk4") if cfg.model.use_odeaction else (None,):

            def forward(key, observation, actions):
                model = cfg.model.create(key)
                if solver is not None:
                    ODEInferenceConfig(solver, 1).apply(model)
                loss = model.compute_loss(key, observation, actions)
                generated = model.sample_actions(key, observation)
                return loss, generated

            loss, actions = jax.eval_shape(forward, jax.random.key(0), *cfg.model.inputs_spec())
            queries = cfg.model.action_horizon * (cfg.model.odeaction_samples_per_segment if solver else 1)
            assert loss.shape == (1, queries), loss.shape
            assert actions.shape == (1, cfg.model.action_horizon, cfg.model.action_dim), actions.shape
            print(f"PASS {name} {solver or 'baseline'}: loss={loss.shape}, actions={actions.shape}")


if __name__ == "__main__":
    main()
