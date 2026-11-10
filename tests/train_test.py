"""Optional full-model training/checkpoint smoke test using synthetic data."""

import dataclasses
import pathlib

import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from scripts import train


@pytest.mark.manual
def test_train(tmp_path: pathlib.Path):
    config = _config.TrainConfig(
        name="test_train",
        model=pi0_config.Pi0Config(pi05=True, use_odeaction=True),
        batch_size=2,
        num_workers=0,
        fsdp_devices=1,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        num_train_steps=2,
        log_interval=1,
        wandb_enabled=False,
    )
    train.main(config)
    train.main(dataclasses.replace(config, resume=True, num_train_steps=4))
