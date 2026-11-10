import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # GPU checks are opt-in; routine tests use the CPU backend.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
