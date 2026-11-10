"""Runtime options for integrating an ODEAction action transition field."""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class ODEInferenceConfig:
    ode_solver: Literal["euler", "rk4"] = "euler"
    integration_steps: int = 1

    def __post_init__(self):
        if self.ode_solver not in ("euler", "rk4"):
            raise ValueError("ode_solver must be 'euler' or 'rk4'.")
        if type(self.integration_steps) is not int or self.integration_steps < 1:
            raise ValueError("integration_steps must be a positive integer.")

    def apply(self, model) -> None:
        """Set runtime options after loading weights and before constructing Policy."""
        if not getattr(model, "use_odeaction", False):
            raise ValueError("ODE inference options require an ODEAction model.")
        model.ode_solver = self.ode_solver
        model.integration_steps = self.integration_steps
