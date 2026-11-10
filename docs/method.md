# Learning an Action Transition Field

[Project overview](../README.md) · [Real-robot guide](real_robot.md)

ODEAction replaces joint prediction of future action chunks with recursive action evolution. Conditioned on images, a task instruction, and robot state, the action expert predicts a transition vector for the current action:

$$
\frac{\mathrm{d}a}{\mathrm{d}\tau}=F_\theta(a,\tau\mid c).
$$

## Demonstrations define the target field

Discrete demonstration actions are linearly interpolated into a trajectory $\gamma(\tau)$. Query actions are sampled in a fixed-scale neighborhood of this trajectory. The target combines progression along the demonstration with correction toward it:

$$
a=\gamma(\tau)+\sigma\epsilon,\qquad
F^*(a,\tau)=\dot\gamma(\tau)+k\bigl(\gamma(\tau)-a\bigr).
$$

The neighborhood scale is fixed throughout evolution. The [Cobot Magic example configurations](../src/openpi/training/config.py) specify the scale and correction gain.

## Direct supervision

The model minimizes squared error between its predicted transition vectors and the targets. Field queries are processed in parallel: each query attends to the visual-language prefix and itself. Evolution time is embedded separately for each query.

The real-robot implementation supervises the 14 physical action dimensions while retaining openpi's 32-dimensional model interface. Both training and rollout use an initial action normalized with the action statistics.

## Recursive generation

At inference, the current action is the initial condition. Euler updates it as

$$
a_{n+1}=a_n+\Delta\tau\,F_\theta(a_n,\tau_n\mid c).
$$

RK4 is also available. The solver and integration substeps are inference settings. The visual-language prefix is cached within a rollout, and each field query predicts one action transition vector. The real-robot service returns a sequence of recursively generated actions for the controller to execute.

The core implementation is in [`pi0.py`](../src/openpi/models/pi0.py); runtime ODE options are in [`inference_config.py`](../src/openpi/policies/inference_config.py).
