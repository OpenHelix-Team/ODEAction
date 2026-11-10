# Run ODEAction on Real-World Robots

[Project overview](../README.md) · [Method](method.md)

This Cobot Magic example uses the JAX π₀.₅ backbone in openpi. It supports multi-task training with full fine-tuning or VLM-LoRA, followed by policy serving and robot execution.

## Environment

Follow the environment setup instructions in [openpi](https://github.com/Physical-Intelligence/openpi). Run the commands below from this repository's root directory.

## Data and normalization

The example combines four tasks: placing bananas on a plate, opening a drawer, placing a water-filled cup on a plate without spilling water throughout execution, and inserting a spoon into a cup.

The [example configurations](../src/openpi/training/config.py) use the LeRobot dataset `magic_4tasks_v1`. Its observations contain top and wrist camera images, a 26-dimensional robot state, and task instructions; actions are 14-dimensional absolute joint and gripper commands. The [data transforms](../src/openpi/policies/magic_policy.py) map the dataset's right-then-left arm order to the model's left-then-right order. The [ODEAction transform](../src/openpi/policies/magic_odeaction_policy.py) also constructs the initial action from measured joints and preceding gripper commands.

Set the dataset parent directory, then compute statistics for the selected configuration:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot/datasets
uv run scripts/compute_norm_stats.py --config-name=odeaction_magic_full
```

The example statistics are in `assets/pi05_magic/` and `assets/odeaction_magic/`. Full and LoRA configurations share their method's statistics. To apply the ODEAction normalization floor to newly computed statistics:

```bash
uv run scripts/floor_action_norm.py \
  --src /path/to/raw/norm_stats.json \
  --dst /path/to/assets/norm_stats.json
```

The supplied ODEAction statistics already include this floor. Training saves the selected statistics with the checkpoint, and inference loads them from there.

## Training

| Configuration | Method | Fine-tuning |
|---|---|---|
| `pi05_magic_full` | π₀.₅ | Full |
| `pi05_magic_lora` | π₀.₅ | VLM-LoRA |
| `odeaction_magic_full` | ODEAction | Full |
| `odeaction_magic_lora` | ODEAction | VLM-LoRA |

```bash
uv run scripts/train.py odeaction_magic_full --exp-name=my_run
```

Training initializes from the openpi π₀.₅ base checkpoint. VLM-LoRA trains vision and language adapters together with the full action expert. Device sharding and batch size are configured through `--fsdp-devices` and `--batch-size`.

Checkpoints are saved under `checkpoints/<config>/<experiment>/<step>/`. Keep their `params/` and `assets/` directories together; for LoRA, also retain the experiment's `vlm_lora_contract.json`.

## Inference

Start the policy service on the inference machine:

```bash
uv run scripts/serve_magic_policy.py \
  --config odeaction_magic_full \
  --checkpoint /path/to/checkpoints/odeaction_magic_full/my_run/STEP \
  --deployment-id odeaction-magic \
  --host 0.0.0.0 --port 8000 \
  --replan-steps 50 \
  --ode-solver euler --integration-steps 1
```

Replace the checkpoint path with an existing step directory. Euler and RK4 use the same trained weights; select `--ode-solver rk4` to switch solvers. `--integration-steps` sets the number of substeps per output action. For π₀.₅, select its configuration and omit the ODE options.

Use the [client example](../examples/cobot_magic/client.py) with the inference machine's address to exchange observations and actions. The default interface accepts `state`, an `images` dictionary with `cam_high`, `cam_left_wrist`, and `cam_right_wrist`, and `prompt`. A server-side `--default-prompt` supplies the task instruction when the request omits it.

Returned actions use left-then-right arm order by default. Execute the first `K` actions and set `--replan-steps K` accordingly. ODEAction uses the preceding executed gripper commands for its next initial action; the client can send `last_executed_action` in the server's output order or `last_gripper_cmd` as `[left, right]`. Send `reset=True` in the first observation of each episode.
