"""Serve a JAX pi0.5 or ODEAction policy for Cobot Magic."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

MAGIC_PROMPTS = (
    "Agilex_Cobot_Magic_Put_the_bananas_on_the_plate_1",
    "Agilex_Cobot_Magic_Open_the_drawer_1",
    "Agilex_Cobot_Magic_Put_the_cup_in_the_plate_1",
    "Agilex_Cobot_Magic_Put_the_spoon_into_the_cup_1",
)
CONFIG_NAMES = ("pi05_magic_full", "pi05_magic_lora", "odeaction_magic_full", "odeaction_magic_lora")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=CONFIG_NAMES)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint step directory.")
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input-schema", choices=("NATIVE", "ROBODRIVER"), default="ROBODRIVER")
    parser.add_argument("--output-action-order", choices=("NATIVE", "LEFT_THEN_RIGHT"), default="LEFT_THEN_RIGHT")
    parser.add_argument("--replan-steps", type=int, help="Actions executed between ODEAction requests.")
    parser.add_argument("--ode-solver", choices=("euler", "rk4"), default="euler")
    parser.add_argument("--integration-steps", type=int, default=1, help="ODE substeps per output action.")
    parser.add_argument("--idle-reset-sec", type=float, default=300.0)
    parser.add_argument("--jump-warn-rad", type=float, default=0.5)
    parser.add_argument("--jump-reset-rad", type=float, default=0.0)
    parser.add_argument("--no-reset-on-connect", action="store_true")
    parser.add_argument("--default-prompt")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-config-ckpt-mismatch", action="store_true")
    parser.add_argument("--allow-missing-ckpt-metadata", action="store_true")
    return parser.parse_args()


def preflight(args):
    """Resolve the recipe and check checkpoint assets without constructing the model."""
    from openpi.policies.inference_config import ODEInferenceConfig
    from openpi.shared import normalize
    from openpi.training import config

    ODEInferenceConfig(args.ode_solver, args.integration_steps)
    if args.default_prompt is not None and not args.default_prompt.strip():
        raise ValueError("Default prompt must be a non-empty string.")
    train_config = config.get_config(args.config)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if args.config not in str(checkpoint) and not args.allow_config_ckpt_mismatch:
        raise ValueError(
            "Checkpoint path does not contain the configuration name. "
            "Use --allow-config-ckpt-mismatch for a relocated checkpoint."
        )
    data_config = train_config.data.create_base_config(train_config.assets_dirs, train_config.model)
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Missing checkpoint parameters: {checkpoint / 'params'}")
    normalize.load(checkpoint / "assets" / data_config.asset_id)
    if not args.allow_missing_ckpt_metadata:
        for relative in ("_CHECKPOINT_METADATA", "params/_METADATA"):
            if not (checkpoint / relative).exists():
                raise FileNotFoundError(
                    f"Missing {checkpoint / relative}. "
                    "Use --allow-missing-ckpt-metadata for a compatible alternative layout."
                )
    if train_config.model.use_odeaction:
        if args.replan_steps is None or not 1 <= args.replan_steps <= train_config.model.action_horizon:
            raise ValueError(
                f"Set --replan-steps between 1 and {train_config.model.action_horizon} "
                "to match the number of actions executed by the client."
            )
    logging.info("Checkpoint structure and normalization assets checked: %s", checkpoint)
    return train_config, data_config, checkpoint


def build_metadata(args, train_config, asset_id, is_odeaction, horizon) -> dict:
    model = train_config.model
    order = "left_then_right" if args.output_action_order == "LEFT_THEN_RIGHT" else "right_then_left"
    md = {
        "protocol": {"name": "openpi_magic", "version": 1, "encoding": "msgpack_numpy", "compression": "none"},
        "deployment": {
            "deployment_id": args.deployment_id,
            "model_family": "pi05_odeaction" if is_odeaction else "pi05",
            "config_name": train_config.name,
            "asset_id": asset_id,
            "action_horizon": horizon,
            "vlm_lora": bool(
                getattr(model, "vlm_lora", None) is not None and getattr(model.vlm_lora, "enabled", False)
            ),
        },
        "observation_schema": {
            "input_schema": args.input_schema,
            "state": {
                "shape": [26],
                "dtype": "float32",
                "order": "right_joints6+right_gripper, right_ee_pose6, left_joints6+left_gripper, left_ee_pose6",
            },
            "images": (
                {
                    "keys": ["observation/image_top", "observation/image_left", "observation/image_right"],
                    "layout": "HWC",
                    "dtype": "uint8",
                    "color": "RGB",
                }
                if args.input_schema == "NATIVE"
                else {
                    "keys": ["images/cam_high", "images/cam_left_wrist", "images/cam_right_wrist"],
                    "layout": "CHW_or_HWC",
                    "dtype": "uint8",
                    "color": "RGB",
                }
            ),
            "prompt_required": args.default_prompt is None,
            "supported_prompts": list(MAGIC_PROMPTS),
        },
        "action_schema": {
            "shape": [horizon, 14],
            "dtype": "float32",
            "order": order,
            "joint_semantics": "absolute_position_target",
            "gripper_semantics": "absolute_position_target",
        },
        "robot_type": "agilex_cobot_magic",
        "native_action_order": "right_then_left",
        "capabilities": {
            "healthz": True,
            "reset_field": True,
            "stateful_gripper_feedback": is_odeaction,
        },
    }
    if is_odeaction:
        md["gripper_state_policy"] = {
            "why": "The initial action uses measured joints and previous executed gripper commands.",
            "server_maintains_previous_chunk": True,
            "replan_steps": args.replan_steps,
            "row_used": args.replan_steps - 1,
            "optional_client_fields": {
                "last_executed_action": "[14] last executed action in the configured output order",
                "last_gripper_cmd": "[left, right] previous executed gripper commands",
            },
            "fallback_on_episode_start": 1.05,
            "reset": {
                "http_endpoint": "GET /reset",
                "on_new_connection": not args.no_reset_on_connect,
                "obs_field": "reset=true",
                "idle_sec": args.idle_reset_sec,
                "jump_warn_rad": args.jump_warn_rad,
                "jump_reset_rad": args.jump_reset_rad,
            },
        }
    if is_odeaction:
        md["ode_solver"] = args.ode_solver
        md["integration_steps"] = args.integration_steps
    return md


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    train_config, data_config, checkpoint = preflight(args)
    is_odeaction = train_config.model.use_odeaction
    horizon = train_config.model.action_horizon
    metadata = build_metadata(args, train_config, data_config.asset_id, is_odeaction, horizon)
    if args.preflight_only:
        print(json.dumps(metadata, indent=2))
        return

    from openpi.policies import policy_config
    from openpi.policies.inference_config import ODEInferenceConfig
    from openpi.serving.magic_policy_adapter import InputSchema
    from openpi.serving.magic_policy_adapter import MagicPolicyAdapter
    from openpi.serving.magic_policy_adapter import OutputActionOrder
    from openpi.serving.magic_websocket_server import MagicWebsocketPolicyServer

    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=args.default_prompt,
        ode_config=ODEInferenceConfig(args.ode_solver, args.integration_steps) if is_odeaction else None,
    )
    adapter = MagicPolicyAdapter(
        policy=policy,
        is_odeaction=is_odeaction,
        input_schema=InputSchema(args.input_schema),
        output_action_order=OutputActionOrder(args.output_action_order),
        action_horizon=horizon,
        replan_steps=args.replan_steps if is_odeaction else horizon,
        supported_prompts=MAGIC_PROMPTS,
        default_prompt=args.default_prompt,
        jump_warn_rad=args.jump_warn_rad,
        jump_reset_rad=args.jump_reset_rad,
        idle_reset_sec=args.idle_reset_sec,
    )
    server = MagicWebsocketPolicyServer(
        policy=adapter,
        host=args.host,
        port=args.port,
        metadata=metadata,
        reset_on_connect=not args.no_reset_on_connect,
    )
    logging.info("Serving %s at ws://%s:%s", args.deployment_id, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
