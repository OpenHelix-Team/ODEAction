#!/usr/bin/env python3


#   python tests/manual/magic_fake_client_roundtrip.py --mode stub --is-odeaction
#   python tests/manual/magic_fake_client_roundtrip.py --mode remote --host 127.0.0.1 --port 8000

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

H_DEFAULT = 50
PROMPT = "Agilex_Cobot_Magic_Put_the_bananas_on_the_plate_1"

_PASS, _FAIL = [], []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        _PASS.append(label)
        print(f"  ✅ {label}")
    else:
        _FAIL.append(f"{label} {detail}")
        print(f"  ❌ {label}  {detail}")


class _StubPolicy:
    """StubPolicy."""

    def __init__(self, horizon: int):
        self.h = horizon
        self.seen: list[dict] = []

        t = np.arange(horizon, dtype=np.float32)[:, None] / max(horizon - 1, 1)
        base = np.arange(14, dtype=np.float32)[None, :]
        self.chunk = (base + 0.8 * t * (1.0 + 0.1 * base)).astype(np.float32)
        self.chunk[:, 6] = 0.31
        self.chunk[:, 13] = 0.73

    def infer(self, obs: dict) -> dict:
        self.seen.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()})
        return {"actions": self.chunk.copy(), "policy_timing": {"infer_ms": 0.1}}

    def reset(self) -> None:
        pass


def _state_from_executed_row(row_out_order: np.ndarray, order: str) -> np.ndarray:
    """state from executed row."""
    row = np.asarray(row_out_order, dtype=np.float32)
    if order == "LEFT_THEN_RIGHT":
        left7, right7 = row[0:7], row[7:14]
    else:
        right7, left7 = row[0:7], row[7:14]
    s = np.zeros(26, dtype=np.float32)
    s[0:7] = right7
    s[13:20] = left7
    return s


def _make_obs(schema: str, h: int = 0) -> dict:
    rng = np.random.default_rng(h)
    state = rng.normal(size=26).astype(np.float32)
    if schema == "ROBODRIVER":
        return {
            "state": state,
            "images": {
                "cam_high": rng.integers(0, 256, (3, 48, 64), dtype=np.uint8),
                "cam_left_wrist": rng.integers(0, 256, (3, 48, 64), dtype=np.uint8),
                "cam_right_wrist": rng.integers(0, 256, (3, 48, 64), dtype=np.uint8),
            },
            "prompt": PROMPT,
        }
    return {
        "observation/state": state,
        "observation/image_top": rng.integers(0, 256, (48, 64, 3), dtype=np.uint8),
        "observation/image_left": rng.integers(0, 256, (48, 64, 3), dtype=np.uint8),
        "observation/image_right": rng.integers(0, 256, (48, 64, 3), dtype=np.uint8),
        "prompt": PROMPT,
    }


def run(args) -> int:
    from openpi_client import websocket_client_policy as _client

    stub = None
    if args.mode == "stub":
        from openpi.serving import magic_policy_adapter as _ad
        from openpi.serving import magic_websocket_server as _srv

        stub = _StubPolicy(args.horizon)
        adapter = _ad.MagicPolicyAdapter(
            policy=stub,
            is_odeaction=args.is_odeaction,
            input_schema=_ad.InputSchema(args.input_schema),
            output_action_order=_ad.OutputActionOrder(args.output_action_order),
            action_horizon=args.horizon,
            replan_steps=args.replan,
            supported_prompts=(PROMPT,),
        )
        md = {
            "protocol": {"name": "openpi_magic", "version": 1, "encoding": "msgpack_numpy"},
            "deployment": {
                "deployment_id": "stub",
                "model_family": "pi05_odeaction" if args.is_odeaction else "pi05",
                "action_horizon": args.horizon,
            },
            "observation_schema": {"input_schema": args.input_schema, "prompt_required": True},
            "action_schema": {
                "shape": [args.horizon, 14],
                "dtype": "float32",
                "order": "left_then_right" if args.output_action_order == "LEFT_THEN_RIGHT" else "right_then_left",
            },
            "robot_type": "agilex_cobot_magic",
            "native_action_order": "right_then_left",
            "capabilities": {"stateful_gripper_feedback": args.is_odeaction},
        }
        server = _srv.MagicWebsocketPolicyServer(policy=adapter, host=args.host, port=args.port, metadata=md)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        time.sleep(1.5)
        print(f"[stub] Local stub server started: ws://{args.host}:{args.port}  (is_odeaction={args.is_odeaction})")

    print("\n===== ① Connection and metadata =====")
    client = _client.WebsocketClientPolicy(host=args.host, port=args.port)
    md = client.get_server_metadata()
    check(isinstance(md, dict) and bool(md), "Server sends metadata first")
    check(
        md.get("protocol", {}).get("name") == "openpi_magic", "protocol.name == openpi_magic", str(md.get("protocol"))
    )
    check(md.get("native_action_order") == "right_then_left", "native_action_order is right_then_left")
    horizon = int(md.get("deployment", {}).get("action_horizon", args.horizon))
    declared_order = md.get("action_schema", {}).get("order")
    is_odeaction = bool(md.get("capabilities", {}).get("stateful_gripper_feedback", args.is_odeaction))
    print(f"     (horizon={horizon}, action order={declared_order}, is_odeaction={is_odeaction})")

    print("\n===== ② First inference =====")
    r1 = client.infer(_make_obs(args.input_schema, 0))
    a1 = np.asarray(r1["actions"])
    check(a1.shape == (horizon, 14), f"Action shape == ({horizon},14)", str(a1.shape))
    check(a1.dtype == np.float32, "Action dtype == float32", str(a1.dtype))
    check(bool(np.all(np.isfinite(a1))), "All action values are finite")

    if args.mode == "stub":
        native = stub.chunk
        expect = (
            np.concatenate([native[:, 7:14], native[:, 0:7]], axis=-1)
            if args.output_action_order == "LEFT_THEN_RIGHT"
            else native
        )
        check(np.allclose(a1, expect), "Arm order matches metadata (one swap)")
        check("last_gripper_cmd" not in stub.seen[0], "First inference uses init_gripper_open without session history")

    print(f"\n===== ③ Client loop(replan={args.replan}, feedback={args.send_last_action}) =====")
    prev = a1
    for i in range(3):
        obs = _make_obs(args.input_schema, 10 + i)

        executed_row = prev[args.replan - 1]
        tracked = _state_from_executed_row(executed_row, args.output_action_order)
        if args.input_schema == "ROBODRIVER":
            obs["state"] = tracked
        else:
            obs["observation/state"] = tracked
        if args.send_last_action:
            obs["last_executed_action"] = executed_row.copy()
        r = client.infer(obs)
        prev = np.asarray(r["actions"])
        check(prev.shape == (horizon, 14), f"Request {i + 2} action shape matches")

    if is_odeaction and args.mode == "stub":
        print("\n===== ④ ODEAction gripper feedback =====")
        native = stub.chunk
        expect = [native[args.replan - 1][13], native[args.replan - 1][6]]
        got = stub.seen[-1].get("last_gripper_cmd")
        check(
            got is not None and np.allclose(got, expect, atol=1e-6),
            f"last_gripper_cmd matches [left, right] grippers at previous row {args.replan - 1}",
            f"got={got} expect={expect}",
        )
        srv = r.get("magic_server", {})
        check(srv.get("replan_steps") == args.replan, "Server replan_steps matches", str(srv))
        d = srv.get("diagnostics", {})
        check(
            "declared_row" in d and d["declared_row"] == args.replan - 1,
            f"Reported declared_row == {args.replan - 1}",
            str(d),
        )

        check(
            d.get("matched_row") == args.replan - 1, f"Joint matching identifies declared row {args.replan - 1}", str(d)
        )
        check("replan_mismatch" not in d, "No replan mismatch when tracking matches", str(d))
        check("jump" not in d, "No displacement warning when tracking matches", str(d))

    if is_odeaction:
        print("\n===== ⑤ Episode reset clears history =====")
        r3 = client.infer({**_make_obs(args.input_schema, 99), "reset": True})
        check(np.asarray(r3["actions"]).shape == (horizon, 14), "Inference succeeds after reset")
        if args.mode == "stub":
            check("last_gripper_cmd" not in stub.seen[-1], "Reset cleared session history")
    else:
        print("\n===== ④ pi0.5: no gripper history =====")
        if args.mode == "stub":
            check(all("last_gripper_cmd" not in s2 for s2 in stub.seen), "pi0.5 never used gripper history")

    if args.mode == "stub":
        import json
        import urllib.request

        base_url = f"http://{args.host}:{args.port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base_url + "/healthz", timeout=5) as response:
            check(response.status == 200, "HTTP health endpoint")
        with opener.open(base_url + "/reset", timeout=5) as response:
            check(json.load(response)["ok"], "HTTP session reset")
        with opener.open(base_url + "/status", timeout=5) as response:
            status = json.load(response)["status"]
            check(not status["has_prev_chunk"], "HTTP reset cleared gripper history")

    print("\n===== ⑥ Invalid observations are rejected =====")
    bad = _make_obs(args.input_schema, 3)
    key = "state" if args.input_schema == "ROBODRIVER" else "observation/state"
    bad[key] = np.full(26, np.nan, dtype=np.float32)
    rejected = False
    try:
        client.infer(bad)
    except Exception:  # noqa: BLE001
        rejected = True
    check(rejected, "Server rejected state containing NaN")

    print("\n" + "=" * 60)
    print(f"Result: {len(_PASS)} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print("  ❌ " + f)
    print("=" * 60)
    return 1 if _FAIL else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Cobot Magic client/server roundtrip checks")
    p.add_argument("--mode", choices=["stub", "remote"], default="stub")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8999)
    p.add_argument("--input-schema", choices=["NATIVE", "ROBODRIVER"], default="ROBODRIVER")
    p.add_argument("--output-action-order", choices=["NATIVE", "LEFT_THEN_RIGHT"], default="LEFT_THEN_RIGHT")
    p.add_argument("--is-odeaction", action="store_true", help="[stub] Simulate ODEAction with gripper feedback")
    p.add_argument(
        "--replan",
        type=int,
        default=30,
        help="Execute K actions between requests; match the server --replan-steps setting",
    )
    p.add_argument(
        "--send-last-action",
        action="store_true",
        help="Send last_executed_action feedback; otherwise use server-side action history",
    )
    p.add_argument("--horizon", type=int, default=H_DEFAULT)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
