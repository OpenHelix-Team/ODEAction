"""Exercise public metadata, errors, and session resets over a real local socket."""

import argparse
import asyncio
import json
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from openpi.serving.magic_policy_adapter import InputSchema
from openpi.serving.magic_policy_adapter import MagicPolicyAdapter
from openpi.serving.magic_websocket_server import MagicWebsocketPolicyServer
from scripts.serve_magic_policy import build_metadata


class StubPolicy:
    def __init__(self, error=None):
        self.error = error
        self.seen = []

    def infer(self, obs):
        self.seen.append(obs)
        if self.error is not None:
            raise self.error
        return {"actions": np.zeros((2, 14), np.float32)}


def observation(schema):
    state = np.zeros(26, np.float32)
    image = np.zeros((4, 4, 3), np.uint8)
    if schema is InputSchema.NATIVE:
        return {
            "observation/state": state,
            **{f"observation/image_{cam}": image for cam in ("top", "left", "right")},
        }
    return {"state": state, "images": {cam: image for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}


def make_service(is_odeaction, schema, error=None, default_prompt="Example task"):
    stub = StubPolicy(error)
    adapter = MagicPolicyAdapter(
        stub,
        is_odeaction=is_odeaction,
        input_schema=schema,
        action_horizon=2,
        replan_steps=2,
        default_prompt=default_prompt,
        idle_reset_sec=0,
    )
    args = argparse.Namespace(
        deployment_id="test",
        input_schema=schema.value,
        output_action_order="LEFT_THEN_RIGHT",
        default_prompt=default_prompt,
        replan_steps=2,
        no_reset_on_connect=False,
        idle_reset_sec=0,
        jump_warn_rad=0.5,
        jump_reset_rad=0,
        ode_solver="rk4",
        integration_steps=2,
    )
    config = SimpleNamespace(
        name="odeaction_magic_full" if is_odeaction else "pi05_magic_full", model=SimpleNamespace(vlm_lora=None)
    )
    metadata = build_metadata(args, config, "magic_4tasks_v1", is_odeaction, 2)
    return MagicWebsocketPolicyServer(adapter, metadata=metadata), stub


async def request(client, obs):
    await client.send(msgpack_numpy.Packer().pack(obs))
    return await client.recv()


async def http_get(port, path):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
    await writer.drain()
    response = await asyncio.wait_for(reader.read(), timeout=5)
    writer.close()
    await writer.wait_closed()
    headers, body = response.split(b"\r\n\r\n", 1)
    assert b"200 OK" in headers
    return json.loads(body)


@pytest.mark.parametrize("is_odeaction", [False, True])
@pytest.mark.parametrize("schema", list(InputSchema))
def test_metadata_prompt_and_all_network_reset_routes(is_odeaction, schema, tmp_path):
    async def run():
        service, stub = make_service(is_odeaction, schema)
        async with serve(service._handler, "127.0.0.1", 0, process_request=service._process_request) as server:
            port = server.sockets[0].getsockname()[1]
            url = f"ws://127.0.0.1:{port}"
            async with connect(url) as client:
                metadata = msgpack_numpy.unpackb(await client.recv())
                assert "checkpoint_dir" not in metadata["deployment"]
                assert "units" not in metadata["action_schema"]
                assert not metadata["observation_schema"]["prompt_required"]
                assert str(tmp_path) not in json.dumps(metadata)
                obs = observation(schema)
                for step in (1, 2):
                    response = msgpack_numpy.unpackb(await request(client, obs))
                    assert response["magic_server"]["step"] == step
                assert stub.seen[-1]["prompt"] == "Example task"
                assert (await http_get(port, "/reset"))["ok"]
                status = (await http_get(port, "/status"))["status"]
                assert status["steps_since_reset"] == 0
                assert status["reset_count"] == 2  # connection, then HTTP reset
                assert msgpack_numpy.unpackb(await request(client, obs))["magic_server"]["step"] == 1
                obs["reset"] = True
                assert msgpack_numpy.unpackb(await request(client, obs))["magic_server"]["step"] == 1
                assert service._policy.status()["reset_count"] == 3
            async with connect(url) as client:
                await client.recv()
                status = service._policy.status()
                assert status["steps_since_reset"] == 0
                assert status["reset_count"] == 4

    asyncio.run(asyncio.wait_for(run(), timeout=15))


@pytest.mark.parametrize("error_type", [RuntimeError, FileNotFoundError, ValueError, TypeError])
def test_error_frames_never_include_local_paths_or_tracebacks(error_type, tmp_path, caplog):
    private_path = tmp_path / "private-checkpoint" / "params"
    error = error_type(f"Cannot read {private_path}")

    async def run():
        service, _ = make_service(True, InputSchema.ROBODRIVER, error=error)
        async with serve(service._handler, "127.0.0.1", 0) as server:
            async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                await client.recv()
                response = await request(client, observation(InputSchema.ROBODRIVER))
                assert response == "Inference request failed. Check the request and server logs."
                assert str(private_path) not in response and "Traceback" not in response
                await client.wait_closed()
                assert client.close_code == 1011
                assert client.close_reason == "Inference request failed."
                assert service._policy.status()["steps_since_reset"] == 0

    asyncio.run(asyncio.wait_for(run(), timeout=15))
    assert str(private_path) in caplog.text  # Detailed diagnostics stay in local logs.


def test_invalid_observation_gets_a_safe_error_frame():
    async def run():
        service, stub = make_service(False, InputSchema.ROBODRIVER, default_prompt=None)
        async with serve(service._handler, "127.0.0.1", 0) as server:
            async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                metadata = msgpack_numpy.unpackb(await client.recv())
                assert metadata["observation_schema"]["prompt_required"]
                response = await request(client, observation(InputSchema.ROBODRIVER))
                assert response == "Inference request failed. Check the request and server logs."
                assert not stub.seen

    asyncio.run(asyncio.wait_for(run(), timeout=15))
