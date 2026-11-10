"""Send one synthetic observation to a running policy service."""

import argparse

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(policy.get_server_metadata())
    observation = {
        "state": np.zeros(26, dtype=np.float32),
        "images": {
            cam: np.zeros((224, 224, 3), dtype=np.uint8) for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "prompt": "Agilex_Cobot_Magic_Put_the_spoon_into_the_cup_1",
    }
    result = policy.infer(observation)
    print("Action sequence shape:", result["actions"].shape)


if __name__ == "__main__":
    main()
