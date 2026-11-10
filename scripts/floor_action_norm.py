"""Apply minimum ranges to action statistics for nearly stationary dimensions."""

import argparse
import copy
import json
from pathlib import Path


def floor_statistics(stats: dict, *, range_floor: float = 0.05, std_floor: float = 0.02) -> dict:
    if range_floor <= 0 or std_floor <= 0:
        raise ValueError("Normalization floors must be positive.")
    result = copy.deepcopy(stats)
    root = result.get("norm_stats", result)
    for key in ("state", "actions"):
        values = root[key]
        q01, q99, std = values["q01"], values["q99"], values["std"]
        if not (len(q01) == len(q99) == len(std)):
            raise ValueError(f"Inconsistent statistics dimensions for {key}.")
        for i in range(len(std)):
            if q99[i] - q01[i] < range_floor:
                center = (q99[i] + q01[i]) / 2
                q01[i], q99[i] = center - range_floor / 2, center + range_floor / 2
            std[i] = max(std[i], std_floor)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--range-floor", type=float, default=0.05)
    parser.add_argument("--std-floor", type=float, default=0.02)
    args = parser.parse_args()
    result = floor_statistics(json.loads(args.src.read_text()), range_floor=args.range_floor, std_floor=args.std_floor)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved normalization statistics to {args.dst}")


if __name__ == "__main__":
    main()
