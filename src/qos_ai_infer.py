"""Run a trained QoS neural scheduler on one new analytical scenario."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .qos_ai import predict_single_scenario
except ImportError:  # pragma: no cover
    from qos_ai import predict_single_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=900001)
    parser.add_argument("--topology-seed", type=int, default=2)
    parser.add_argument("--sa-maxiter", type=int, default=2)
    parser.add_argument("--sa-maxfun", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", default="results/qos_ai_inference.json")
    args = parser.parse_args()

    result = predict_single_scenario(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        topology_seed=args.topology_seed,
        sa_maxiter=args.sa_maxiter,
        sa_maxfun=args.sa_maxfun,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
