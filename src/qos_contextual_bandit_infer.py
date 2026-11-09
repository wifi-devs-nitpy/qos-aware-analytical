"""Run a trained LinUCB QoS scheduler on a new unseen scenario."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

try:
    from .qos_contextual_bandit import (
        LinUCBContextualBandit,
        build_group_context_features,
        select_bandit_schedule,
        N_STAS,
        N_LINKS,
    )
    from .qos_ai import generate_scenario
    from .qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from .qos_scheduler import validate_selection
except ImportError:  # pragma: no cover
    from qos_contextual_bandit import LinUCBContextualBandit, build_group_context_features, select_bandit_schedule, N_STAS, N_LINKS
    from qos_ai import generate_scenario
    from qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from qos_scheduler import validate_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=900001)
    parser.add_argument("--topology-seed", type=int, default=2)
    parser.add_argument("--sa-maxiter", type=int, default=20)
    parser.add_argument("--sa-maxfun", type=int, default=10_000)
    parser.add_argument("--output", default="results/qos_final_three_method_comparison/inference_new_scenario.json")
    args = parser.parse_args()

    model, metadata = LinUCBContextualBandit.load(args.checkpoint)
    ap, sta, walls = matlab_qos_topology(
        DEFAULT_CONFIG["ap_number"], DEFAULT_CONFIG["sta_number"],
        DEFAULT_CONFIG["grid_value"], np.random.default_rng(args.topology_seed),
    )
    scenario = generate_scenario(
        seed=args.seed, ap_matrix=ap, sta_matrix=sta, walls=walls,
        sa_maxiter=args.sa_maxiter, sa_maxfun=args.sa_maxfun,
        reward_units_mode="matlab_as_written", include_expert=False,
    )
    features = build_group_context_features(scenario.groups, scenario.sta_ac)
    selected, scores, decision_time = select_bandit_schedule(
        model, features, scenario.groups, explore=False
    )
    validate_selection(scenario.groups, selected, N_STAS, N_LINKS)
    rows = []
    for index in selected:
        group = scenario.groups[index]
        rows.append({
            "link_id": group.link_id,
            "stas_zero_based": list(group.sta_ids),
            "predicted_group_utility_gbps": float(scores[index]),
            "measured_after_selection_utility_mbps": float(scenario.rewards_bps[index] / 1e6),
        })
    result = {
        "scheduler": "LinUCB contextual combinatorial bandit",
        "ilp_called_at_inference": False,
        "scenario_seed": args.seed,
        "selected_groups": rows,
        "post_selection_measured_qos_utility_mbps": float(scenario.rewards_bps[selected].sum() / 1e6),
        "decision_time_ms": decision_time * 1e3,
        "constraint_violations": 0,
        "model_metadata": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
