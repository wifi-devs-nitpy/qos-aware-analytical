"""Run conventional and QoS-aware ILP baselines on the exact LinUCB test scenarios.

This baseline-only runner is optional: the main qos_contextual_bandit.py final
experiment already evaluates both ILPs on the same held-out scenarios. Use this
script only when you need to reproduce the ILP baselines separately (for example,
if LinUCB was trained/evaluated using a standalone workflow).

The test scenario seed formula and topology generation are deliberately kept
identical to qos_contextual_bandit.run_contextual_bandit_experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .qos_ai import generate_scenario
    from .qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from .qos_contextual_bandit import N_AC, N_LINKS, N_STAS, _selection_metrics, _mean_ci
    from .qos_scheduler import evaluate_conventional_group, select_groups_ilp, validate_selection
except ImportError:  # script-style imports
    from qos_ai import generate_scenario
    from qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from qos_contextual_bandit import N_AC, N_LINKS, N_STAS, _selection_metrics, _mean_ci
    from qos_scheduler import evaluate_conventional_group, select_groups_ilp, validate_selection


def run_ilp_baselines(
    *,
    test_scenarios_count: int = 100,
    seed: int = 20261110,
    topology_seed: int = 2,
    sa_maxiter: int = 20,
    sa_maxfun: int = 10_000,
    output_dir: str | Path = "results/qos_ilp_baselines",
) -> dict[str, Any]:
    if test_scenarios_count < 1:
        raise ValueError("test_scenarios_count must be positive")
    if sa_maxiter < 1 or sa_maxfun < 1:
        raise ValueError("SA budgets must be positive")

    ap_matrix, sta_matrix, walls = matlab_qos_topology(
        DEFAULT_CONFIG["ap_number"],
        DEFAULT_CONFIG["sta_number"],
        DEFAULT_CONFIG["grid_value"],
        np.random.default_rng(topology_seed),
    )

    method_names = ["Conventional CSR (ILP)", "QoS-aware CSR (ILP)"]
    ac_names = list(DEFAULT_CONFIG["ac_names"])
    throughput_rows: list[list[float]] = []
    qos_utility_rows: list[list[float]] = []
    ac_avg_rows: list[np.ndarray] = []
    ac_total_rows: list[np.ndarray] = []
    ac_count_rows: list[np.ndarray] = []
    scheduler_time_rows: list[list[float]] = []
    scenario_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for i in range(test_scenarios_count):
        # Same held-out seed formula as qos_contextual_bandit.py.
        scenario_seed = seed + 1_000_000 + i * 1009
        print(
            f"ILP baseline scenario {i + 1}/{test_scenarios_count} (seed={scenario_seed})...", flush=True)
        scenario = generate_scenario(
            seed=scenario_seed,
            ap_matrix=ap_matrix,
            sta_matrix=sta_matrix,
            walls=walls,
            sa_maxiter=sa_maxiter,
            sa_maxfun=sa_maxfun,
            reward_units_mode="matlab_as_written",
            include_expert=False,
        )

        # Conventional CSR: unweighted delivered rates, each STA's own TXOP.
        conventional_rewards = np.asarray([
            evaluate_conventional_group(
                group,
                scenario.sta_ac,
                DEFAULT_CONFIG["ac_weights"],
                DEFAULT_CONFIG["txop_by_ac_s"],
                int(DEFAULT_CONFIG["packet_bits"]),
            ).reward
            for group in scenario.groups
        ], dtype=np.float64)
        start = time.perf_counter()
        conventional_selected, conventional_objective = select_groups_ilp(
            scenario.groups, conventional_rewards, n_stas=N_STAS, n_links=N_LINKS
        )
        conventional_time = time.perf_counter() - start
        validate_selection(
            scenario.groups, conventional_selected, N_STAS, N_LINKS)

        # QoS-aware CSR: AC-weighted rewards with group-aligned TXOP.
        start = time.perf_counter()
        qos_selected, qos_objective = select_groups_ilp(
            scenario.groups, scenario.rewards_bps, n_stas=N_STAS, n_links=N_LINKS
        )
        qos_time = time.perf_counter() - start
        validate_selection(scenario.groups, qos_selected, N_STAS, N_LINKS)

        conventional_metrics = _selection_metrics(
            scenario, conventional_selected, conventional=True)
        qos_metrics = _selection_metrics(
            scenario, qos_selected, conventional=False)
        metrics = [conventional_metrics, qos_metrics]

        throughput_rows.append(
            [m["network_throughput_bps"] / 1e6 for m in metrics])
        qos_utility_rows.append([
            conventional_metrics["qos_weighted_utility_bps"] / 1e6,
            qos_metrics["qos_weighted_utility_bps"] / 1e6,
        ])
        ac_avg_rows.append(
            np.vstack([m["ac_average_per_sta_bps"] for m in metrics]) / 1e6)
        ac_total_rows.append(
            np.vstack([m["ac_rate_totals_bps"] for m in metrics]) / 1e6)
        ac_count_rows.append(
            np.vstack([m["ac_scheduled_counts"] for m in metrics]))
        scheduler_time_rows.append([conventional_time, qos_time])

        scenario_rows.append({
            "test_scenario": i + 1,
            "scenario_seed": scenario_seed,
            "candidate_group_count": len(scenario.groups),
            "sta_access_categories_zero_based": np.asarray(scenario.sta_ac, dtype=int).tolist(),
            "conventional_throughput_mbps": throughput_rows[-1][0],
            "qos_ilp_throughput_mbps": throughput_rows[-1][1],
            "conventional_qos_weighted_utility_mbps": qos_utility_rows[-1][0],
            "qos_ilp_weighted_utility_mbps": qos_utility_rows[-1][1],
            "conventional_selected_indices": [int(x) for x in conventional_selected],
            "qos_ilp_selected_indices": [int(x) for x in qos_selected],
            "conventional_solver_objective_bps": float(conventional_objective),
            "qos_ilp_solver_objective_bps": float(qos_objective),
        })
        selected_rows.append({
            "test_scenario": i + 1,
            "scenario_seed": scenario_seed,
            "methods": [
                {"method": method_names[0], "selected_groups": [
                    {"link_id": int(scenario.groups[j].link_id),
                     "stas_zero_based": list(scenario.groups[j].sta_ids),
                     "aps_zero_based": list(scenario.groups[j].ap_ids)}
                    for j in conventional_selected]},
                {"method": method_names[1], "selected_groups": [
                    {"link_id": int(scenario.groups[j].link_id),
                     "stas_zero_based": list(scenario.groups[j].sta_ids),
                     "aps_zero_based": list(scenario.groups[j].ap_ids)}
                    for j in qos_selected]},
            ],
        })

    throughput = np.asarray(throughput_rows, dtype=float)
    qos_util = np.asarray(qos_utility_rows, dtype=float)
    ac_avg = np.asarray(ac_avg_rows, dtype=float)
    ac_total = np.asarray(ac_total_rows, dtype=float)
    ac_counts_per_scenario = np.asarray(ac_count_rows, dtype=np.int64)
    scheduler_times = np.asarray(scheduler_time_rows, dtype=float)

    mean_throughput = throughput.mean(axis=0)
    throughput_ci = [_mean_ci(throughput[:, j]) for j in range(2)]
    baseline = max(float(mean_throughput[0]), 1e-12)
    improvements = 100.0 * (mean_throughput - mean_throughput[0]) / baseline
    total_times = scheduler_times.sum(axis=0)
    mean_times = scheduler_times.mean(axis=0)
    total_ac_counts = ac_counts_per_scenario.sum(axis=0)
    mean_ac_avg = ac_avg.mean(axis=0)
    mean_ac_total = ac_total.mean(axis=0)
    violations = 0  # validate_selection raises if any schedule is infeasible.

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "qos_ilp_baselines.npz",
        throughput_results_mbps=throughput,
        qos_weighted_utility_mbps=qos_util,
        ac_average_per_sta_throughput_mbps=ac_avg,
        ac_total_throughput_mbps=ac_total,
        ac_scheduled_counts_per_scenario=ac_counts_per_scenario,
        ac_scheduled_counts_total=total_ac_counts,
        scheduler_times_per_scenario_s=scheduler_times,
        scheduler_total_times_s=total_times,
        scheduler_mean_times_s=mean_times,
    )
    summary = {
        "experiment": "Conventional and QoS-aware ILP baselines on LinUCB held-out scenarios",
        "methods": method_names,
        "method_order": ["conventional_ilp", "qos_aware_ilp"],
        "scenario_counts": {"test": test_scenarios_count},
        "seed": seed,
        "topology_seed": topology_seed,
        "test_scenario_seed_formula": "seed + 1000000 + test_index * 1009",
        "sa_budget": {"maxiter": sa_maxiter, "maxfun": sa_maxfun},
        "reward_units_mode": "matlab_as_written",
        "throughput_summary": {
            "mean_throughput_mbps": mean_throughput.tolist(),
            "mean_throughput_gbps": (mean_throughput / 1000.0).tolist(),
            "improvement_vs_conventional_pct": improvements.tolist(),
            "per_method_statistics_mbps": throughput_ci,
        },
        "qos_weighted_utility_mean_mbps": qos_util.mean(axis=0).tolist(),
        "mean_per_ac_average_throughput_mbps": mean_ac_avg.tolist(),
        "mean_per_ac_total_throughput_mbps_per_scenario": mean_ac_total.tolist(),
        "ac_scheduled_counts": total_ac_counts.tolist(),
        "mean_scheduler_time_per_scenario_s": mean_times.tolist(),
        "total_scheduler_time_s": total_times.tolist(),
        "constraint_violations": violations,
        "topology": {
            "ap_positions": ap_matrix.tolist(),
            "sta_positions": sta_matrix.tolist(),
            "walls": walls.tolist(),
            "ap_count": int(DEFAULT_CONFIG["ap_number"]),
            "sta_count": int(DEFAULT_CONFIG["sta_number"]),
            "grid_value_m": float(DEFAULT_CONFIG["grid_value"]),
        },
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "test_scenarios.json").write_text(json.dumps(scenario_rows,
                                                                indent=2), encoding="utf-8")
    (destination / "selected_groups.json").write_text(json.dumps(selected_rows,
                                                                 indent=2), encoding="utf-8")
    with (destination / "scenario_throughput.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "scenario_seed", *method_names])
        for row in scenario_rows:
            writer.writerow([row["test_scenario"], row["scenario_seed"],
                             row["conventional_throughput_mbps"], row["qos_ilp_throughput_mbps"]])

    print("\nILP baseline summary")
    for i, name in enumerate(method_names):
        print(f"{name:28s}: {mean_throughput[i] / 1000.0:.6f} Gbps "
              f"({improvements[i]:+.2f}% vs conventional), mean solver={mean_times[i]:.6f} s/scenario")
    print(f"Constraint violations: {violations}")
    print(f"Saved to: {destination.resolve()}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--topology-seed", type=int, default=2)
    parser.add_argument("--sa-maxiter", type=int, default=20)
    parser.add_argument("--sa-maxfun", type=int, default=10_000)
    parser.add_argument("--output-dir", default="results/qos_ilp_baselines")
    args = parser.parse_args()
    run_ilp_baselines(
        test_scenarios_count=args.test_scenarios,
        seed=args.seed,
        topology_seed=args.topology_seed,
        sa_maxiter=args.sa_maxiter,
        sa_maxfun=args.sa_maxfun,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
