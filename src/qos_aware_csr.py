"""Run the MATLAB QoS-aware CSR experiment using the Python simulator PHY.

Run from the Python project root:
    python src/qos_aware_csr.py --iterations 1 --sa-maxiter 5 --sa-maxfun 100

For the MATLAB experiment defaults, omit the tuning flags. The complete
100-iteration / full simulated-annealing experiment can be computationally
expensive because MATLAB also optimizes power for many candidate groups.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support both ``python src/qos_aware_csr.py`` and package imports.
    from . import utils
    from .constants import SYSTEM
    from .qos_scheduler import (
        assign_access_categories,
        build_feasible_candidate_groups,
        evaluate_conventional_group,
        evaluate_group,
        apply_matlab_as_written_reward_units,
        select_groups_greedy,
        select_groups_ilp,
    )
except ImportError:  # pragma: no cover - exercised by script execution
    import utils
    from constants import SYSTEM
    from qos_scheduler import (
        assign_access_categories,
        build_feasible_candidate_groups,
        evaluate_conventional_group,
        evaluate_group,
        apply_matlab_as_written_reward_units,
        select_groups_greedy,
        select_groups_ilp,
    )


DEFAULT_CONFIG: dict[str, Any] = {
    "ap_number": 4,
    "sta_number": 16,
    "grid_value": 48.0,
    "scenario_type": "grid",
    "pn_dbm": -95.0,
    "cca_dbm": -82.0,
    "nss": 2,
    "iterations": 100,
    "seed": 2,
    "ac_names": ["VO", "VI", "BE", "BK"],
    "ac_weights": [2.5, 1.40, 1.00, 0.45],
    "ac_sampling_weights": [1.0, 1.0, 1.0, 2.5],
    "txop_by_ac_s": [1.5e-3, 3.0e-3, 1.0e-3, 1.0e-3],
    "packet_bits": 12_000,
    "links": [
        {"name": "6 GHz", "bandwidth_mhz": 80, "frequency_ghz": 6.035},
        {"name": "5 GHz", "bandwidth_mhz": 40, "frequency_ghz": 5.22},
        {"name": "2.4 GHz", "bandwidth_mhz": 20, "frequency_ghz": 2.412},
    ],
    "cg_size": 4,
    "sa_maxiter": 500,
    "sa_maxfun": 10_000,
    "reward_units_mode": "matlab_as_written",
}


def _matlab_linspace(start: float, stop: float, num: int) -> np.ndarray:
    """NumPy-compatible array with MATLAB's special one-point linspace rule."""
    if num <= 0:
        return np.empty(0, dtype=float)
    if num == 1:
        return np.array([stop], dtype=float)
    return np.linspace(start, stop, num, dtype=float)


def matlab_qos_topology(
    ap_number: int = 4,
    sta_number: int = 16,
    grid_value: float = 48.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the placement distributions in QoS/AP_STA_coordinates.m."""
    if ap_number != 4:
        raise ValueError(
            "The MATLAB reference hardcodes four AP positions; ap_number must be 4")
    if sta_number % ap_number != 0:
        raise ValueError(
            "sta_number must be divisible by ap_number for the grid association")
    if rng is None:
        rng = np.random.default_rng(2)

    quarter = grid_value / 4.0
    ap_matrix = np.array(
        [
            [quarter, quarter],
            [quarter, 3.0 * quarter],
            [3.0 * quarter, quarter],
            [3.0 * quarter, 3.0 * quarter],
        ],
        dtype=float,
    )
    sta_matrix = np.zeros((sta_number, 2), dtype=float)
    stations_per_ap = sta_number // ap_number
    max_distance = grid_value / 4.0

    for ap_index in range(ap_number):
        # Preserve the exact count and distance expressions in the MATLAB file.
        # MATLAB round() uses half-away-from-zero; floor(x + 0.5) matches it for these positive counts.
        def matlab_round(value): return int(np.floor(value + 0.5))
        num_mid = max(1, matlab_round(0.02 * stations_per_ap))
        num_near = matlab_round((stations_per_ap - num_mid) / 2.0)
        num_far = stations_per_ap - num_near - num_mid

        near = _matlab_linspace(1.0, 0.15 * max_distance,
                                num_near) + 0.2 * rng.random(num_near)
        mid = 0.4 * max_distance + 0.05 * max_distance * rng.random(num_mid)
        far = _matlab_linspace(
            0.75 * max_distance, max_distance, num_far) + 0.2 * rng.random(num_far)
        distances = np.concatenate((near, mid, far))
        rng.shuffle(distances)

        for local_index, distance in enumerate(distances):
            theta = 2.0 * np.pi * rng.random()
            x = ap_matrix[ap_index, 0] + distance * np.cos(theta)
            y = ap_matrix[ap_index, 1] + distance * np.sin(theta)
            global_index = ap_index * stations_per_ap + local_index
            sta_matrix[global_index, 0] = np.clip(x, 0.0, grid_value)
            sta_matrix[global_index, 1] = np.clip(y, 0.0, grid_value)

    # MATLAB rerolls the entire placement only for exact duplicate coordinates
    # or a STA exactly coinciding with an AP; these events are vanishingly rare.
    if np.unique(sta_matrix, axis=0).shape[0] != sta_number:
        raise RuntimeError(
            "MATLAB-compatible station placement produced duplicate coordinates")
    if any(np.any(np.all(np.isclose(sta_matrix, ap), axis=1)) for ap in ap_matrix):
        raise RuntimeError("A STA coincides with an AP")

    walls = np.array(
        [
            [0.0, grid_value, grid_value / 2.0, grid_value / 2.0],
            [grid_value / 2.0, grid_value / 2.0, 0.0, grid_value],
        ],
        dtype=float,
    )
    return ap_matrix, sta_matrix, walls


def _print_iteration(iteration: int, throughput_bps: np.ndarray) -> None:
    values = " | ".join(f"{value / 1e6:.2f} Mbps" for value in throughput_bps)
    print(f"Iteration {iteration:03d}: Conventional-ILP={values.split(' | ')[0]} | "
          f"QoS-ILP={values.split(' | ')[1]} | QoS-Greedy={values.split(' | ')[2]}")


def run_experiment(
    *,
    iterations: int = 100,
    seed: int = 2,
    sa_maxiter: int = 500,
    sa_maxfun: int = 10_000,
    output_dir: str | Path | None = "results/qos_aware_csr",
    verbose: bool = True,
    reward_units_mode: str = "matlab_as_written",
) -> dict[str, Any]:
    """Execute conventional ILP, proposed QoS ILP and proposed QoS Greedy."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if sa_maxiter < 1 or sa_maxfun < 1:
        raise ValueError("SA budgets must be positive")
    if reward_units_mode not in {"matlab_as_written", "physical_consistent"}:
        raise ValueError(
            "reward_units_mode must be matlab_as_written or physical_consistent")

    # MATLAB rng(2) controls topology, shadowing and AC sampling. NumPy's RNG
    # is deterministic but will not generate the same random stream as MATLAB.
    # Preserve legacy utility behaviour outside the explicit RNG path.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    ap_matrix, sta_matrix, walls = matlab_qos_topology(
        DEFAULT_CONFIG["ap_number"], DEFAULT_CONFIG["sta_number"],
        DEFAULT_CONFIG["grid_value"], rng,
    )
    association = utils.ap_sta_association(
        DEFAULT_CONFIG["ap_number"], DEFAULT_CONFIG["sta_number"], "grid"
    )

    n_aps = DEFAULT_CONFIG["ap_number"]
    n_stas = DEFAULT_CONFIG["sta_number"]
    n_links = len(DEFAULT_CONFIG["links"])
    n_methods = 3
    ac_weights = np.asarray(DEFAULT_CONFIG["ac_weights"], dtype=float)
    txop_by_ac = np.asarray(DEFAULT_CONFIG["txop_by_ac_s"], dtype=float)
    packet_bits = int(DEFAULT_CONFIG["packet_bits"])
    noise_power_mw = 10.0 ** (DEFAULT_CONFIG["pn_dbm"] / 10.0)
    csr_overhead = float(utils.overheads_calc("BE")["csr_overheads"])

    throughput_results = np.zeros((iterations, n_methods), dtype=float)
    avg_thr_all_ac = np.zeros((n_methods, iterations, 4), dtype=float)
    ac_scheduled_counts = np.zeros((n_methods, 4), dtype=int)
    execution_times = np.zeros(n_methods, dtype=float)
    selected_group_log: list[list[dict[str, Any]]] = []
    max_tx_power_mw_by_link = np.zeros(n_links, dtype=float)

    if verbose:
        print("\n=== MATLAB QoS reference topology generated ===")
        print(f"AP positions:\n{ap_matrix}")
        print(f"STA positions:\n{sta_matrix}")
        print(f"CSR overhead (BE): {csr_overhead * 1e6:.3f} us")
        print(f"Candidate-group size limit: {DEFAULT_CONFIG['cg_size']}")

    for iteration in range(iterations):
        # MATLAB randsample(1:4, STA_number, true, [1 1 1 2.5]).
        sta_ac = assign_access_categories(
            n_stas, rng, DEFAULT_CONFIG["ac_sampling_weights"]
        )

        all_groups = []
        for link_id, link in enumerate(DEFAULT_CONFIG["links"]):
            max_tx_power_dbm, nsc = utils.tx_power_calc(
                link["bandwidth_mhz"], DEFAULT_CONFIG["nss"])
            max_tx_power_mw_by_link[link_id] = 10.0 ** (
                max_tx_power_dbm / 10.0)
            channel_matrix = utils.get_channel_matrix(
                max_tx_power_dbm,
                ap_matrix,
                sta_matrix,
                "grid",
                walls,
                cca=DEFAULT_CONFIG["cca_dbm"],
                frequency_ghz=link["frequency_ghz"],
                rng=rng,
            )
            link_groups = build_feasible_candidate_groups(
                link_id=link_id,
                association=association,
                channel_matrix=channel_matrix,
                max_tx_power_dbm=max_tx_power_dbm,
                noise_power_mw=noise_power_mw,
                n_subcarriers=nsc,
                nss=DEFAULT_CONFIG["nss"],
                csr_overhead_s=csr_overhead,
                cg_size=DEFAULT_CONFIG["cg_size"],
                rng=rng,
                sa_maxiter=sa_maxiter,
                sa_maxfun=sa_maxfun,
            )
            all_groups.extend(link_groups)
            if verbose:
                print(
                    f"  {link['name']}: BW={link['bandwidth_mhz']} MHz, "
                    f"f={link['frequency_ghz']} GHz, Nsc={nsc}, "
                    f"Pmax={max_tx_power_dbm:.3f} dBm, feasible groups={len(link_groups)}"
                )

        if not all_groups:
            raise RuntimeError("No candidate groups generated")

        if reward_units_mode == "matlab_as_written":
            # Match Final_Code_010926.m's reward-loop unit handling: noise and
            # singleton power are W; optimized multi-STA powers remain mW.
            noise_power_w = noise_power_mw / 1000.0
            evaluation_groups = [
                apply_matlab_as_written_reward_units(
                    group,
                    noise_power_w=noise_power_w,
                    max_tx_power_w=max_tx_power_mw_by_link[group.link_id] / 1000.0,
                    nss=DEFAULT_CONFIG["nss"],
                )
                for group in all_groups
            ]
        else:
            evaluation_groups = all_groups

        conventional_evaluations = [
            evaluate_conventional_group(
                group, sta_ac, ac_weights, txop_by_ac, packet_bits)
            for group in evaluation_groups
        ]
        qos_evaluations = [
            evaluate_group(group, sta_ac, ac_weights, txop_by_ac, packet_bits)
            for group in evaluation_groups
        ]
        conventional_rewards = np.array(
            [item.reward for item in conventional_evaluations])
        qos_rewards = np.array(
            [item.weighted_reward for item in qos_evaluations])

        start = time.perf_counter()
        conventional_selected, _ = select_groups_ilp(
            all_groups, conventional_rewards, n_stas, n_links
        )
        execution_times[0] += time.perf_counter() - start

        start = time.perf_counter()
        qos_ilp_selected, _ = select_groups_ilp(
            all_groups, qos_rewards, n_stas, n_links)
        execution_times[1] += time.perf_counter() - start

        start = time.perf_counter()
        qos_greedy_selected, _ = select_groups_greedy(
            all_groups, qos_rewards, n_stas, n_links)
        execution_times[2] += time.perf_counter() - start

        selections = [conventional_selected,
                      qos_ilp_selected, qos_greedy_selected]
        iteration_log: list[dict[str, Any]] = []
        for method_index, selected_indices in enumerate(selections):
            ac_throughput = np.zeros(4, dtype=float)
            ac_served = np.zeros(4, dtype=int)
            total_throughput = 0.0
            method_groups: list[dict[str, Any]] = []

            for group_index in selected_indices:
                group = evaluation_groups[group_index]
                evaluation = (
                    conventional_evaluations[group_index]
                    if method_index == 0
                    else qos_evaluations[group_index]
                )
                total_throughput += float(evaluation.rates_bps.sum())
                for local_index, sta in enumerate(group.sta_ids):
                    ac_index = int(sta_ac[sta])
                    rate = float(evaluation.rates_bps[local_index])
                    ac_throughput[ac_index] += rate
                    ac_served[ac_index] += 1
                    ac_scheduled_counts[method_index, ac_index] += 1

                method_groups.append(
                    {
                        "link_id": group.link_id,
                        "stas_zero_based": list(group.sta_ids),
                        "aps_zero_based": list(group.ap_ids),
                        "sinr_db": group.sinr_db.tolist(),
                        "mcs": group.mcs.tolist(),
                        "tx_power_mw": group.tx_power_mw.tolist(),
                        "filter_sinr_db": all_groups[group_index].sinr_db.tolist(),
                    }
                )

            throughput_results[iteration, method_index] = total_throughput
            avg_thr_all_ac[method_index, iteration,
                           :] = ac_throughput / np.maximum(ac_served, 1)
            iteration_log.append(
                {"method_index": method_index, "selected_groups": method_groups})

        selected_group_log.append(iteration_log)
        if verbose:
            _print_iteration(iteration + 1, throughput_results[iteration])

    method_names = [
        "Conventional (ILP)", "Proposed QoS (ILP)", "Proposed QoS (Greedy)"]
    mean_bps = throughput_results.mean(axis=0)
    baseline = mean_bps[0]
    improvement_pct = np.zeros_like(mean_bps) if baseline == 0 else (
        mean_bps - baseline) / baseline * 100.0
    summary = {
        "method_names": method_names,
        "mean_throughput_mbps": (mean_bps / 1e6).tolist(),
        "improvement_vs_conventional_pct": improvement_pct.tolist(),
        "mean_scheduler_time_per_iteration_s": (execution_times / iterations).tolist(),
        "total_scheduler_time_s": execution_times.tolist(),
        "ac_scheduled_counts": ac_scheduled_counts.tolist(),
        "mean_per_ac_throughput_mbps": (avg_thr_all_ac.mean(axis=1) / 1e6).tolist(),
        "iterations": iterations,
        "seed": seed,
        "sa_maxiter": sa_maxiter,
        "sa_maxfun": sa_maxfun,
        "reward_units_mode": reward_units_mode,
        "notes": [
            "NumPy random streams are deterministic but not identical to MATLAB rng(2).",
            "SciPy dual_annealing matches the bounded sum(log(rate)) SA objective and key budgets, but not MATLAB solver trajectories.",
            "matlab_as_written reproduces the reward-loop mixed W/mW handling in Final_Code_010926.m; physical_consistent keeps all powers/noise in mW.",
            "This reproduces the static analytical multi-link experiment, not MAPCsim's packet-level event loop.",
        ],
    }

    if verbose:
        print("\n=== Mean network throughput ===")
        for name, value, improvement in zip(method_names, summary["mean_throughput_mbps"], improvement_pct):
            print(
                f"{name:26s}: {value:9.3f} Mbps ({improvement:+.2f}% vs conventional)")
        print("\n=== Mean scheduler time per iteration ===")
        for name, value in zip(method_names, summary["mean_scheduler_time_per_iteration_s"]):
            print(f"{name:26s}: {value:.6f} s")
        print("\n=== AC scheduling counts ===")
        print("Rows=methods; columns=VO, VI, BE, BK")
        print(ac_scheduled_counts)

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "qos_aware_csr_results.npz",
            throughput_results_bps=throughput_results,
            avg_thr_all_ac_bps=avg_thr_all_ac,
            ac_scheduled_counts=ac_scheduled_counts,
            execution_times_s=execution_times,
        )
        (destination / "summary.json").write_text(json.dumps(summary,
                                                             indent=2), encoding="utf-8")
        (destination / "selected_groups.json").write_text(
            json.dumps(selected_group_log, indent=2), encoding="utf-8")
        if verbose:
            print(f"\nResults saved to: {destination.resolve()}")

    return {
        "throughput_results_bps": throughput_results,
        "avg_thr_all_ac_bps": avg_thr_all_ac,
        "ac_scheduled_counts": ac_scheduled_counts,
        "execution_times_s": execution_times,
        "summary": summary,
        "ap_matrix": ap_matrix,
        "sta_matrix": sta_matrix,
        "walls": walls,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int,
                        default=DEFAULT_CONFIG["iterations"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--sa-maxiter", type=int,
                        default=DEFAULT_CONFIG["sa_maxiter"])
    parser.add_argument("--sa-maxfun", type=int,
                        default=DEFAULT_CONFIG["sa_maxfun"])
    parser.add_argument("--output-dir", type=str,
                        default="results/qos_aware_csr")
    parser.add_argument("--reward-units-mode", choices=("matlab_as_written", "physical_consistent"),
                        default=DEFAULT_CONFIG["reward_units_mode"], help="Use exact Final_Code_010926.m reward-loop units or consistent physical units")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress iteration and summary output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_experiment(
            iterations=args.iterations,
            seed=args.seed,
            sa_maxiter=args.sa_maxiter,
            sa_maxfun=args.sa_maxfun,
            output_dir=args.output_dir,
            verbose=not args.quiet,
            reward_units_mode=args.reward_units_mode,
        )
    except Exception as exc:
        print(f"QoS-aware CSR experiment failed: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
