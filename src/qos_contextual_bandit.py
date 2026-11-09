"""Reward-driven contextual combinatorial bandit for QoS-aware CSR scheduling.

The learner receives candidate-group context and observes reward only for the
three groups it selects in each scenario (semi-bandit feedback). It does not
use ILP labels during training. At inference, a LinUCB model estimates each
candidate group's QoS utility; an exhaustive three-link feasibility decoder
selects the highest-scoring compatible schedule.

This is an analytical, static-scenario learning experiment. It does not use
MAPCsim, queues, packet events, or ILP during bandit training/inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import t as student_t

try:
    from .qos_ai import generate_scenario
    from .qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from .qos_scheduler import (
        CandidateGroup,
        calculate_ber_per,
        evaluate_group,
        evaluate_conventional_group,
        select_groups_ilp,
        validate_selection,
    )
except ImportError:  # pragma: no cover - script-style imports
    from qos_ai import generate_scenario
    from qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from qos_scheduler import (
        CandidateGroup,
        calculate_ber_per,
        evaluate_group,
        evaluate_conventional_group,
        select_groups_ilp,
        validate_selection,
    )

N_STAS = int(DEFAULT_CONFIG["sta_number"])
N_LINKS = len(DEFAULT_CONFIG["links"])
N_AC = len(DEFAULT_CONFIG["ac_weights"])
N_APS = int(DEFAULT_CONFIG["ap_number"])

# Feature layout:
# bias(1) + link(3) + group size(1) + AC counts(4) + per-AC effective PHY
# rate(4) + SINR stats(3) + MCS stats(3) + PER stats(2) + TXOP(1) + CSR
# overhead(1) + mean log-power(1) + AP counts(4) + STA mask(16) + per-STA
# candidate conflict degree(16) + candidates/link(3).
FEATURE_DIM = (
    1 + N_LINKS + 1 + N_AC + N_AC + 3 + 3 + 2 + 1 + 1 + 1
    + N_APS + N_STAS + N_STAS + N_LINKS
)


@dataclass
class LinUCBContextualBandit:
    """Linear UCB model with semi-bandit feedback for selected candidate groups."""

    dimension: int = FEATURE_DIM
    alpha: float = 0.7
    ridge: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        self.A = np.eye(self.dimension, dtype=np.float64) * self.ridge
        self.b = np.zeros(self.dimension, dtype=np.float64)
        self.n_observations = 0

    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def predict(self, features: np.ndarray, *, explore: bool = True) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"features must have shape (n, {self.dimension})")
        theta = self.theta
        mean = x @ theta
        try:
            solved = np.linalg.solve(self.A, x.T).T
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise RuntimeError("LinUCB covariance matrix is singular") from exc
        uncertainty = np.sqrt(np.maximum(np.sum(x * solved, axis=1), 0.0))
        scores = mean + self.alpha * uncertainty if explore else mean
        return scores, uncertainty

    def update(self, feature: np.ndarray, observed_reward_gbps: float) -> None:
        """Update from one selected group's observed reward (not unselected arms)."""
        x = np.asarray(feature, dtype=np.float64).reshape(-1)
        if x.shape != (self.dimension,):
            raise ValueError(f"feature must have shape ({self.dimension},)")
        if not np.all(np.isfinite(x)) or not np.isfinite(observed_reward_gbps):
            raise ValueError("feature and reward must be finite")
        self.A += np.outer(x, x)
        self.b += x * float(observed_reward_gbps)
        self.n_observations += 1

    def update_selected(
        self,
        features: np.ndarray,
        selected_indices: Sequence[int],
        observed_rewards_bps: Sequence[float],
    ) -> None:
        """Consume semi-bandit feedback for selected groups only."""
        rewards = np.asarray(observed_rewards_bps, dtype=float)
        if len(selected_indices) != rewards.size:
            raise ValueError("one observed reward is required per selected group")
        for index, reward_bps in zip(selected_indices, rewards):
            self.update(features[int(index)], float(reward_bps) / 1e9)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        np.savez_compressed(
            path,
            A=self.A,
            b=self.b,
            dimension=np.asarray(self.dimension),
            alpha=np.asarray(self.alpha),
            ridge=np.asarray(self.ridge),
            n_observations=np.asarray(self.n_observations),
            metadata_json=np.asarray(json.dumps(metadata or {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["LinUCBContextualBandit", dict[str, Any]]:
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                dimension=int(data["dimension"]),
                alpha=float(data["alpha"]),
                ridge=float(data["ridge"]),
            )
            model.A = np.asarray(data["A"], dtype=np.float64)
            model.b = np.asarray(data["b"], dtype=np.float64)
            model.n_observations = int(data["n_observations"])
            metadata = json.loads(str(data["metadata_json"].item()))
        return model, metadata


def build_group_context_features(
    groups: Sequence[CandidateGroup], sta_ac: Sequence[int]
) -> np.ndarray:
    """Build arm context without exposing exact reward or expert labels.

    Features contain link/group composition, AC-specific PHY potential,
    SINR/MCS/PER summaries, TXOP/overhead, AP membership and conflict density.
    The actual weighted reward is never passed into this function.
    """
    ac = np.asarray(sta_ac, dtype=int)
    if ac.shape != (N_STAS,) or np.any((ac < 0) | (ac >= N_AC)):
        raise ValueError(f"sta_ac must contain {N_STAS} valid access-category indices")
    if not groups:
        raise ValueError("groups must not be empty")

    group_counts_by_link = np.zeros(N_LINKS, dtype=np.float64)
    station_candidate_counts = np.zeros(N_STAS, dtype=np.float64)
    for group in groups:
        if not 0 <= group.link_id < N_LINKS:
            raise ValueError(f"invalid link id {group.link_id}")
        group_counts_by_link[group.link_id] += 1.0
        for sta in group.sta_ids:
            station_candidate_counts[sta] += 1.0

    rows: list[np.ndarray] = []
    for group in groups:
        members = np.asarray(group.sta_ids, dtype=int)
        n = len(members)
        if n == 0:
            raise ValueError("candidate groups cannot be empty")
        member_ac = ac[members]
        ac_counts = np.bincount(member_ac, minlength=N_AC).astype(np.float64) / 4.0

        per_values = np.ones(n, dtype=np.float64)
        effective_phy = np.zeros(n, dtype=np.float64)
        for local in range(n):
            if int(group.mcs[local]) >= 0:
                _, per = calculate_ber_per(float(group.sinr_db[local]), int(group.n_bps[local]), int(DEFAULT_CONFIG["packet_bits"]))
                per_values[local] = per
            effective_phy[local] = max(float(group.phy_rate_bps[local]), 0.0) * (1.0 - per_values[local])

        per_ac_phy = np.zeros(N_AC, dtype=np.float64)
        for local, ac_index in enumerate(member_ac):
            per_ac_phy[int(ac_index)] += effective_phy[local] / 1e9

        sinr = np.asarray(group.sinr_db, dtype=np.float64)
        mcs = np.asarray(group.mcs, dtype=np.float64)
        pwr = np.maximum(np.asarray(group.tx_power_mw, dtype=np.float64), 1e-12)
        txop = max(float(DEFAULT_CONFIG["txop_by_ac_s"][int(k)]) for k in member_ac)
        ap_counts = np.bincount(np.asarray(group.ap_ids, dtype=int), minlength=N_APS).astype(np.float64)[:N_APS] / 4.0
        sta_mask = np.zeros(N_STAS, dtype=np.float64)
        sta_mask[members] = 1.0
        conflicts = np.zeros(N_STAS, dtype=np.float64)
        conflicts[members] = station_candidate_counts[members] / 100.0

        row = np.concatenate([
            np.array([1.0]),
            np.eye(N_LINKS, dtype=np.float64)[group.link_id],
            np.array([n / 4.0]),
            ac_counts,
            np.clip(per_ac_phy, 0.0, 10.0),
            np.array([
                np.clip(np.min(sinr) / 80.0, -1.0, 1.0),
                np.clip(np.mean(sinr) / 80.0, -1.0, 1.0),
                np.clip(np.max(sinr) / 80.0, -1.0, 1.0),
            ]),
            np.array([
                np.clip(np.min(mcs) / 13.0, -1.0, 1.0),
                np.clip(np.mean(mcs) / 13.0, -1.0, 1.0),
                np.clip(np.max(mcs) / 13.0, -1.0, 1.0),
            ]),
            np.array([float(np.mean(per_values)), float(np.max(per_values))]),
            np.array([txop / 3e-3]),
            np.array([float(group.csr_overhead_s) / 3e-3]),
            np.array([float(np.mean(np.log10(pwr + 1e-12)) / 3.0)]),
            ap_counts,
            sta_mask,
            conflicts,
            group_counts_by_link / 200.0,
        ])
        if row.shape != (FEATURE_DIM,):
            raise RuntimeError(f"feature shape {row.shape} != expected {(FEATURE_DIM,)}")
        rows.append(row.astype(np.float64))

    result = np.vstack(rows)
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite contextual features generated")
    return result


def select_best_feasible_schedule(
    groups: Sequence[CandidateGroup], scores: Sequence[float], n_stas: int = N_STAS
) -> list[int]:
    """Exactly maximize additive model scores under cross-link STA constraints.

    There are only three links and typically tens of candidate groups/link, so
    exhaustive feasible-combination enumeration is small (usually <100k
    combinations). This decoder optimizes *predicted* scores, never true rewards.
    """
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(groups),) or not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be a finite vector with one value per group")

    by_link: list[list[int]] = [[] for _ in range(N_LINKS)]
    masks: list[int] = []
    for idx, group in enumerate(groups):
        if not 0 <= group.link_id < N_LINKS:
            raise ValueError(f"invalid link id {group.link_id}")
        if any(sta < 0 or sta >= n_stas for sta in group.sta_ids):
            raise ValueError("group contains an invalid STA index")
        by_link[group.link_id].append(idx)
        mask = 0
        for sta in group.sta_ids:
            mask |= 1 << int(sta)
        masks.append(mask)

    if any(not indices for indices in by_link):
        raise RuntimeError("each link must have at least one candidate group")

    best_score = -np.inf
    best_selection: list[int] | None = None
    for i0 in by_link[0]:
        m0 = masks[i0]
        s0 = score_array[i0]
        for i1 in by_link[1]:
            m1 = masks[i1]
            if m0 & m1:
                continue
            partial = s0 + score_array[i1]
            for i2 in by_link[2]:
                if (m0 | m1) & masks[i2]:
                    continue
                total = partial + score_array[i2]
                if total > best_score:
                    best_score = float(total)
                    best_selection = [i0, i1, i2]
    if best_selection is None:
        raise RuntimeError("no feasible one-group-per-link schedule exists")
    validate_selection(groups, best_selection, n_stas, N_LINKS)
    return best_selection


def select_bandit_schedule(
    model: LinUCBContextualBandit,
    features: np.ndarray,
    groups: Sequence[CandidateGroup],
    *,
    explore: bool,
) -> tuple[list[int], np.ndarray, float]:
    start = time.perf_counter()
    scores, uncertainty = model.predict(features, explore=explore)
    selected = select_best_feasible_schedule(groups, scores)
    elapsed = time.perf_counter() - start
    return selected, scores, elapsed


def _mean_ci(values: Sequence[float]) -> dict[str, float | str]:
    """Mean, sample standard deviation and two-sided 95% Student-t interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0, "std": 0.0, "ci95_half_width": 0.0,
            "ci95_low": 0.0, "ci95_high": 0.0,
            "ci95_method": "Student-t",
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    critical = float(student_t.ppf(0.975, df=arr.size - 1)) if arr.size > 1 else 0.0
    half = critical * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_half_width": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_method": "Student-t",
    }


def _selection_metrics(
    scenario: Any,
    selected_indices: Sequence[int],
    *,
    conventional: bool,
) -> dict[str, Any]:
    """Evaluate a selected schedule using the matching MATLAB method semantics.

    Conventional CSR uses per-STA TXOP and unweighted group throughput.
    QoS-aware CSR and LinUCB use group-aligned TXOP and AC-weighted selection
    utility. For every method, reported network throughput is the unweighted
    sum of the per-STA delivered rates, as in Final_Code_010926.m.
    """
    ac_rate_totals = np.zeros(N_AC, dtype=np.float64)
    ac_counts = np.zeros(N_AC, dtype=np.int64)
    network_throughput = 0.0
    qos_weighted_utility = 0.0
    for idx in selected_indices:
        group = scenario.groups[int(idx)]
        if conventional:
            evaluation = evaluate_conventional_group(
                group,
                scenario.sta_ac,
                DEFAULT_CONFIG["ac_weights"],
                DEFAULT_CONFIG["txop_by_ac_s"],
                int(DEFAULT_CONFIG["packet_bits"]),
            )
        else:
            evaluation = evaluate_group(
                group,
                scenario.sta_ac,
                DEFAULT_CONFIG["ac_weights"],
                DEFAULT_CONFIG["txop_by_ac_s"],
                int(DEFAULT_CONFIG["packet_bits"]),
            )
            qos_weighted_utility += float(evaluation.weighted_reward)

        network_throughput += float(np.sum(evaluation.rates_bps))
        for local, sta in enumerate(group.sta_ids):
            ac = int(scenario.sta_ac[sta])
            ac_rate_totals[ac] += float(evaluation.rates_bps[local])
            ac_counts[ac] += 1

    # This reproduces MATLAB's avgThr_all_AC calculation: each scenario first
    # computes mean throughput per served STA of an AC, using zero if unserved;
    # the report then averages those per-scenario values.
    ac_average_per_sta = ac_rate_totals / np.maximum(ac_counts, 1)
    return {
        "network_throughput_bps": float(network_throughput),
        "qos_weighted_utility_bps": float(qos_weighted_utility),
        "ac_rate_totals_bps": ac_rate_totals,
        "ac_scheduled_counts": ac_counts,
        "ac_average_per_sta_bps": ac_average_per_sta,
    }


def _save_matlab_style_figures(
    destination: Path,
    *,
    throughput_results_mbps: np.ndarray,
    ac_average_per_sta_mbps: np.ndarray,
    ac_scheduled_counts: np.ndarray,
    execution_times_s: np.ndarray,
    training_history: Sequence[dict[str, float]],
    ap_matrix: np.ndarray,
    sta_matrix: np.ndarray,
    walls: np.ndarray,
    method_names: Sequence[str],
    ac_names: Sequence[str],
) -> list[str]:
    """Create MATLAB-style figures for the three-method final comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        return []

    figure_names: list[str] = []

    # 1. First five test scenarios: network throughput comparison.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n = min(5, throughput_results_mbps.shape[0])
    x = np.arange(1, n + 1)
    width = 0.24
    for method_index, method_name in enumerate(method_names):
        ax.bar(x + (method_index - 1) * width,
               throughput_results_mbps[:n, method_index] / 1000.0,
               width, label=method_name)
    ax.set_xlabel("Test scenario", fontsize=15, fontweight="bold")
    ax.set_ylabel("Total network throughput (Gbps)", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=10)
    fig.tight_layout()
    name = "01_network_throughput_first_5_scenarios.png"
    fig.savefig(destination / name, dpi=180)
    plt.close(fig)
    figure_names.append(name)

    # 2. MATLAB-style mean per-AC throughput (mean per served STA per scenario).
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(N_AC)
    width = 0.24
    for method_index, method_name in enumerate(method_names):
        ax.bar(x + (method_index - 1) * width,
               ac_average_per_sta_mbps[method_index],
               width, label=method_name)
    ax.set_xlabel("Access category", fontsize=15, fontweight="bold")
    ax.set_ylabel("Average per-STA AC throughput (Mbps)", fontsize=15, fontweight="bold")
    ax.set_xticks(x, ac_names)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=10)
    fig.tight_layout()
    name = "02_average_throughput_per_access_category.png"
    fig.savefig(destination / name, dpi=180)
    plt.close(fig)
    figure_names.append(name)

    # 3. Fixed topology visualization.
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for wall in np.asarray(walls):
        ax.plot([wall[0], wall[1]], [wall[2], wall[3]], linewidth=3, label="Wall")
    ax.scatter(ap_matrix[:, 0], ap_matrix[:, 1], marker="^", s=150, label="Access point")
    ax.scatter(sta_matrix[:, 0], sta_matrix[:, 1], marker="o", s=65, label="Station")
    for i, (x0, y0) in enumerate(ap_matrix):
        ax.annotate(f"AP{i + 1}", (x0, y0), xytext=(-8, 8), textcoords="offset points")
    for i, (x0, y0) in enumerate(sta_matrix):
        ax.annotate(f"S{i + 1}", (x0, y0), xytext=(0, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel("X position (m)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y position (m)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, float(DEFAULT_CONFIG["grid_value"]))
    ax.set_ylim(0, float(DEFAULT_CONFIG["grid_value"]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    # De-duplicate wall labels in the legend.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=10)
    fig.tight_layout()
    name = "03_network_topology.png"
    fig.savefig(destination / name, dpi=180)
    plt.close(fig)
    figure_names.append(name)

    # 4. Total access-category scheduling counts over the test set.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(N_AC)
    width = 0.24
    for method_index, method_name in enumerate(method_names):
        ax.bar(x + (method_index - 1) * width,
               ac_scheduled_counts[method_index],
               width, label=method_name)
    ax.set_xlabel("Access category", fontsize=15, fontweight="bold")
    ax.set_ylabel("Total station allocations", fontsize=15, fontweight="bold")
    ax.set_xticks(x, ac_names)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=10)
    fig.tight_layout()
    name = "04_access_category_scheduling_counts.png"
    fig.savefig(destination / name, dpi=180)
    plt.close(fig)
    figure_names.append(name)

    # 5. Scheduler execution time over the full test set.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(np.arange(len(method_names)), execution_times_s)
    ax.set_xlabel("Scheduling method", fontsize=15, fontweight="bold")
    ax.set_ylabel("Total scheduler execution time (seconds)", fontsize=15, fontweight="bold")
    ax.set_xticks(np.arange(len(method_names)), method_names, rotation=12, ha="right")
    ax.grid(True, axis="y", alpha=0.35)
    for bar, value in zip(bars, execution_times_s):
        ax.annotate(f"{value:.3f} s", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    name = "05_scheduler_execution_time.png"
    fig.savefig(destination / name, dpi=180)
    plt.close(fig)
    figure_names.append(name)

    # 6. Bandit training reward trace (additional AI-specific plot).
    if training_history:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        episodes = np.asarray([row["episode"] for row in training_history], dtype=int)
        rewards = np.asarray([row["selected_qos_utility_mbps"] for row in training_history], dtype=float)
        ax.plot(episodes, rewards, linewidth=1.0, alpha=0.55, label="Episode reward")
        window = min(50, max(1, len(rewards) // 10))
        if len(rewards) >= window:
            moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(episodes[window - 1:], moving, linewidth=2.0, label=f"{window}-episode moving average")
        ax.set_xlabel("Training scenario", fontsize=15, fontweight="bold")
        ax.set_ylabel("Selected QoS utility (Mbps)", fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        name = "06_linucb_training_reward.png"
        fig.savefig(destination / name, dpi=180)
        plt.close(fig)
        figure_names.append(name)

    return figure_names


def run_contextual_bandit_experiment(
    *,
    train_scenarios_count: int = 1_000,
    test_scenarios_count: int = 100,
    seed: int = 20261110,
    topology_seed: int = 2,
    sa_maxiter: int = 20,
    sa_maxfun: int = 10_000,
    alpha: float = 0.7,
    ridge: float = 1.0,
    output_dir: str | Path = "results/qos_contextual_bandit_final",
) -> dict[str, Any]:
    """Train LinUCB and compare Conventional ILP, QoS-aware ILP and LinUCB.

    The test set is shared: every method receives the exact same generated
    scenario (topology, AC assignment, channel matrices, groups, powers and PHY
    quantities). Conventional ILP uses unweighted per-STA-TXOP group rewards;
    QoS-aware ILP uses AC-weighted, aligned-TXOP rewards; LinUCB selects from
    learned predicted scores and is evaluated using the same QoS-aligned rates.
    Greedy is intentionally excluded from the final comparison.
    """
    if train_scenarios_count < 1 or test_scenarios_count < 1:
        raise ValueError("train_scenarios_count and test_scenarios_count must be positive")
    if sa_maxiter < 1 or sa_maxfun < 1:
        raise ValueError("SA budgets must be positive")

    method_names = [
        "Conventional CSR (ILP)",
        "QoS-aware CSR (ILP)",
        "QoS-aware CSR (LinUCB)",
    ]
    ac_names = list(DEFAULT_CONFIG["ac_names"])
    n_methods = len(method_names)

    ap_matrix, sta_matrix, walls = matlab_qos_topology(
        DEFAULT_CONFIG["ap_number"], DEFAULT_CONFIG["sta_number"],
        DEFAULT_CONFIG["grid_value"], np.random.default_rng(topology_seed),
    )
    bandit = LinUCBContextualBandit(dimension=FEATURE_DIM, alpha=alpha, ridge=ridge)
    training_rows: list[dict[str, float]] = []
    training_started = time.perf_counter()

    # Online reward-driven training: no expert actions/ILP calls.
    for episode in range(train_scenarios_count):
        scenario_seed = seed + episode * 1009
        print(f"Bandit training scenario {episode + 1}/{train_scenarios_count} (seed={scenario_seed})...", flush=True)
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
        features = build_group_context_features(scenario.groups, scenario.sta_ac)
        selected, _scores, decision_time = select_bandit_schedule(
            bandit, features, scenario.groups, explore=True
        )
        validate_selection(scenario.groups, selected, N_STAS, N_LINKS)
        observed = [float(scenario.rewards_bps[idx]) for idx in selected]
        bandit.update_selected(features, selected, observed)
        episode_utility_mbps = sum(observed) / 1e6
        training_rows.append({
            "episode": episode + 1,
            "scenario_seed": scenario_seed,
            "selected_qos_utility_mbps": episode_utility_mbps,
            "decision_time_ms": decision_time * 1e3,
            "observed_group_rewards": len(selected),
            "cumulative_observations": bandit.n_observations,
        })
        if (episode + 1) % 25 == 0 or episode == 0 or episode + 1 == train_scenarios_count:
            print(
                f"  selected QoS utility={episode_utility_mbps:.2f} Mbps; "
                f"observations={bandit.n_observations}", flush=True
            )
    training_time_s = time.perf_counter() - training_started

    # Test: all three methods use the same scenario instance and candidates.
    test_rows: list[dict[str, Any]] = []
    throughput_rows_mbps: list[list[float]] = []
    qos_utility_rows_mbps: list[list[float | None]] = []
    ac_avg_rows_mbps: list[np.ndarray] = []
    ac_total_rows_mbps: list[np.ndarray] = []
    ac_count_rows: list[np.ndarray] = []
    scheduler_time_rows_s: list[list[float]] = []
    linucb_inference_times_ms: list[float] = []
    exact_matches = 0
    constraint_violations = 0
    total_candidate_groups = 0
    selected_group_log: list[dict[str, Any]] = []

    for test_index in range(test_scenarios_count):
        scenario_seed = seed + 1_000_000 + test_index * 1009
        print(f"Evaluating held-out scenario {test_index + 1}/{test_scenarios_count} (seed={scenario_seed})...", flush=True)
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
        total_candidate_groups += len(scenario.groups)

        # Conventional ILP objective: unweighted rates and individual STA TXOPs.
        conventional_evaluations = [
            evaluate_conventional_group(
                group, scenario.sta_ac, DEFAULT_CONFIG["ac_weights"],
                DEFAULT_CONFIG["txop_by_ac_s"], int(DEFAULT_CONFIG["packet_bits"]),
            )
            for group in scenario.groups
        ]
        conventional_rewards = np.asarray(
            [evaluation.reward for evaluation in conventional_evaluations], dtype=np.float64
        )
        start = time.perf_counter()
        conventional_selected, _ = select_groups_ilp(
            scenario.groups, conventional_rewards, n_stas=N_STAS, n_links=N_LINKS
        )
        conventional_time = time.perf_counter() - start
        validate_selection(scenario.groups, conventional_selected, N_STAS, N_LINKS)

        # QoS-aware ILP objective: weighted rates with common group TXOP.
        start = time.perf_counter()
        qos_ilp_selected, _ = select_groups_ilp(
            scenario.groups, scenario.rewards_bps, n_stas=N_STAS, n_links=N_LINKS
        )
        qos_ilp_time = time.perf_counter() - start
        validate_selection(scenario.groups, qos_ilp_selected, N_STAS, N_LINKS)

        # LinUCB uses its posterior mean at test time: no exploration bonus.
        features = build_group_context_features(scenario.groups, scenario.sta_ac)
        linucb_selected, predicted_scores, linucb_time = select_bandit_schedule(
            bandit, features, scenario.groups, explore=False
        )
        validate_selection(scenario.groups, linucb_selected, N_STAS, N_LINKS)
        linucb_inference_times_ms.append(linucb_time * 1e3)

        selections = [conventional_selected, qos_ilp_selected, linucb_selected]
        conventional_metrics = _selection_metrics(scenario, conventional_selected, conventional=True)
        qos_ilp_metrics = _selection_metrics(scenario, qos_ilp_selected, conventional=False)
        linucb_metrics = _selection_metrics(scenario, linucb_selected, conventional=False)
        metrics = [conventional_metrics, qos_ilp_metrics, linucb_metrics]

        scenario_throughput_mbps = [m["network_throughput_bps"] / 1e6 for m in metrics]
        scenario_qos_utility_mbps: list[float | None] = [
            None,
            qos_ilp_metrics["qos_weighted_utility_bps"] / 1e6,
            linucb_metrics["qos_weighted_utility_bps"] / 1e6,
        ]
        throughput_rows_mbps.append(scenario_throughput_mbps)
        qos_utility_rows_mbps.append(scenario_qos_utility_mbps)
        ac_avg_rows_mbps.append(np.vstack([m["ac_average_per_sta_bps"] for m in metrics]) / 1e6)
        ac_total_rows_mbps.append(np.vstack([m["ac_rate_totals_bps"] for m in metrics]) / 1e6)
        ac_count_rows.append(np.vstack([m["ac_scheduled_counts"] for m in metrics]))
        scheduler_time_rows_s.append([conventional_time, qos_ilp_time, linucb_time])

        exact_match = set(linucb_selected) == set(qos_ilp_selected)
        exact_matches += int(exact_match)
        scenario_violations = 0
        for method_selection in selections:
            try:
                validate_selection(scenario.groups, method_selection, N_STAS, N_LINKS)
            except Exception:
                scenario_violations += 1
        constraint_violations += scenario_violations

        test_rows.append({
            "test_scenario": test_index + 1,
            "scenario_seed": scenario_seed,
            "candidate_group_count": len(scenario.groups),
            "conventional_selected_indices": conventional_selected,
            "qos_ilp_selected_indices": qos_ilp_selected,
            "linucb_selected_indices": linucb_selected,
            "conventional_throughput_mbps": scenario_throughput_mbps[0],
            "qos_ilp_throughput_mbps": scenario_throughput_mbps[1],
            "linucb_throughput_mbps": scenario_throughput_mbps[2],
            "qos_ilp_weighted_utility_mbps": scenario_qos_utility_mbps[1],
            "linucb_weighted_utility_mbps": scenario_qos_utility_mbps[2],
            "linucb_to_qos_ilp_utility_pct": 100.0 * linucb_metrics["qos_weighted_utility_bps"] / max(qos_ilp_metrics["qos_weighted_utility_bps"], 1e-12),
            "linucb_to_qos_ilp_throughput_pct": 100.0 * linucb_metrics["network_throughput_bps"] / max(qos_ilp_metrics["network_throughput_bps"], 1e-12),
            "conventional_ac_throughput_mbps_vo_vi_be_bk": (conventional_metrics["ac_rate_totals_bps"] / 1e6).tolist(),
            "qos_ilp_ac_throughput_mbps_vo_vi_be_bk": (qos_ilp_metrics["ac_rate_totals_bps"] / 1e6).tolist(),
            "linucb_ac_throughput_mbps_vo_vi_be_bk": (linucb_metrics["ac_rate_totals_bps"] / 1e6).tolist(),
            "conventional_ac_scheduled_counts_vo_vi_be_bk": conventional_metrics["ac_scheduled_counts"].tolist(),
            "qos_ilp_ac_scheduled_counts_vo_vi_be_bk": qos_ilp_metrics["ac_scheduled_counts"].tolist(),
            "linucb_ac_scheduled_counts_vo_vi_be_bk": linucb_metrics["ac_scheduled_counts"].tolist(),
            "conventional_scheduler_time_s": conventional_time,
            "qos_ilp_scheduler_time_s": qos_ilp_time,
            "linucb_scheduler_time_s": linucb_time,
            "linucb_exact_match_qos_ilp": exact_match,
            "constraint_violations": scenario_violations,
        })

        group_details = []
        for method_name, method_selection in zip(method_names, selections):
            selected_detail = []
            for group_index in method_selection:
                group = scenario.groups[group_index]
                selected_detail.append({
                    "link_id": int(group.link_id),
                    "stas_zero_based": list(group.sta_ids),
                    "aps_zero_based": list(group.ap_ids),
                    "sinr_db": np.asarray(group.sinr_db).tolist(),
                    "mcs": np.asarray(group.mcs).tolist(),
                    "tx_power_mw": np.asarray(group.tx_power_mw).tolist(),
                })
            group_details.append({"method": method_name, "selected_groups": selected_detail})
        selected_group_log.append({"test_scenario": test_index + 1, "methods": group_details})

        print(
            f"Test {test_index + 1:03d}: Conv-ILP={scenario_throughput_mbps[0]:.2f} Mbps | "
            f"QoS-ILP={scenario_throughput_mbps[1]:.2f} Mbps | "
            f"LinUCB={scenario_throughput_mbps[2]:.2f} Mbps",
            flush=True,
        )

    throughput_arr = np.asarray(throughput_rows_mbps, dtype=np.float64)
    ac_avg_arr = np.asarray(ac_avg_rows_mbps, dtype=np.float64)  # test x method x AC
    ac_total_arr = np.asarray(ac_total_rows_mbps, dtype=np.float64)
    ac_count_arr = np.asarray(ac_count_rows, dtype=np.int64)
    scheduler_time_arr = np.asarray(scheduler_time_rows_s, dtype=np.float64)
    qos_utility_arr = np.asarray([
        [row[1] if row[1] is not None else np.nan, row[2] if row[2] is not None else np.nan]
        for row in qos_utility_rows_mbps
    ], dtype=np.float64)

    throughput_statistics = [_mean_ci(throughput_arr[:, i]) for i in range(n_methods)]
    mean_throughput_mbps = np.asarray([item["mean"] for item in throughput_statistics])
    baseline = max(float(mean_throughput_mbps[0]), 1e-12)
    improvement_pct = 100.0 * (mean_throughput_mbps - mean_throughput_mbps[0]) / baseline
    mean_scheduler_time = scheduler_time_arr.mean(axis=0)
    total_scheduler_time = scheduler_time_arr.sum(axis=0)
    ac_scheduled_counts = ac_count_arr.sum(axis=0)  # methods x AC, total over test scenarios
    mean_per_ac_avg = ac_avg_arr.mean(axis=0)  # methods x AC, MATLAB-style averaging
    mean_per_ac_total = ac_total_arr.mean(axis=0)  # methods x AC, mean total AC Mbps per scenario
    qos_utility_stats = {
        "qos_ilp_mbps": _mean_ci(qos_utility_arr[:, 0]),
        "linucb_mbps": _mean_ci(qos_utility_arr[:, 1]),
        "linucb_to_qos_ilp_pct": _mean_ci(100.0 * qos_utility_arr[:, 1] / np.maximum(qos_utility_arr[:, 0], 1e-12)),
        "linucb_to_qos_ilp_throughput_pct": _mean_ci(
            100.0 * throughput_arr[:, 2] / np.maximum(throughput_arr[:, 1], 1e-12)
        ),
    }
    exact_match_rate = 100.0 * exact_matches / test_scenarios_count

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure_names = _save_matlab_style_figures(
        destination,
        throughput_results_mbps=throughput_arr,
        ac_average_per_sta_mbps=mean_per_ac_avg,
        ac_scheduled_counts=ac_scheduled_counts,
        execution_times_s=total_scheduler_time,
        training_history=training_rows,
        ap_matrix=ap_matrix,
        sta_matrix=sta_matrix,
        walls=walls,
        method_names=method_names,
        ac_names=ac_names,
    )

    report: dict[str, Any] = {
        "experiment": "Final three-method QoS-aware multi-link CSR comparison",
        "methods": method_names,
        "method_order": ["conventional_ilp", "qos_aware_ilp", "linucb"],
        "learning_algorithm": "LinUCB contextual combinatorial bandit with semi-bandit feedback",
        "training_feedback": "Only selected candidate-group QoS rewards are revealed to LinUCB; unselected group rewards are not used for updates.",
        "training_ilp_calls": 0,
        "test_ilp_usage": "Conventional ILP and QoS-aware ILP are reference baselines on the same test scenarios; neither is used during LinUCB training or inference.",
        "action_decoder": "Exact maximum predicted-score schedule over three links, subject to disjoint STA membership.",
        "feature_dimension": FEATURE_DIM,
        "feature_excludes_exact_reward": True,
        "reward_units_mode": "matlab_as_written",
        "seed": seed,
        "topology_seed": topology_seed,
        "scenario_counts": {"train": train_scenarios_count, "test": test_scenarios_count},
        "bandit": {"algorithm": "LinUCB", "alpha": alpha, "ridge": ridge, "observations": bandit.n_observations},
        "sa_budget": {"maxiter": sa_maxiter, "maxfun": sa_maxfun},
        "topology": {
            "ap_positions": ap_matrix.tolist(),
            "sta_positions": sta_matrix.tolist(),
            "walls": walls.tolist(),
            "ap_count": N_APS,
            "sta_count": N_STAS,
            "grid_value_m": float(DEFAULT_CONFIG["grid_value"]),
        },
        "throughput_summary": {
            "mean_throughput_mbps": mean_throughput_mbps.tolist(),
            "mean_throughput_gbps": (mean_throughput_mbps / 1000.0).tolist(),
            "improvement_vs_conventional_pct": improvement_pct.tolist(),
            "per_method_statistics_mbps": throughput_statistics,
        },
        "qos_weighted_utility_summary": qos_utility_stats,
        "mean_scheduler_time_per_scenario_s": mean_scheduler_time.tolist(),
        "total_scheduler_time_s": total_scheduler_time.tolist(),
        "mean_linucb_inference_time_ms": float(np.mean(linucb_inference_times_ms)),
        "ac_scheduled_counts": ac_scheduled_counts.tolist(),
        "mean_per_ac_average_throughput_mbps": mean_per_ac_avg.tolist(),
        "mean_per_ac_total_throughput_mbps_per_scenario": mean_per_ac_total.tolist(),
        "ac_names": ac_names,
        "exact_linucb_schedule_matches_qos_ilp": exact_matches,
        "linucb_exact_match_rate_pct": exact_match_rate,
        "constraint_violations": constraint_violations,
        "candidate_group_count_total": total_candidate_groups,
        "training_time_s": training_time_s,
        "training_history": training_rows,
        "test_scenarios": test_rows,
        "figures": figure_names,
        "limitations": [
            "This is a static-scenario contextual combinatorial bandit, not temporal network-control RL.",
            "The topology geometry is fixed; access-category and channel realizations vary across scenarios.",
            "LinUCB is updated only from selected-group semi-bandit feedback and makes no ILP calls during training or inference.",
            "Conventional ILP uses unweighted per-STA TXOP rewards; QoS-aware ILP and LinUCB use AC-weighted aligned-TXOP rewards.",
            "Transmit-power optimization uses SciPy dual_annealing with the configured iteration and function-evaluation budgets; it is not guaranteed to find the global optimum and does not reproduce MATLAB solver trajectories exactly.",
            "The MATLAB-as-written mixed W/mW reward behavior is preserved for comparison parity.",
            "Weighted QoS preference is not a hard minimum-throughput guarantee for every access category.",
        ],
    }

    # Preserve the trained LinUCB state for standalone inference and reproducibility.
    bandit.save(
        destination / "qos_contextual_bandit.npz",
        {
            "algorithm": "LinUCB",
            "alpha": alpha,
            "ridge": ridge,
            "feature_dimension": FEATURE_DIM,
            "feature_description": "PHY/AC/link/group/conflict descriptors; exact reward excluded",
            "reward_units_mode": "matlab_as_written",
            "seed": seed,
            "topology_seed": topology_seed,
            "train_scenarios": train_scenarios_count,
            "test_scenarios": test_scenarios_count,
            "sa_maxiter": sa_maxiter,
            "sa_maxfun": sa_maxfun,
        },
    )

    np.savez_compressed(
        destination / "qos_final_results.npz",
        throughput_results_mbps=throughput_arr,
        qos_weighted_utility_mbps=qos_utility_arr,
        ac_average_per_sta_throughput_mbps=ac_avg_arr,
        ac_total_throughput_mbps=ac_total_arr,
        ac_scheduled_counts_per_scenario=ac_count_arr,
        ac_scheduled_counts_total=ac_scheduled_counts,
        scheduler_times_per_scenario_s=scheduler_time_arr,
        scheduler_total_times_s=total_scheduler_time,
        scheduler_mean_times_s=mean_scheduler_time,
        training_reward_mbps=np.asarray([row["selected_qos_utility_mbps"] for row in training_rows]),
    )
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (destination / "test_scenarios.json").write_text(json.dumps(test_rows, indent=2), encoding="utf-8")
    (destination / "selected_groups.json").write_text(json.dumps(selected_group_log, indent=2), encoding="utf-8")
    (destination / "topology.json").write_text(json.dumps(report["topology"], indent=2), encoding="utf-8")
    with (destination / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_rows[0].keys()))
        writer.writeheader()
        writer.writerows(training_rows)

    # MATLAB-style tabular exports for direct report/presentation use.
    with (destination / "final_comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", "Mean throughput (Gbps)", "Improvement vs Conventional (%)", "Mean scheduler time/scenario (s)", "Total scheduler time (s)"])
        for i, name in enumerate(method_names):
            writer.writerow([name, mean_throughput_mbps[i] / 1000.0, improvement_pct[i], mean_scheduler_time[i], total_scheduler_time[i]])
    with (destination / "per_ac_throughput.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", *ac_names])
        for i, name in enumerate(method_names):
            writer.writerow([name, *mean_per_ac_avg[i].tolist()])
    with (destination / "per_ac_total_throughput.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", *ac_names])
        for i, name in enumerate(method_names):
            writer.writerow([name, *mean_per_ac_total[i].tolist()])
    with (destination / "ac_scheduling_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", *ac_names])
        for i, name in enumerate(method_names):
            writer.writerow([name, *ac_scheduled_counts[i].tolist()])
    with (destination / "scheduler_execution_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", "Total time (s)", "Average time per scenario (s)"])
        for i, name in enumerate(method_names):
            writer.writerow([name, total_scheduler_time[i], mean_scheduler_time[i]])

    # Concise final report in the same order as the MATLAB console summaries.
    report_lines = [
        "# Final QoS-aware Multi-Link CSR Results",
        "",
        f"Training scenarios: {train_scenarios_count}  ",
        f"Test scenarios: {test_scenarios_count}  ",
        f"Simulated annealing budget: maxiter={sa_maxiter}, maxfun={sa_maxfun}  ",
        f"LinUCB observations: {bandit.n_observations}  ",
        "",
        "## Average network throughput",
        "",
        "| Method | Mean throughput (Gbps) | Improvement vs conventional |",
        "|---|---:|---:|",
    ]
    for i, name in enumerate(method_names):
        report_lines.append(f"| {name} | {mean_throughput_mbps[i] / 1000.0:.6f} | {improvement_pct[i]:+.2f}% |")
    report_lines += [
        "",
        "## QoS-weighted utility",
        "",
        f"- QoS-aware ILP: {qos_utility_stats['qos_ilp_mbps']['mean']:.3f} Mbps",
        f"- LinUCB: {qos_utility_stats['linucb_mbps']['mean']:.3f} Mbps",
        f"- LinUCB / QoS-aware ILP: {qos_utility_stats['linucb_to_qos_ilp_pct']['mean']:.2f}%",
        "",
        "## Average per-STA throughput by access category",
        "",
        "| Method | VO (Mbps) | VI (Mbps) | BE (Mbps) | BK (Mbps) |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, name in enumerate(method_names):
        vals = " | ".join(f"{v:.3f}" for v in mean_per_ac_avg[i])
        report_lines.append(f"| {name} | {vals} |")
    report_lines += [
        "",
        "## Access-category scheduling counts",
        "",
        "| Method | VO | VI | BE | BK |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, name in enumerate(method_names):
        vals = " | ".join(str(int(v)) for v in ac_scheduled_counts[i])
        report_lines.append(f"| {name} | {vals} |")
    report_lines += [
        "",
        "## Scheduler execution time",
        "",
        "| Method | Total time (s) | Mean per scenario (s) |",
        "|---|---:|---:|",
    ]
    for i, name in enumerate(method_names):
        report_lines.append(f"| {name} | {total_scheduler_time[i]:.6f} | {mean_scheduler_time[i]:.6f} |")
    report_lines += [
        "",
        "## Additional checks",
        "",
        f"- LinUCB exact schedule matches with QoS-aware ILP: {exact_matches}/{test_scenarios_count} ({exact_match_rate:.2f}%).",
        f"- Scheduling constraint violations: {constraint_violations}.",
        f"- Mean LinUCB inference time: {np.mean(linucb_inference_times_ms):.4f} ms.",
        f"- ILP calls during LinUCB training: 0.",
        "",
        "## Figures",
        "",
    ]
    report_lines.extend(f"- `{name}`" for name in figure_names)
    report_lines += [
        "",
        "## Interpretation and limitations",
        "",
        "This is a static-scenario contextual combinatorial bandit evaluation, not temporal packet-level network control. The AP/STA topology is fixed while AC assignments and channel realizations vary. LinUCB is trained using selected-group semi-bandit rewards only. Conventional ILP uses unweighted per-STA TXOP; QoS-aware ILP and LinUCB use AC-weighted, aligned-TXOP rewards. Simulated annealing is a heuristic and does not guarantee globally optimal transmit powers. The MATLAB-as-written mixed W/mW reward behavior is preserved for parity. AC weights provide preferences, not hard minimum-throughput guarantees.",
    ]
    (destination / "FINAL_RESULTS_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Plain-text summary follows the layout of the MATLAB reference output.
    print("\n==================================================")
    print("       AC SCHEDULING SUMMARY (TEST SCENARIOS)     ")
    print("==================================================")
    print("Rows=methods; columns=VO, VI, BE, BK")
    print(ac_scheduled_counts)
    print("\n==================================================")
    print("       COMPUTATIONAL EXECUTION TIME SUMMARY      ")
    print("==================================================")
    for name, total, average in zip(method_names, total_scheduler_time, mean_scheduler_time):
        print(f"{name:30s}: total={total:.6f} s | average/scenario={average:.6f} s")
    print("\n==================================================")
    print("       AVERAGE NETWORK THROUGHPUT SUMMARY        ")
    print("==================================================")
    for name, value, improvement in zip(method_names, mean_throughput_mbps, improvement_pct):
        print(f"{name:30s}: {value / 1000.0:.6f} Gbps ({improvement:+.2f}% vs conventional)")
    print("\n=== Mean throughput per access category (Mbps; MATLAB-style per-served-STA average) ===")
    print("Columns=VO, VI, BE, BK")
    print(mean_per_ac_avg)
    print("\n=== QoS-weighted utility comparison ===")
    print(f"QoS-aware ILP: {qos_utility_stats['qos_ilp_mbps']['mean']:.3f} Mbps")
    print(f"LinUCB:        {qos_utility_stats['linucb_mbps']['mean']:.3f} Mbps")
    print(f"LinUCB / QoS-ILP utility: {qos_utility_stats['linucb_to_qos_ilp_pct']['mean']:.2f}%")
    print(f"LinUCB exact schedule matches QoS-ILP: {exact_matches}/{test_scenarios_count} ({exact_match_rate:.2f}%)")
    print(f"Constraint violations: {constraint_violations}")
    print(f"Training observations: {bandit.n_observations}; training ILP calls: 0")
    print(f"Mean LinUCB inference: {np.mean(linucb_inference_times_ms):.4f} ms")
    print(f"\nResults and MATLAB-style figures saved to: {destination.resolve()}")
    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenarios", type=int, default=1_000)
    parser.add_argument("--test-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--topology-seed", type=int, default=2)
    parser.add_argument("--sa-maxiter", type=int, default=20)
    parser.add_argument("--sa-maxfun", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/qos_final_three_method_comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_contextual_bandit_experiment(
        train_scenarios_count=args.train_scenarios,
        test_scenarios_count=args.test_scenarios,
        seed=args.seed,
        topology_seed=args.topology_seed,
        sa_maxiter=args.sa_maxiter,
        sa_maxfun=args.sa_maxfun,
        alpha=args.alpha,
        ridge=args.ridge,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
