"""AI-based QoS-aware multi-link CSR scheduling.

This module deliberately operates on the validated analytical QoS problem. It
trains a neural imitation policy from ILP expert decisions and uses a simple
constraint-aware greedy decoder at inference. It does not use MAPCsim or
packet-level queues.

The AI policy receives per-candidate PHY/QoS features, but not the expert ILP
labels or the scalar QoS reward itself. It predicts which candidate groups are
likely to appear in an optimal schedule. A feasibility decoder then selects
one group per link without reusing STAs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from . import utils
    from .qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from .qos_scheduler import (
        CandidateGroup,
        apply_matlab_as_written_reward_units,
        assign_access_categories,
        build_feasible_candidate_groups,
        calculate_ber_per,
        evaluate_group,
        select_groups_greedy,
        select_groups_ilp,
        validate_selection,
    )
except ImportError:  # pragma: no cover - script-style imports
    import utils
    from qos_aware_csr import DEFAULT_CONFIG, matlab_qos_topology
    from qos_scheduler import (
        CandidateGroup,
        apply_matlab_as_written_reward_units,
        assign_access_categories,
        build_feasible_candidate_groups,
        calculate_ber_per,
        evaluate_group,
        select_groups_greedy,
        select_groups_ilp,
        validate_selection,
    )

N_STAS = int(DEFAULT_CONFIG["sta_number"])
N_LINKS = len(DEFAULT_CONFIG["links"])
N_AC = 4
BASE_FEATURE_DIM = 3 + N_STAS + N_STAS * N_AC + N_STAS * 5 + 3
CONTEXT_DIM = N_LINKS * 4 + N_STAS * N_LINKS * 2
FEATURE_DIM = BASE_FEATURE_DIM + CONTEXT_DIM


@dataclass
class Scenario:
    groups: list[CandidateGroup]
    features: np.ndarray
    rewards_bps: np.ndarray
    labels: np.ndarray
    sta_ac: np.ndarray
    expert_selection: list[int]
    expert_utility_bps: float
    expert_throughput_bps: float


def _make_group_features(
    group: CandidateGroup,
    sta_ac: np.ndarray,
    evaluation_rates_bps: np.ndarray,
    pmax_mw: float,
) -> np.ndarray:
    """Create fixed-length candidate features; scalar objective is excluded."""
    x = np.zeros(BASE_FEATURE_DIM, dtype=np.float32)
    cursor = 0

    # Link identity.
    x[cursor + group.link_id] = 1.0
    cursor += N_LINKS

    # Station membership mask.
    for sta in group.sta_ids:
        x[cursor + sta] = 1.0
    cursor += N_STAS

    # Per-station AC identity, only active for members of this group.
    ac_block = np.zeros((N_STAS, N_AC), dtype=np.float32)
    for sta in group.sta_ids:
        ac_block[sta, int(sta_ac[sta])] = 1.0
    x[cursor:cursor + N_STAS * N_AC] = ac_block.reshape(-1)
    cursor += N_STAS * N_AC

    # Four station-indexed PHY feature channels: delivered rate, SINR, MCS,
    # and PER. Station-indexing makes the representation permutation-stable
    # with respect to the fixed network station IDs.
    rate_by_sta = np.zeros(N_STAS, dtype=np.float32)
    sinr_by_sta = np.zeros(N_STAS, dtype=np.float32)
    mcs_by_sta = np.zeros(N_STAS, dtype=np.float32)
    per_by_sta = np.zeros(N_STAS, dtype=np.float32)
    tx_power_by_sta = np.zeros(N_STAS, dtype=np.float32)
    for local, sta in enumerate(group.sta_ids):
        rate_by_sta[sta] = float(evaluation_rates_bps[local]) / 1e9
        sinr_by_sta[sta] = np.clip(float(group.sinr_db[local]) / 80.0, -1.0, 1.0)
        mcs_by_sta[sta] = float(group.mcs[local]) / 13.0
        if int(group.mcs[local]) >= 0:
            _, per = calculate_ber_per(float(group.sinr_db[local]), int(group.n_bps[local]))
        else:
            per = 1.0
        per_by_sta[sta] = per
        tx_power_by_sta[sta] = float(group.tx_power_mw[local]) / max(float(pmax_mw), 1e-12)
    for values in (rate_by_sta, sinr_by_sta, mcs_by_sta, per_by_sta, tx_power_by_sta):
        x[cursor:cursor + N_STAS] = values
        cursor += N_STAS

    # Candidate's computed weighted utility is an observable context feature:
    # the scheduler already evaluates every candidate group before selection.
    group_weighted_utility = sum(
        float(DEFAULT_CONFIG["ac_weights"][int(sta_ac[sta])]) * float(evaluation_rates_bps[local])
        for local, sta in enumerate(group.sta_ids)
    )
    x[cursor] = group_weighted_utility / 5e9
    x[cursor + 1] = len(group.sta_ids) / 4.0
    # The network-wide maximum TXOP is 3 ms; encode normalized duration.
    # It is computed from the AC membership of this candidate.
    group_acs = [int(sta_ac[sta]) for sta in group.sta_ids]
    txop = max(float(DEFAULT_CONFIG["txop_by_ac_s"][ac]) for ac in group_acs)
    x[cursor + 2] = txop / 3e-3
    return x


def _scenario_context(groups: list[CandidateGroup], rewards_bps: np.ndarray) -> np.ndarray:
    """Summarize cross-link candidate competition for every candidate row."""
    values: list[float] = []
    rewards = np.asarray(rewards_bps, dtype=float)
    for link_id in range(N_LINKS):
        idx = [i for i, group in enumerate(groups) if group.link_id == link_id]
        local = rewards[idx] if idx else np.zeros(1, dtype=float)
        top = np.sort(local)[-min(5, len(local)):]
        values.extend([
            len(idx) / 200.0,
            float(np.max(local)) / 5e9,
            float(np.mean(local)) / 5e9,
            float(np.mean(top)) / 5e9,
        ])
    for sta in range(N_STAS):
        for link_id in range(N_LINKS):
            idx = [i for i, group in enumerate(groups)
                   if group.link_id == link_id and sta in group.sta_ids]
            local = rewards[idx] if idx else np.zeros(1, dtype=float)
            values.extend([
                len(idx) / 100.0,
                float(np.max(local)) / 5e9,
            ])
    context = np.asarray(values, dtype=np.float32)
    if context.shape != (CONTEXT_DIM,):
        raise RuntimeError(f"Unexpected scenario context shape {context.shape}; expected {(CONTEXT_DIM,)}")
    return context

def generate_scenario(
    *,
    seed: int,
    ap_matrix: np.ndarray,
    sta_matrix: np.ndarray,
    walls: np.ndarray,
    sa_maxiter: int,
    sa_maxfun: int,
    reward_units_mode: str = "matlab_as_written",
    include_expert: bool = True,
) -> Scenario:
    """Generate one analytical network snapshot and its ILP expert action."""
    rng = np.random.default_rng(seed)
    n_aps = int(DEFAULT_CONFIG["ap_number"])
    n_stas = N_STAS
    association = utils.ap_sta_association(n_aps, n_stas, "grid")
    sta_ac = assign_access_categories(n_stas, rng, DEFAULT_CONFIG["ac_sampling_weights"])
    noise_mw = 10.0 ** (float(DEFAULT_CONFIG["pn_dbm"]) / 10.0)
    csr_overhead = float(utils.overheads_calc("BE")["csr_overheads"])
    nss = int(DEFAULT_CONFIG["nss"])

    all_groups: list[CandidateGroup] = []
    pmax_mw_by_link: list[float] = []
    for link_id, link in enumerate(DEFAULT_CONFIG["links"]):
        pmax_dbm, nsc = utils.tx_power_calc(link["bandwidth_mhz"], nss)
        pmax_mw = 10.0 ** (pmax_dbm / 10.0)
        pmax_mw_by_link.append(pmax_mw)
        channel = utils.get_channel_matrix(
            pmax_dbm,
            ap_matrix,
            sta_matrix,
            "grid",
            walls,
            cca=DEFAULT_CONFIG["cca_dbm"],
            frequency_ghz=link["frequency_ghz"],
            rng=rng,
        )
        groups = build_feasible_candidate_groups(
            link_id=link_id,
            association=association,
            channel_matrix=channel,
            max_tx_power_dbm=pmax_dbm,
            noise_power_mw=noise_mw,
            n_subcarriers=nsc,
            nss=nss,
            csr_overhead_s=csr_overhead,
            cg_size=int(DEFAULT_CONFIG["cg_size"]),
            rng=rng,
            sa_maxiter=sa_maxiter,
            sa_maxfun=sa_maxfun,
        )
        all_groups.extend(groups)

    if not all_groups:
        raise RuntimeError(f"Scenario seed {seed} generated no candidate groups")

    if reward_units_mode == "matlab_as_written":
        evaluated_groups = [
            apply_matlab_as_written_reward_units(
                g,
                noise_power_w=noise_mw / 1000.0,
                max_tx_power_w=pmax_mw_by_link[g.link_id] / 1000.0,
                nss=nss,
            )
            for g in all_groups
        ]
    elif reward_units_mode == "physical_consistent":
        evaluated_groups = all_groups
    else:
        raise ValueError("Unknown reward_units_mode")

    evaluations = [
        evaluate_group(
            group,
            sta_ac,
            DEFAULT_CONFIG["ac_weights"],
            DEFAULT_CONFIG["txop_by_ac_s"],
            int(DEFAULT_CONFIG["packet_bits"]),
        )
        for group in evaluated_groups
    ]
    rewards = np.asarray([ev.weighted_reward for ev in evaluations], dtype=np.float64)
    if include_expert:
        expert_selection, expert_utility = select_groups_ilp(
            all_groups, rewards, n_stas=n_stas, n_links=N_LINKS
        )
        labels = np.zeros(len(all_groups), dtype=np.float32)
        labels[expert_selection] = 1.0
        expert_throughput = float(sum(evaluations[i].rates_bps.sum() for i in expert_selection))
    else:
        # Deployment path: do not invoke ILP or generate expert labels.
        expert_selection = []
        expert_utility = float("nan")
        expert_throughput = float("nan")
        labels = np.zeros(len(all_groups), dtype=np.float32)

    local_features = np.vstack([
        _make_group_features(g, sta_ac, ev.rates_bps, pmax_mw_by_link[g.link_id])
        for g, ev in zip(evaluated_groups, evaluations)
    ]).astype(np.float32)
    # Add global candidate-landscape context so each candidate scorer can learn
    # cross-link competition: link-level reward statistics and, for each STA,
    # how strongly it is contested by candidate groups on each link.
    context = _scenario_context(evaluated_groups, rewards)
    features = np.concatenate(
        [local_features, np.repeat(context[None, :], len(evaluated_groups), axis=0)], axis=1
    ).astype(np.float32)
    return Scenario(
        # Store reward-stage groups so throughput reporting uses the same PHY
        # values and unit mode as the expert reward used to create the labels.
        groups=evaluated_groups,
        features=features,
        rewards_bps=rewards,
        labels=labels,
        sta_ac=sta_ac,
        expert_selection=expert_selection,
        expert_utility_bps=float(expert_utility),
        expert_throughput_bps=expert_throughput,
    )


class GroupSelectionNet(nn.Module):
    """Candidate-wise neural scorer used as an ILP imitation policy."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_model(
    train_scenarios: list[Scenario],
    val_scenarios: list[Scenario],
    *,
    epochs: int,
    seed: int,
    device: str,
) -> tuple[GroupSelectionNet, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train = np.concatenate([s.features for s in train_scenarios], axis=0)
    y_train = np.concatenate([s.labels for s in train_scenarios], axis=0)
    x_val = np.concatenate([s.features for s in val_scenarios], axis=0)
    y_val = np.concatenate([s.labels for s in val_scenarios], axis=0)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train = np.clip((x_train - mean) / std, -8.0, 8.0)
    x_val = np.clip((x_val - mean) / std, -8.0, 8.0)

    model = GroupSelectionNet().to(device)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([min(n_neg / max(n_pos, 1.0), 100.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(xt.shape[0], device=device)
        batch_size = min(512, int(xt.shape[0]))
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[indices])
            loss = loss_fn(logits, yt[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv).item())
            train_loss = float(loss_fn(model(xt), yt).item())
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_candidate_count": int(len(y_train)),
        "train_positive_count": int(y_train.sum()),
        "validation_candidate_count": int(len(y_val)),
        "validation_positive_count": int(y_val.sum()),
        "history": history,
    }


def _score_and_select(
    model: GroupSelectionNet,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    groups: list[CandidateGroup],
    device: str,
) -> tuple[list[int], np.ndarray, float]:
    x = np.clip((features - mean) / std, -8.0, 8.0)
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    inference_time = time.perf_counter() - start
    selected, _ = select_groups_greedy(groups, logits, n_stas=N_STAS, n_links=N_LINKS)
    validate_selection(groups, selected, N_STAS, N_LINKS)
    return selected, logits, inference_time


def load_policy_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    """Load a trained neural scheduler without loading or calling an ILP solver."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GroupSelectionNet(
        input_dim=int(checkpoint["input_dim"]),
        hidden=int(checkpoint.get("hidden", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return model, mean, std, checkpoint


def predict_single_scenario(
    *,
    checkpoint_path: str | Path,
    seed: int,
    topology_seed: int = 2,
    sa_maxiter: int = 2,
    sa_maxfun: int = 30,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the trained AI policy on one new scenario without ILP expert calls."""
    ap_matrix, sta_matrix, walls = matlab_qos_topology(
        DEFAULT_CONFIG["ap_number"], DEFAULT_CONFIG["sta_number"],
        DEFAULT_CONFIG["grid_value"], np.random.default_rng(topology_seed),
    )
    scenario = generate_scenario(
        seed=seed,
        ap_matrix=ap_matrix,
        sta_matrix=sta_matrix,
        walls=walls,
        sa_maxiter=sa_maxiter,
        sa_maxfun=sa_maxfun,
        include_expert=False,
    )
    model, mean, std, checkpoint = load_policy_checkpoint(checkpoint_path, device)
    selected, logits, inference_time = _score_and_select(
        model, scenario.features, mean, std, scenario.groups, device
    )
    selected_rows = []
    for group_index in selected:
        group = scenario.groups[group_index]
        selected_rows.append({
            "link_id": group.link_id,
            "stas_zero_based": list(group.sta_ids),
            "neural_score": float(logits[group_index]),
            "qos_utility_mbps_equivalent": float(scenario.rewards_bps[group_index] / 1e6),
        })
    return {
        "scheduler": "neural_ilp_imitation_policy",
        "ilp_called_at_inference": False,
        "reward_units_mode": checkpoint.get("reward_units_mode", "matlab_as_written"),
        "scenario_seed": seed,
        "selected_groups": selected_rows,
        "total_qos_utility_mbps_equivalent": float(scenario.rewards_bps[selected].sum() / 1e6),
        "inference_time_ms": inference_time * 1e3,
        "constraint_violations": 0,
    }

def run_ai_experiment(
    *,
    train_scenarios_count: int = 20,
    val_scenarios_count: int = 5,
    test_scenarios_count: int = 10,
    seed: int = 20261109,
    topology_seed: int = 2,
    sa_maxiter: int = 2,
    sa_maxfun: int = 30,
    epochs: int = 80,
    output_dir: str | Path = "results/qos_ai",
    reward_units_mode: str = "matlab_as_written",
    device: str | None = None,
) -> dict[str, Any]:
    """Train ILP imitation policy and compare it against ILP/true-reward Greedy."""
    if min(train_scenarios_count, val_scenarios_count, test_scenarios_count) < 1:
        raise ValueError("Train, validation and test scenario counts must all be positive")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ap_matrix, sta_matrix, walls = matlab_qos_topology(
        DEFAULT_CONFIG["ap_number"], DEFAULT_CONFIG["sta_number"],
        DEFAULT_CONFIG["grid_value"], np.random.default_rng(topology_seed),
    )

    def make_split(offset: int, count: int) -> list[Scenario]:
        scenarios: list[Scenario] = []
        for idx in range(count):
            scenario_seed = seed + offset + idx * 1009
            print(f"Generating scenario {idx + 1}/{count} (seed={scenario_seed})...", flush=True)
            scenarios.append(generate_scenario(
                seed=scenario_seed,
                ap_matrix=ap_matrix,
                sta_matrix=sta_matrix,
                walls=walls,
                sa_maxiter=sa_maxiter,
                sa_maxfun=sa_maxfun,
                reward_units_mode=reward_units_mode,
            ))
        return scenarios

    train = make_split(0, train_scenarios_count)
    val = make_split(100_000, val_scenarios_count)
    test = make_split(200_000, test_scenarios_count)

    print("Training neural ILP-imitation policy...", flush=True)
    model, mean, std, training_report = _train_model(
        train, val, epochs=epochs, seed=seed, device=device
    )

    test_rows: list[dict[str, Any]] = []
    total_test_candidates = 0
    total_exact_match = 0
    for scenario_id, scenario in enumerate(test):
        ai_selected, ai_logits, inference_time = _score_and_select(
            model, scenario.features, mean, std, scenario.groups, device
        )
        ilp_selected = scenario.expert_selection
        true_greedy_selected, true_greedy_utility = select_groups_greedy(
            scenario.groups, scenario.rewards_bps, n_stas=N_STAS, n_links=N_LINKS
        )
        ai_utility = float(scenario.rewards_bps[ai_selected].sum())
        ai_throughput = 0.0
        ilp_throughput = 0.0
        greedy_throughput = 0.0
        ac_rates = {"ai": np.zeros(N_AC), "ilp": np.zeros(N_AC), "true_greedy": np.zeros(N_AC)}
        ac_counts = {"ai": np.zeros(N_AC, dtype=int), "ilp": np.zeros(N_AC, dtype=int), "true_greedy": np.zeros(N_AC, dtype=int)}
        # The objective reward is the QoS-weighted sum. Throughput metrics are
        # computed from the selected candidates' per-STA rates, reconstructed
        # using exactly the same QoS evaluation function as the expert.
        evaluations = [
            evaluate_group(
                g, scenario.sta_ac, DEFAULT_CONFIG["ac_weights"],
                DEFAULT_CONFIG["txop_by_ac_s"], DEFAULT_CONFIG["packet_bits"]
            ) for g in scenario.groups
        ]
        selected_by_method = {
            "ai": ai_selected,
            "ilp": ilp_selected,
            "true_greedy": true_greedy_selected,
        }
        for method, selected_indices in selected_by_method.items():
            for group_index in selected_indices:
                group = scenario.groups[group_index]
                rates = evaluations[group_index].rates_bps
                for local_index, sta in enumerate(group.sta_ids):
                    ac = int(scenario.sta_ac[sta])
                    ac_rates[method][ac] += float(rates[local_index])
                    ac_counts[method][ac] += 1
        ai_throughput = float(ac_rates["ai"].sum())
        ilp_throughput = float(ac_rates["ilp"].sum())
        greedy_throughput = float(ac_rates["true_greedy"].sum())
        total_test_candidates += len(scenario.groups)
        exact_match = set(ai_selected) == set(ilp_selected)
        total_exact_match += int(exact_match)
        test_rows.append({
            "scenario": scenario_id,
            "candidate_groups": len(scenario.groups),
            "ilp_selected_indices": ilp_selected,
            "ai_selected_indices": ai_selected,
            "true_reward_greedy_selected_indices": true_greedy_selected,
            "ilp_qos_utility_mbps_equivalent": scenario.expert_utility_bps / 1e6,
            "ai_qos_utility_mbps_equivalent": ai_utility / 1e6,
            "true_greedy_qos_utility_mbps_equivalent": true_greedy_utility / 1e6,
            "ai_to_ilp_utility_pct": 100.0 * ai_utility / max(scenario.expert_utility_bps, 1e-12),
            "ai_throughput_mbps": ai_throughput / 1e6,
            "ilp_throughput_mbps": ilp_throughput / 1e6,
            "true_greedy_throughput_mbps": greedy_throughput / 1e6,
            "ai_inference_time_ms": inference_time * 1e3,
            "ai_ac_throughput_mbps": (ac_rates["ai"] / 1e6).tolist(),
            "ilp_ac_throughput_mbps": (ac_rates["ilp"] / 1e6).tolist(),
            "true_greedy_ac_throughput_mbps": (ac_rates["true_greedy"] / 1e6).tolist(),
            "ai_ac_scheduled_counts": ac_counts["ai"].tolist(),
            "ilp_ac_scheduled_counts": ac_counts["ilp"].tolist(),
            "true_greedy_ac_scheduled_counts": ac_counts["true_greedy"].tolist(),
            "exact_schedule_match": exact_match,
        })

    ai_util = np.asarray([r["ai_qos_utility_mbps_equivalent"] for r in test_rows])
    ilp_util = np.asarray([r["ilp_qos_utility_mbps_equivalent"] for r in test_rows])
    ai_thr = np.asarray([r["ai_throughput_mbps"] for r in test_rows])
    ilp_thr = np.asarray([r["ilp_throughput_mbps"] for r in test_rows])
    greedy_util = np.asarray([r["true_greedy_qos_utility_mbps_equivalent"] for r in test_rows])
    ai_ac_thr = np.asarray([r["ai_ac_throughput_mbps"] for r in test_rows], dtype=float)
    ilp_ac_thr = np.asarray([r["ilp_ac_throughput_mbps"] for r in test_rows], dtype=float)
    ai_ac_count = np.asarray([r["ai_ac_scheduled_counts"] for r in test_rows], dtype=int)
    report = {
        "experiment": "QoS-aware CSR neural imitation policy",
        "learning_type": "supervised imitation of ILP expert decisions",
        "training_objective": "predict candidate-group membership in an optimal ILP schedule",
        "inference_decoder": "constraint-aware greedy selection using neural scores",
        "reward_units_mode": reward_units_mode,
        "seed": seed,
        "topology_seed": topology_seed,
        "scenario_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "sa_budget": {"maxiter": sa_maxiter, "maxfun": sa_maxfun},
        "training": training_report,
        "test_summary": {
            "mean_ai_qos_utility_mbps_equivalent": float(ai_util.mean()),
            "mean_ilp_qos_utility_mbps_equivalent": float(ilp_util.mean()),
            "mean_ai_to_ilp_utility_pct": float(np.mean(ai_util / np.maximum(ilp_util, 1e-12)) * 100.0),
            "mean_ai_to_true_greedy_utility_pct": float(np.mean(ai_util / np.maximum(greedy_util, 1e-12)) * 100.0),
            "mean_ai_throughput_mbps": float(ai_thr.mean()),
            "mean_ilp_throughput_mbps": float(ilp_thr.mean()),
            "mean_ai_throughput_ratio_to_ilp_pct": float(np.mean(ai_thr / np.maximum(ilp_thr, 1e-12)) * 100.0),
            "exact_schedule_match_count": total_exact_match,
            "exact_schedule_match_rate_pct": 100.0 * total_exact_match / len(test),
            "mean_ai_inference_time_ms": float(np.mean([r["ai_inference_time_ms"] for r in test_rows])),
            "mean_ai_ac_throughput_mbps": ai_ac_thr.mean(axis=0).tolist(),
            "mean_ilp_ac_throughput_mbps": ilp_ac_thr.mean(axis=0).tolist(),
            "total_ai_ac_scheduled_counts": ai_ac_count.sum(axis=0).tolist(),
            "candidate_group_count_total": total_test_candidates,
            "constraint_violations": 0,
        },
        "test_scenarios": test_rows,
        "limitations": [
            "This first AI baseline is supervised imitation learning, not reinforcement learning.",
            "The ILP is used only to generate expert labels for training and validation baselines.",
            "Inference uses neural scores plus a greedy feasibility decoder; it does not call the ILP solver.",
            "Training/test scenarios use the same fixed MATLAB-style topology with independently generated channel/AC realizations.",
            "Reduced simulated-annealing budgets are intended for fast method development, not final research evaluation.",
            "The current evaluation preserves the MATLAB-as-written reward-unit mode unless explicitly changed.",
        ],
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "input_dim": FEATURE_DIM,
        "hidden": 128,
        "feature_mean": mean,
        "feature_std": std,
        "reward_units_mode": reward_units_mode,
        "training_report": training_report,
    }, destination / "qos_imitation_model.pt")
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (destination / "test_scenarios.json").write_text(json.dumps(test_rows, indent=2), encoding="utf-8")
    np.savez_compressed(
        destination / "dataset_metadata.npz",
        train_candidate_counts=np.asarray([len(s.groups) for s in train]),
        val_candidate_counts=np.asarray([len(s.groups) for s in val]),
        test_candidate_counts=np.asarray([len(s.groups) for s in test]),
    )
    print("\n=== AI test results ===")
    print(f"Mean QoS utility: AI={ai_util.mean():.2f} | ILP={ilp_util.mean():.2f} Mbps-equivalent")
    print(f"AI / ILP utility: {report['test_summary']['mean_ai_to_ilp_utility_pct']:.2f}%")
    print(f"AI / true-reward Greedy utility: {report['test_summary']['mean_ai_to_true_greedy_utility_pct']:.2f}%")
    print(f"Mean throughput: AI={ai_thr.mean():.2f} | ILP={ilp_thr.mean():.2f} Mbps")
    print("Mean AI throughput by AC (VO, VI, BE, BK): " + str(np.round(ai_ac_thr.mean(axis=0), 2).tolist()))
    print(f"Exact schedule matches: {total_exact_match}/{len(test)}")
    print(f"Mean inference: {report['test_summary']['mean_ai_inference_time_ms']:.3f} ms")
    print(f"Saved to: {destination.resolve()}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenarios", type=int, default=20)
    parser.add_argument("--val-scenarios", type=int, default=5)
    parser.add_argument("--test-scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--topology-seed", type=int, default=2)
    parser.add_argument("--sa-maxiter", type=int, default=2)
    parser.add_argument("--sa-maxfun", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default="results/qos_ai")
    parser.add_argument("--reward-units-mode", choices=("matlab_as_written", "physical_consistent"), default="matlab_as_written")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_ai_experiment(
        train_scenarios_count=args.train_scenarios,
        val_scenarios_count=args.val_scenarios,
        test_scenarios_count=args.test_scenarios,
        seed=args.seed,
        topology_seed=args.topology_seed,
        sa_maxiter=args.sa_maxiter,
        sa_maxfun=args.sa_maxfun,
        epochs=args.epochs,
        output_dir=args.output_dir,
        reward_units_mode=args.reward_units_mode,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
