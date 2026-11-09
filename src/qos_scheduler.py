"""QoS-aware coordinated spatial reuse scheduling primitives.

This module ports the scheduling mathematics used by the MATLAB reference
``Final_Code_010926.m``. It is deliberately independent of MAPCsim's event
loop: the MATLAB reference is an analytical multi-link experiment, not a
packet-level event simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import erfc, floor, log2, sqrt
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, dual_annealing, milp
from scipy.sparse import lil_matrix

try:
    from .utils import mcs_cal_PER_001 as _python_simulator_mcs
except ImportError:  # pragma: no cover - script-style import
    from utils import mcs_cal_PER_001 as _python_simulator_mcs


@dataclass(frozen=True)
class CandidateGroup:
    """A feasible coordinated group on one frequency link.

    STA and AP indices are zero-based in Python. Arrays are ordered so that
    element ``k`` of ``sta_ids`` is served by ``ap_ids[k]``.
    """

    link_id: int
    sta_ids: tuple[int, ...]
    ap_ids: tuple[int, ...]
    tx_power_mw: np.ndarray
    sinr_db: np.ndarray
    mcs: np.ndarray
    n_bps: np.ndarray
    coding_rate: np.ndarray
    phy_rate_bps: np.ndarray
    csr_overhead_s: float
    n_subcarriers: int
    channel_submatrix: np.ndarray | None = None


@dataclass(frozen=True)
class GroupEvaluation:
    reward: float
    rates_bps: np.ndarray
    weighted_reward: float
    group_txop_s: float


def assign_access_categories(
    n_stas: int,
    rng: np.random.Generator,
    sampling_weights: Sequence[float] = (1.0, 1.0, 1.0, 2.5),
) -> np.ndarray:
    """Sample per-STA AC indices 0..3 using the MATLAB ``pro`` weights."""
    probabilities = np.asarray(sampling_weights, dtype=float)
    if probabilities.shape != (4,) or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("sampling_weights must contain four non-negative values with positive sum")
    probabilities /= probabilities.sum()
    return rng.choice(4, size=n_stas, replace=True, p=probabilities)


def mcs_from_sinr(sinr_db: float) -> tuple[int, int, float]:
    """Delegate to the original Python simulator's MCS implementation."""
    mcs, n_bps, coding_rate = _python_simulator_mcs(float(sinr_db))
    return int(mcs), int(n_bps), float(coding_rate)


def calculate_ber_per(sinr_db: float, n_bps: int, packet_bits: int = 12_000) -> tuple[float, float]:
    """Calculate the analytical BER/PER expression in Final_Code_010926.m."""
    if packet_bits <= 0:
        raise ValueError("packet_bits must be positive")
    if n_bps < 1:
        raise ValueError("n_bps must be at least 1")
    if not np.isfinite(sinr_db):
        return 0.5, 1.0

    constellation_size = 2**int(n_bps)
    sinr_linear = 10.0 ** (sinr_db / 10.0)
    q_arg = sqrt((3.0 * log2(constellation_size) * sinr_linear) / (constellation_size - 1))
    ber = (4.0 / log2(constellation_size)) * (1.0 - 1.0 / sqrt(constellation_size)) * 0.5 * erfc(q_arg / sqrt(2.0))
    ber = float(np.clip(ber, 0.0, 0.5))
    per = float(np.clip(1.0 - (1.0 - ber) ** packet_bits, 0.0, 1.0))
    return ber, per


def calculate_sinr(channel_submatrix: np.ndarray, tx_power_mw: np.ndarray, noise_power_mw: float) -> np.ndarray:
    """Calculate one SINR per STA, using the MATLAB row-wise interference sum."""
    h = np.asarray(channel_submatrix, dtype=float)
    p = np.asarray(tx_power_mw, dtype=float).reshape(-1)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] != p.size:
        raise ValueError("channel_submatrix must be square and match the power vector")
    if np.any(h < 0) or np.any(p <= 0) or noise_power_mw <= 0:
        raise ValueError("channel gains and powers must be non-negative/positive")
    signal = p * np.diag(h)
    interference = np.sum(h * p[np.newaxis, :], axis=1) - signal
    denominator = noise_power_mw + interference
    return signal / denominator


def proportional_fair_utility(
    powers_mw: np.ndarray,
    channel_submatrix: np.ndarray,
    noise_power_mw: float,
    n_subcarriers: int,
    nss: int,
) -> float:
    """Sum(log(PHY rate)) used by the MATLAB power optimizer."""
    sinr = calculate_sinr(channel_submatrix, powers_mw, noise_power_mw)
    sinr_db = 10.0 * np.log10(sinr)
    rates = np.zeros(sinr_db.size, dtype=float)
    symbol_time = 12.8e-6 + 0.8e-6
    for index, value in enumerate(sinr_db):
        mcs, n_bps, code_rate = mcs_from_sinr(float(value))
        if mcs >= 0:
            rates[index] = n_subcarriers * n_bps * code_rate * nss / symbol_time
    # MATLAB uses sum(log(max(rates, eps))).
    return float(np.log(np.maximum(rates, np.finfo(float).eps)).sum())


def optimize_power_simulated_annealing(
    channel_submatrix: np.ndarray,
    noise_power_mw: float,
    pmax_mw: float,
    n_subcarriers: int,
    nss: int,
    rng: np.random.Generator,
    maxiter: int = 500,
    maxfun: int = 10_000,
    initial_temperature: float = 100.0,
) -> np.ndarray:
    """Bounded simulated-annealing power allocation.

    MATLAB uses ``simulannealbnd`` with P0=(1+Pmax)/2, 500 iterations,
    10,000 function evaluations and initial temperature 100. SciPy's
    ``dual_annealing`` is a related bounded generalized simulated-annealing
    implementation. We match the objective, bounds, initial point and stated
    budgets, but the two solvers do not have identical trajectories.
    """
    n_links = int(np.asarray(channel_submatrix).shape[0])
    lower = np.ones(n_links, dtype=float)
    upper = np.full(n_links, float(pmax_mw), dtype=float)
    if pmax_mw < 1.0:
        raise ValueError(f"Pmax must be >= 1 mW for the MATLAB bounds; got {pmax_mw}")
    if maxiter < 1 or maxfun < 1:
        raise ValueError("maxiter and maxfun must be positive")

    initial_point = (lower + upper) / 2.0

    def objective(power: np.ndarray) -> float:
        return -proportional_fair_utility(
            power, channel_submatrix, noise_power_mw, n_subcarriers, nss
        )

    result = dual_annealing(
        objective,
        bounds=list(zip(lower, upper)),
        x0=initial_point,
        maxiter=maxiter,
        maxfun=maxfun,
        initial_temp=initial_temperature,
        rng=rng,
        no_local_search=True,
    )
    return np.asarray(result.x, dtype=float)


def generate_matlab_group_rows(
    association: Sequence[Sequence[int]], n_stas: int, cg_size: int
) -> list[tuple[int, ...]]:
    """Port the row construction/order in MATLAB CGcreation.m.

    Association and returned STA IDs are zero-based. Each row has one STA
    slot per AP; -1 represents a zero/unused AP. Singleton rows are placed
    first in ascending STA order, matching ``[one_STA_per_tx; map_matrix1]``.
    """
    n_aps = len(association)
    if cg_size < 1 or cg_size > n_aps:
        raise ValueError("cg_size must be between 1 and the number of APs")

    rows: list[tuple[int, ...]] = []
    for sta in range(n_stas):
        row = [-1] * n_aps
        ap = next((idx for idx, members in enumerate(association) if sta in members), None)
        if ap is not None:
            row[ap] = sta
            rows.append(tuple(row))

    per_ap_choices = [[-1, *[int(sta) for sta in members]] for members in association]
    # MATLAB's combvec(u{:}) makes the first input vary fastest. Python's
    # itertools.product varies its last input fastest, so enumerate the
    # reversed inputs and reverse each tuple to preserve MATLAB row order.
    # This ordering matters because the MATLAB script runs the stochastic
    # power optimizer sequentially over the candidate rows.
    for reversed_combination in product(*reversed(per_ap_choices)):
        combination = tuple(reversed(reversed_combination))
        members = [sta for sta in combination if sta >= 0]
        if 2 <= len(members) <= cg_size:
            rows.append(tuple(int(sta) for sta in combination))
    return rows


def build_feasible_candidate_groups(
    *,
    link_id: int,
    association: Sequence[Sequence[int]],
    channel_matrix: np.ndarray,
    max_tx_power_dbm: float,
    noise_power_mw: float,
    n_subcarriers: int,
    nss: int,
    csr_overhead_s: float,
    cg_size: int,
    rng: np.random.Generator,
    sa_maxiter: int = 500,
    sa_maxfun: int = 10_000,
) -> list[CandidateGroup]:
    """Generate/filter candidate groups like MATLAB CGcreation.m.

    The pruning test is based on PHY rate (not PER-adjusted throughput): a
    group survives only when ``group_size * group_PHY_rate >= singleton_rate``
    for every member. A failed group's supersets are pruned.
    """
    channel_matrix = np.asarray(channel_matrix, dtype=float)
    n_stas, n_aps = channel_matrix.shape
    if n_aps != len(association):
        raise ValueError("association length must equal the channel matrix AP dimension")

    pmax_mw = 10.0 ** (float(max_tx_power_dbm) / 10.0)
    rows = generate_matlab_group_rows(association, n_stas, cg_size)
    discard = np.ones(len(rows), dtype=bool)
    baseline_phy_rate = np.full(n_stas, np.nan, dtype=float)
    feasible: list[CandidateGroup] = []
    symbol_time = 12.8e-6 + 0.8e-6

    for row_index, row in enumerate(rows):
        if not discard[row_index]:
            continue
        ap_ids = tuple(ap for ap, sta in enumerate(row) if sta >= 0)
        sta_ids = tuple(row[ap] for ap in ap_ids)
        if not sta_ids:
            continue

        h = channel_matrix[np.ix_(sta_ids, ap_ids)]
        if len(sta_ids) == 1:
            powers = np.array([pmax_mw], dtype=float)
        else:
            powers = optimize_power_simulated_annealing(
                h,
                noise_power_mw,
                pmax_mw,
                n_subcarriers,
                nss,
                rng,
                maxiter=sa_maxiter,
                maxfun=sa_maxfun,
            )

        sinr_linear = calculate_sinr(h, powers, noise_power_mw)
        sinr_db = 10.0 * np.log10(sinr_linear)
        mcs = np.full(len(sta_ids), -1, dtype=int)
        n_bps = np.ones(len(sta_ids), dtype=int)
        coding_rate = np.full(len(sta_ids), 0.5, dtype=float)
        phy_rate = np.zeros(len(sta_ids), dtype=float)

        for local_index, (sta, sinr_value) in enumerate(zip(sta_ids, sinr_db)):
            mcs_value, bits_per_subcarrier, rate_code = mcs_from_sinr(float(sinr_value))
            mcs[local_index] = mcs_value
            n_bps[local_index] = bits_per_subcarrier
            coding_rate[local_index] = rate_code
            if mcs_value >= 0:
                phy_rate[local_index] = (
                    n_subcarriers * bits_per_subcarrier * rate_code * nss / symbol_time
                )

        # The first n_stas rows are the singleton baselines. This code also
        # handles an arbitrary input order safely, although the generated rows
        # follow the MATLAB singleton-first construction.
        if len(sta_ids) == 1:
            baseline_phy_rate[sta_ids[0]] = phy_rate[0]

        accepted = True
        for local_index, sta in enumerate(sta_ids):
            base_rate = baseline_phy_rate[sta]
            if np.isnan(base_rate):
                # Should not occur with singleton-first rows; fail loudly if
                # the generator is changed and breaks this dependency.
                raise RuntimeError(f"Missing singleton baseline for STA {sta}")
            if len(sta_ids) * phy_rate[local_index] < base_rate:
                accepted = False
                sta_set = set(sta_ids)
                for candidate_index, candidate_row in enumerate(rows):
                    candidate_members = {sta for sta in candidate_row if sta >= 0}
                    if sta_set.issubset(candidate_members):
                        discard[candidate_index] = False
                break

        if accepted:
            feasible.append(
                CandidateGroup(
                    link_id=link_id,
                    sta_ids=sta_ids,
                    ap_ids=ap_ids,
                    tx_power_mw=powers.copy(),
                    sinr_db=sinr_db.copy(),
                    mcs=mcs.copy(),
                    n_bps=n_bps.copy(),
                    coding_rate=coding_rate.copy(),
                    phy_rate_bps=phy_rate.copy(),
                    csr_overhead_s=float(csr_overhead_s),
                    n_subcarriers=int(n_subcarriers),
                    channel_submatrix=h.copy(),
                )
            )

    return feasible


def apply_matlab_as_written_reward_units(
    group: CandidateGroup,
    *,
    noise_power_w: float,
    max_tx_power_w: float,
    nss: int,
) -> CandidateGroup:
    """Recompute reward-stage PHY values exactly as Final_Code_010926.m does.

    CGcreation.m optimizes and stores powers in mW. In Final_Code_010926.m,
    the reward loop converts noise and singleton max power to W, but reuses
    multi-STA TxPowerMatrix rows (still mW) without conversion. This helper
    deliberately reproduces that mixed-unit behavior for source-code parity.

    This is not the physically consistent model. Use the candidate group's
    existing SINR (built with mW/mW) for the consistent model.
    """
    if group.channel_submatrix is None:
        raise ValueError("channel_submatrix is required for MATLAB reward-unit emulation")
    if noise_power_w <= 0 or max_tx_power_w <= 0:
        raise ValueError("noise_power_w and max_tx_power_w must be positive")

    if len(group.sta_ids) == 1:
        # MATLAB Final_Code_010926.m replaces singleton TxPowerMatrix values
        # with MaxTxPower_all, which is expressed in W.
        powers_for_reward = np.array([max_tx_power_w], dtype=float)
    else:
        # MATLAB reuses CGcreation's optimized power vector, which is in mW,
        # while the reward-loop noise value is in W. Preserve this behavior.
        powers_for_reward = np.asarray(group.tx_power_mw, dtype=float)

    sinr_linear = calculate_sinr(
        group.channel_submatrix, powers_for_reward, noise_power_w
    )
    sinr_db = 10.0 * np.log10(sinr_linear)
    mcs = np.full(len(group.sta_ids), -1, dtype=int)
    n_bps = np.ones(len(group.sta_ids), dtype=int)
    coding_rate = np.full(len(group.sta_ids), 0.5, dtype=float)
    phy_rate = np.zeros(len(group.sta_ids), dtype=float)
    symbol_time = 12.8e-6 + 0.8e-6

    for index, value in enumerate(sinr_db):
        mcs_value, bits_per_subcarrier, rate_code = mcs_from_sinr(float(value))
        mcs[index] = mcs_value
        n_bps[index] = bits_per_subcarrier
        coding_rate[index] = rate_code
        if mcs_value >= 0:
            phy_rate[index] = (
                group.n_subcarriers * bits_per_subcarrier * rate_code * nss / symbol_time
            )

    return replace(
        group,
        sinr_db=sinr_db,
        mcs=mcs,
        n_bps=n_bps,
        coding_rate=coding_rate,
        phy_rate_bps=phy_rate,
    )


def evaluate_group(
    group: CandidateGroup,
    sta_ac_indices: Sequence[int],
    ac_weights: Sequence[float],
    txop_by_ac_s: Sequence[float],
    packet_bits: int = 12_000,
) -> GroupEvaluation:
    """Compute the aligned-TXOP QoS reward and per-STA delivered rates."""
    sta_ac = np.asarray(sta_ac_indices, dtype=int)
    weights = np.asarray(ac_weights, dtype=float)
    txops = np.asarray(txop_by_ac_s, dtype=float)
    group_acs = sta_ac[np.asarray(group.sta_ids, dtype=int)]
    group_txop = float(np.max(txops[group_acs]))

    rates = np.zeros(len(group.sta_ids), dtype=float)
    weighted_reward = 0.0
    for k, sta in enumerate(group.sta_ids):
        mcs = int(group.mcs[k])
        if mcs < 0:
            continue
        _, per = calculate_ber_per(float(group.sinr_db[k]), int(group.n_bps[k]), packet_bits)
        phy_rate = float(group.phy_rate_bps[k])
        packets = floor((phy_rate * (group_txop - group.csr_overhead_s)) / packet_bits)
        packets = max(packets, 1)
        rate = packets * packet_bits * (1.0 - per) / group_txop
        rates[k] = rate
        weighted_reward += weights[sta_ac[sta]] * rate

    return GroupEvaluation(
        reward=float(rates.sum()),
        rates_bps=rates,
        weighted_reward=float(weighted_reward),
        group_txop_s=group_txop,
    )


def evaluate_conventional_group(
    group: CandidateGroup,
    sta_ac_indices: Sequence[int],
    ac_weights: Sequence[float],
    txop_by_ac_s: Sequence[float],
    packet_bits: int = 12_000,
) -> GroupEvaluation:
    """Compute conventional (unweighted, per-STA TXOP) reward/rates."""
    sta_ac = np.asarray(sta_ac_indices, dtype=int)
    weights = np.asarray(ac_weights, dtype=float)
    txops = np.asarray(txop_by_ac_s, dtype=float)
    group_acs = sta_ac[np.asarray(group.sta_ids, dtype=int)]
    group_txop = float(np.max(txops[group_acs]))

    rates = np.zeros(len(group.sta_ids), dtype=float)
    for k, sta in enumerate(group.sta_ids):
        if int(group.mcs[k]) < 0:
            continue
        _, per = calculate_ber_per(float(group.sinr_db[k]), int(group.n_bps[k]), packet_bits)
        txop = float(txops[sta_ac[sta]])
        packets = floor((float(group.phy_rate_bps[k]) * (txop - group.csr_overhead_s)) / packet_bits)
        packets = max(packets, 1)
        rates[k] = packets * packet_bits * (1.0 - per) / txop

    weighted = sum(weights[sta_ac[sta]] * rates[k] for k, sta in enumerate(group.sta_ids))
    return GroupEvaluation(
        reward=float(rates.sum()),
        rates_bps=rates,
        weighted_reward=float(weighted),
        group_txop_s=group_txop,
    )


def select_groups_ilp(
    groups: Sequence[CandidateGroup],
    rewards: Sequence[float],
    n_stas: int,
    n_links: int = 3,
) -> tuple[list[int], float]:
    """Select exactly one group/link, with no STA selected more than once."""
    n_groups = len(groups)
    reward_array = np.asarray(rewards, dtype=float)
    if n_groups == 0 or reward_array.shape != (n_groups,):
        raise ValueError("groups and rewards must be non-empty and have equal lengths")
    if not np.all(np.isfinite(reward_array)):
        raise ValueError("all ILP rewards must be finite")

    constraints_matrix = lil_matrix((n_links + n_stas, n_groups), dtype=float)
    for column, group in enumerate(groups):
        if not 0 <= group.link_id < n_links:
            raise ValueError(f"group link_id {group.link_id} is outside 0..{n_links - 1}")
        constraints_matrix[group.link_id, column] = 1.0
        for sta in group.sta_ids:
            constraints_matrix[n_links + sta, column] = 1.0

    lower = np.concatenate((np.ones(n_links), np.zeros(n_stas)))
    upper = np.ones(n_links + n_stas)
    constraints = LinearConstraint(constraints_matrix.tocsr(), lower, upper)
    result = milp(
        c=-reward_array,
        integrality=np.ones(n_groups, dtype=int),
        bounds=Bounds(np.zeros(n_groups), np.ones(n_groups)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"ILP scheduling failed: {result.message}")

    selected = np.flatnonzero(result.x > 0.5).tolist()
    validate_selection(groups, selected, n_stas, n_links)
    return selected, float(reward_array[selected].sum())


def select_groups_greedy(
    groups: Sequence[CandidateGroup],
    rewards: Sequence[float],
    n_stas: int,
    n_links: int = 3,
) -> tuple[list[int], float]:
    """Greedy reward-ordered scheduler from Final_Code_010926.m."""
    reward_array = np.asarray(rewards, dtype=float)
    if reward_array.shape != (len(groups),):
        raise ValueError("groups and rewards must have equal lengths")
    selected: list[int] = []
    used_links: set[int] = set()
    used_stas: set[int] = set()

    order = np.argsort(-reward_array, kind="stable")
    for candidate_index in order:
        group = groups[int(candidate_index)]
        if group.link_id in used_links:
            continue
        if used_stas.intersection(group.sta_ids):
            continue
        selected.append(int(candidate_index))
        used_links.add(group.link_id)
        used_stas.update(group.sta_ids)
        if len(used_links) == n_links:
            break

    validate_selection(groups, selected, n_stas, n_links)
    return selected, float(reward_array[selected].sum())


def validate_selection(
    groups: Sequence[CandidateGroup], selected_indices: Iterable[int], n_stas: int, n_links: int = 3
) -> None:
    """Raise if a scheduler violates the MATLAB selection constraints."""
    selected = [groups[index] for index in selected_indices]
    link_ids = [group.link_id for group in selected]
    if len(selected) != n_links or set(link_ids) != set(range(n_links)):
        raise RuntimeError(f"Expected exactly one selected group per link; selected links={link_ids}")
    used_stas: list[int] = [sta for group in selected for sta in group.sta_ids]
    if len(used_stas) != len(set(used_stas)):
        raise RuntimeError("A STA was selected in more than one group")
    if any(sta < 0 or sta >= n_stas for sta in used_stas):
        raise RuntimeError("Selected group contains an invalid STA index")
