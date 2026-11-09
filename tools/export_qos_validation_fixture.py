"""Create a deterministic cross-language PHY/reward fixture for MATLAB validation.

Run from the project root:
    python tools/export_qos_validation_fixture.py

The script writes validation/qos_validation_fixture.mat and
validation/python_expected.mat. These contain fixed inputs and Python outputs;
MATLAB can independently calculate and export the same named outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
VALIDATION = PROJECT_ROOT / "validation"
sys.path.insert(0, str(SRC))

import utils  # noqa: E402
from qos_scheduler import (  # noqa: E402
    CandidateGroup, calculate_ber_per, calculate_sinr, mcs_from_sinr,
    generate_matlab_group_rows,
    select_groups_greedy, select_groups_ilp,
)


def _phy_values(h: np.ndarray, p: np.ndarray, noise: float, nsc: int, nss: int):
    sinr_linear = calculate_sinr(h, p, noise)
    sinr_db = 10.0 * np.log10(sinr_linear)
    mcs = np.full(sinr_db.size, -1, dtype=np.int32)
    n_bps = np.ones(sinr_db.size, dtype=np.int32)
    coding_rate = np.full(sinr_db.size, 0.5, dtype=float)
    ber = np.full(sinr_db.size, np.nan, dtype=float)
    per = np.full(sinr_db.size, np.nan, dtype=float)
    phy_rate = np.zeros(sinr_db.size, dtype=float)

    for i, value in enumerate(sinr_db):
        m, bits, rc = mcs_from_sinr(float(value))
        mcs[i], n_bps[i], coding_rate[i] = m, bits, rc
        if m >= 0:
            ber[i], per[i] = calculate_ber_per(float(value), bits, 12_000)
            phy_rate[i] = nsc * bits * rc * nss / (12.8e-6 + 0.8e-6)
    return {
        "sinr_db": sinr_db,
        "mcs": mcs,
        "n_bps": n_bps,
        "coding_rate": coding_rate,
        "ber": ber,
        "per": per,
        "phy_rate_bps": phy_rate,
    }


def _delivered_rates(phy: dict[str, np.ndarray], txops: np.ndarray, ac_idx: np.ndarray, csr: float):
    group_txop = float(np.max(txops[ac_idx]))
    aligned_rates = np.zeros(ac_idx.size, dtype=float)
    conventional_rates = np.zeros(ac_idx.size, dtype=float)
    for i in range(ac_idx.size):
        if phy["mcs"][i] < 0:
            continue
        packets_aligned = max(int(np.floor(phy["phy_rate_bps"][i] * (group_txop - csr) / 12_000)), 1)
        aligned_rates[i] = packets_aligned * 12_000 * (1.0 - phy["per"][i]) / group_txop
        own_txop = float(txops[ac_idx[i]])
        packets_conventional = max(int(np.floor(phy["phy_rate_bps"][i] * (own_txop - csr) / 12_000)), 1)
        conventional_rates[i] = packets_conventional * 12_000 * (1.0 - phy["per"][i]) / own_txop
    return group_txop, aligned_rates, conventional_rates


def main() -> None:
    VALIDATION.mkdir(parents=True, exist_ok=True)

    # A fixed three-STA channel block with weak inter-STA coupling. This makes
    # the reference's mixed mW/W reward-stage behavior observable.
    h = np.array(
        [
            [1.20e-7, 1.0e-14, 1.5e-14],
            [2.0e-14, 9.00e-8, 1.2e-14],
            [1.0e-14, 2.5e-14, 6.00e-8],
        ],
        dtype=float,
    )
    tx_power_mw = np.array([50.0, 40.0, 30.0], dtype=float)
    pn_dbm = -95.0
    noise_mw = 10.0 ** (pn_dbm / 10.0)
    nsc, nss, packet_bits = 980, 2, 12_000
    ac_idx = np.array([0, 1, 3], dtype=np.int32)  # VO, VI, BK (Python zero-based)
    ac_weights = np.array([1.85, 1.40, 1.00, 0.45], dtype=float)
    txops = np.array([1.5e-3, 3.0e-3, 1.0e-3, 1.0e-3], dtype=float)
    csr = float(utils.overheads_calc("BE")["csr_overheads"])

    physical = _phy_values(h, tx_power_mw, noise_mw, nsc, nss)
    aligned_txop, aligned_rates, conventional_rates = _delivered_rates(physical, txops, ac_idx, csr)
    aligned_weighted_reward = float(np.sum(ac_weights[ac_idx] * aligned_rates))
    conventional_reward = float(np.sum(conventional_rates))

    # Reproduce Final_Code_010926.m reward-loop units exactly for a multi-STA
    # group: P is numerically mW while noise is numerically W.
    legacy_multi = _phy_values(h, tx_power_mw, noise_mw / 1000.0, nsc, nss)
    legacy_txop, legacy_aligned_rates, legacy_conventional_rates = _delivered_rates(
        legacy_multi, txops, ac_idx, csr
    )

    # Singleton branch in Final_Code_010926.m uses MaxTxPower_all in W and
    # noise in W, so its SINR is dimensionally consistent.
    singleton_h = np.array([[1.20e-7]], dtype=float)
    singleton_pmax_mw = 50.0
    singleton_physical = _phy_values(singleton_h, np.array([singleton_pmax_mw]), noise_mw, nsc, nss)
    singleton_matlab = _phy_values(
        singleton_h,
        np.array([singleton_pmax_mw / 1000.0]),
        noise_mw / 1000.0,
        nsc,
        nss,
    )

    # Verify path-loss formula at fixed coordinates, without random shadowing.
    pos_a = np.array([5.0, 5.0])
    pos_b = np.array([9.0, 8.0])
    frequencies = np.array([6.035, 5.22, 2.412], dtype=float)
    pathloss = np.array(
        [utils.get_loss(pos_a, pos_b, 1, std_dev=0.0, frequency_ghz=f) for f in frequencies]
    )
    txpower_dbm = np.array([utils.tx_power_calc(bw, nss)[0] for bw in (80, 40, 20)])
    nsc_by_link = np.array([utils.tx_power_calc(bw, nss)[1] for bw in (80, 40, 20)], dtype=np.int32)

    # Segment/wall intersection cases, including crossing, no-crossing,
    # endpoint contact and collinear overlap. Fields follow MATLAB's argument order.
    wall_cases = np.array(
        [
            [0, 10, 0, 10, 5, 5, 0, 10],
            [0, 4, 0, 3, 5, 5, 0, 10],
            [0, 5, 0, 5, 5, 5, 0, 10],
            [0, 10, 5, 5, 2, 8, 5, 5],
        ],
        dtype=float,
    )
    wall_intersections = np.array(
        [
            utils.check_segment_intersection(*case[:4], case[4:])
            for case in wall_cases
        ],
        dtype=np.int32,
    )

    # Small ILP instance with a unique optimum. Python and MATLAB use the same
    # one-group-per-link and one-use-per-STA constraints.
    ilp_links = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    ilp_stas = [(0,), (1,), (0,), (2,), (1,), (2,)]
    ilp_rewards = np.array([12.0, 8.0, 10.0, 9.0, 11.0, 7.0])
    ilp_groups = [
        CandidateGroup(
            link_id=int(ilp_links[i]), sta_ids=stas, ap_ids=(0,),
            tx_power_mw=np.array([1.0]), sinr_db=np.array([30.0]),
            mcs=np.array([3]), n_bps=np.array([4]), coding_rate=np.array([0.5]),
            phy_rate_bps=np.array([1.0]), csr_overhead_s=0.0, n_subcarriers=980,
        )
        for i, stas in enumerate(ilp_stas)
    ]
    ilp_selected, ilp_objective = select_groups_ilp(
        ilp_groups, ilp_rewards, n_stas=3, n_links=3
    )
    greedy_selected, greedy_objective = select_groups_greedy(
        ilp_groups, ilp_rewards, n_stas=3, n_links=3
    )

    default_association = [list(range(ap * 4, (ap + 1) * 4)) for ap in range(4)]
    candidate_group_rows = np.asarray(
        [[0 if sta < 0 else sta + 1 for sta in row]
         for row in generate_matlab_group_rows(default_association, n_stas=16, cg_size=4)],
        dtype=np.int32,
    )

    fixture = {
        "channel_submatrix": h,
        "tx_power_mw": tx_power_mw,
        "noise_power_mw": np.array([[noise_mw]]),
        "pn_dbm": np.array([[pn_dbm]]),
        "n_subcarriers": np.array([[nsc]], dtype=np.int32),
        "nss": np.array([[nss]], dtype=np.int32),
        "packet_bits": np.array([[packet_bits]], dtype=np.int32),
        "ac_idx_matlab": (ac_idx + 1).reshape(1, -1),
        "ac_weights": ac_weights.reshape(1, -1),
        "txop_by_ac_s": txops.reshape(1, -1),
        "csr_overhead_s": np.array([[csr]]),
        "wall_intersection_cases": wall_cases,
        "ilp_group_links_matlab": (ilp_links + 1).reshape(1, -1),
        "ilp_group_sta_matrix_matlab": np.array(
            [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.int32
        ),
        "ilp_rewards": ilp_rewards.reshape(1, -1),
        "candidate_group_rows_matlab": candidate_group_rows,
        "singleton_channel": singleton_h,
        "singleton_pmax_mw": np.array([[singleton_pmax_mw]]),
        "pathloss_pos_a": pos_a.reshape(1, -1),
        "pathloss_pos_b": pos_b.reshape(1, -1),
        "pathloss_n_walls": np.array([[1]], dtype=np.int32),
        "pathloss_frequencies_ghz": frequencies.reshape(1, -1),
    }

    expected = {
        "physical_sinr_db": physical["sinr_db"],
        "physical_mcs": physical["mcs"],
        "physical_n_bps": physical["n_bps"],
        "physical_coding_rate": physical["coding_rate"],
        "physical_ber": physical["ber"],
        "physical_per": physical["per"],
        "physical_phy_rate_bps": physical["phy_rate_bps"],
        "aligned_group_txop_s": np.array([aligned_txop]),
        "aligned_rates_bps": aligned_rates,
        "aligned_weighted_reward": np.array([aligned_weighted_reward]),
        "conventional_rates_bps": conventional_rates,
        "conventional_reward": np.array([conventional_reward]),
        "legacy_multi_sinr_db": legacy_multi["sinr_db"],
        "legacy_multi_mcs": legacy_multi["mcs"],
        "legacy_multi_n_bps": legacy_multi["n_bps"],
        "legacy_multi_per": legacy_multi["per"],
        "legacy_multi_phy_rate_bps": legacy_multi["phy_rate_bps"],
        "legacy_aligned_rates_bps": legacy_aligned_rates,
        "legacy_conventional_rates_bps": legacy_conventional_rates,
        "singleton_physical_sinr_db": singleton_physical["sinr_db"],
        "singleton_legacy_sinr_db": singleton_matlab["sinr_db"],
        "pathloss_db": pathloss,
        "txpower_dbm": txpower_dbm,
        "nsc": nsc_by_link,
        "csr_be": np.array([csr]),
        "wall_intersections": wall_intersections,
        "ilp_selected_indices": np.asarray(ilp_selected, dtype=np.int32) + 1,
        "ilp_objective": np.array([ilp_objective]),
        "greedy_selected_indices": np.asarray(greedy_selected, dtype=np.int32) + 1,
        "greedy_objective": np.array([greedy_objective]),
        "candidate_group_rows": candidate_group_rows,
    }

    savemat(VALIDATION / "qos_validation_fixture.mat", fixture, do_compression=True)
    savemat(VALIDATION / "python_expected.mat", expected, do_compression=True)
    summary = {
        "csr_overhead_us": csr * 1e6,
        "physical_sinr_db": physical["sinr_db"].tolist(),
        "txpower_dbm": txpower_dbm.tolist(),
        "nsc": nsc_by_link.tolist(),
        "physical_mcs": physical["mcs"].tolist(),
        "legacy_mixed_unit_sinr_db": legacy_multi["sinr_db"].tolist(),
        "legacy_minus_physical_sinr_db": (legacy_multi["sinr_db"] - physical["sinr_db"]).tolist(),
        "physical_aligned_weighted_reward_mbps": aligned_weighted_reward / 1e6,
        "physical_conventional_reward_mbps": conventional_reward / 1e6,
        "legacy_aligned_reward_mbps": float(np.sum(ac_weights[ac_idx] * legacy_aligned_rates)) / 1e6,
        "legacy_conventional_reward_mbps": float(np.sum(legacy_conventional_rates)) / 1e6,
        "note": "Synthetic deterministic validation fixture; not a simulation result.",
    }
    (VALIDATION / "python_expected_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote fixture and expected outputs under: {VALIDATION}")


if __name__ == "__main__":
    main()
