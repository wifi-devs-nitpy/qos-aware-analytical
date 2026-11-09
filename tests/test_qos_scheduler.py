import numpy as np

import utils
from qos_aware_csr import _matlab_linspace

from qos_scheduler import (
    CandidateGroup,
    calculate_ber_per,
    calculate_sinr,
    evaluate_conventional_group,
    evaluate_group,
    generate_matlab_group_rows,
    mcs_from_sinr,
    select_groups_greedy,
    select_groups_ilp,
)


def _group(link, stas, aps, sinr=(30.0,), rates=(100e6,)):
    n = len(stas)
    return CandidateGroup(
        link_id=link,
        sta_ids=tuple(stas),
        ap_ids=tuple(aps),
        tx_power_mw=np.ones(n),
        sinr_db=np.full(n, sinr[0]),
        mcs=np.full(n, 3),
        n_bps=np.full(n, 4),
        coding_rate=np.full(n, 0.5),
        phy_rate_bps=np.asarray(rates if len(rates) == n else [rates[0]] * n),
        csr_overhead_s=0.0005,
        n_subcarriers=980,
    )


def test_mcs_threshold_edges_match_reference():
    assert mcs_from_sinr(14.2861)[0] == -1
    assert mcs_from_sinr(14.2862)[0] == 0
    assert mcs_from_sinr(19.5154)[0] == 1
    assert mcs_from_sinr(57.3929)[0] == 13


def test_ber_per_is_bounded_and_monotonic_with_sinr():
    _, per_low = calculate_ber_per(20.0, 2)
    _, per_high = calculate_ber_per(30.0, 2)
    assert 0.0 <= per_high <= per_low <= 1.0


def test_sinr_interference_calculation():
    h = np.array([[1.0, 0.1], [0.2, 1.0]])
    p = np.array([10.0, 10.0])
    sinr = calculate_sinr(h, p, noise_power_mw=1.0)
    np.testing.assert_allclose(sinr, [10.0 / 2.0, 10.0 / 3.0])


def test_group_rows_start_with_singletons_and_respect_cg_size():
    association = [[0, 1], [2, 3]]
    rows = generate_matlab_group_rows(association, n_stas=4, cg_size=2)
    assert rows[:4] == [(0, -1), (1, -1), (-1, 2), (-1, 3)]
    assert all(sum(sta >= 0 for sta in row) <= 2 for row in rows)
    assert len(rows) == 8  # 4 singleton + 4 cross-AP pairs
    # MATLAB combvec makes the first AP dimension vary fastest.
    assert rows[4:] == [
        (0, 2), (1, 2), (0, 3), (1, 3)
    ]


def test_qos_reward_uses_aligned_txop_and_weights():
    group = _group(0, (0, 1), (0, 1), sinr=(30.0,))
    ac = np.array([0, 1])  # VO + VI; max TXOP is 3 ms
    txops = [1.5e-3, 3.0e-3, 1.0e-3, 1.0e-3]
    qos = evaluate_group(group, ac, [1.85, 1.4, 1.0, 0.45], txops)
    conventional = evaluate_conventional_group(group, ac, [1.85, 1.4, 1.0, 0.45], txops)
    assert qos.group_txop_s == 3.0e-3
    assert qos.weighted_reward > 0
    assert conventional.rates_bps.shape == (2,)
    assert not np.allclose(qos.rates_bps, conventional.rates_bps)


def test_ilp_and_greedy_return_one_group_per_link_without_station_conflicts():
    groups = [
        _group(0, (0,), (0,)),
        _group(0, (1,), (0,)),
        _group(1, (2,), (1,)),
        _group(1, (3,), (1,)),
        _group(2, (4,), (2,)),
        _group(2, (5,), (2,)),
    ]
    rewards = np.array([10.0, 12.0, 9.0, 8.0, 7.0, 11.0])
    ilp_selected, ilp_reward = select_groups_ilp(groups, rewards, n_stas=6, n_links=3)
    greedy_selected, greedy_reward = select_groups_greedy(groups, rewards, n_stas=6, n_links=3)
    assert len(ilp_selected) == len(greedy_selected) == 3
    assert ilp_reward == 32.0
    assert greedy_reward == 32.0


def test_default_matlab_group_count_is_624():
    association = [list(range(i * 4, (i + 1) * 4)) for i in range(4)]
    rows = generate_matlab_group_rows(association, n_stas=16, cg_size=4)
    assert len(rows) == 624
    assert sum(sum(sta >= 0 for sta in row) == 1 for row in rows) == 16
    assert sum(sum(sta >= 0 for sta in row) > 1 for row in rows) == 608


def test_frequency_aware_pathloss_matches_log_distance_term():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    loss_6 = utils.get_loss(a, b, 0, std_dev=0, frequency_ghz=6.035)
    loss_5 = utils.get_loss(a, b, 0, std_dev=0, frequency_ghz=5.22)
    expected_delta = 20.0 * np.log10(6.035 / 5.22)
    np.testing.assert_allclose(loss_6 - loss_5, expected_delta, rtol=0, atol=1e-12)


def test_matlab_linspace_single_point_returns_second_endpoint():
    np.testing.assert_array_equal(_matlab_linspace(9.0, 12.0, 1), [12.0])
    assert _matlab_linspace(9.0, 12.0, 0).size == 0


def test_matlab_as_written_mixed_units_are_explicit_and_singleton_is_consistent():
    from qos_scheduler import apply_matlab_as_written_reward_units

    h = np.array(
        [
            [1.20e-7, 1.0e-14, 1.5e-14],
            [2.0e-14, 9.00e-8, 1.2e-14],
            [1.0e-14, 2.5e-14, 6.00e-8],
        ],
        dtype=float,
    )
    powers_mw = np.array([50.0, 40.0, 30.0])
    noise_mw = 10 ** (-95 / 10)
    group = CandidateGroup(
        link_id=0,
        sta_ids=(0, 1, 2),
        ap_ids=(0, 1, 2),
        tx_power_mw=powers_mw,
        sinr_db=10 * np.log10(calculate_sinr(h, powers_mw, noise_mw)),
        mcs=np.array([7, 6, 5]),
        n_bps=np.array([6, 6, 6]),
        coding_rate=np.array([5 / 6, 3 / 4, 2 / 3]),
        phy_rate_bps=np.ones(3),
        csr_overhead_s=552.8e-6,
        n_subcarriers=980,
        channel_submatrix=h,
    )

    legacy = apply_matlab_as_written_reward_units(
        group,
        noise_power_w=noise_mw / 1000,
        max_tx_power_w=50.0 / 1000,
        nss=2,
    )
    physical_sinr = 10 * np.log10(calculate_sinr(h, powers_mw, noise_mw))
    assert np.all(legacy.sinr_db - physical_sinr > 20.0)
    assert np.all(legacy.sinr_db - physical_sinr < 30.0)

    # The MATLAB reference uses W for both singleton transmit power and noise.
    singleton = CandidateGroup(
        link_id=0,
        sta_ids=(0,),
        ap_ids=(0,),
        tx_power_mw=np.array([50.0]),
        sinr_db=np.array([0.0]),
        mcs=np.array([0]),
        n_bps=np.array([1]),
        coding_rate=np.array([0.5]),
        phy_rate_bps=np.array([1.0]),
        csr_overhead_s=552.8e-6,
        n_subcarriers=980,
        channel_submatrix=np.array([[1.2e-7]]),
    )
    singleton_legacy = apply_matlab_as_written_reward_units(
        singleton,
        noise_power_w=noise_mw / 1000,
        max_tx_power_w=50.0 / 1000,
        nss=2,
    )
    singleton_physical = 10 * np.log10(50.0 * 1.2e-7 / noise_mw)
    np.testing.assert_allclose(singleton_legacy.sinr_db, [singleton_physical], rtol=0, atol=1e-12)
