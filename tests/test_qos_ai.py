import numpy as np

from qos_ai import BASE_FEATURE_DIM, FEATURE_DIM, CONTEXT_DIM, GroupSelectionNet, _make_group_features, _scenario_context
from qos_scheduler import CandidateGroup, validate_selection
import torch


def _candidate(link, stations):
    n = len(stations)
    return CandidateGroup(
        link_id=link,
        sta_ids=tuple(stations),
        ap_ids=tuple(range(n)),
        tx_power_mw=np.ones(n),
        sinr_db=np.full(n, 35.0),
        mcs=np.full(n, 5),
        n_bps=np.full(n, 4),
        coding_rate=np.full(n, 0.5),
        phy_rate_bps=np.full(n, 100e6),
        csr_overhead_s=0.0005,
        n_subcarriers=980,
    )


def test_feature_vector_has_expected_shape_and_station_membership():
    group = _candidate(1, (2, 5))
    sta_ac = np.zeros(16, dtype=int)
    features = _make_group_features(group, sta_ac, np.array([100e6, 200e6]), 60.0)
    assert features.shape == (BASE_FEATURE_DIM,)
    # Link one-hot starts at 0; link 1 is active.
    assert features[1] == 1.0
    station_offset = 3
    assert features[station_offset + 2] == 1.0
    assert features[station_offset + 5] == 1.0


def test_group_selection_network_output_shape():
    model = GroupSelectionNet()
    output = model(torch.zeros((7, FEATURE_DIM), dtype=torch.float32))
    assert output.shape == (7,)


def test_global_context_encodes_three_links_and_station_conflicts():
    groups = [_candidate(0, (0, 1)), _candidate(1, (1, 2)), _candidate(2, (3,))]
    context = _scenario_context(groups, np.array([100e6, 200e6, 300e6]))
    assert context.shape == (CONTEXT_DIM,)
    assert np.all(np.isfinite(context))


def test_ai_inference_decoder_returns_feasible_schedule_without_solver():
    from qos_ai import _score_and_select
    from qos_scheduler import CandidateGroup

    groups = []
    for link, stations in [(0, (0,)), (0, (1,)), (1, (2,)), (1, (3,)), (2, (4,)), (2, (5,))]:
        groups.append(CandidateGroup(
            link_id=link,
            sta_ids=stations,
            ap_ids=(0,),
            tx_power_mw=np.ones(1),
            sinr_db=np.array([30.0]),
            mcs=np.array([3]),
            n_bps=np.array([4]),
            coding_rate=np.array([0.5]),
            phy_rate_bps=np.array([100e6]),
            csr_overhead_s=0.0005,
            n_subcarriers=980,
        ))
    model = GroupSelectionNet()
    features = np.random.default_rng(7).normal(size=(len(groups), FEATURE_DIM)).astype(np.float32)
    mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    std = np.ones(FEATURE_DIM, dtype=np.float32)
    selected, _, _ = _score_and_select(model, features, mean, std, groups, "cpu")
    assert len(selected) == 3
    chosen = [groups[i] for i in selected]
    assert len({g.link_id for g in chosen}) == 3
    stas = [sta for g in chosen for sta in g.sta_ids]
    assert len(stas) == len(set(stas))
