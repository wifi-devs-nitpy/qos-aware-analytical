import numpy as np
import pytest

from qos_contextual_bandit import (
    FEATURE_DIM,
    LinUCBContextualBandit,
    build_group_context_features,
    select_best_feasible_schedule,
)
from qos_scheduler import CandidateGroup, validate_selection


def candidate(link, stations, reward_marker=1.0):
    n = len(stations)
    return CandidateGroup(
        link_id=link,
        sta_ids=tuple(stations),
        ap_ids=tuple(i % 4 for i in range(n)),
        tx_power_mw=np.full(n, 10.0 * reward_marker),
        sinr_db=np.full(n, 35.0 + reward_marker),
        mcs=np.full(n, 5),
        n_bps=np.full(n, 4),
        coding_rate=np.full(n, 0.5),
        phy_rate_bps=np.full(n, 200e6 * reward_marker),
        csr_overhead_s=552.8e-6,
        n_subcarriers=980,
    )


def test_context_features_are_finite_and_exclude_reward_argument():
    groups = [candidate(0, (0, 1)), candidate(1, (2, 3)), candidate(2, (4,))]
    ac = np.array([0, 1, 2, 3] + [0] * 12)
    features = build_group_context_features(groups, ac)
    assert features.shape == (3, FEATURE_DIM)
    assert np.all(np.isfinite(features))
    assert features.shape[1] == FEATURE_DIM


def test_linucb_predict_and_update_uses_observed_selected_arms_only():
    groups = [candidate(0, (0,)), candidate(1, (1,)), candidate(2, (2,))]
    ac = np.zeros(16, dtype=int)
    x = build_group_context_features(groups, ac)
    model = LinUCBContextualBandit(alpha=0.5)
    before = model.n_observations
    scores, uncertainty = model.predict(x, explore=True)
    assert scores.shape == (3,)
    assert uncertainty.shape == (3,)
    model.update_selected(x, [0, 2], [100e6, 300e6])
    assert model.n_observations == before + 2
    assert not np.allclose(model.theta, 0.0)


def test_exhaustive_decoder_returns_best_feasible_joint_schedule():
    groups = [
        candidate(0, (0,)), candidate(0, (1,)),
        candidate(1, (0, 2)), candidate(1, (2,)),
        candidate(2, (1, 3)), candidate(2, (3,)),
    ]
    scores = np.array([100.0, 90.0, 200.0, 80.0, 200.0, 70.0])
    selected = select_best_feasible_schedule(groups, scores)
    validate_selection(groups, selected, 16, 3)
    assert set(selected) == {0, 3, 4}
    assert scores[selected].sum() == 380.0


def test_decoder_rejects_infeasible_candidate_set():
    groups = [candidate(0, (0,)), candidate(1, (0,)), candidate(2, (0,))]
    with pytest.raises(RuntimeError, match="no feasible"):
        select_best_feasible_schedule(groups, [1.0, 1.0, 1.0])


def test_model_checkpoint_round_trip(tmp_path):
    model = LinUCBContextualBandit(alpha=0.9, ridge=2.0)
    x = np.zeros(FEATURE_DIM)
    x[0] = 1.0
    model.update(x, 0.75)
    path = tmp_path / "bandit.npz"
    model.save(path, {"purpose": "test"})
    loaded, metadata = LinUCBContextualBandit.load(path)
    assert loaded.n_observations == 1
    assert loaded.alpha == 0.9
    assert loaded.ridge == 2.0
    assert metadata["purpose"] == "test"
    np.testing.assert_allclose(loaded.A, model.A)
    np.testing.assert_allclose(loaded.b, model.b)


def test_final_three_method_report_smoke(tmp_path):
    from qos_contextual_bandit import run_contextual_bandit_experiment
    import json

    report = run_contextual_bandit_experiment(
        train_scenarios_count=1,
        test_scenarios_count=1,
        seed=314159,
        topology_seed=2,
        sa_maxiter=1,
        sa_maxfun=15,
        output_dir=tmp_path / "final_report",
    )
    assert report["methods"] == [
        "Conventional CSR (ILP)",
        "QoS-aware CSR (ILP)",
        "QoS-aware CSR (LinUCB)",
    ]
    assert report["bandit"]["observations"] == 3
    assert report["training_ilp_calls"] == 0
    assert report["constraint_violations"] == 0

    output = tmp_path / "final_report"
    expected_files = {
        "summary.json",
        "FINAL_RESULTS_REPORT.md",
        "qos_contextual_bandit.npz",
        "qos_final_results.npz",
        "final_comparison_table.csv",
        "per_ac_throughput.csv",
        "ac_scheduling_counts.csv",
        "scheduler_execution_time.csv",
        "selected_groups.json",
        "topology.json",
    }
    assert expected_files.issubset({path.name for path in output.iterdir()})
    saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(saved_summary["methods"]) == 3
    assert "Greedy" not in " ".join(saved_summary["methods"])
