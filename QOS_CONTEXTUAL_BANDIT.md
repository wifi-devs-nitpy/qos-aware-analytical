# QoS-Aware CSR Scheduling with LinUCB

## Final comparison scope

The final reported comparison contains exactly three methods:

1. Conventional CSR using ILP (unweighted throughput objective and per-STA TXOP).
2. Proposed QoS-aware CSR using ILP (AC-weighted throughput objective and group-aligned TXOP).
3. Proposed QoS-aware CSR using LinUCB (learned predicted group scores, evaluated with the same QoS-aligned rate model).

Greedy is excluded from the final comparison.

## Recommended final experiment

Run from the Python project root:

```bash
PYTHONPATH=src python src/qos_contextual_bandit.py \
    --train-scenarios 1000 \
    --test-scenarios 100 \
    --seed 20261110 \
    --topology-seed 2 \
    --sa-maxiter 20 \
    --sa-maxfun 10000 \
    --alpha 0.7 \
    --ridge 1.0 \
    --output-dir results/qos_final_1000x100
```

This creates 1,000 training scenarios and 100 held-out test scenarios. LinUCB receives three selected-group reward observations per training scenario, for 3,000 semi-bandit updates. No ILP is called during training or LinUCB inference.

Each held-out test scenario is generated once and is shared by all three methods. The conventional and QoS-aware ILPs use different objectives on the same candidate groups. LinUCB uses its learned posterior-mean scores and the feasibility decoder; its selected schedule is evaluated using the QoS-aligned reward model.

## Meaning of the three methods

### Conventional CSR (ILP)
- Candidate-group objective: sum of unweighted per-STA rates.
- TXOP: each STA uses its own AC-specific TXOP.
- Used as the conventional baseline.

### QoS-aware CSR (ILP)
- Candidate-group objective: sum of AC-weighted per-STA rates.
- TXOP: all STAs in a candidate group use the maximum TXOP required by any member.
- Used as the optimal reference for the implemented weighted objective.

### QoS-aware CSR (LinUCB)
- Context: 63 PHY, AC, link, group and conflict features.
- Policy: LinUCB estimates per-group expected utility; the decoder chooses the maximum-scoring feasible three-link combination.
- Training feedback: rewards of the three selected groups only. Unselected candidate rewards are not used for model updates.
- Test-time action selection uses the posterior mean (no exploration bonus).

## What is reported

The runner produces MATLAB-style summaries and figures:

- Mean aggregate network throughput in Gbps and percentage improvement over conventional CSR-ILP.
- Throughput per test scenario (first five plotted, all scenarios saved).
- Mean per-STA throughput by access category, computed in the same per-scenario-then-average manner as the MATLAB reference.
- Total number of scheduled stations per access category.
- Total and average scheduler execution time.
- QoS-weighted utility for QoS-ILP and LinUCB, including a LinUCB/QoS-ILP ratio.
- LinUCB exact schedule-match rate against QoS-ILP.
- Constraint violations and LinUCB inference time.
- LinUCB training reward trace.
- Fixed network topology plot.

## Output files

The output directory contains:

- `summary.json`: complete machine-readable summary, scenario counts, SA budget, topology and limitations.
- `FINAL_RESULTS_REPORT.md`: compact report with final tables.
- `qos_final_results.npz`: arrays for independent plotting and analysis.
- `qos_contextual_bandit.npz`: trained LinUCB checkpoint.
- `training_history.csv`: one row per training scenario.
- `test_scenarios.json`: per-scenario results for all three methods.
- `selected_groups.json`: selected groups for all three methods and all test scenarios.
- `final_comparison_table.csv`, `per_ac_throughput.csv`, `per_ac_total_throughput.csv`, `ac_scheduling_counts.csv`, `scheduler_execution_time.csv`.
- `topology.json`.
- Six PNG figures: throughput, per-AC throughput, topology, AC scheduling counts, scheduler execution time and LinUCB training reward.

## Standalone LinUCB inference

```bash
PYTHONPATH=src python src/qos_contextual_bandit_infer.py \
    --checkpoint results/qos_final_1000x100/qos_contextual_bandit.npz \
    --seed 900777 \
    --topology-seed 2 \
    --sa-maxiter 20 \
    --sa-maxfun 10000
```

No ILP is invoked by this inference script.

## Scientific qualifications

- This is a static-scenario contextual combinatorial bandit, not temporal packet-level network control.
- The AP/STA topology is fixed across training and testing; AC assignments and channel realizations vary. The current experiment does not establish generalization to unseen deployment geometries.
- The Python optimizer uses SciPy `dual_annealing`; it is not guaranteed to find global power-allocation optima and does not share MATLAB's exact random/solver trajectory.
- The `matlab_as_written` mixed W/mW reward calculation is retained for parity with the supplied MATLAB reference.
- AC weights are soft scheduling preferences, not hard minimum-throughput guarantees.
- The final experiment uses the explicitly recorded SA budget. Results should not be compared to MATLAB's 500-iteration SA results as if their power optimization budgets were identical.
