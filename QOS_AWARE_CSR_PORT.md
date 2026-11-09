# QoS-aware CSR: MATLAB-to-Python reproduction

This implementation ports the analytical multi-link experiment in
`qos_aware_scheduling/Final_Code_010926.m`. It is intentionally a separate
entry point from `MAPCsim`: the MATLAB reference computes candidate-group PHY
rewards and performs a three-link selection in an iteration; it does not run
the event-driven queue / backoff loop in `MAPCsim.m`.

## Files added or changed

- `src/qos_scheduler.py` — candidate groups, power allocation, SINR/MCS/BER/PER,
  throughput/reward functions, ILP and Greedy selection.
- `src/qos_aware_csr.py` — experiment runner and command-line interface.
- `tests/test_qos_scheduler.py` — unit tests for formulas, group generation,
  ILP/Greedy feasibility, frequency-aware path loss and MATLAB unit compatibility.
- `tools/export_qos_validation_fixture.py` / `tools/compare_matlab_validation.py` —
  deterministic cross-language fixture generation and result comparison.
- `validation/matlab/validate_qos_fixture.m` — MATLAB-side validator for shared inputs.
- `src/utils.py` — backward-compatible optional `frequency_ghz` and `rng`
  parameters for `get_loss()` / `get_channel_matrix()`. The old defaults remain
  unchanged. `pyswarm` is now an optional import used only by the existing PSO
  code path.

## Run

From the `python_code` project root:

```bash
# Fast integration smoke test (reduced SA search budget; not research results)
PYTHONPATH=src python src/qos_aware_csr.py \
  --iterations 1 --sa-maxiter 5 --sa-maxfun 40

# MATLAB-configured iteration count and stated annealing budgets
PYTHONPATH=src python src/qos_aware_csr.py

# Unit tests
PYTHONPATH=src pytest -q
```

Outputs are written under `results/qos_aware_csr/` by default:

- `qos_aware_csr_results.npz`
- `summary.json`
- `selected_groups.json`

## Reference configuration implemented

- 4 APs, 16 STAs, 48 m square grid; two perpendicular walls.
- 2 spatial streams; noise = -95 dBm; CCA threshold = -82 dBm.
- 100 iterations, seed = 2.
- Links: 6.035 GHz / 80 MHz / 980 subcarriers; 5.22 GHz / 40 MHz / 468;
  2.412 GHz / 20 MHz / 234.
- AC weights: VO=1.85, VI=1.40, BE=1.00, BK=0.45.
- AC sampling weights: [1, 1, 1, 2.5]. This is deliberately separate from
  the QoS reward weights.
- TXOPs: VO=1.5 ms, VI=3.0 ms, BE=1.0 ms, BK=1.0 ms.
- Candidate groups: up to four STAs, at most one STA per AP.
- CSR overhead is calculated using `OverheadsCalc('BE')` for all links, as the
  reference script actually does.

## Algorithm details preserved

1. The topology is generated once, before the iteration loop, using the
   distance distributions in the QoS archive's `AP_STA_coordinates.m`.
2. Each iteration assigns one AC per STA, then generates a separate channel
   matrix for each link frequency.
3. Candidate groups start with singleton rows, followed by the multi-STA
   Cartesian combinations. The reference pruning rule is preserved:
   `group_size * group_PHY_rate >= singleton_PHY_rate` for each member;
   a failed group's supersets are pruned.
4. Multi-STA groups use bounded proportional-fair power allocation:
   maximize `sum(log(PHY_rate_i))` with power bounds [1 mW, Pmax].
5. The conventional reward uses unweighted per-STA throughput and each STA's
   own TXOP. The proposed reward uses weighted throughput and the maximum TXOP
   among the stations in the group.
6. ILP enforces exactly one group per link and at most one appearance per STA.
   Greedy sorts by reward and applies the same link/STA conflict rules.
7. Reported throughput is unweighted successful throughput, even for the
   QoS-weighted methods, matching the MATLAB evaluation loop.

## Important fidelity limits / not yet validated

- NumPy and MATLAB use different random-number implementations. A fixed seed
  makes this Python run repeatable, but does not reproduce MATLAB's exact
  topology, shadowing or AC sequence.
- MATLAB `simulannealbnd` and SciPy `dual_annealing` are related bounded
  simulated-annealing solvers, but their internal trajectories differ. The
  Python port matches the objective, [1, Pmax] bounds, midpoint initial point,
  initial temperature 100, maximum iterations 500 and function-evaluation
  budget 10,000. MATLAB's `ReannealInterval=100` is not a one-to-one option in
  SciPy's solver.
- No MATLAB reference `.mat`, `.csv` or printed numeric result file was
  supplied, so numeric parity has not yet been demonstrated. The code and
  intermediate quantities need a same-input comparison before claiming
  identical numerical results.
- The port is the analytical three-link MATLAB experiment. Integrating it into
  the Python event-driven packet/queue simulator is a separate Phase 2 task.

## MATLAB-specific topology edge case

The topology port explicitly preserves MATLAB's `linspace(a, b, 1) == b`
rule. NumPy's `linspace(a, b, 1)` returns `a`, so using it unchanged would
place the one far STA per AP at the wrong radius in the default 4-STA/AP
configuration. A regression test covers this difference.

## Reward-unit compatibility modes

The reference MATLAB files contain a reward-stage unit inconsistency:

- `CGcreation.m` converts `MaxTxPower_dBm` to mW and stores optimized powers in mW.
- `Final_Code_010926.m` converts `Pn_dBm` to W for the reward loop and converts singleton maximum power to W.
- The same reward loop reuses multi-STA `TxPowerMatrix` values without converting mW to W.

The discrepancy affects the noise term relative to signal/interference, so its SINR effect depends on the ratio of co-channel interference to noise. It can approach 30 dB in a noise-limited case; it is much smaller in interference-limited cases. The singleton branch is consistent because it converts both transmit power and noise to W.

The runner now makes the choice explicit:

```bash
# Literal reward-loop behavior in Final_Code_010926.m (default)
PYTHONPATH=src python src/qos_aware_csr.py --iterations 1 --sa-maxiter 5 --sa-maxfun 40 \
  --reward-units-mode matlab_as_written

# Dimensionally consistent physical calculation (corrected variant)
PYTHONPATH=src python src/qos_aware_csr.py --iterations 1 --sa-maxiter 5 --sa-maxfun 40 \
  --reward-units-mode physical_consistent
```

Do not present these two modes as equivalent. Use `matlab_as_written` for literal comparison to the scholar's current `Final_Code_010926.m`; discuss `physical_consistent` as a correction that should be approved before changing the research baseline.

## Shared MATLAB/Python validation fixture

See `validation/MATLAB_PYTHON_VALIDATION.md`. The `tools/export_qos_validation_fixture.py` script creates deterministic `.mat` inputs and Python expected outputs. Run `validation/matlab/validate_qos_fixture.m` in MATLAB with the QoS archive's helper functions on the path, then run `tools/compare_matlab_validation.py`.

This isolates formulas and units from random-stream and optimizer differences. It checks SINR, MCS, BER/PER, PHY rate, TXOP rewards, the mixed-unit reference path, singleton handling, path loss, wall intersections, 624 candidate-group rows/order, TX power/subcarrier constants, and a small ILP/Greedy instance.
