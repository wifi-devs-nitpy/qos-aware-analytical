# AI-based QoS-aware Multi-Link CSR Scheduling

## Goal

Learn to make the same analytical scheduling decision as the MATLAB QoS-aware CSR experiment, without integrating with `MAPCsim` and without packet-level latency/queue analysis.

For each network snapshot, the input contains three frequency links, candidate coordinated groups, PHY information and per-STA access categories. The schedule must select exactly one candidate group per link and must not select a station more than once.

## First AI baseline: supervised neural imitation

The first implemented AI method is a **neural imitation policy**:

1. Generate an analytical network scenario using the existing validated QoS code.
2. During offline dataset construction only, solve the scenario with the existing ILP scheduler. The selected candidate groups become expert labels.
3. Train a neural network to score each candidate group using PHY/QoS features and global candidate-competition context.
4. At inference, the neural network scores candidate groups. A constraint-aware greedy decoder ranks those learned scores and selects one group per link without station conflicts.

**The ILP is not called by the inference path.** `qos_ai_infer.py` calls `generate_scenario(..., include_expert=False)` and the neural-score decoder only.

This is supervised learning / imitation learning, not reinforcement learning. It is the first AI baseline. A contextual-bandit or PPO agent can be added later and compared using the same scenario generator and reward definition.

## Neural input features

The current feature vector contains:

- Link identity (6 GHz, 5 GHz, 2.4 GHz).
- Candidate station-membership mask.
- Per-station access-category identity.
- Per-station delivered-rate estimate, SINR, MCS, PER and normalized transmit power.
- Candidate weighted QoS utility, group size and aligned TXOP.
- Global candidate context: per-link reward statistics and per-STA/per-link candidate competition statistics.

The model does **not** receive the expert selected/not-selected labels as input. Labels are used only for the training loss.

## Important distinction: QoS preference versus guarantee

The reproduced objective uses the reference AC weights VO=1.85, VI=1.40, BE=1.00, BK=0.45. These weights encourage differentiated service but do not impose a hard minimum throughput or minimum scheduling count for each AC. Therefore, report per-AC throughput and selection counts; do not claim strict QoS guarantees unless explicit minimum-service constraints are added and evaluated.

## Files

- `src/qos_ai.py`: scenario generation, candidate features, neural model, training, held-out evaluation and checkpoint loading.
- `src/qos_ai_infer.py`: runs a trained neural scheduler on a new scenario without calling ILP.
- `tests/test_qos_ai.py`: tests feature encoding, model output and feasibility decoding.
- `src/qos_scheduler.py`: unchanged mathematical/optimization foundation used by the analytical experiments.
- `src/qos_aware_csr.py`: unchanged MATLAB analytical baseline.

## Fast development experiment

From the project root in Git Bash:

```bash
PYTHONPATH=src python src/qos_ai.py \
  --train-scenarios 50 \
  --val-scenarios 10 \
  --test-scenarios 15 \
  --sa-maxiter 1 \
  --sa-maxfun 15 \
  --epochs 100 \
  --output-dir results/qos_ai_context
```

The low simulated-annealing budget is only for fast AI pipeline development. It is not the full MATLAB reference optimization budget.

## Run inference without ILP

```bash
PYTHONPATH=src python src/qos_ai_infer.py \
  --checkpoint results/qos_ai_context/qos_imitation_model.pt \
  --seed 999001 \
  --sa-maxiter 1 \
  --sa-maxfun 15 \
  --output results/qos_ai_context/inference_demo.json
```

The JSON output includes `ilp_called_at_inference: false` and the selected groups. Candidate generation and PHY/QoS feature calculation still run; only the expert ILP selection is omitted.

## Development results (not final research results)

Run configuration:

- Training scenarios: 50
- Validation scenarios: 10
- Held-out test scenarios: 15
- Fixed MATLAB-style topology generated with topology seed 2
- Independently generated channel and AC realizations across scenarios
- Simulated annealing: 1 iteration / 15 function evaluations per optimization (reduced development budget)
- Reward mode: `matlab_as_written`

Observed held-out results:

| Metric | Neural imitation policy | ILP expert |
|---|---:|---:|
| Mean QoS-weighted utility (Mbps-equivalent) | 3753.97 | 3975.01 |
| Mean unweighted throughput (Mbps) | 2357.60 | 2637.87 |
| Mean utility as percentage of ILP | 94.27% | 100% |
| Mean inference time | 1.66 ms | Not directly comparable; ILP is an optimizer |

Additional observations:

- Exact full-schedule matches: 2/15.
- Constraint violations: 0/15.
- The learned policy reached 94.85% of the true-reward Greedy utility on average.
- The true-reward Greedy baseline reached approximately 99.42% of ILP utility on these test scenarios.
- The mean per-AC AI throughput was VO=1134.40, VI=1095.73, BE=116.27 and BK=11.20 Mbps. The low BK value is a warning that the soft-weight objective does not guarantee minimum service.

These numbers are a proof-of-pipeline only. The reduced power-optimization budget and 15 held-out scenarios are not sufficient for a research-performance claim. The learned policy does run without ILP during inference, but it does not yet outperform the true-reward Greedy baseline and does not exactly reproduce most expert schedules.

## Next experiments

1. Increase scenario diversity and use the reference power-optimization budget for final train/test data.
2. Compare neural imitation with contextual-bandit learning and PPO using the same split protocol.
3. Consider a graph/set model if generalization across topologies is required.
4. Report throughput, weighted utility, per-AC throughput, AC selection counts, optimality gap, feasibility and inference time.
5. Add hard per-AC minimum-service constraints only as a separate extension; preserve the original weighted objective as the faithful baseline.

## Second AI method: reward-driven contextual combinatorial LinUCB

A second learning path is implemented in `src/qos_contextual_bandit.py`. Unlike
the neural imitation model, this learner does not use ILP labels during
training. It uses a **LinUCB contextual combinatorial bandit with semi-bandit
feedback**:

1. Each candidate coordinated group is an arm with a context vector.
2. The context includes link identity, group size, AC composition, per-AC
   PER-adjusted PHY potential, SINR/MCS/PER summaries, aligned TXOP, CSR
   overhead, power, AP membership, STA membership, and candidate-conflict
   density.
3. The exact weighted candidate reward is not included in the feature vector.
4. The policy chooses one group per link while enforcing no STA reuse.
5. After selection, it observes the reward of each selected group only. Rewards
   for unselected candidate groups are not used to update the bandit.
6. LinUCB updates its linear reward estimate from this selected-arm feedback.

The three-link decoder exhaustively maximizes the **predicted** sum of group
scores over feasible combinations. It does not use the true reward in the
selection stage. Since there are only three links and tens of candidate groups
per link, this small combinatorial enumeration is tractable. It is not an ILP
call.

This is a contextual combinatorial bandit / semi-bandit formulation for a
static network snapshot. It is not long-horizon temporal network-control RL.
The training loop has zero ILP calls; ILP is used only as a held-out comparison
baseline, and the deployment-style inference entry point does not call ILP.

### Train and evaluate the contextual bandit

```bash
PYTHONPATH=src python src/qos_contextual_bandit.py \
  --train-scenarios 60 \
  --test-scenarios 20 \
  --seed 20261110 \
  --sa-maxiter 1 \
  --sa-maxfun 15 \
  --alpha 0.7 \
  --ridge 1.0 \
  --output-dir results/qos_contextual_bandit
```

### Run the trained contextual bandit without ILP

```bash
PYTHONPATH=src python src/qos_contextual_bandit_infer.py \
  --checkpoint results/qos_contextual_bandit/qos_contextual_bandit.npz \
  --seed 900777 \
  --sa-maxiter 1 \
  --sa-maxfun 15 \
  --output results/qos_contextual_bandit/inference.json
```

### Initial development experiment

Configuration: 60 reward-feedback training scenarios, 20 held-out test
scenarios, topology seed 2, scenario seed 20261110, SA budget 1 iteration / 15
function evaluations per candidate power optimization, LinUCB alpha 0.7 and
ridge 1.0. The network geometry is fixed to the MATLAB reference topology;
channel and access-category realizations vary by scenario.

| Held-out metric | LinUCB | QoS ILP | True-reward Greedy |
|---|---:|---:|---:|
| Mean QoS-weighted utility | 4090.27 Mbps | 4140.59 Mbps | 4093.54 Mbps |
| Mean unweighted throughput | 2857.80 Mbps | 2905.40 Mbps | 2880.40 Mbps |
| Mean utility relative to ILP | 98.73% | 100% | 98.87% |

Other observations:

- 180 selected-group reward observations were used for training (3 per scenario).
- ILP calls during bandit training: 0.
- Exact schedule matches with ILP: 9/20 (45%).
- Scheduling constraint violations: 0.
- Mean decision time: 3.78 ms, including LinUCB scoring and feasible-combination decoding.
- Mean AI throughput by AC (VO, VI, BE, BK): 1032.4, 1197.8, 401.8, 225.8 Mbps.
- Mean ILP throughput by AC (VO, VI, BE, BK): 1015.2, 1168.0, 549.6, 172.6 Mbps.
- Mean ratio of AI to ILP QoS utility: 98.73%, with a normal-approximation 95% CI half-width of 1.07 percentage points across 20 test scenarios.

These are promising development results, not final scientific claims: the
simulated-annealing budget is reduced, the held-out set has only 20 scenarios,
and only one fixed topology geometry is tested. Repeat with reference SA
budgets, larger independent test sets, multiple topology seeds, and sensitivity
analysis for LinUCB alpha/ridge before publication claims.

The semi-bandit feedback assumption is that selected-link/group rewards can be
measured separately after scheduling. If only the total combined reward is
observable, use a full-bandit feedback formulation instead.
