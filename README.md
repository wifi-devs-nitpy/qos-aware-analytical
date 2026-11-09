# 11bn-py-Simulator

A Python-based simulator for IEEE 802.11bn candidate features with integrated Deep Reinforcement Learning (DRL) framework for intelligent scheduling optimization.

## Authors

- David Nunez, david.nunez@upf.edu

## Introduction

The 11bn-py-Simulator is a comprehensive simulation platform designed to model and evaluate IEEE 802.11bn wireless networks. It provides a flexible environment for developing and testing scheduling algorithms, with built-in support for Deep Reinforcement Learning (DRL) agents. The simulator enables researchers and developers to train Proximal Policy Optimization (PPO) agents that can learn to make optimal scheduling decisions in complex network scenarios. 

>Note that, the current PPO implementation uses MaskablePPO from https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

This project bridges the gap between network simulation and machine learning, allowing for the development of intelligent, data-driven scheduling solutions for next-generation wireless networks.

## Features

- **IEEE 802.11bn Simulation**: Comprehensive modeling of 802.11bn network features and behaviors
- **Traffic Generation**: Realistic traffic pattern generation for various network conditions
- **Deployment Scenarios**: Multiple deployment configurations with scalable network topologies
- **DRL Integration**: Built-in support for training reinforcement learning agents
- **PPO Agent**: Pre-configured Proximal Policy Optimization (MaskablePPO) agent for scheduling optimization
- **Custom Environment**: Gymnasium-compatible custom environment for RL agent training
- **Dataset Support**: Pre-generated traffic and deployment datasets for consistent evaluation
- **Training Monitoring**: Wandb integration for real-time training visualization
- **Model Evaluation**: Comprehensive evaluation metrics and analysis tools

## Overview

### Architecture

The simulator is organized into the following core components:

- **`simulator_main.py`**: Main entry point for running simulations
- **`custom_env.py`**: Custom Gymnasium (https://gymnasium.farama.org/api/env/) environment for DRL agent interaction
- **`rl_agent.py`**: RL agent implementation and training logic
- **`traffic_generator.py`**: Generates realistic network traffic patterns
- **`deployment_generator.py`**: Creates network deployment scenarios
- **`mapc_sim.py`**: MAPC simulator integration (main simulator class)
- **`constants.py`**: Project-wide constants and configuration parameters
- **`utils.py`**: Utility functions for logging, data handling, and processing

### Project Structure

```
├── code/                          # Source code modules
│   ├── constants.py              # Configuration constants
│   ├── custom_env.py             # Custom Gym environment
│   ├── deployment_generator.py   # Deployment generation
│   ├── mapc_sim.py               # MAPC simulator
│   ├── rl_agent.py               # DRL agent implementation
│   ├── simulator_main.py          # Main simulator
│   ├── traffic_generator.py       # Traffic pattern generation
│   └── utils.py                  # Utility functions
├── deployments_datasets/          # Pre-generated deployment configurations
├── traffic_datasets/              # Pre-generated traffic patterns
├── trained_models/                # Trained models and evaluation results
│   ├── models/                   # Saved model checkpoints
│   ├── evaluations.npz           # Evaluation metrics
│   ├── monitor.csv               # Training monitor logs
│   └── tensorboard/              # TensorBoard logs
├── results/                      # To store the results
└── README.md                      # This file
```

## Usage

### Prerequisites

- Python 3.7+
- Required dependencies (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 11bn-py-Simulator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulator

#### Basic Simulation

To run a basic simulation:

```bash
python code/simulator_main.py
```

For reproducibility, both the simulator and the learning agent support configurable seeds. For simulation, deployments (i.e., device locations) and traffic patterns can either be generated dynamically during execution or pre-generated and stored as datasets for reuse in subsequent simulations by configuring the following parameters:
- `sim_config['use_preloaded_deployments']: True`
- `sim_config['use_preloaded_traffic']: True`

#### Training a DRL Agent

To train a PPO agent for scheduling optimization:

```bash
python code/rl_agent.py
```

The training process will:
- Initialize the custom environment with deployment and traffic datasets
- Train the PPO agent using the specified hyperparameters
- Log training metrics to Wandb (if configured)
- Save model checkpoints to `trained_models/models/`


The training support parallel environments through `SubprocVecEnv`.

##### Wandb Integration

The training process can be monitored via wandb API from https://wandb.ai. For that purpose, in rl_agent.py set:
- `learning_config['wandb_log']: True`

The following two files have been customized in order to show /custom statistics in Wandb:
- Change `stable_baselines3.common.monitor:102` by:

            ep_info = {
                "r": round(ep_rew, 6), 
                "l": ep_len, 
                "t": round(time.time() - self.t_start, 6),
                "total_percentile99": round(info['total_percentile99'],6),
                "worst_percentile99": round(info['worst_percentile99'],6),
                "mean_rew_shaping": round(info['mean_rew_shaping'],6),
                "mean_long_term_rew": round(info['mean_long_term_rew'],6),
                "mean_reward": round(info['mean_reward'],6),
                } 
- In `stable_baselines3.common.on_policy_algorithm:293` add:
            
            self.logger.record("custom/ep_total_percentile99", safe_mean([ep_info["total_percentile99"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/ep_worst_percentile99", safe_mean([ep_info["worst_percentile99"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/step_mean_rew_shaping", safe_mean([ep_info["mean_rew_shaping"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/step_mean_long_term_rew", safe_mean([ep_info["mean_long_term_rew"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/step_mean_reward", safe_mean([ep_info["mean_reward"] for ep_info in self.ep_info_buffer])) 



##### Saving models
For saving the model, two methods are available:
1. `learning_config['save_best_model']: True` -> Save the best model through EvalCallback
2. `learning_config['checkpoint_log']: True` -> Save the model periodically

#### Evaluation

The model is evaluated from `rl_agent.evaluation()` method, which is also called from `simulator_main.py`
Don't forget to add the proper name of the model to `model` list in `simulator_main.py`.    

### Configuration

The constants groups, SYSTEM, MAC, CHANNEL are stored in `code/constants.py`.

Apart from that, some settings can be set in:
- traffic_config
- sim_config
- learning_config

## Contribute

Contributions are welcome! But first, send an email to david.nunez@upf.edu to discuss in advance the scope/goal of the contribution

After that:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows the project's style guidelines
- Changes include appropriate documentation
- Tests are added for new functionality

## Acknowledgements

This work is supported by the following projects:
MLDR (Chist-ERA ANR-23-CHR4-0005) PCI2023-145958-2 MCIU/AEI/10.13039/
TRUE Wi-Fi PID2024-155470NB-I00 (MICIU/AEI/10,13039/501100011033/FEDER, UE)
Wi-XR Wi-XR PID2021-123995NB-I00 (MCIU/AEI/FEDER,UE)
AGAUR ICREA Academia 00077
Maria de Maeztu Units of Excellence Programme (CEX2021-001195-M)

---

## AI-based QoS-aware CSR scheduling

The analytical QoS baseline can be used to train a neural imitation scheduler that learns from ILP expert labels and makes new scheduling decisions without invoking ILP at inference. See [`AI_QOS_SCHEDULING.md`](AI_QOS_SCHEDULING.md) for the formulation, fast training command, inference command, limitations and development results.

Quick start (Git Bash, from the project root):

```bash
PYTHONPATH=src python src/qos_ai.py --train-scenarios 50 --val-scenarios 10 --test-scenarios 15 --sa-maxiter 1 --sa-maxfun 15 --epochs 100 --output-dir results/qos_ai
```

This reduced-budget command is for pipeline development only, not final research evaluation.

## QoS-aware multi-link CSR AI research experiment

The repository also contains an analytical QoS-aware multi-link CSR experiment
ported from the supplied MATLAB reference. It is intentionally separate from
`MAPCsim` and focuses on selecting one coordinated group per frequency link,
without reusing a station across links.

The final result comparison contains exactly three methods:

1. Conventional CSR using ILP (unweighted throughput objective, per-STA TXOP).
2. QoS-aware CSR using ILP (AC-weighted objective, aligned group TXOP).
3. QoS-aware CSR using LinUCB (reward-driven contextual combinatorial bandit).

Greedy is not included in the final comparison.

### Final experiment command

From the project root (Git Bash):

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

The 1,000 training scenarios provide 3,000 selected-group semi-bandit feedback
updates. No ILP calls are made during LinUCB training or inference. All three
methods are evaluated on exactly the same 100 held-out scenarios.

- `src/qos_aware_csr.py`: MATLAB-aligned analytical CSR reference and configuration.
- `src/qos_scheduler.py`: candidate-group generation, PHY calculations, SA power allocation, rewards and ILP constraints.
- `src/qos_ai.py`: neural ILP-imitation scheduler (separate AI baseline).
- `src/qos_contextual_bandit.py`: final three-method experiment runner, LinUCB training, evaluation and MATLAB-style reporting.
- `src/qos_contextual_bandit_infer.py`: standalone LinUCB inference with no ILP call.
- `QOS_CONTEXTUAL_BANDIT.md`: formulation and scientific limitations.
- `FINAL_RESULTS_GUIDE.md`: final run command, expected accounting and report outputs.

The output includes MATLAB-style throughput, per-AC throughput, AC allocation,
execution-time and topology figures; per-scenario results; summary tables; and
the trained LinUCB checkpoint. See `FINAL_RESULTS_GUIDE.md` for details.
