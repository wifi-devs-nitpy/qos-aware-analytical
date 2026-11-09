import os
import h5py
import numpy as np
from numpy.random import SeedSequence
import time
import re
import math
from typing import Callable
import torch as th
from torch import nn
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.env_checker import check_env 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv #, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from sb3_contrib import MaskablePPO
# For logging and monitoring
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import uuid

from constants import SYSTEM, MAC, CHANNEL
import utils as utils
from mapc_sim import *
from custom_env import * 
from traffic_generator import traffic_generator
from deployment_generator import deployment_generator
 

def parse_args_from_slurm():
    """
    Parse arguments from command line
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='new_project_cleaning_11bn-py-Simulator')  # WandB project name
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=128)                 # default --- 2048                   
    parser.add_argument('--batch_size', type=int, default=64)              # default --- 64        
    parser.add_argument('--n_epochs', type=int, default=10)        # default --- 10
    parser.add_argument('--initial_lr', type=float, default=4E-4)       # default --- 3e-4   || 4E-4             
    parser.add_argument('--learning_decay', type=str, default='cosine') # default --- 'cosine', 'linear', 'exp', 'square'
    parser.add_argument('--gamma', type=float, default=0.99)               # default --- 0.99
    parser.add_argument('--gae_lambda', type=float, default=0.95)          # default --- 0.95
    parser.add_argument('--clip_range', type=float, default=0.2)           # default --- 0.2
    parser.add_argument('--episode_threshold', type=int, default=0) 
    parser.add_argument('--w_long_term', type=float, default=1E-2)
    parser.add_argument('--window_size', type=int, default=100)

    args = parser.parse_args()

    return vars(args)

def schedule_clip_range(
    clip_min_phase1: float = 0.1,
    clip_max: float        = 0.4,
    clip_min_phase2: float = 0.1,
    switch_point: float    = 0.5,  # when to switch from phase1 → phase2
) -> Callable[[float], float]:
    """
    Returns a schedule function that:
      - Increases clip_range from clip_min_phase1 → clip_max as 
        progress_remaining goes from 1.0 → switch_point
      - Decreases clip_range from clip_max → clip_min_phase2 as
        progress_remaining goes from switch_point → 0.0
    """
    def schedule(progress_remaining: float) -> float:
        # Phase 1: [0.0, switch_point]  →  clip_min_phase1 → clip_max
        if progress_remaining > switch_point:
            frac = (1.0 - progress_remaining) / (1.0 - switch_point)
            return clip_min_phase1 + (clip_max - clip_min_phase1) * frac
        # Phase 2: [switch_point, 1.0] → clip_max → clip_min_phase2
        else:
            frac = progress_remaining / switch_point
            return clip_min_phase2 + (clip_max - clip_min_phase2) * frac
    return schedule

def schedule_learning_rate(initial_lr, learning_decay=None):
    def schedule(progress_remaining):
        if learning_decay is None:
            return initial_lr
        elif learning_decay == 'linear':
            return initial_lr * progress_remaining
        elif learning_decay == 'cosine':
            return initial_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        elif learning_decay == 'square':
            return initial_lr * (progress_remaining)**2
        elif learning_decay == 'exp':
            return initial_lr * np.exp(-1.38 * (1-progress_remaining))     # k=1.38 ---> log(1/2)*2  to achieve 50% decay at 50% of the training
        else:
            raise ValueError(f"Unknown decay type: {learning_decay}")
    return schedule

def create_env(traffic_config, sim_config, learning_config,  mobility_config=None, sta_mobility=None, seed=None, channel_matrix=None, map_matrix=None, tx_power_matrix_temp=None, comb_ok=None, monitor_gym=False, training_flag=True):
    """
    Create a custom environment for RL training or evaluation
    """

    if channel_matrix is None:
        # Deployment-dependent data
        sim_config['ap_matrix'], sta_matrix, sim_config['association'], channel_matrix = deployment_generator(sim_config, seed)

        # utils.plot_deployment(ap_matrix, sta_matrix, sim_config['association'], sim_config['grid_value'], sim_config['walls'])

        # Compute the CGs and TxPowerMatrix
        map_matrix, tx_power_matrix_temp, comb_ok = utils.cg_creation_tpc(
            sim_config['association'],
            channel_matrix,
            sim_config['max_tx_power_dbm'],
            sim_config['nsc'],
            is_filtering=sim_config['filtering'], 
            tpc_method=sim_config['tpc_method'], # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
            cg_size=sim_config['cg_size']
        )
    if mobility_config and (sta_mobility is None):
        sta_mobility = utils.generate_sta_mobility(
            sim_config['ap_matrix'],
            sta_matrix,
            sim_config['walls'],
            sim_config['grid_value'],
            sim_config['association'],
            sim_config['timestamp_to_stop'],
            mobility_config['ch_realization_duration'],
            mobility_config['speed'],
            min_dist_to_ap=1.0,
            max_attempts=30,
            rng=np.random.default_rng(seed),
            )
        # utils.plot_mobility_trajectories(sta_mobility, sim_config['ap_matrix'], sim_config['association'], sim_config['walls'], sim_config['grid_value'])


    # Create copy of config to avoid mutation issues
    env_traffic_config = copy.deepcopy(traffic_config)

    env_sim_config = copy.deepcopy(sim_config)
    env_sim_config["training_flag"] = training_flag  # Set flag per environment

    env_learning_config = copy.deepcopy(learning_config)

    if not training_flag:
        env_sim_config['timestamp_to_stop'] = 5  # default evaluation duration
    
    # Creating the simulator instance  

    simulator = MAPCsim(env_sim_config, mobility_config=mobility_config)  # new "MAPC simulator" object
    simulator.sta_mobility = sta_mobility
    simulator.simulation_system = 'rl'                 # 'EDCA' or 'CSR'
    simulator.channel_matrix = channel_matrix.copy()  # Channel matrix
    simulator.cgs_stas = copy.deepcopy(map_matrix)         # Entire groups matrix (all posible combinations)
    simulator.tx_power_matrix = copy.deepcopy(tx_power_matrix_temp)  # Entire Tx power matrix (all posible combinations)
    simulator.comb_ok = copy.deepcopy(comb_ok) # Combinations ok 
    simulator.access_category = traffic_config['edca_access_category']  # Access category of devices in the network
    simulator.timestamp_to_stop = sim_config['timestamp_to_stop'] # training episode duration

    # Creating the custom environment
    env = CustomEnv(env_traffic_config, env_sim_config, env_learning_config, simulator)  
    # check_env(env)  # Check the environment
    if monitor_gym:
        env = Monitor(env, env_learning_config['log_dir'], info_keywords=('total_percentile99', 'worst_percentile99','mean_rew_shaping', 'mean_long_term_rew','mean_reward'))  # Wrap the environment
    return env

def training(traffic_config, sim_config, learning_config, mobility_config=None):
    """
    Train the RL agent using the specified configuration
    """

    # Set the seed
    seed = sim_config['seed']
    np.random.seed(seed)

    # Create log dir
    os.makedirs(learning_config['log_dir'], exist_ok=True)

    # Initialize W&B logging
    if learning_config['wandb_log']:
        # Start W&B run 
        logging_run = wandb.init(project=learning_config['project_name'], 
                        id=learning_config['run_id'], 
                        sync_tensorboard=True,
                        monitor_gym=False,
                        save_code=True,
                        )
        run_id = logging_run.id
        wandb.config.update({
            "n_steps": learning_config['n_steps'],
            "batch_size": learning_config['batch_size'],
            "n_epochs": learning_config['n_epochs'],
            "initial_lr": learning_config['initial_lr'],
            "learning_decay": learning_config['learning_decay'],
            "gamma": learning_config['gamma'],
            "gae_lambda": learning_config['gae_lambda'],
            "clip_range": learning_config['clip_range'],
            "episode_threshold": learning_config['episode_threshold'],
            "w_long_term": learning_config['w_long_term'],
            "window_size": learning_config['window_size'],
        })
    
        ### Add WandbCallback
        logging_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=os.path.join(learning_config['log_dir'], "models", run_id),
            model_save_freq=100,
            verbose=2,
        )
    else:
        run_id = uuid.uuid4().hex[:8]
        print(f"W&B logging is disabled. Using run_id: {run_id}")


    # # Create evaluation environments with training_flag=False
    eval_env = None
    if learning_config['save_best_model']:

        total_envs = learning_config['parallel_envs']
        eval_envs = learning_config['eval_envs']
        train_envs = total_envs - eval_envs

        assert train_envs > 0, "Not enough environments for training"

        eval_env = make_vec_env(
            lambda: create_env(
                traffic_config, 
                sim_config, 
                learning_config, 
                mobility_config=mobility_config,
                seed=np.random.randint(1, int(1E8)), 
                monitor_gym=True,
                training_flag=False
            ),
            n_envs=eval_envs,
            vec_env_cls=SubprocVecEnv
        )
    else:
        train_envs = learning_config['parallel_envs']

    # Create training environments with training_flag=True
    train_env = make_vec_env(
        lambda: create_env(
            traffic_config, 
            sim_config, 
            learning_config, 
            mobility_config=mobility_config,
            seed=np.random.randint(1, int(1E8)), 
            monitor_gym=True,
            training_flag=True  # Explicit training flag
        ),
        n_envs=train_envs,    # for training
        vec_env_cls=SubprocVecEnv
    ) 


    learning_rate_scheduled = schedule_learning_rate(
        initial_lr=learning_config['initial_lr'],
        learning_decay=learning_config['learning_decay']
    )

    clip_range_scheduled = schedule_clip_range(clip_min_phase1=0.1,
                                                clip_max=0.3,
                                                clip_min_phase2=0.2,
                                                switch_point=0.5
                                            )
    
    # Paths
    model_dir = os.path.join(learning_config['log_dir'], "models", run_id)
    tensorboard_dir = os.path.join(learning_config['log_dir'], "tensorboard", run_id)

    # Uncomment the following lines to load a pre-trained model
    # model_name = 'n505nvdg'       
    # model = PPO.load(os.path.join(learning_config['log_dir'], "models", model_name, "model_5401600_steps.zip"), env=train_env)

    # Set device to GPU if available, else CPU
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Initialize PPO model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log=tensorboard_dir,
        seed=sim_config['seed'],
        n_steps=learning_config['n_steps'],
        batch_size=learning_config['batch_size'],
        n_epochs=learning_config['n_epochs'],
        learning_rate=learning_rate_scheduled,
        clip_range=learning_config['clip_range'],
        gamma=learning_config['gamma'],
        gae_lambda=learning_config['gae_lambda'],
        device=device,
    )

    # Callback container
    callbacks = []

    # Early stopping callback
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=30,
        min_evals=20,
        verbose=1
    )

    # Evaluation callback
    if learning_config['save_best_model']:

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=learning_config['log_dir'],
            eval_freq=5120,
            deterministic=False,
            render=False,
            callback_after_eval=stop_train_callback,
            n_eval_episodes=20,
        )

        callbacks.append(eval_callback)

    # Checkpoint callback
    if learning_config['checkpoint_log']:

        checkpoint_callback = CheckpointCallback(
            save_freq=2560,
            save_path=model_dir,
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=2,
        )

        callbacks.append(checkpoint_callback)

    # W&B callback
    if learning_config['wandb_log']:
        callbacks.append(logging_callback)

    # Convert to SB3 CallbackList if needed
    callbacklist = CallbackList(callbacks) if callbacks else None

    # Train
    model.learn(
        total_timesteps=learning_config['total_timesteps'],
        callback=callbacklist,
        reset_num_timesteps=False
    )

    # Save the model after training
    model.save(os.path.join(learning_config['log_dir'], "models", run_id, "final_model.zip"))

    if learning_config['wandb_log']:
        logging_run.finish()

def evaluation(traffic_config, sim_config, learning_config,  mobility_config, sta_mobility, channel_matrix, map_matrix, tx_power_matrix_temp, comb_ok, stas_arrivals_matrix, model):
    """
    Evaluate the trained model using the specified configuration
    """
    # Create the environment with training_flag=False
    env = create_env(
        traffic_config, 
        sim_config, 
        learning_config, 
        mobility_config,
        sta_mobility,
        seed=sim_config['seed'],
        channel_matrix=channel_matrix.copy(), 
        map_matrix=copy.deepcopy(map_matrix), 
        tx_power_matrix_temp=copy.deepcopy(tx_power_matrix_temp), 
        comb_ok=copy.deepcopy(comb_ok), 
        monitor_gym=False, 
        training_flag=False
        )
    obs, _ = env.reset(
        seed=sim_config['seed'], 
        stas_arrivals_matrix=copy.deepcopy(stas_arrivals_matrix), 
        is_deployment_fixed=True
        )  # Reset the environment

    terminated = False

    # Load the trained model
    loaded_model = MaskablePPO.load(os.path.join(learning_config['log_dir'], "models", model['model_id'],  model['model_type']), env=env)
    # loaded_model = TRPO.load(os.path.join(learning_config['log_dir'], "models", model['model_id'],  model['model_type']), env=env)
    while not terminated:
        action_masks = env.action_masks()
        action, _states = loaded_model.predict(obs, action_masks=action_masks, deterministic=True)
        # action, _states = loaded_model.predict(obs,  deterministic=False)
        obs, _, terminated, _, _ = env.step(action)

    return env


if __name__ == '__main__':

    # Start Timer
    start_time = time.time()

    args = parse_args_from_slurm()

    sim = '30-16'  # Simulation name: 'APtoAPdistance-STA_NUMBER'
    numbers = re.findall(r'\d+', sim) # Extract numbers from the simulation name

    # Scenario-related
    ap_number = 4
    sta_number = int(numbers[1]) 
    grid_value = int(numbers[0]) * 2
    scenario_type = 'grid'

    walls = np.array([
        [0, grid_value, grid_value/2, grid_value/2],
        [grid_value/2, grid_value/2, 0, grid_value]
    ])

    ### Channel-related parameters
    max_tx_power_dbm, nsc = utils.tx_power_calc() # default bw=80 MHz, nss=2 spatial streams 

    # Deployment data path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up one level
    h5file_deployments_path = os.path.join(base_dir, 'deployments_datasets', sim, 'deployment_datasets.h5')

    match sta_number:
        case 8:
            min_load = 43.42
            max_load = 156.58
        case 12:
            min_load = 20.47
            max_load = 112.87
        case 16:
            min_load = 10.0
            max_load = 90.0
        case 20:
            min_load = 4.21
            max_load = 75.79 


    # Traffic Configuration 
    traffic_config = {
        'load_min': min_load,  # Minimum load in Mbps
        'load_max': max_load,  # Maximum load in Mbps
        'edca_access_category': 'BE'
    }

    # Simulation Configuration
    sim_config = {
        'filtering': True,
        'use_preloaded_deployments': False,
        'use_preloaded_traffic': False,
        'ap_number': ap_number,
        'sta_number': sta_number,
        'scenario_type': scenario_type,
        'grid_value': grid_value,
        'walls': walls,
        'max_tx_power_dbm': max_tx_power_dbm,
        'tpc_method': None,  # TPC Optimization method: None, 'PSO'
        'cg_size': 2,
        'txop_duration': SYSTEM.TXOP_DURATION,
        'pn_dbm': SYSTEM.PN_DBM,
        'cca': SYSTEM.CCA,
        'nss': SYSTEM.NSS,
        'nsc': nsc,
        'training_flag': True,
        'timestamp_to_stop': 5, # typically [1,5] seconds, set equal to 5 for better generalization, but it will increase the training time
        'frame_length': MAC.FRAME_LENGTH,
        'event_number': int(1E5), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(base_dir, 'results', sim),
        'overheads' : utils.overheads_calc(traffic_config['edca_access_category']),   
    }

    # Learning Configuration — base + overrides from CLI args
    learning_config = {
        'log_dir': os.path.join(base_dir, 'trained_models'),
        'wandb_log': True,  # Set to True to enable W&B logging
        'save_best_model': True,  # Set to True to enable best-model saving via EvalCallback
        'checkpoint_log': False,  # Set to True to enable checkpoint logging, saves model periodically
        'parallel_envs': min(os.cpu_count(), 10),  # Number of parallel environments
        'eval_envs': 4,
        'total_timesteps': int(5E6),
        'simulator_attr': 'simulator',
        'project_name': args['project_name'],
        'run_id': args['run_id'],
        'n_steps': args['n_steps'],
        'batch_size': args['batch_size'],
        'n_epochs': args['n_epochs'],
        'initial_lr': args['initial_lr'],
        'learning_decay': args['learning_decay'],
        'gamma': args['gamma'],
        'gae_lambda': args['gae_lambda'],
        'clip_range': args['clip_range'],  
        'episode_threshold': args['episode_threshold'],
        'w_long_term': args['w_long_term'],
        'window_size': args['window_size'],
    }

    mobility_config = None
    ### Uncomment the following lines to enable mobility configuration
    # mobility_config = {
    #     'ch_realization_duration': 0.1,  # seconds
    #     'ch_realizations_per_update': 10,
    #     'speed' : 1  # meters per second
    # }
    
    try:
        # Simulate the iterations
        training(traffic_config, sim_config, learning_config, mobility_config)
    finally:
        if wandb.run:
            wandb.finish()



