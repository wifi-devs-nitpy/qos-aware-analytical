
"""
######################################
Simulator for IEEE 802.11bn
"""

import time
import os
import h5py
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from numpy.random import SeedSequence
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import utils as utils
from constants import SYSTEM, MAC, CHANNEL
from mapc_sim import *
from traffic_generator import traffic_generator
from deployment_generator import deployment_generator
import rl_agent as rl_agent
from utils import plot_histogram
from custom_env import * 

global STA_matrix_save, channelMatrix_save, summary_data

def parse_args_from_slurm():
    """
    Parse arguments from command line
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='NN_tests_mypc')  # WandB project name
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=128)                 # default --- 2048                   
    parser.add_argument('--batch_size', type=int, default=256)              # default --- 64        
    parser.add_argument('--n_epochs', type=int, default=10)        # default --- 10
    parser.add_argument('--initial_lr', type=float, default=6.5E-4)       # default --- 3e-4              
    parser.add_argument('--learning_decay', type=str, default='cosine') # default --- 'cosine', 'linear', 'exp', 'square'
    parser.add_argument('--gamma', type=float, default=0.99)               # default --- 0.99
    parser.add_argument('--gae_lambda', type=float, default=0.95)          # default --- 0.95
    parser.add_argument('--clip_range', type=float, default=0.2)           # default --- 0.2
    parser.add_argument('--episode_threshold', type=int, default=0) 
    parser.add_argument('--w_throughput', type=float, default=4E-7)
    parser.add_argument('--w_long_term', type=float, default=1E-3)
    parser.add_argument('--w_shaping_coef', type=float, default=1.5)
    parser.add_argument('--window_size', type=int, default=10)


    args = parser.parse_args()

    return vars(args)

def simulate_iterations(traffic_config, sim_config, learning_config, mobility_config, seed, iter_number=None):
    """
    Simulates one iterations and returns the delay vectors for the different strategies.
    Parameters:
    traffic_config (dict): Configuration for traffic generation.
    sim_config (dict): Configuration for the simulation.
    learning_config (dict): Configuration for the learning agent.
    seed (int): Random seed for the simulation.
    iter_number (int): Iteration number for the deployment.

    Returns:
    deployment_key (str): Key for the deployment.
    deployment_summary (dict): Summary of the deployment.
    """ 

    print('-----------------------------------------')
    print('-----------------------------------------')

    # # Set the seed
    np.random.seed(seed)

    deployment_key = f"deployment{iter_number}"

    sim_config['ap_matrix'], sta_matrix, sim_config['association'], channel_matrix = deployment_generator(sim_config, seed)

    if sim_config['use_preloaded_deployments']:
        sta_matrix = sta_matrix_save[:, :, iter_number]
        channel_matrix= channel_matrix_save[:, :, iter_number]

    # utils.plot_deployment(sim_config['ap_matrix'], sta_matrix, sim_config['association'], sim_config['grid_value'], sim_config['walls'])
    
    if mobility_config:
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
            rng=np.random.default_rng(seed+5),
            )
        # utils.plot_mobility_trajectories(sta_mobility, sim_config['ap_matrix'], sim_config['association'], sim_config['walls'], sim_config['grid_value'])
    else:
        sta_mobility = None
    
    
    # Compute the CGs and TxPowerMatrix
    map_matrix, tx_power_matrix_temp, comb_ok = utils.cg_creation_tpc(
        sim_config['association'], 
        channel_matrix, 
        sim_config['max_tx_power_dbm'],
        sim_config['nsc'], 
        is_filtering=sim_config['filtering'], 
        tpc_method=sim_config['tpc_method'], # TPC Optimization method: None, 'PSO'
        cg_size=sim_config['cg_size']
    )  

    traffic_iterations = 1 # Number of traffic iterations per deployment
    seed_seq = SeedSequence(seeds[iter_number]) if len(seeds) > 1 else SeedSequence(seeds[0])
    traffic_seeds = seed_seq.generate_state(traffic_iterations)

    # Init the result
    deployment_summary = {}

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                run_traffic_iteration,
                traffic_iter,
                traffic_seeds[traffic_iter],
                traffic_config,
                sim_config,
                learning_config,
                mobility_config,
                sta_mobility,
                channel_matrix,
                map_matrix,
                tx_power_matrix_temp,
                comb_ok,
                sim,
                iter_number
            )
            for traffic_iter in range(traffic_iterations)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Deployment {iter_number}"):
            try:
                traffic_iter, traffic_summary = future.result()
                deployment_summary[f"traffic{traffic_iter}"] = traffic_summary
            except Exception as e:
                print(f"Error in traffic iteration: {e}")

    return deployment_key, deployment_summary

def run_traffic_iteration(
        traffic_iter, 
        traffic_seed, 
        traffic_config, 
        sim_config, 
        learning_config,
        mobility_config,
        sta_mobility,
        channel_matrix, 
        map_matrix, 
        tx_power_matrix_temp, 
        comb_ok, 
        sim, 
        iter_number
    ):
    np.random.seed(traffic_seed)
    if sim_config['use_preloaded_traffic']:
        h5_file_path = os.path.join(base_dir, 'traffic_datasets', campaign, sim, f"deployment{iter_number}", f"stas_arrivals_matrix{traffic_iter}.h5")
        with h5py.File(h5_file_path, 'r') as h5file:
            stas_arrivals_matrix = [h5file[key][:] for key in h5file.keys()]
            traffic_profile_per_sta = h5file.attrs['traffic_profile_per_sta']
    else:
        # traffic_profile_per_sta = np.random.choice(['C','D'], size=sim_config['sta_number']).tolist()
        traffic_profile_per_sta = [
                {
                    'traffic_load': np.random.uniform(traffic_config['load_min'], traffic_config['load_max']),  # Load in Mbps
                    'traffic_model': str(np.random.choice(['poisson', 'bursty']))  # Traffic model
                }
                for i in range(sim_config['sta_number'])
        ]

        stas_arrivals_matrix = traffic_generator(traffic_config, sim_config, traffic_profile_per_sta)

    delay_dict = {}

    # # Evaluate models
    models = []

    ml_results = evaluate_models(models, sim_config, traffic_config, learning_config, mobility_config, sta_mobility,
                                channel_matrix, map_matrix, tx_power_matrix_temp, comb_ok, stas_arrivals_matrix, iter_number)
    
    delay_dict.update(ml_results)

    # # # # EDCA
    # # np.random.seed(sim_config['seed'])
    # # sim_edca = MAPCsim(sim_config, mobility_config=mobility_config)
    # # sim_edca.sta_mobility = sta_mobility
    # # sim_edca.timestamp_to_stop = sim_config['timestamp_to_stop']
    # # sim_edca.channel_matrix = channel_matrix.copy()
    # # sim_edca.sta_queue_timeline = copy.deepcopy(stas_arrivals_matrix)
    # # sim_edca.simulation_system = 'edca'
    # # sim_edca.access_category = traffic_config['edca_access_category']
    # # sim_edca.init_settings()
    # # sim_edca.run()
    # # edca_delay = sim_edca.delay_vector


    # reducing the tx power matrix and cgs_stas according to comb_ok
    tx_power_matrix = [row.tolist() for i, row in enumerate(tx_power_matrix_temp) if comb_ok[i]]
    cgs_stas = [row.tolist() for i, row in enumerate(map_matrix) if comb_ok[i]]
    if len(tx_power_matrix) != len(cgs_stas):
        raise ValueError('Mismatch between TxPowerMatrix and CGs_STAs')
    
    # MNP
    np.random.seed(sim_config['seed'])
    sim_mnp = MAPCsim(sim_config, mobility_config=mobility_config)
    sim_mnp.sta_mobility = sta_mobility
    sim_mnp.timestamp_to_stop = sim_config['timestamp_to_stop']
    sim_mnp.channel_matrix = channel_matrix.copy()
    sim_mnp.sta_queue_timeline = copy.deepcopy(stas_arrivals_matrix)
    sim_mnp.simulation_system = 'csr'
    sim_mnp.scheduler = 'mnp'
    sim_mnp.cgs_stas = copy.deepcopy(cgs_stas)
    sim_mnp.tx_power_matrix = copy.deepcopy(tx_power_matrix)
    sim_mnp.access_category = traffic_config['edca_access_category']
    sim_mnp.init_settings()
    sim_mnp.run()
    mnp_delay = sim_mnp.delay_vector
    # plot_histogram(sim_mnp.priority_selection_counter / sim_mnp.suc_txops, name='MNP')

    # OP
    np.random.seed(sim_config['seed'])
    sim_op = MAPCsim(sim_config, mobility_config=mobility_config)
    sim_op.sta_mobility = sta_mobility
    sim_op.timestamp_to_stop = sim_config['timestamp_to_stop']
    sim_op.channel_matrix = channel_matrix.copy()
    sim_op.sta_queue_timeline = copy.deepcopy(stas_arrivals_matrix)
    sim_op.simulation_system = 'csr'
    sim_op.scheduler = 'op'
    sim_op.cgs_stas = copy.deepcopy(cgs_stas)
    sim_op.tx_power_matrix = copy.deepcopy(tx_power_matrix)
    sim_op.access_category = traffic_config['edca_access_category']
    sim_op.init_settings()
    sim_op.run()
    op_delay = sim_op.delay_vector
    # plot_histogram(sim_op.priority_selection_counter / sim_op.suc_txops, name='OP')

    # TAT
    np.random.seed(sim_config['seed'])
    sim_tat = MAPCsim(sim_config, mobility_config=mobility_config)
    sim_tat.sta_mobility = sta_mobility
    sim_tat.timestamp_to_stop = sim_config['timestamp_to_stop']
    sim_tat.channel_matrix = channel_matrix.copy()
    sim_tat.sta_queue_timeline = copy.deepcopy(stas_arrivals_matrix)
    sim_tat.simulation_system = 'csr'
    sim_tat.scheduler = 'tat'
    sim_tat.cgs_stas = copy.deepcopy(cgs_stas)
    sim_tat.tx_power_matrix = copy.deepcopy(tx_power_matrix)
    sim_tat.access_category = traffic_config['edca_access_category']
    sim_tat.alpha = 0.5
    sim_tat.beta = 0.5
    sim_tat.init_settings()
    sim_tat.run()
    tat_delay = sim_tat.delay_vector
    # plot_histogram(sim_tat.priority_selection_counter / sim_tat.suc_txops, name='TAT')

    ### Add the other strategies
    delay_dict.update({
        # 'edca': edca_delay,
        'mnp': mnp_delay,
        'op': op_delay,
        'tat': tat_delay,
    })

    print(f'Iteration: {traffic_iter}')
    print('--- 99th percentile delay results ---')
    percentiles = {strategy: np.percentile(delay, 99) * 1000 for strategy, delay in delay_dict.items()}
    for strategy, p99 in sorted(percentiles.items(), key=lambda x: x[1]):
        print(f'{strategy:10s}: {p99:.2f} ms')

    # print(f'Traffic Profile per STA: {traffic_profile_per_sta}')
    print('-----------------------------------------')

    # Save the delay metrics to an HDF5 file
    # save_to_h5(sim_config['output_dir'], sim, iter_number, traffic_iter, delay_dict)

    # Return traffic summary
    traffic_summary = {
        "traffic_profile": traffic_profile_per_sta,  # <--- compact
        "p99_delays_ms": {strategy: float(np.percentile(delay, 99) * 1000) for strategy, delay in delay_dict.items()}
    }

    return traffic_iter, traffic_summary 



def evaluate_models(models, sim_config, traffic_config, learning_config, mobility_config, sta_mobility,
                    channel_matrix, map_matrix, tx_power_matrix_temp, comb_ok, stas_arrivals_matrix, iter_number):
    delays = {}
    for model_id in models:
        np.random.seed(sim_config['seed'])  # Ensure fairness across runs
        
        model = {'model_id': model_id, 'model_type': 'best_model.zip'}
        name = 'ML-G'

        sim_ML = rl_agent.evaluation(
            traffic_config, 
            sim_config, 
            learning_config,
            mobility_config,
            sta_mobility,
            channel_matrix, 
            map_matrix, 
            tx_power_matrix_temp, 
            comb_ok,
            stas_arrivals_matrix, 
            model
        )
        sim_ML.simulator.traffic_analysis()
        delay_vector = sim_ML.simulator.delay_vector
        # plot_histogram(sim_ML.simulator.priority_selection_counter / sim_ML.simulator.suc_txops, name=name)

        
        delays[model_id] = delay_vector
    
    return delays

def save_to_h5(output_dir, sim, iter_number, traffic_iter, delay_dict):
    """
    Saves delay metrics into an HDF5 file named 'delay.h5' in a structured directory.

    Parameters:
    - output_dir: str, base directory to save the output.
    - sim: unused (kept for compatibility).
    - iter_number: int or str, current deployment iteration.
    - traffic_iter: int or str, current traffic iteration.
    - delay_dict: dict, mapping from delay metric names to their corresponding arrays.
                  Example: {'EDCAdelay': ..., 'ModelA': ..., 'ModelB': ..., ...}
    """
    # Create the directory structure
    output_path = os.path.join(output_dir, f"deployment{iter_number}", f"traffic{traffic_iter}")
    os.makedirs(output_path, exist_ok=True)

    # Define the HDF5 file path
    h5_file_path = os.path.join(output_path, "delay.h5")
    
    # Write all delay metrics to the HDF5 file
    with h5py.File(h5_file_path, 'a') as f:
        for name, data in delay_dict.items():
            if name in f:
                del f[name]  # Optional: overwrite specific dataset
            f.create_dataset(name, data=data)

def init_pool_processes(h5_path, use_preloaded):
    """Load HDF5 data only if required by the simulation mode"""
    global STA_matrix_save, channelMatrix_save
    if use_preloaded:
        with h5py.File(h5_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]

# Main function
if __name__ == "__main__":
    start_time = time.time()

    args = parse_args_from_slurm()

    sim = '30-16'
    campaign = 'general[10,90]'

    ap_number = 4
    sta_number = int(re.findall(r'\d+', sim)[1])
    grid_value = int(re.findall(r'\d+', sim)[0]) * 2
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

    # Simulation parameters for parallel processing
    iterations = 100

    # For reproducibility
    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(max(iterations, 100))

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
            # min_load = 50.0
            # max_load = 70.0
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
        'save_model': False,
        'use_preloaded_deployments': True,
        'use_preloaded_traffic': True,
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
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'frame_length': MAC.FRAME_LENGTH,
        'event_number': int(1E5), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(base_dir, 'results', campaign, sim),
        # 'output_dir': os.path.join(base_dir, 'results/mobility/0.1-5', sim),
        'overheads' : utils.overheads_calc(traffic_config['edca_access_category']),   
    }

    # Learning Configuration — base + overrides from CLI args
    learning_config = {
        'log_dir': os.path.join(base_dir, 'trained_models'),
        'parallel_envs': min(os.cpu_count(), 10),  # Number of parallel environments
        'total_timesteps': int(10E6),
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
        'w_throughput': args['w_throughput'],
        'w_long_term': args['w_long_term'],
        'w_shaping_coef': args['w_shaping_coef'],
        'window_size': args['window_size'],
    }

    mobility_config = None
    # mobility_config = {
    #     'ch_realization_duration': 0.1,  # seconds
    #     'ch_realizations_per_update': 5,  # number of channel realizations per position update
    #     'speed' : 1  # meters per second
    # }


    if sim_config['use_preloaded_deployments']:
        with h5py.File(h5file_deployments_path, 'r') as f:
            sta_matrix_save = f['sta_matrix_save'][:]
            channel_matrix_save = f['channel_matrix_save'][:]

    summary_data = {}

    for iter_number in range(iterations):
        deployment_key, deployment_summary = simulate_iterations(
            traffic_config, sim_config, learning_config, mobility_config=mobility_config, seed=seeds[iter_number], iter_number=iter_number
        )
        summary_data[deployment_key] = deployment_summary

    summary_path = os.path.join(sim_config['output_dir'], "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"Summary saved to {summary_path}")

    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")