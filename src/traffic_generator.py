import time
import os
import h5py
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from numpy.random import SeedSequence
import utils as utils
from constants import SYSTEM, MAC
from deployment_generator import deployment_generator

global sta_matrix_save, channel_matrix_save

def traffic_generator(traffic_config, sim_config, traffic_profile_per_sta):
    """
    Generates a list of arrival times for each STA based on the specified traffic model.
    """
    # Initialize the stas_arrivals_matrix
    stas_arrivals_matrix = []

    for sta in range(sim_config['sta_number']):
        
        # Loading STA profile
        traffic_profile = traffic_profile_per_sta[sta]

        # Loading the traffic model
        traffic_model = traffic_profile['traffic_model']

        # Loading the traffic load
        traffic_load = traffic_profile['traffic_load']

        traffic_generation_rate = traffic_load  * 1E6 / sim_config['frame_length']  # in packets/sec
        match traffic_model:
            case 'poisson': # Poisson traffic model
                arrivals = poisson_fixed_events(sim_config['event_number'], traffic_generation_rate)
            case 'bursty': # Bursty traffic model
                arrivals = generate_burst_traffic(sim_config['event_number'], traffic_generation_rate)
            case 'cbr': # CBR traffic model
                fps = traffic_config['traffic_profiles'][traffic_profile_per_sta[sta]]['fps']
                arrivals = generate_cbr_traffic(sim_config['frame_length'], traffic_load, fps)
            case _:
                raise ValueError("Invalid traffic type specified.")

        stas_arrivals_matrix.append(arrivals)

    return stas_arrivals_matrix

def poisson_fixed_events(event_number, traffic_generation_rate):
    """
    Generate arrival times using a Poisson process for each STA.
    """

    # Generate exponential inter-arrival times
    w = np.random.exponential(scale=1/traffic_generation_rate, size=event_number)
    arrivals = np.cumsum(w)

    # import matplotlib.pyplot as plt
    # # # Plot the exponential inter-arrival times histogram
    # plt.figure()
    # plt.hist(w, bins=50, density=True, alpha=0.75)
    # x = np.linspace(0, np.max(w), 100)
    # plt.plot(x, traffic_generation_rate * np.exp(-traffic_generation_rate * x), 'r-', lw=2)
    # plt.title('Histogram of Inter-Arrival Times with Exponential PDF')
    # plt.xlabel('Inter-Arrival Time')
    # plt.ylabel('Probability Density')
    # plt.show()

    return arrivals

def generate_burst_traffic(event_number, traffic_generation_rate):
    """
    Generate bursty traffic arrivals for each STA.
    """

    # Average ON and OFF times
    average_on_time = 1E-3
    average_off_time = 10E-3

    # Expected proportion of time spent in the ON state
    on_off_ratio = average_on_time / (average_on_time + average_off_time)

    # Adjusted generation rate during ON periods
    adjusted_generation_rate = traffic_generation_rate / on_off_ratio


    arrivals = np.zeros(event_number)  # Preallocate space for arrival times
    current_time = 0  # Start at time 0
    total_packets_generated = 0  # Track the total number of packets generated

    while total_packets_generated < event_number:
        # ON period: Generate packets based on adjusted_generation_rate
        on_period_duration = np.random.exponential(average_on_time)
        packets_in_burst = int(on_period_duration * adjusted_generation_rate)

        for _ in range(packets_in_burst):
            if total_packets_generated >= event_number:
                break
            inter_arrival_time = np.random.exponential(1 / adjusted_generation_rate)
            current_time += inter_arrival_time
            arrivals[total_packets_generated] = current_time
            total_packets_generated += 1

        # OFF period: No packets generated
        off_period_duration = np.random.exponential(average_off_time)
        current_time += off_period_duration



    # # # Check the actual rate of packets generated
    # total_time = arrivals[-1] - arrivals[0]
    # actual_rate = len(arrivals) / total_time
    # print(f"Actual rate = {actual_rate:.2f} packets/sec and expected rate = {traffic_generation_rate:.2f} packets/sec")


    ##### Verifying the code:
    # # Check if the arrival times are in increasing order
    # assert np.all(np.diff(arrivals) > 0), f"Non-monotonic times found in STA"


    return arrivals

def generate_cbr_traffic(frame_length, traffic_load, fps):
    """
    Generate CBR traffic arrivals for each STA.
    """
    bitrate = traffic_load
    frame_interval = 1 / fps
    frames_per_burst = int(np.ceil((bitrate * 1E6 * frame_interval) / frame_length))
    frame_spacing = 5E-6
    stop_timestamp = 20

    current_time = np.random.uniform(0, frame_interval)
    arrivals = []
    while current_time < stop_timestamp:
        burst_times = current_time + np.arange(frames_per_burst) * frame_spacing
        arrivals.extend(burst_times)
        current_time += frame_interval

    return np.array(arrivals)

def save_to_h5(sim_config, traffic_config, traffic_profile_per_sta, stas_arrivals_matrix, deployment_iteration, traffic_iteration):
    """
    Saves the stas_arrivals_matrix into individual HDF5 files in a structured directory.
    """
    # Create the directory structure
    output_path = os.path.join(sim_config['output_dir'], f"deployment{deployment_iteration}")
    os.makedirs(output_path, exist_ok=True)

    # Save the current iteration to its own HDF5 file
    h5_file_path = os.path.join(output_path, f"stas_arrivals_matrix{traffic_iteration}.h5")
    with h5py.File(h5_file_path, 'w') as h5file:
        for i, arrivals in enumerate(stas_arrivals_matrix):
            h5file.create_dataset(f"sta_{i}", data=arrivals) # no compression
            # h5file.create_dataset(f"sta_{i}", data=arrivals, compression="gzip")   # with compression

        # Store traffic profile per STA as h5 attribute
        h5file.attrs['traffic_profile_per_sta'] = str(traffic_profile_per_sta)

def simulate_iterations(traffic_config, sim_config, seed, iter_number=None):
    """
    Simulates one iterations and returns the delay vectors for the different strategies.
    Parameters:
    traffic_config (dict): Configuration for traffic generation.
    sim_config (dict): Configuration for the simulation.
    seed (int): Random seed for the simulation.
    iter_number (int): Iteration number for the deployment.

    """

    # # Set the seed
    np.random.seed(seed)

    _, sta_matrix, _, channel_matrix = deployment_generator(sim_config, seed)

    if sim_config['use_preloaded_deployments']:
        sta_matrix = sta_matrix_save[:, :, iter_number]
        channel_matrix = channel_matrix_save[:, :, iter_number]

    traffic_iterations = 1
    seed_seq = SeedSequence(seeds[iter_number]) if len(seeds) > 1 else SeedSequence(seeds[0])
    traffic_seeds = seed_seq.generate_state(traffic_iterations)

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                run_traffic_iteration,
                traffic_iter,
                traffic_seeds[traffic_iter],
                traffic_config,
                sim_config,
                sim,
                iter_number
            )
            for traffic_iter in range(traffic_iterations)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"deployment {iter_number}"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in traffic iteration: {e}")

    return

def run_traffic_iteration(
        traffic_iter, 
        traffic_seed, 
        traffic_config, 
        sim_config, 
        sim, 
        iter_number
    ):
    np.random.seed(traffic_seed)
    # traffic_config['traffic_profile_per_sta'] = np.random.choice(['A','B'], size=sim_config['sta_number']).tolist()

    traffic_profile_per_sta = [
            {
                'traffic_load': np.random.uniform(traffic_config['load_min'], traffic_config['load_max']),  # Load in Mbps
                'traffic_model': str(np.random.choice(['poisson', 'bursty']))  # Traffic model
            }
            for i in range(sim_config['sta_number'])
    ]
    stas_arrivals_matrix = traffic_generator(traffic_config, sim_config, traffic_profile_per_sta)

    ### Uncoment to save the delay vectors into HDF5 files for each iterations, traffic type, and traffic load in a structured directory
    save_to_h5(sim_config, traffic_config, traffic_profile_per_sta, stas_arrivals_matrix, iter_number, traffic_iter)


# main function
if __name__ == "__main__":
    # Start Timer
    start_time = time.time()

    sim = '30-16'  # Simulation name: 'APtoAPdistance-STA_NUMBER'
    campaign = 'general[10,90]'

    # Scenario-related
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

    # Number of iterations
    iterations = 100
    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(iterations)

    # Traffic Configuration 
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
        'use_preloaded_deployments': True,
        'ap_number': ap_number,
        'sta_number': sta_number,
        'scenario_type': scenario_type,
        'grid_value': grid_value,
        'walls': walls,
        'max_tx_power_dbm': max_tx_power_dbm,
        'tpc_method': None,  # TPC Optimization method: None, 'PSO'
        'cg_size': ap_number,
        'txop_duration': SYSTEM.TXOP_DURATION,
        'pn_dbm': SYSTEM.PN_DBM,
        'cca': SYSTEM.CCA,  
        'nss': SYSTEM.NSS,
        'nsc': nsc,
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'frame_length': MAC.FRAME_LENGTH,
        'event_number': int(1E6), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(base_dir, 'traffic_datasets', campaign, sim),
        'overheads' : utils.overheads_calc(traffic_config['edca_access_category'])
    }

    if sim_config['use_preloaded_deployments']:
        with h5py.File(h5file_deployments_path, 'r') as f:
            sta_matrix_save = f['sta_matrix_save'][:]
            channel_matrix_save = f['channel_matrix_save'][:]

    for iter_number in range(iterations):
        simulate_iterations(
            traffic_config, sim_config, seeds[iter_number], iter_number
        )

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")