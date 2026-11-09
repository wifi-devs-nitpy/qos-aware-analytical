import numpy as np
import re
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from numpy.random import SeedSequence
import utils as utils
from constants import SYSTEM
import h5py
import os

################## Deployment Generator ###########################

# Helper function to simulate one iteration
def deployment_generator(sim_config, seed, show_plot=False):
    """ 
    Simulate one iteration and return the deployment matrices.
    """
    # Set the seed
    np.random.seed(seed)

    # Generate the AP and STA coordinates
    ap_matrix, sta_matrix = utils.ap_sta_coordinates(
        sim_config['ap_number'], 
        sim_config['sta_number'], 
        sim_config['scenario_type'], 
        sim_config['grid_value']
        )
    
    # Compute the channel_matrix
    channel_matrix = utils.get_channel_matrix(
        sim_config['max_tx_power_dbm'], 
        ap_matrix,
        sta_matrix,
        sim_config['scenario_type'],
        sim_config['walls']
    )

    # Association list
    association = utils.ap_sta_association(
        sim_config['ap_number'], 
        sim_config['sta_number'],
        sim_config['scenario_type']
    )

    # Call the function to plot
    if show_plot:
        utils.plot_deployment(
            ap_matrix, 
            sta_matrix, 
            association, 
            sim_config['grid_value'], 
            sim_config['walls']
            )

    return ap_matrix, sta_matrix, association, channel_matrix

if __name__ == '__main__':
    # Start Timer
    start_time = time.time()

    # Simulation parameters
    edca_access_category = 'BE'

    sim = '30-16'  # Simulation name: 'APtoAPdistance-STA_NUMBER'
    numbers = re.findall(r'\d+', sim) # Extract numbers from the simulation name

    # Scenario-related
    ap_number = 4
    sta_number = int(numbers[1]) 
    grid_value = int(numbers[0]) * 2
    scenario_type = 'grid'

    walls = np.array([[0, grid_value, grid_value/2, grid_value/2], 
                    [grid_value/2, grid_value/2, 0, grid_value]])

    ### Channel-related parameters
    max_tx_power_dbm, nsc = utils.tx_power_calc() # default bw=80 MHz, nss=2 spatial streams

    # Number of iterations
    iterations = 100

    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(iterations)

    # Pre-allocate variables
    sta_matrix_save = np.empty((sta_number, 2, iterations))        
    channel_matrix_save = np.empty((sta_number, ap_number, iterations))

    # Simulation Configuration
    sim_config = {
        'ap_number': ap_number,
        'sta_number': sta_number,
        'scenario_type': scenario_type,
        'grid_value': grid_value,
        'walls': walls,
        'max_tx_power_dbm': max_tx_power_dbm,
        'txop_duration': SYSTEM.TXOP_DURATION,
        'pn_dbm': SYSTEM.PN_DBM,
        'cca': SYSTEM.CCA,
        'nss': SYSTEM.NSS,
        'nsc': nsc,
        'seed': 1,
    }

    max_workers = min(os.cpu_count(), iterations)  # Optimize worker count
    # max_workers = 1  # Optimize worker count
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                deployment_generator, sim_config, seeds[iter_number], show_plot=False
            )
            for iter_number in range(iterations)
        ]        
        # Process results as they complete
        for iter_number, future in enumerate(tqdm(futures, desc="Processing", unit=" iterations")):
            try:
                _, sta_matrix, _, channel_matrix = future.result()
                sta_matrix_save[:, :, iter_number] = sta_matrix
                channel_matrix_save[:, :, iter_number] = channel_matrix
            except Exception as e:
                print(f"Error in iteration {iter_number}")

    # Define output folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up one level
    output_folder = os.path.join(base_dir, 'deployments_datasets' , sim)
    os.makedirs(output_folder, exist_ok=True)

    # Define file path
    h5file_path = os.path.join(output_folder, 'deployment_datasets.h5')

    # Save data to HDF5 file
    with h5py.File(h5file_path, 'w') as f:
        f.create_dataset('sta_matrix_save', data=sta_matrix_save)
        f.create_dataset('channel_matrix_save', data=channel_matrix_save)

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")