import math  # Import math module here
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import shutil
import json

# optimization problem
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
try:
    from pyswarm import pso  # Optional: only needed by the original PSO path.
except ImportError:  # QoS reproduction does not depend on PSO.
    pso = None

# For CGcreationTPC
from itertools import product

# Import constants
from constants import SYSTEM, MAC, CHANNEL


####################################################################################################################
# Function to calculate the TX power and the number of subcarriers 
def tx_power_calc(bw=None, nss=None):
    """
    Compute the number of subcarriers, nsc, as well as the total power [dBm] used depending on the bandwidth and the number of spatial streams
    """

    bw = SYSTEM.BW if bw is None else bw
    nss = SYSTEM.NSS if nss is None else nss

    bw_to_nsc = {20: 234, 40: 468, 80: 980, 160: 1960}
    nsc = bw_to_nsc.get(bw, None)
    if nsc is None:
        raise ValueError(f"Unsupported bandwidth: {bw}")

    ps_density = 5               # power spectral density in dBm/MHz (it cannot exceed 10 dBm/MHz by regulation, and the EIRP is even limited to 23 dBm in Europe)
    eirp = ps_density + 10 * math.log10(bw)                      # EIRP is constant by regulation in the 6Gz band
    eirp = min(eirp, 23)                                        # By ETSI constraint EIRP cannot exceed 23 dBm in Europe
    tx_power_ss_dbm = eirp - 10 * math.log10(nss)                   # transmission power per spatial stream

    return tx_power_ss_dbm, nsc
####################################################################################################################

# Function to calculate the overheads
def overheads_calc(edca_access_category):
    """
    Computes the needed overheads for the transmission process.
    Returns: pre_tx_overheads_edca, info_overheads_csr, pre_tx_overheads_csr, edca_overheads, csr_overheads
    """


    aifsn = 2 if edca_access_category in ['VI', 'VO'] else 3
    aifs = aifsn * SYSTEM.TE + SYSTEM.TSIFS

    # EDCA Overheads
    pre_tx_overheads_edca = SYSTEM.TRTS + SYSTEM.TSIFS + SYSTEM.TCTS + SYSTEM.TSIFS + SYSTEM.TIME_PREAMBLE_DATA
    edca_overheads = pre_tx_overheads_edca + SYSTEM.TSIFS + SYSTEM.TBACK + aifs + SYSTEM.TE

    # CSR Overheads
    info_overheads_csr = SYSTEM.TMAPC_ICF + SYSTEM.TSIFS + SYSTEM.TMAPC_ICR + SYSTEM.TSIFS
    pre_tx_overheads_csr = SYSTEM.TMAPC_TF + SYSTEM.TSIFS + SYSTEM.TIME_PREAMBLE_DATA
    csr_overheads = info_overheads_csr + pre_tx_overheads_csr + SYSTEM.TSIFS + SYSTEM.TBACK + aifs + SYSTEM.TE

    return {'pre_tx_overheads_edca': pre_tx_overheads_edca, 'info_overheads_csr': info_overheads_csr, 'pre_tx_overheads_csr': pre_tx_overheads_csr,
            'edca_overheads': edca_overheads, 'csr_overheads': csr_overheads}
####################################################################################################################

# Function to calculate the AP-STA coordinates
def ap_sta_coordinates(ap_number, sta_number, scenario_type, grid_value):
    """
    Computes the matrices with the coordinates of the devices (ap_matrix and sta_matrix).
    Returns: ap_matrix, sta_matrix
    """
    stas_per_ap = sta_number // ap_number

    # Max Distance between AP and STA (used only in 'grid')
    ap_sta_max_distance = min(10, grid_value / 4)  # in meters

    # Initialize matrices
    ap_matrix = np.zeros((ap_number, 2))
    sta_matrix = np.zeros((sta_number, 2))

    if scenario_type == 'grid':
        # APs are manually placed in a grid
        ap_matrix = np.array([[grid_value / 4, grid_value / 4],
                              [grid_value / 4, 3 * grid_value / 4],
                              [3 * grid_value / 4, grid_value / 4],
                              [3 * grid_value / 4, 3 * grid_value / 4]])

        # STA placement around each AP
        t = 2 * np.pi * np.random.rand(sta_number)
        g = 1 + (ap_sta_max_distance - 1) * np.random.rand(sta_number)
        ap_indices = np.repeat(np.arange(ap_number), stas_per_ap)
        sta_matrix[:, 0] = ap_matrix[ap_indices, 0] + g * np.cos(t)
        sta_matrix[:, 1] = ap_matrix[ap_indices, 1] + g * np.sin(t)

    elif scenario_type == 'random':
        # APs randomly placed all over the area
        ap_matrix = grid_value * np.random.rand(ap_number, 2)

        # STAs randomly placed all over the area
        sta_matrix = grid_value * np.random.rand(sta_number, 2)

    # Validations
    if ap_matrix.shape[0] != ap_number:
        raise ValueError('ap_number does not match with ap_matrix dimension')
    if sta_matrix.shape[0] != sta_number:
        raise ValueError('sta_number does not match with sta_matrix dimension')

    return ap_matrix, sta_matrix
####################################################################################################################

def generate_sta_mobility(
        ap_matrix,
        sta_matrix,
        walls,
        grid_value,
        association,
        timestamp_to_stop,
        ch_realization_duration,
        speed,
        min_dist_to_ap=1.0,
        max_attempts=30,
        rng=None
        ):
        """
        Mobility traces for STAs using persistent-heading random walk.

        Returns
        -------
        sta_mobility : list of np.ndarray
            List of length num_steps, each element is (sta_number, 2) positions.
        """
        if rng is None:
            rng = np.random.default_rng()
            

        # sanity / config
        ap_number = ap_matrix.shape[0]
        sta_number = sta_matrix.shape[0]

        # map STA -> AP index
        sta_to_ap = get_association(association, np.arange(sta_number))

        # number of mobility steps and per-step distance
        num_steps = int(np.ceil(timestamp_to_stop / ch_realization_duration))
        step_dist = speed * ch_realization_duration

        # local helpers
        def crosses_any_wall(p_from, p_to):
            x1, y1 = p_from
            x2, y2 = p_to
            for w in walls:
                if check_segment_intersection(x1, x2, y1, y2, w):
                    return True
            return False

        def in_bounds(p):
            gv = grid_value
            return 0.0 <= p[0] <= gv and 0.0 <= p[1] <= gv

        # current positions and storage
        curr = sta_matrix.copy()
        sta_mobility = []

        # persistent heading (theta) per STA (radians)
        thetas = rng.uniform(0, 2*np.pi, size=sta_number)

        for step in range(num_steps):
            sta_mobility.append(curr.copy())

            next_positions = curr.copy()
            for s in range(sta_number):
                origin = curr[s]
                ap_idx = sta_to_ap[s]
                ap_pos = ap_matrix[ap_idx]

                moved = False

                # First try the persistent heading; if it fails, try replacing heading up to max_attempts times.
                # We'll attempt at most (1 + max_attempts) candidate headings: current one + replacements.
                attempts_allowed = 1 + max_attempts
                for attempt in range(attempts_allowed):
                    # On attempt 0 use existing heading, otherwise sample a new heading and overwrite persistent heading
                    if attempt == 0:
                        theta = thetas[s]
                    else:
                        theta = rng.uniform(0, 2*np.pi)
                        thetas[s] = theta  # make this new heading persistent for future steps

                    delta = np.array([np.cos(theta), np.sin(theta)]) * step_dist
                    cand = origin + delta

                    # quick bounds clamp (simple bounce)
                    if not in_bounds(cand):
                        cand[0] = np.clip(cand[0], 0.0, grid_value)
                        cand[1] = np.clip(cand[1], 0.0, grid_value)

                    # check wall crossing
                    if crosses_any_wall(origin, cand):
                        # try next heading
                        continue

                    # enforce min distance to AP
                    vec_ap = cand - ap_pos
                    dist_ap = np.linalg.norm(vec_ap)
                    if dist_ap < min_dist_to_ap:
                        # compute away direction; if candidate coincides with AP, pick random away dir
                        if np.allclose(vec_ap, 0):
                            away_dir = rng.uniform(-1, 1, 2)
                            norm = np.linalg.norm(away_dir)
                            if norm == 0:
                                # can't fix this heading — try a new heading
                                continue
                            away_dir = away_dir / norm
                        else:
                            away_dir = vec_ap / (np.linalg.norm(vec_ap) + 1e-12)

                        cand = ap_pos + away_dir * min_dist_to_ap

                        # ensure bounds and no-wall-crossing after projection
                        if not in_bounds(cand) or crosses_any_wall(origin, cand):
                            continue

                    # If we reach here, candidate is valid
                    next_positions[s] = cand
                    moved = True
                    break

                if not moved:
                    # nothing valid found — STA stays in place. Keep the current theta so it may try again next timestep.
                    next_positions[s] = origin

            # advance
            curr = next_positions

        return sta_mobility
####################################################################################################################

def plot_mobility_trajectories(sta_mobility, ap_matrix, association, walls, grid_value):

        # Set font
        plt.rcParams['font.family'] = 'Noto Sans Mono'
        
        # Number of STAs and APs
        num_times = len(sta_mobility)
        num_stas = sta_mobility[0].shape[0]
        num_aps = ap_matrix.shape[0]

        # Prepare color list
        if num_aps < 10:
            ap_colours = np.array([
                [0.0763082893739572, 0.499882500825560, 0.931206019689022],
                [0.779918792240115, 0.679229996120941, 0.0248992275503480],
                [0.438409231440894, 0.803739036104376, 0.600548917464123],
                [0.723465177830941, 0.380941133148538, 0.950129500413646],
                [0.977989511996603, 0.0659363469059051, 0.230302879020965],
                [0.538495870410434, 0.288145599307994, 0.548489919236030],
                [0.501120463659938, 0.909593527719614, 0.909128374886731],
                [0.0720511333597615, 0.213385353579916, 0.133169445759250],
                [0.268438980101871, 0.452123961817683, 0.523412580673766]
            ])
        else:
            ap_colours = np.random.rand(num_aps, 3)

        # Create figure
        plt.figure(figsize=(8, 8))

        # Plot APs
        for k in range(num_aps):
            plt.plot(ap_matrix[k, 0], ap_matrix[k, 1], 'v', markersize=6, color=ap_colours[k])
            plt.text(ap_matrix[k, 0], ap_matrix[k, 1], f'AP${{{k}}}$', fontsize=14,
                    ha='center', va='bottom')

        # Pre-extract STA trajectories
        # trajectories[i] = Nx2 array of STA i for all times
        trajectories = {i: np.array([sta_mobility[t][i] for t in range(num_times)])
                        for i in range(num_stas)}

        # Plot STA mobility trails
        for sta_idx in range(num_stas):

            # Identify associated AP (color grouping)
            ap_idx = None
            for k, assoc in enumerate(association):
                if sta_idx in assoc:
                    ap_idx = k
                    break

            color = ap_colours[ap_idx] if ap_idx is not None else 'gray'
            traj = trajectories[sta_idx]

            # Draw the entire path
            plt.plot(traj[:, 0], traj[:, 1], '--', linewidth=1.5, color=color, alpha=0.6)
            # Draw start and end points
            plt.plot(traj[0, 0], traj[0, 1], 'o', markersize=5, color=color)
            plt.plot(traj[-1, 0], traj[-1, 1], 's', markersize=6, color=color)

            # Label
            plt.text(traj[-1, 0], traj[-1, 1], f'STA${{{sta_idx}}}$', fontsize=12,
                    ha='left', va='bottom')

        # Plot walls
        for w in walls:
            plt.plot([w[0], w[1]], [w[2], w[3]], color='k', linewidth=2)

        # Aesthetic setup
        plt.grid(True)
        plt.xlabel('X-axis [m]', fontsize=16)
        plt.ylabel('Y-axis [m]', fontsize=16)
        plt.xticks(np.arange(0, grid_value + 1, 5))
        plt.yticks(np.arange(0, grid_value + 1, 5))
        plt.xlim([0, grid_value])
        plt.ylim([0, grid_value])
        plt.title('STA Mobility Traces', fontsize=18)
        plt.show()
####################################################################################################################



# Function to calculate the AP-STA association
def ap_sta_association(ap_number, sta_number, scenario_type):
    """
    Association process. STAs are associated independently of the distance to their corresponding AP.
    Returns a list with the list of STAs by AP.
    """
    if scenario_type == 'grid':
        stas_per_ap = sta_number // ap_number
        # Create a list of STAs and reshape it to match the AP-STA association
        association = np.arange(sta_number).reshape(ap_number, stas_per_ap).tolist()

    return association
####################################################################################################################

# Function to plot the devices' locations in the deployment
def plot_deployment(ap_matrix, sta_matrix, association, grid_value, walls):
    """
    Plots the deployment with APs, STAs, and walls.
    """
    # Set font family
    plt.rcParams['font.family'] = 'Noto Sans Mono'
    # plt.rcParams['font.family'] = 'Tlwg Typewriter'

    # Create figure
    plt.figure(figsize=(8, 8))

    # Define AP colors
    if ap_matrix.shape[0] < 10:
        ap_colours = np.array([[0.0763082893739572, 0.499882500825560, 0.931206019689022],
                               [0.779918792240115, 0.679229996120941, 0.0248992275503480],
                               [0.438409231440894, 0.803739036104376, 0.600548917464123],
                               [0.723465177830941, 0.380941133148538, 0.950129500413646],
                               [0.977989511996603, 0.0659363469059051, 0.230302879020965],
                               [0.538495870410434, 0.288145599307994, 0.548489919236030],
                               [0.501120463659938, 0.909593527719614, 0.909128374886731],
                               [0.0720511333597615, 0.213385353579916, 0.133169445759250],
                               [0.268438980101871, 0.452123961817683, 0.523412580673766]])
    else:
        ap_colours = np.random.rand(ap_matrix.shape[0], 3)

    # Plot APs
    for k in range(ap_matrix.shape[0]):
        plt.plot(ap_matrix[k, 0], ap_matrix[k, 1], 'v', markersize=6, color=ap_colours[k, :])
        plt.text(ap_matrix[k, 0], ap_matrix[k, 1], f'AP${{ {k} }}$', fontsize=14, ha='center', va='bottom')

    # Plot STAs
    for j in range(sta_matrix.shape[0]):
        # Find the position where the current STA is associated
        idxCol = None
        for i, assoc in enumerate(association):
            if j in assoc:
                idxCol = i
                break
        
        if idxCol is not None:  # If a valid association was found
            plt.plot(sta_matrix[j, 0], sta_matrix[j, 1], 's', markersize=6, color=ap_colours[idxCol, :])
            plt.text(sta_matrix[j, 0], sta_matrix[j, 1], f'STA${{ {j} }}$', fontsize=14, ha='center', va='bottom')
        else:
            print(f"STA {j} is not associated with any AP.")

    # Plot walls
    for i in range(walls.shape[0]):
        plt.plot([walls[i, 0], walls[i, 1]], [walls[i, 2], walls[i, 3]], color='k', linewidth=2)

    # Add grid and labels
    plt.grid(True)
    plt.xlabel('X-axis [meters]', fontsize=16)
    plt.ylabel('Y-axis [meters]', fontsize=16)
    plt.xticks(np.arange(0, grid_value + 1, 5))
    plt.yticks(np.arange(0, grid_value + 1, 5))
    plt.xlim([0, grid_value])
    plt.ylim([0, grid_value])

    plt.show()
####################################################################################################################

# Function to calculate the pathloss
def get_loss(a_position, b_position, m_number_of_walls, std_dev=None, frequency_ghz=None, rng=None):
    """
    Calculates the path loss between two devices using the Enterprise model 
    defined for 802.11ax.
    
    Parameters:
    - a_position (array-like): Coordinates (x, y) of device A.
    - b_position (array-like): Coordinates (x, y) of device B.
    - m_number_of_walls (int): Number of walls between the devices.
    - std_dev (float): Standard deviation for shadowing in dB.

    Returns:
    - loss (float): Path loss in dB.
    """
    loss = 0.0      
    distance = np.linalg.norm(np.array(a_position) - np.array(b_position))
    std_dev = CHANNEL.STD_DEV if std_dev is None else std_dev
    frequency_ghz = CHANNEL.FREQUENCY if frequency_ghz is None else frequency_ghz

    # Additional loss (when distance is greater than the breaking point)
    add_loss = 0.0
    if distance >= CHANNEL.DBP:
        add_loss = 35 * np.log10(distance / CHANNEL.DBP)

    shadowing_sample = np.random.randn() if rng is None else rng.standard_normal()
    shadowing = std_dev * shadowing_sample  # Shadowing in dB

    loss = 40.05 + 20 * np.log10(frequency_ghz / 2.4) + 20 * np.log10(min(distance, CHANNEL.DBP)) + add_loss + 7 * m_number_of_walls + shadowing

    return loss
####################################################################################################################

# Function to check if two segments intersect (number of walls between devices)
def check_segment_intersection(x1a1, x2a1, y1a1, y2a1, a2):
    """
    Returns True if segment a1 crosses segment a2. This function is used to determine
    whether the transmission defined by the two points in a1 intersects with the wall
    defined by segment a2.

    Parameters:
    x1a1 (float): x-coordinate of the first point of segment a1.
    x2a1 (float): x-coordinate of the second point of segment a1.
    y1a1 (float): y-coordinate of the first point of segment a1.
    y2a1 (float): y-coordinate of the second point of segment a1.
    a2 (array-like): Coordinates [x1, x2, y1, y2] defining the second segment a2.

    Returns:
    bool: True if segments intersect, False otherwise.
    """

    x1a2, x2a2, y1a2, y2a2 = a2
    # Helper function to calculate orientation
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    # Helper function to check if a point is on a segment
    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    # Define the endpoints
    p1, q1 = (x1a1, y1a1), (x2a1, y2a1)
    p2, q2 = (x1a2, y1a2), (x2a2, y2a2)

    # Calculate orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False
####################################################################################################################

# Function to calculate channelMatrix and RSSI_dB_vector_to_export
def get_channel_matrix(max_tx_power_dbm, ap_matrix, sta_matrix, scenario_type, walls, cca=None, frequency_ghz=None, rng=None):
    """
    Calculates the channel matrix and RSSI [dB] values for each AP-STA and AP-AP pairs. Validate RSSI between APs is above the CCA

    returns: channel_matrix
    """

    cca = SYSTEM.CCA if cca is None else cca
    frequency_ghz = CHANNEL.FREQUENCY if frequency_ghz is None else frequency_ghz

    # Matrix to store all AP-STA channel coefficients
    channel_matrix = np.zeros((sta_matrix.shape[0], ap_matrix.shape[0]))

    number_of_walls_ap_sta_matrix = np.zeros((sta_matrix.shape[0], ap_matrix.shape[0]))  # Matrix for walls between APs and STAs
    number_of_walls_ap_ap_matrix = np.zeros((ap_matrix.shape[0], ap_matrix.shape[0]))  # Matrix for walls between APs

    ap_to_ap_rssi_matrix = np.zeros((ap_matrix.shape[0], ap_matrix.shape[0]))  # AP to AP RSSI matrix

    # Loop to calculate channel matrix and RSSI values for each STA-AP pair
    for k in range(ap_matrix.shape[0]):
        for kk in range(sta_matrix.shape[0]):
            for kkk in range(walls.shape[0]):  # Checking the number of walls between AP_k and STA_kk
                is_intersecting = check_segment_intersection(ap_matrix[k, 0], sta_matrix[kk, 0], ap_matrix[k, 1], sta_matrix[kk, 1], walls[kkk, :])
                if is_intersecting:
                    number_of_walls_ap_sta_matrix[kk, k] += 1

            channel_coefficient_db = get_loss(ap_matrix[k, :], sta_matrix[kk, :], number_of_walls_ap_sta_matrix[kk, k], frequency_ghz=frequency_ghz, rng=rng)  # default standard deviation std_dev for shadowing=5
            channel_matrix[kk, k] = 1 / 10 ** (channel_coefficient_db / 10)

        # AP to AP interactions
        ap_other_vector = list(set(range(ap_matrix.shape[0])) - set([k]))  # All APs except AP k
        for i in ap_other_vector:
            for ii in range(walls.shape[0]):  # Checking the number of walls between AP_k and AP_i
                is_intersecting = check_segment_intersection(ap_matrix[k, 0], ap_matrix[i, 0], ap_matrix[k, 1], ap_matrix[i, 1], walls[ii, :])
                if is_intersecting:
                    number_of_walls_ap_ap_matrix[k, i] += 1

            # Calculate RSSI for AP to AP connections
            if scenario_type == 'random':
                channel_coefficient_db = get_loss(ap_matrix[k, :], ap_matrix[i, :], number_of_walls_ap_ap_matrix[k, i], std_dev=0, frequency_ghz=frequency_ghz, rng=rng)  # Shadowing std_dev=0 between APs
                ap_to_ap_rssi_matrix[k, i] = max_tx_power_dbm - channel_coefficient_db
            elif scenario_type == 'grid':
                channel_coefficient_db = get_loss(ap_matrix[k, :], ap_matrix[i, :], 1, std_dev=0, frequency_ghz=frequency_ghz, rng=rng)  # Only 1 wall between APs
                ap_to_ap_rssi_matrix[k, i] = max_tx_power_dbm - channel_coefficient_db

    # Validation check for RSSI between APs
    if np.min(ap_to_ap_rssi_matrix) < cca:
        raise ValueError('Scenario constraint: RSSI between APs is under the CCA threshold. All APs should be in the coverage area of the others')

    return channel_matrix
####################################################################################################################

def mcs_cal_PER_001(sinr_db):
    """
    Calculates the 802.11be MCS and related parameters (n_bps, rc) for a given SINR value.
    
    Args:
    sinr_db (float): SINR value in dB

    Returns:
    mcs (int): Modulation and Coding Scheme index (-1 if invalid)
    n_bps (int): Number of coded bits per subcarrier per stream (1 if invalid)
    rc (float): Rate of coding (1 / 2 if invalid)
    """

    if 14.2862 <= sinr_db < 19.5154:
        mcs = int(0)
        n_bps = 1
        rc = 1 / 2
    elif 19.5154 <= sinr_db < 25.5501:
        mcs = int(1)
        n_bps = 2
        rc = 1 / 2
    elif 25.5501 <= sinr_db < 27.9312:
        mcs = int(2)
        n_bps = 2
        rc = 3 / 4
    elif 27.9312 <= sinr_db < 33.7179:
        mcs = int(3)
        n_bps = 4
        rc = 1 / 2
    elif 33.7179 <= sinr_db < 36.6008:
        mcs = int(4)
        n_bps = 4
        rc = 3 / 4
    elif 36.6008 <= sinr_db < 38.8428:
        mcs = int(5)
        n_bps = 6
        rc = 2 / 3
    elif 38.8428 <= sinr_db < 41.9447:
        mcs = int(6)
        n_bps = 6
        rc = 3 / 4
    elif 41.9447 <= sinr_db < 43.9603:
        mcs = int(7)
        n_bps = 6
        rc = 5 / 6
    elif 43.9603 <= sinr_db < 46.5902:
        mcs = int(8)
        n_bps = 8
        rc = 3 / 4
    elif 46.5902 <= sinr_db < 49.1915:
        mcs = int(9)
        n_bps = 8
        rc = 5 / 6
    elif 49.1915 <= sinr_db < 52.3450:
        mcs = int(10)
        n_bps = 10
        rc = 3 / 4
    elif 52.3450 <= sinr_db < 53.8530:
        mcs = int(11)
        n_bps = 10
        rc = 5 / 6
    elif 53.8530 <= sinr_db < 57.3929:
        mcs = int(12)
        n_bps = 12
        rc = 3 / 4
    elif sinr_db >= 57.3929:
        mcs = int(13)
        n_bps = 12
        rc = 5 / 6
    else:
        # invalid MCS
        mcs = int(-1)
        n_bps = 1
        rc = 1 / 2

    return mcs, n_bps, rc
####################################################################################################################

def power_allocation(n_stas, noise_power, channel_matrix_reduced, pmax, nsc, tpc_method, nss=None):
    """
    Optimizes power allocation using the specified method to maximize proportional fairness.
    Returns: popt
    """

    # Function to compute the product of rates for the given power allocation
    def compute_rates(p, channel_matrix_reduced, noise_power, nsc, nss):
        """
        Computes the product of rates for the given power allocation.
        
        Args:
        p: Power allocation vector (n_stas,)
        channel_matrix_reduced: Channel gain matrix (n_stas x n_stas)
        noise_power: Noise power (scalar)
        n_stas: Number of links
        nsc: Number of subcarriers
        nss: Number of spatial streams

        Returns:
        product_rate: Product of rates for all links
        """
        
        rates = np.zeros(n_stas)
        
        # Calculate SINR in linear scale
        sinr = (p * np.diag(channel_matrix_reduced)) / (noise_power + np.sum(channel_matrix_reduced * p, axis=1) - np.diag(channel_matrix_reduced) * p)
        sinr_db = 10 * np.log10(sinr)
        
        for i in range(n_stas):
            mcs, n_bps, rc = mcs_cal_PER_001(sinr_db[i])
            if mcs == -1:
                rates[i] = 0
            else:
                rates[i] = (nsc * n_bps * rc * nss) / (SYSTEM.TDFT + SYSTEM.TGI)
                # rates[i] = np.log2(1 + sinr[i])
        
        # Return the product of all rates

        if np.any(rates == 0):
            product_rate = 0  # Penalize invalid rates
        else:
            # product_rate = np.prod(rates)
            product_rate = np.sum(np.log(rates))
        
        return product_rate
        # return np.min(rates)
    
    # Function to calculate the optimal power allocation using the Particle Swarm Optimization (PSO) algorithm
    def power_allocation_particleswarm(n_stas, noise_power, channel_matrix_reduced, pmax, nsc, nss):
        """
        Optimizes power allocation using Particle Swarm Optimization (PSO)
        to maximize proportional fairness.
        
        Args:
        n_stas: Number of links (transmitters)
        noise_power: Noise power (scalar)
        channel_matrix_reduced: Channel gain matrix (n_stas x n_stas)
        pmax: Maximum power constraint (scalar)
        nsc: Number of subcarriers
        nss: Number of spatial streams
        
        Returns:
        popt: Optimized power allocation vector (n_stas,)
        """
        

        # Bounds: 0 <= P <= pmax for each power allocation
        lb = np.ones(n_stas)  # Lower bound for power allocation
        ub = pmax * np.ones(n_stas)  # Upper bound for power allocation

        # Use PSO with optimized swarm size and iterations
        SWARM_SIZE = 20  # Reduced swarm size (can be tuned)            default ---- 20      
        MAX_ITER = 20   # Reduced number of iterations (can be tuned)    default ---- 20 

        def objective(p):
            return -compute_rates(p, channel_matrix_reduced, noise_power, nsc, nss)

        # Silence stdout and stderr
        with open(os.devnull, 'w') as fnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = fnull
                sys.stderr = fnull

                popt, _ = pso(objective, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
                    swarmsize=SWARM_SIZE, omega=0.5, phip=0.5, phig=0.5, maxiter=MAX_ITER, 
                    minstep=1e-8, minfunc=1e-8, debug=False)  # minstep=1e-8, minfunc=1e-8
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        # popt, _ = pso(objective, lb, ub, swarmsize=SWARM_SIZE, maxiter=MAX_ITER, debug=False, 
        #                minfunc=1e-6, omega=0.5, phip=0.5, phig=0.5)
        

        return popt

    

    nss = SYSTEM.NSS if nss is None else nss

    # Main logic to select the optimization method
    if tpc_method is None:
        popt = np.full(n_stas, pmax)
    elif tpc_method == 'PSO':   # Particle Swarm Optimization
        if pso is None:
            raise ImportError("The 'pyswarm' package is required for tpc_method='PSO'.")
        popt = power_allocation_particleswarm(n_stas, noise_power, channel_matrix_reduced, pmax, nsc, nss)
    else:
        raise ValueError(f"Unsupported optimization method: {tpc_method}")

    return popt
####################################################################################################################

# Function to compute the number of aggregated packets that can be transmitted in a given time
def tx_packets(nsc, n_bps, rc, data_tx_time, nss=None):
    """
    Calculates the number of aggregated packets (A-MPDU length) that can be transmitted 
    within the given data transmission time.

    Args:
    nsc (int): Number of subcarriers
    n_bps (int): Number of coded bits per subcarrier per stream
    rc (float): Rate of coding
    nss (int): Number of spatial streams
    data_tx_time (float): Data transmission time in seconds

    Returns:
    int: Number of packets that can be transmitted
    """

    nss = SYSTEM.NSS if nss is None else nss

    # Calculate the number of packets that can be transmitted
    bits_available = np.floor(data_tx_time / (SYSTEM.TDFT + SYSTEM.TGI)) * (nsc * n_bps * rc * nss)              # Total bits that can be transmitted
    agg_packets = int(np.floor((bits_available - MAC.LSF - MAC.LTAIL) / (MAC.LMD + MAC.LMH + MAC.FRAME_LENGTH)))  # Number of aggregated packets
    
    if agg_packets > 1024:
        raise ValueError('Number of aggregated packets exceeds the maximum limit')
    
    return agg_packets
####################################################################################################################

# Function to compute the elapsed time for transmitting the given number of aggregated packets
def elapsed_time_tx(nsc, n_bps, rc, tx_packets, nss=None):
    """
    Calculates the elapsed time for transmitting the given number of aggregated packets.

    Args:
    nsc (int): Number of subcarriers
    n_bps (int): Number of coded bits per subcarrier per stream
    rc (float): Rate of coding
    nss (int): Number of spatial streams
    tx_packets (int): Number of aggregated packets

    Returns:
    float: time_tx is the elapsed time for transmitting data packets
    """
    nss = SYSTEM.NSS if nss is None else nss
    lmd = 0 if tx_packets == 1 else MAC.LMD 

    # Calculate the elapsed time for transmitting the packets
    time_tx = np.ceil((MAC.LSF + tx_packets*(lmd + MAC.LMH + MAC.FRAME_LENGTH) + MAC.LTAIL)/(nsc*n_bps*rc*nss))*(SYSTEM.TDFT + SYSTEM.TGI)

    return time_tx
####################################################################################################################

def generate_combinations(association, cg_size):
    """
    Generates all possible combinations of AP-STAs for a given association.
    """

    placeholder = -1
    # Step 1: Generate all possible combinations of STAs for each AP, including np.nan for NaN replacement
    u_with_placeholders = [np.array([placeholder] + stas, dtype=int) for stas in association]

    # Step 2: Generate all possible combinations of AP-STAs using itertools.product
    all_combinations = np.array(list(product(*u_with_placeholders)), dtype=int)

    # Step 3: Filter out rows where all elements are the placeholder (np.nan)
    valid_combinations = all_combinations[~np.all(all_combinations==placeholder, axis=1)]

    # Step 4: Apply filtering based on cg_size
    # Only keep rows where the number of non-placeholder elements is <= cg_size
    idx_row = np.sum((valid_combinations!=placeholder), axis=1)
    valid_combinations = valid_combinations[idx_row <= cg_size]

    # Step 5: Remove duplicate rows
    valid_combinations = np.unique(valid_combinations, axis=0)

    # Step 6: Sort rows by the number of placeholder elements (descending) and smallest non-placeholder integers
    valid_combinations = valid_combinations[
        np.lexsort((
            np.min(np.where(valid_combinations!=placeholder, valid_combinations, np.inf), axis=1),
            -np.sum(valid_combinations==placeholder, axis=1)
        ))
    ]

    # removing the placeholder
    combs_list = [combs[np.where(combs!=placeholder)] for _, combs in enumerate(valid_combinations)]

    return combs_list
####################################################################################################################

def get_association(association, stas):
    """
    The corresponding APs which the stations in STAs are associated to
    Return:
    aps : list
    """
    aps = [
        next((idx for idx, assoc in enumerate(association) if sta in assoc), -1)
        for sta in stas
    ]

    return aps

def get_rssi(max_tx_power_db, channel_matrix, stas, aps):
    """
    The corresponding RSSI values for the stations in STAs
    Return:
    rssi : list
    """
    rssi = [
        channel_matrix[sta, ap]*10**(max_tx_power_db / 10)
        for sta, ap in zip(stas, aps)
    ]

    return rssi

def get_channel_coefficient(channel_matrix):
    """
    Compute the channel coefficients (flattened) for the given channel matrix all BSSs
    
    """
    return channel_matrix.flatten().tolist()

def get_channel_coefficient_bss(channel_matrix, STAs, APs):
    """
    The corresponding intra-BSS channel coefficients for the stations in STAs
    Return:
    channel_coeff : list
    """
    channel_coeff = [
        channel_matrix[sta, ap]
        for sta, ap in zip(STAs, APs)
    ]

    return channel_coeff


# Function to compute the C-SR groups and the corresponding power allocation
def cg_creation_tpc(association, channel_matrix, max_tx_power_dbm, nsc, is_filtering, tpc_method, cg_size, nss=None): 
    """
    Computes the Co-SR groups and the corresponding power allocation.
    Returns: map_matrix, TxPowerMatrixTemp, comb_ok
    """

    nss = SYSTEM.NSS if nss is None else nss  # Number of spatial streams

    # Initialization
    noise_power = 10**(SYSTEM.PN_DBM / 10)
    max_tx_power = 10**(max_tx_power_dbm / 10)

    # Function to generate all possible combinations of AP-STAs for a given association
    map_matrix = generate_combinations(association, cg_size)

    # Create TxPowerMatrixTemp with the same shape as map_matrix
    tx_power_matrix_temp = [np.full_like(row, max_tx_power, dtype=float) for _, row in enumerate(map_matrix)]

    if not is_filtering:
        return map_matrix, tx_power_matrix_temp, np.ones(len(map_matrix), dtype=bool)

    # Other matrices initialization
    datarate_temp = [np.zeros(len(row), dtype=float) for _, row in enumerate(map_matrix)]
    comb_ok = np.zeros(len(map_matrix), dtype=bool)
    discard_list = np.ones(len(map_matrix), dtype=bool)

    # Main loop for verifying groups
    for i in range(len(map_matrix)):
        if discard_list[i] == False:
            continue
        
        stas = map_matrix[i]
        aps = get_association(association, stas)

        channel_matrix_reduced = channel_matrix[np.ix_(stas, aps)]

        if len(stas) == 1:  # Use maximum power for a single STA
            popt = max_tx_power
        else:  # Compute the subset of power that maximizes the proportional fair transmission
            # TPC (Power allocation)
            
            # Solving the Opt problem with the selected method
            popt = power_allocation(len(stas), noise_power, channel_matrix_reduced, max_tx_power, nsc, tpc_method)

            # Store the power vector in TxPowerMatrixTemp
            tx_power_matrix_temp[i] = popt

        # Compute the SINR and datarate for each STA
        sinr = (popt * np.diag(channel_matrix_reduced)) / (noise_power + np.sum(channel_matrix_reduced * popt, axis=1) - np.diag(channel_matrix_reduced) * popt)
        sinr_db = 10 * np.log10(sinr)

        for k, _ in enumerate(stas):
            mcs, n_bps, rc = mcs_cal_PER_001(sinr_db[k])
            if mcs == -1:
                datarate_temp[i][k] = 0 #SINR under the threshold
            else:
                datarate_temp[i][k] = n_bps * rc / (12 * 5/6)  # Number of bits to code a symbol
                # datarateTemp[i][k] = NSC * N_bps * Rc * NSS / (12.8e-6 + 0.8e-6) # Datarate in bps
            
            # if the datarate using CSR is lower than the datarate without CSR, discard the combination
            if len(stas) * datarate_temp[i][k] >= datarate_temp[stas[k]]:
                continue
            else:
                # # Discarding the rest of combs where this STAs appear (Prunning)
                true_indices = np.where(discard_list)[0]
                mask = np.array([set(stas).issubset(set(map_matrix[idx])) for idx in true_indices])
                discard_list[true_indices[mask]] = False
                break


                # discard_list[i] = False
                # break

        if discard_list[i] == True:
            comb_ok[i] = True
            # datarate[i] = len(datarateTemp[i])*np.prod(datarateTemp[i])

    return map_matrix, tx_power_matrix_temp, comb_ok

####################################################################################################################

def plot_histogram(data, name):
    positions = np.arange(len(data))  # X-axis: positions (priorities)
    plt.bar(positions, data)

    plt.xlabel("Priority Rank (the higher the index, the higher the priority)", fontsize=16)
    plt.ylabel("Number of Selections / Number of TXOPs", fontsize=16)
    # plt.title("Priority Selection Frequency")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.legend([name], loc='upper right')
    plt.show()

