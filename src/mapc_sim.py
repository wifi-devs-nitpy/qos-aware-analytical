import numpy as np
import math
import matplotlib.pyplot as plt
from utils import mcs_cal_PER_001, tx_packets, elapsed_time_tx, get_association, cg_creation_tpc, check_segment_intersection, plot_deployment, get_channel_matrix
# Import constants
from constants import SYSTEM, MAC, CHANNEL

import copy

class MAPCsim:
    """
    MAPCsim class simulates a wireless network with multiple access points (APs) and stations (STAs).
    It handles the simulation configuration, mobility, traffic generation, backoff mechanism, scheduling, and results analysis.
    """

    def __init__(self, sim_config, mobility_config=None):

        self.sim_config = sim_config

        # System-related
        self.txop_duration = sim_config['txop_duration']                              # Duration of a TXOP
        self.noise_power = 10 ** (sim_config['pn_dbm'] / 10)                          # Noise power in mW
        self.max_tx_power = 10 ** (sim_config['max_tx_power_dbm'] / 10)                   # Maximum transmission power in mW
        self.nss = sim_config['nss']                                                  # Number of spatial streams
        self.nsc = sim_config['nsc']                                                  # Number of subcarriers

        # Scenario-related
        self.ap_matrix = sim_config['ap_matrix']                                              # AP positions
        self.ap_number = sim_config['ap_number']                                                # Number of APs                                                   
        self.sta_number = sim_config['sta_number']                                              # Number of STAs
        self.association = sim_config['association']                                   # Association matrix
        self.channel_matrix = np.zeros((self.sta_number, self.ap_number))               # Channel matrix
        self.channel_matrix_fading = np.zeros((self.sta_number, self.ap_number))                   # Channel matrix
        self.channel_matrix_last_estimation = np.zeros((self.sta_number, self.ap_number))          # Channel matrix at the last estimation

        # mobility-related
        self.mobility_config = mobility_config                                              # Mobility object
        self.sta_mobility : list = None                                # Mobility traces for STAs
        self.ch_realization_duration = self.mobility_config['ch_realization_duration'] if self.mobility_config else None  # Duration of each channel realization
        self.last_channel_idx : int = 0
        self.ch_realizations_per_update : int = self.mobility_config['ch_realizations_per_update'] if self.mobility_config else None  # Number of channel realizations per update
        
        
        # Simulation-related
        self.sim_timeline = 0                                                # Simulation timeline
        self.timestamp_to_stop : float                        # Timestamp to stop the simulation

        # Traffic-related
        self.access_category : str                                         # Access category                  
        self.sta_queue_timeline = []                                     # Stores the arrival time of the packets of all STAs
        self.delivery_timestamp_record : list                               # Stores the delivery time of the packets of all STAs
        self.sta_queue_state : list                                        # Stores the state of the packets of all STAs. True = not transmitted, False = transmitted
        
        self.first_pos_timestamp : np.ndarray                                # Stores the timestamp of the first non-transmitted packet of each STA
        self.first_pos_position : np.ndarray                                 # Stores the position of the first non-transmitted packet of each STA




        # Simulation-related
        self.simulation_system : str                                      # Simulation system -> EDCA or CSR

        # backoff-related
        self.txop_winner : int                                              # Winner of the TXOP
        self.aps_packet_indicator : np.ndarray                             # Vector indicating whether each AP has packets to transmit
        self.backoff_values : np.ndarray                                    # backoff values for each AP
        self.backoff_stage : np.ndarray                                     # backoff stage for each AP
        self.cw_min : int                                                 # Minimum contention window
        self.max_backoff_stage : int                                       # Maximum backoff stage
        self.aifs : float                                                # Arbitration Inter-Frame Space
        

        # TXOP-related
        self.txop_win_number : np.ndarray                                  # Number of TXOP wins for each AP
        self.txop_collision : np.ndarray                                  # Number of TXOP collisions for each AP
        self.pre_tx_overheads_edca = sim_config['overheads']['pre_tx_overheads_edca']                      # Amount of time per TXOP before the data transmission begins using EDCA 
        self.edca_overheads = sim_config['overheads']['edca_overheads']                                  # Total amount of EDCA overheads 
        self.info_overheads_csr = sim_config['overheads']['info_overheads_csr']                            # Amount of time per TXOP for the information exchange in CSR
        self.pre_tx_overheads_csr = sim_config['overheads']['pre_tx_overheads_csr']                      # Amount of time per TXOP before the data transmission begins using CSR
        self.csr_overheads = sim_config['overheads']['csr_overheads']                                  # Total amount of CSR overheads

        # CSR-related
        self.map_matrix : np.ndarray                                      # Matrix with the mapping of the STAs to the APs
        self.cgs_stas : list                                            # C-SR compatible groups of STAs
        self.tx_power_matrix : list                                 # Transmission power matrix
        self.comb_ok : np.ndarray                                 # Combination vector to indicate whether the combination is selected or not
        self.scheduler : str                            # scheduling: - Number of packets: 'MNP' 
                                                        #             - Oldest packet: 'OP'
                                                        #             - Random selection: 'Random'
                                                        #             - TAT selection: 'TAT'
                                                        #             - Hybrid selection: 'Hybrid'
        self.alpha = 0.5                                                  # Alpha value for TAT. Default value is 0.5
        self.beta = 0.5                                                   # Beta value for TAT. Default value is 0.5
        
        # Results
        self.throughput_txop : float                                         # Throughput of the TXOP
        self.last_txop_queue_sizes = np.zeros(self.sta_number, dtype=int)  # Queue sizes at the last TXOP
        self.per_txop_sta_tx_packets : np.ndarray                         # Number of packets transmitted per STA per TXOP
        self.sta_selection_counter : np.ndarray                             # Counter for the number of times each STA is selected
        self.throughput_sim : np.ndarray                                  # Throughput of each STA
        self.delay_per_sta : list                                         # Delay of each STA
        self.delay_vector : list                                           # Delay matrix (all STAs)         
        self.ap_collision_prob : np.ndarray                                # Collision probability of each AP

        self.suc_txops = int(0)  # Counter for successful transmissions
        self.priority_selection_counter = np.zeros((self.sta_number,), dtype=int)  # Priority selection for each STA

    def update_ap(self):
        """
        Updates the vector indicating whether each AP has packets to transmit.
        """
        # Vectorized operation to check if any associated STA has packets to transmit
        self.aps_packet_indicator = np.array([
            np.any(self.first_pos_timestamp[self.association[k]] <= self.sim_timeline)
            for k in range(self.ap_number)
        ], dtype=bool)

    def update_sta(self, sta_rx, rx_vector_pos, temp_elapsed_time):
        """
        Updates STA properties when packets are received.
        """
        if rx_vector_pos:
            # Update the queue state of the STA
            self.sta_queue_state[sta_rx][rx_vector_pos] = False
            
            # Update the delivery timestamp record
            self.delivery_timestamp_record[sta_rx][rx_vector_pos] = self.sim_timeline + temp_elapsed_time
            
            # Update the first position timestamp and position
            queue_state = self.sta_queue_state[sta_rx]
            self.first_pos_position[sta_rx] = (
                np.argmax(queue_state) if np.any(queue_state) else len(queue_state) - 1
            )
            # if sum(self.sta_queue_state[sta_rx]) == 0:
            #     raise ValueError("All packets have been transmitted for STA {}")

            self.first_pos_timestamp[sta_rx] = self.sta_queue_timeline[sta_rx][self.first_pos_position[sta_rx]]

        # Update the number of times each STA is selected    
        self.sta_selection_counter[sta_rx] += 1

    def update_channel_matrix(self, sta_matrix=None):
        """
        Updates the channel matrix, likely due to mobility.
        """
        channel_matrix = get_channel_matrix(
        self.sim_config['max_tx_power_dbm'], 
        self.ap_matrix,
        sta_matrix,
        self.sim_config['scenario_type'],
        self.sim_config['walls']
        )
        return  channel_matrix


    def update_cgs_and_tx_power(self):
        """
        Updates the CGs and transmission power matrix.
        """
        map_matrix, tx_power_matrix_temp, self.comb_ok = cg_creation_tpc(
        self.association, 
        self.channel_matrix_fading, 
        self.sim_config['max_tx_power_dbm'], 
        self.nsc, 
        self.sim_config['filtering'], 
        tpc_method=self.sim_config['tpc_method'], # TPC Optimization method: None, 'PSO'
        cg_size=self.sim_config['cg_size']
        )
        match self.simulation_system: 
            case 'csr':                                          
                self.tx_power_matrix = [row.tolist() for i, row in enumerate(tx_power_matrix_temp) if self.comb_ok[i]]
                self.cgs_stas = [row.tolist() for i, row in enumerate(map_matrix) if self.comb_ok[i]]
            case 'rl':
                self.tx_power_matrix = tx_power_matrix_temp
                self.cgs_stas = map_matrix

    def backoff(self):
        """
        Executes the backoff process.
        """
        if  self.simulation_system == 'edca':
            Tc = SYSTEM.TRTS + SYSTEM.TSIFS + SYSTEM.TCTS + self.aifs + SYSTEM.TE;      # collision duration
        elif (self.simulation_system == 'csr') or (self.simulation_system == 'rl'):
            Tc = SYSTEM.TMAPC_ICF + SYSTEM.TSIFS + SYSTEM.TMAPC_ICR + self.aifs + SYSTEM.TE;      # collision duration
        
        # Extract the minimum backoff value among the APs with packets
        aps_with_packets = np.where(self.aps_packet_indicator)[0]
        # print(f'backoff values = {self.backoff_values}')
        # print(f'APs with packets = {aps_with_packets}')

        slotnum = np.min(self.backoff_values[aps_with_packets])

        self.backoff_values[aps_with_packets] -= slotnum
        idx = np.where(self.backoff_values[aps_with_packets] == 0)[0]

        collision_counter = 0
        while True:
            if len(idx) == 1:
                self.txop_winner = aps_with_packets[idx[0]]
                # print(f'TXOP winner - AP{self.txop_winner}')
                self.txop_win_number[self.txop_winner] += 1
                self.backoff_values[self.txop_winner] = np.random.randint(0, self.cw_min)
                self.backoff_stage[self.txop_winner] = 0
                # print(f'Slot number = {slotnum}')
                # print(f'Number of collisions = {collision_counter}')
                # print(f'backoff values = {self.backoff_values}')
                # print(f'backoff stage = {self.backoff_stage}')
                # print('----------------------------------')

                return slotnum * 9e-6 + collision_counter * Tc, self.txop_winner
            else:
                # print(f'Collisioned APs = {idx}')
                for i in idx:
                    ap = aps_with_packets[i]
                    self.txop_win_number[ap] += 1
                    self.txop_collision[ap] += 1
                    if self.backoff_stage[ap] < self.max_backoff_stage:
                        self.backoff_stage[ap] += 1
                    self.backoff_values[ap] = np.random.randint(0, self.cw_min * (2 ** self.backoff_stage[ap]))
                    # print(f'Selected backoff value: {self.backoff_values[aps_with_packets[i]]}')
                    
                
                # print(f'backoff values = {self.backoff_values}')
                slotnum += np.min(self.backoff_values[aps_with_packets])
                collision_counter += 1
                self.backoff_values[aps_with_packets] -= np.min(self.backoff_values[aps_with_packets])
                idx = np.where(self.backoff_values[aps_with_packets] == 0)[0]
                # print(f'backoff values = {self.backoff_values}')
                # print('')

    def get_queue(self, sta):
        """ Vector with the available packets to be transmitted to a given STA at sim_timeline. 
        Returns:
        tx_vector_pos : list   -----   List of the available packets to be transmitted to sta at sim_timeline
        
        """

        # Find the first packet available to be transmitted 
        first_pos = self.first_pos_position[sta]

        # Find the last packet available to be transmitted
        last_pos = np.where(self.sta_queue_timeline[sta] <= self.sim_timeline)[0][-1]

        # Find the vector of packets that can be transmitted
        packet_range = range(first_pos, last_pos + 1)
        tx_vector_pos = [p for p in packet_range if self.sta_queue_state[sta][p] == True]

        return tx_vector_pos

    def scheduling_v1(self, agent_decision=None):
        """
        Scheduling logic.
        
        Returns:
        sta_rx : list   -----   List of STAs to be served
        aps : list      -----   List of the corresponding APs to served the STAs in sta_rx
        """
        # If an external agent decision is provided
        if agent_decision is not None:
            sta_rx, aps = agent_decision
        else:
            if self.simulation_system == 'edca':
                # STA selection based on the oldest packet
                sta_idx = np.argmin(self.first_pos_timestamp[self.association[self.txop_winner]])
                sta_rx = np.atleast_1d(self.association[self.txop_winner][sta_idx])

                aps = np.atleast_1d(self.txop_winner)
            elif self.simulation_system == 'csr':

                # Initialize variables
                cgs = copy.deepcopy(self.cgs_stas)
                per_sta_score_packets = np.zeros(self.sta_number, dtype=int)
                placeholder = -1 # Placeholder for inactive STAs

                # Identify inactive STAs and update CGs
                inactive_stas = np.where(self.first_pos_timestamp > self.sim_timeline)[0]
                inactive_set = set(inactive_stas)
                for cg in cgs:
                    cg[:] = [placeholder if sta in inactive_set else sta for sta in cg]

                # Calculate packets for all STAs
                per_sta_score_packets = np.array([len(self.get_queue(j)) if j not in inactive_set else 0 for j in range(self.sta_number)])

                # Scoring and unique STA extraction
                uni = []                           # Unique STAs
                score_packets = []                  # Score based on the number of packets
                score_time_oldest = []               # Score based on the oldest packet (timestamp)
                score_tat = []                      # Score based on TAT

                # Iterate through CGs that contain valid STAs
                for cg in cgs:
                    u = [sta for sta in cg if sta != placeholder]
                    allow = True
                    if len(cg) > 2:
                        allow = any([True if np.array_equal(x, u) else False for x in self.cgs_stas])
                        
                    if not u or not allow:
                        continue

                    # u = np.array(active_u)
                    uni.append(u)
                    score_packets.append(np.sum(per_sta_score_packets[u]))
                    score_time_oldest.append(np.min(self.first_pos_timestamp[u]))

                    if self.scheduler in ['tat', 'hybrid']:
                        ei_min = np.min(self.first_pos_timestamp[u])
                        ei_max = np.max(self.first_pos_timestamp[u])
                        delta_nt = self.sim_timeline - ei_min
                        Delta_nt = self.sim_timeline - ei_max
                        score_tat.append(
                            delta_nt if len(u) == 1 else delta_nt + self.beta * (Delta_nt - self.alpha * delta_nt)
                        )

                if self.scheduler == 'mnp':  # Maximum number of packets
                    max_score = max(score_packets)
                    idx_score = np.argmax(score_packets)
                    equal_score_idx = [i for i, score in enumerate(score_packets) if score == max_score]

                    if len(equal_score_idx) != 1:
                        idx_score = equal_score_idx[np.argmin([score_time_oldest[i] for i in equal_score_idx])]
                elif self.scheduler == 'op': # Oldest packet
                    min_oldest_score = min(score_time_oldest)
                    idx_score = np.argmin(score_time_oldest)
                    equal_score_idx = [i for i, score in enumerate(score_time_oldest) if score == min_oldest_score]

                    if len(equal_score_idx) != 1:
                        idx_score = equal_score_idx[np.argmax([score_packets[i] for i in equal_score_idx])]
                elif self.scheduler == 'random':
                    idx_score = np.random.randint(len(uni))
                elif self.scheduler == 'tat': # Traffic-Alignment Tracker
                    max_score = max(score_tat)
                    idx_score = np.argmax(score_tat)
                    equal_score_idx = [i for i, score in enumerate(score_tat) if score == max_score]
                    if len(equal_score_idx) != 1:
                        idx_score = equal_score_idx[np.argmax([score_packets[i] for i in equal_score_idx])]

                sta_rx = uni[idx_score]

                # compute the corresponding APs which the STAs in sta_rx are associated with
                aps = get_association(self.association, sta_rx)

        return sta_rx, aps

    def tx_time_calc(self, sta_rx, aps, data_tx_time):
        """
        Transmission time calculation.
        """
        # Initialize variables
        # agg_packets = np.zeros(len(sta_rx), dtype=int)
        # tx_Packets = np.zeros(len(sta_rx), dtype=int)
        temp_elapsed_time = np.zeros(len(sta_rx))
        agg_packets_group = []  # List to store the number of aggregated packets for each STA

        # Channel matrix for the corresponding APs and sta_rx
        channel_matrix_reduced = self.channel_matrix_fading[np.ix_(sta_rx, aps)]

        # If sta_rx == 1, no TPC
        if len(sta_rx) == 1:
            popt = self.max_tx_power
        else:
            try:
                idx = next(i for i, cg in enumerate(self.cgs_stas) if np.array_equal(cg, sta_rx))
            except StopIteration:
                raise ValueError('Not found this stas in the available cgs') # sta_rx not found
            
            popt = self.tx_power_matrix[idx]

        # Computing the SINR in dB 
        sinr_db = 10 * np.log10((popt * np.diag(channel_matrix_reduced)) / (self.noise_power + np.sum(channel_matrix_reduced * popt, axis=1) - np.diag(channel_matrix_reduced) * popt))
        
        for k, sta in enumerate(sta_rx):
            # Compute MCS and related parameters
            mcs, n_bps, Rc = mcs_cal_PER_001(sinr_db[k])
            
            # Fetch the queue of available packets for the current STA
            tx_vector_pos = self.get_queue(sta)

            if mcs != -1:
                Pe = 1e-2  # Packet error probability for valid MCS
            else:
                Pe = 1  # All packets are lost

            agg_packets = tx_packets(self.nsc, n_bps, Rc, data_tx_time)
                          
            # Determine the number of packets to transmit
            tx_Packets = min(len(tx_vector_pos), agg_packets)

            # Updating the tx_vector_pos considering the minimum between the available packets and the packets that can be aggregated
            tx_vector_pos = tx_vector_pos[:tx_Packets]

            if Pe < 1:  # If transmission is possible
                # Packets successfully transmitted
                received_packets = np.random.binomial(tx_Packets, 1 - Pe)
                lost_packets = np.random.choice(tx_vector_pos, tx_Packets - received_packets, replace=False)
                rx_vector_pos = [p for p in tx_vector_pos if p not in lost_packets]
            else:
                rx_vector_pos = []  # No packets received

            # Compute transmission time
            temp_elapsed_time[k] = elapsed_time_tx(self.nsc, n_bps, Rc, tx_Packets)

            # Update the number of transmitted packets in the current TXOP
            self.per_txop_sta_tx_packets[sta] = tx_Packets if Pe < 1 else -1
            agg_packets_group.append(agg_packets)  # Store the number of aggregated packets

            # Update STA state
            self.update_sta(sta, rx_vector_pos, temp_elapsed_time[k])
        
        self.throughput_txop = sum(self.per_txop_sta_tx_packets[np.where(self.per_txop_sta_tx_packets>0)]) / (self.info_overheads_csr + self.pre_tx_overheads_csr + max(temp_elapsed_time) + SYSTEM.TSIFS + SYSTEM.TBACK + self.aifs + SYSTEM.TE)  # Throughput of the TXOP

        return max(temp_elapsed_time)
    
    def apply_fading(self, fading_type="rician", K_dB=10):
        """
        Apply fast fading to a base channel matrix.
        
        Parameters:
        - channel_matrix: matrix with pathloss-based channel gains
        - fading_type: 'rayleigh' or 'rician'
        - K_dB: Rician K-factor in dB (ignored if rayleigh)

        Returns:
        - faded_channel_matrix: same shape with fast fading applied
        """
        shape = self.channel_matrix.shape

        if fading_type == "rayleigh":
            fading = (np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape)) / np.sqrt(2)
        elif fading_type == "rician":
            K = 10**(K_dB / 10)
            s = np.sqrt(K / (K + 1))   # LOS component
            sigma = np.sqrt(1 / (2 * (K + 1)))  # NLOS component

            fading = (
                s + np.random.normal(0, sigma, shape) + 1j * np.random.normal(0, sigma, shape)
            )
        else:
            raise ValueError("Unsupported fading type.")

        self.channel_matrix_fading = self.channel_matrix * np.abs(fading) ** 2  # Power gain

        return 

    def traffic_analysis(self, warm_up_time=0.0):
        """Analysis of the results, considering only packets that arrived after warm_up_time.

        Parameters
        ----------
        warm_up_time : float, optional
            Packets with arrival time < warm_up_time are excluded from the analysis.
        """

        self.delay_per_sta = [[] for _ in range(self.sta_number)]
        self.delay_vector = []

        for j in range(self.sta_number):  # Per STA Analysis

            queue_timeline = self.sta_queue_timeline[j]
            delivery_record = self.delivery_timestamp_record[j]
            state_record = self.sta_queue_state[j]

            # ---- Filter packets that arrived after warm_up_time ----
            arrival_mask = queue_timeline >= warm_up_time
            filtered_queue = queue_timeline[arrival_mask]
            filtered_delivery = delivery_record[arrival_mask]
            filtered_state = state_record[arrival_mask]

            # Separate transmitted and non-transmitted packets
            transmitted_mask = (filtered_state == False)
            not_transmitted_mask = (filtered_state == True)

            # Ensure packets that arrived after the simulation stop are not considered pending
            not_transmitted_mask[filtered_queue > self.timestamp_to_stop] = False

            # Calculate delays for transmitted packets
            transmitted_delays = filtered_delivery[transmitted_mask] - filtered_queue[transmitted_mask]

            if np.any(transmitted_delays <= 0):
                raise ValueError("Delay cannot be negative or zero for transmitted packets")

            # For untransmitted packets (pending)
            pending_arrivals = filtered_queue[not_transmitted_mask]
            pending_delays = self.timestamp_to_stop - pending_arrivals

            if np.any(pending_delays <= 0):
                raise ValueError("Pending packet delay is negative or zero; check this.")

            # Concatenate both
            all_delays = np.concatenate((transmitted_delays, pending_delays))

            self.delay_per_sta[j] = all_delays
            self.delay_vector = np.concatenate((self.delay_vector, all_delays))

        # Per AP analysis (unchanged)
        for jj in range(self.ap_number):
            if self.txop_win_number[jj] == 0:
                self.ap_collision_prob[jj] = 0
            else:
                self.ap_collision_prob[jj] = self.txop_collision[jj] / self.txop_win_number[jj]

    def init_settings(self):
        """Initialization. Restarts the parameters to start a new simulation"""
     
        if self.access_category == 'VO':
            self.cw_min = 4
            self.max_backoff_stage = 1
            aifsn = 2
        elif self.access_category == 'VI':
            self.cw_min = 8
            self.max_backoff_stage = 1
            aifsn = 2
        elif self.access_category == 'BE':
            self.cw_min = 16
            self.max_backoff_stage = 6
            aifsn = 3
        else:
            raise ValueError("Invalid access category")
    
        # Simulation related
        self.sim_timeline = 0

        # Traffic-related
        self.first_pos_timestamp = np.zeros((self.sta_number,), dtype=float)     # Stores the timestamp
        self.first_pos_position = np.zeros((self.sta_number,), dtype=int)      # Stores the position of the first non-transmitted packet of each STA
        self.delivery_timestamp_record = []                               # Stores the delivery time of the packets of all STAs
        self.sta_queue_state = []                                       # Stores the state of the packets of all STAs. True = not transmitted, False = transmitted

        # Initialize queues and traffic-related vars
        for i, timeline in enumerate(self.sta_queue_timeline):
            # Convert and filter
            timeline = np.array(timeline, dtype=float)
            pos = np.where(timeline > self.timestamp_to_stop)[0][0]
            timeline = timeline[:pos+1]  # ✅ Remove late arrivals

            # Ensure the last packet arrival time is >= timestamp_to_stop
            if timeline[-1] < self.timestamp_to_stop:
                raise ValueError("The last packet arrival time must be greater than or equal to timestamp_to_stop.")

            actual_length = len(timeline)

            # Initialize state arrays with correct length
            self.delivery_timestamp_record.append(np.full(actual_length, np.nan, dtype=float))
            self.sta_queue_state.append(np.ones(actual_length, dtype=bool))

            # Other assignments
            self.first_pos_timestamp[i] = timeline[0] if actual_length > 0 else np.nan
            self.sta_queue_timeline[i] = timeline

        # Scenario-related
        self.channel_matrix_fading = self.channel_matrix.copy()  # Copy the channel matrix for fading
        self.channel_matrix_last_estimation = self.channel_matrix.copy()  # Initial channel matrix estimation

        # Mobility-related
        self.last_channel_idx = int(0)
  
        # backoff-related
        self.aps_packet_indicator = np.zeros((self.ap_number,), dtype=bool)  # Vector indicating whether each AP has packets to transmit
        self.backoff_values = np.random.randint(0, self.cw_min, size=self.ap_number)
        self.backoff_stage = np.zeros((self.ap_number,), dtype=int)           # backoff stage for each AP
        self.aifs = aifsn * 9e-6 + 16e-6
        

        # TXOP-related
        self.txop_win_number = np.zeros((self.ap_number,), dtype=int)          # Number of TXOP wins for each AP
        self.txop_collision = np.zeros((self.ap_number,), dtype=int)          # Number of TXOP collisions for each AP
        

        # Results
        self.last_txop_queue_sizes = np.zeros(self.sta_number, dtype=int)  # Queue sizes at the last TXOP
        self.per_txop_sta_tx_packets = np.zeros((self.sta_number,), dtype=int)          # Number of packets transmitted per STA per TXOP
        self.sta_selection_counter = np.zeros((self.sta_number,), dtype=int)    # Counter for the number of times each STA is selected
        self.throughput_sim = np.zeros((self.sta_number,), dtype=float)                    # Throughput of each STA       
        self.ap_collision_prob = np.zeros((self.ap_number,), dtype=float)      # Collision probability of each AP

        self.suc_txops = int(0)  # Counter for successful transmissions
        self.priority_selection_counter = np.zeros((self.sta_number,), dtype=int)  # Priority selection for each STA

    def sim_forward(self):
        """ Forward the simulation just right before the scheduling process. """

        if min(self.first_pos_timestamp) > self.sim_timeline:
            self.sim_timeline = min(self.first_pos_timestamp)
        
        # Update APs_packet_indicator vector to indicate whether each AP has packets to transmit
        self.update_ap()

        # backoff process
        backofftime, self.txop_winner = self.backoff()

        # Update simulation timeline with backoff time and backoff collision time
        self.sim_timeline += backofftime

        if self.simulation_system in ['csr', 'rl']:
            self.sim_timeline += self.info_overheads_csr           # ICF + SIFS + ICR + SIFS

    def run_step(self, agent_decision=None):
        """Run a single step of the simulation, optionally with external intervention."""
        
        # Reset the number of packets transmitted per STA per TXOP and the total queue size
        self.per_txop_sta_tx_packets = np.zeros((self.sta_number,), dtype=int)
        self.throughput_txop = 0.0
        self.last_txop_queue_sizes = np.array([len(self.get_queue(sta)) if self.first_pos_timestamp[sta] <= self.sim_timeline else 0 for sta in range(self.sim_config['sta_number'])])

        delay = np.array([self.sim_timeline - self.first_pos_timestamp[sta] if self.first_pos_timestamp[sta] <= self.sim_timeline else 0.0 for sta in range(self.sta_number)])
        ordered = np.argsort(delay)
        

        # Verify whether agent_decision is None (non-agent decision) or is not empty (agent decision with valid decision) 
        if (agent_decision not in [None, [], [[], []]]) or self.simulation_system in ['edca', 'csr']:
   
            # Scheduling
            sta_rx, APs = self.scheduling_v1(agent_decision)

            self.suc_txops += 1  # Increment the successful TXOP counter
            pos_priority = [np.where(ordered==sta)[0] for sta in sta_rx]
            self.priority_selection_counter[pos_priority] += 1

            # Pre-TX overheads and data transmission time calculation
            if self.simulation_system == "edca":
                self.sim_timeline += self.pre_tx_overheads_edca
                data_tx_time = self.txop_duration - self.edca_overheads
            elif self.simulation_system in ['csr', 'rl']:
                self.sim_timeline += self.pre_tx_overheads_csr
                data_tx_time = self.txop_duration - self.csr_overheads

            
            # Compute the transmission time and update the simulation timeline
            elapsed_time = self.tx_time_calc(sta_rx, APs, data_tx_time) # Transmission time calculation

            if elapsed_time > data_tx_time:
                raise ValueError("Transmission time cannot be greater than the allowed transmission time")

            self.sim_timeline += elapsed_time + SYSTEM.TSIFS + SYSTEM.TBACK + self.aifs + SYSTEM.TE # elapsed time + SIFS + ACK + AIFS + Te
            
            
            # # Mobility
            if self.mobility_config:
                cur_idx = int(math.floor(self.sim_timeline / self.ch_realization_duration + 1e-12))
                if cur_idx != self.last_channel_idx and self.sim_timeline <= self.timestamp_to_stop:
                    self.last_channel_idx = cur_idx
                    self.channel_matrix_fading = self.update_channel_matrix(self.sta_mobility[cur_idx])
                    
                    if self.simulation_system in ['csr', 'rl']:
                        if cur_idx % self.ch_realizations_per_update == 0:  # Time to update CGs and Tx power
                            self.channel_matrix_last_estimation = self.channel_matrix_fading.copy()
                            self.update_cgs_and_tx_power()         
        return 
        
    def run(self):
        """Simulation process."""

        while self.sim_timeline < self.timestamp_to_stop:

            self.sim_forward() # Forward the simulation just right before the scheduling process.
            self.run_step() # Run a single step of the simulation, optionally with external agent decision.

        self.traffic_analysis()
