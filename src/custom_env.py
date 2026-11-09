from typing import List, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from traffic_generator import traffic_generator
from utils import get_association, cg_creation_tpc, get_channel_coefficient, get_channel_coefficient_bss, generate_sta_mobility, plot_mobility_trajectories
from deployment_generator import deployment_generator
import pandas as pd
import matplotlib.pyplot as plt
from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, traffic_config, sim_config, learning_config, simulator):
        super().__init__()

        # Simulation configuration
        self.sim_config = sim_config
        self.traffic_config = traffic_config

        self.training_flag = sim_config["training_flag"]  # Store flag

        # Set the duration of each episode
        self.timestamp_to_stop = self.sim_config['timestamp_to_stop']

        # Loading the simulator into the environment
        self.simulator = simulator

        # Compute assoc_ids once using your helper
        n_aps = self.sim_config['ap_number']      # number of aps
        n_stas = self.sim_config['sta_number']    # number of STAs

        # Precompute one-hot as well
        self.assoc_onehot = np.eye(n_aps)[self.association()]  # shape (n_stas, n_aps)


        # Define the action space
        self.action_space = spaces.Discrete(len(self.simulator.comb_ok))  # Number of valid actions

        # Initialize the state
        self.delays = np.zeros(n_stas, dtype=float)                      # delays
        self.queue_sizes = np.zeros(n_stas, dtype=float)                 # queue sizes
        # self.channel_coef = np.zeros(self.sim_config['sta_number'], dtype=float)
        self.channel_coef = np.zeros(n_stas*n_aps, dtype=float)


        # obs space
        # self.observation_space = spaces.Box(low=0, high=1, shape=(sim_config['sta_number']*3,), dtype=float)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2*n_stas + len(self.channel_coef),), dtype=float)

        self.step_number = int(0)

        self.agent_decision = [[], []]  # Initialize agent decision
 
        # Masking the actions
        self.mask = np.array(self.simulator.comb_ok, dtype=bool)
        self.valid_indices = np.flatnonzero(self.mask)

        self.reward_shaping: list = []
        self.long_term_reward: list = []
        self.reward: list = []

        self.w_long_term: float = learning_config['w_long_term']

        self.window_size: int = learning_config['window_size']
        self.historical_delay: list = []
        self.reward_records: list = []

        self.training_episode_counter = int(0)  # Counter for the number of episodes
        self.episode_counter_threshold =  learning_config['episode_threshold']  # Number of episodes to wait before generating a new deployment

    def reset(self, seed=None, stas_arrivals_matrix=None, is_deployment_fixed=False):
        """
        Resets the environment to the initial state. Get the observation in the initial state
        """

        # Seed the environment if seed is provided
        super().reset(seed=seed)  #
        
        ########### deployment generation 
        if (self.training_episode_counter > self.episode_counter_threshold) or (self.training_flag == False) and (not is_deployment_fixed):
            # Generate a new deployment
            self._deployment_generator(seed=seed)

            if self.training_flag:
                self.training_episode_counter = int(0)  # Reset the episode counter

        if self.training_flag:
            self.training_episode_counter += 1  # Increment the episode counter


        ########### traffic generation
        if stas_arrivals_matrix is None:
            # Set the seed
            # np.random.seed(seed)
            # traffic_profile_per_sta = np.random.choice(['A','B'], size=self.sim_config['sta_number']).tolist()
            traffic_profile_per_sta = [
                    {
                        'traffic_load': np.random.uniform(self.traffic_config['load_min'], self.traffic_config['load_max']),  # Load in Mbps
                        'traffic_model': str(np.random.choice(['poisson', 'bursty']))  # Traffic model
                    }
                    for i in range(self.sim_config['sta_number'])
            ]

            stas_arrivals_matrix = traffic_generator(
                self.traffic_config,
                self.sim_config,
                traffic_profile_per_sta,
                )  

        # Validate traffic
        if any(stas_arrivals_matrix[i][-1] < self.timestamp_to_stop for i in range(self.sim_config['sta_number'])):
            raise ValueError(f"Traffic should last more than {self.timestamp_to_stop} seconds")                  

        # Loading the traffic dataset into the buffers in the simulator
        self.simulator.sta_queue_timeline = stas_arrivals_matrix

        # Reset the simulator (initialize the settings)
        self.simulator.init_settings()

        # Advance until the first event
        self.simulator.sim_forward()

        # Initialize the state
        self.delays = np.zeros(self.sim_config['sta_number'], dtype=float)                      # delays
        self.queue_sizes = np.zeros(self.sim_config['sta_number'], dtype=float)                 # queue sizes
        self.agent_decision = [[], []]  # Initialize agent decision  

        # Masking the bad actions (actions that are not possible independently of the queue ocupancy)
        self.mask = np.array(self.simulator.comb_ok, dtype=bool)
        self.valid_indices = np.flatnonzero(self.mask)

        self.reward_shaping = []
        self.long_term_reward = []
        self.reward = []

        # Get the observation
        obs = self.get_state()

        self.last_txop_timestamp = np.min(self.simulator.first_pos_timestamp)
        self.step_number = int(0)

        self.historical_delay = [self.simulator.sim_timeline-self.last_txop_timestamp]
        self.reward_records = []
        # validation of historical delay initialization
        if self.historical_delay[0] <= 0:
            raise ValueError(
                "Initial historical delay must contain positive values.")

        # Optionally we can pass additional info
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        Executes one time step in the simulator and returns:
        - next_state: The new state of the environment.
        - reward: The reward received for the action.
        - done: Whether the episode has ended.
        """

        # Get the action
        self.get_action(action)

        # increase the step number
        self.step_number += 1
        
        # Execute the action
        self.simulator.run_step(self.agent_decision)
            
        # Forward the simulation
        self.simulator.sim_forward()

        # Get the observation
        obs = self.get_state()

        # Get the reward
        reward = self.get_reward()

        # Check termination conditions
        terminated = truncated = bool(self.simulator.sim_timeline >= self.timestamp_to_stop)

        info = {}
        if terminated or truncated:
            self.simulator.traffic_analysis()
            prctile99 = [np.percentile(self.simulator.delay_per_sta[sta],99) for sta in range(self.sim_config['sta_number'])]

            info['total_percentile99'] = np.percentile(self.simulator.delay_vector,99)
            info['worst_percentile99'] = max(prctile99)
            info['mean_rew_shaping'] = np.mean(self.reward_shaping)
            info['mean_long_term_rew'] = np.mean(self.long_term_reward)
            info['mean_reward'] = np.mean(self.reward)

            # Uncomment the following line to plot the reward trend at the end of each episode
            # self.plot_reward_trend()
        return obs, reward, terminated, truncated, info
    
    def get_action(self, action):
        """
        Get the action.
        Returns:
            agent_decision (list): The action. [sta_rx, aps] where sta_rx is the list with the index of the STAs that will be served and 
                                                                       aps is the list of aps that will transmit.
        """
        uni = self.simulator.cgs_stas[action]

        if uni is None:    # EMPTY ACTION, TODO: IT COULD BE USED FOR WAITING MORE TIME TO ALLOW THE QUEUES TO FILL
            # MaskeablePPO should not select an empty action, but if it happens, we can handle it here
            ValueError("Empty action received, this should not happen. The agent should always choose a valid action.")
            self.agent_decision = None
        else:
            sta_rx = [sta for sta in uni if self.simulator.first_pos_timestamp[sta] <= self.simulator.sim_timeline]
            aps = get_association(self.simulator.association, sta_rx)
                
            # Agent decision to be passed to the simulator
            self.agent_decision = [sta_rx, aps]

        return self.agent_decision

    def get_state(self):
        """
        Get the observation.
        Returns:
            observation : The observation.
        """
        # # Queue sizes Normalized 
        self.queue_sizes = np.array([min(len(self.simulator.get_queue(sta))/1E4,1) if self.simulator.first_pos_timestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['sta_number'])])
 
        # Delays
        self.delays = np.array([(self.simulator.sim_timeline - self.simulator.first_pos_timestamp[sta]) / float(self.timestamp_to_stop) if self.simulator.first_pos_timestamp[sta] <= self.simulator.sim_timeline else 0.0 for sta in range(self.sim_config['sta_number'])])
        
        ### Channel coefficients
        # Consider only self channel coefficients, i.e., the channel coefficients of the STAs with their associated APs
        # self.channel_coef = [r / 0.005 for r in get_channel_coefficient_bss(self.simulator.channel_matrix_last_estimation, range(self.sim_config['sta_number']), get_association(self.sim_config['association'], range(self.sim_config['sta_number'])))] # 0.005 is the minimum channel coeficient, i.e., no walls, distance= 1m
        
        # Consider all channel coefficients, i.e., the channel coefficients of the STAs with all APs
        self.channel_coef = [r / 0.005 for r in get_channel_coefficient(self.simulator.channel_matrix_last_estimation)] # 0.005 is the minimum channel coeficient, i.e., no walls, distance= 1m


        obs = np.concatenate(
            (self.delays, self.queue_sizes, self.channel_coef), 
            axis=0
        ).astype(np.float32)

        return obs
    
    def get_reward(self):
        """ Compute the reward """

        ################################################################
        # # # Compute the short term reward (for reward shaping)
        reward_shaping = np.min(self.simulator.first_pos_timestamp) - self.last_txop_timestamp   

        # Debugging for negative reward shaping
        if (reward_shaping < 0) and (reward_shaping < -1E-6):   # tiny tolerance
            print("NEGATIVE reward_shaping detected!")            
            print(" Reward shaping:", reward_shaping)
            print(" step_number:", self.step_number)
            print(" last_txop_timestamp (prev):", self.last_txop_timestamp)
            print(" current_min_firstPosTimestamp:", np.min(self.simulator.first_pos_timestamp))
            print(" full firstPosTimestamp array:", self.simulator.first_pos_timestamp)

        # Update last txop timestamp
        self.last_txop_timestamp = np.min(self.simulator.first_pos_timestamp)
        
        # long term reward
        current_worst_delay = self.simulator.sim_timeline - np.min(self.simulator.first_pos_timestamp)
        D_cutoff, k = 0.1, 5        # scaled to [-1, 0.23]
        long_term_reward = self.w_long_term*(2 / (1 + np.exp(k * (current_worst_delay - D_cutoff))) - 1)
        # long_term_reward = min(0.001/(current_worst_delay + 1E-6),1.0)   # paper version (out of date)


        delay_reward = reward_shaping + long_term_reward
        reward = delay_reward

        # Store results for plotting
        self.historical_delay.append(current_worst_delay)
        self.reward_records.append({
            'step': self.step_number,
            'mean': np.mean(self.historical_delay[-self.window_size:]),
            'new_point': current_worst_delay,
            'reward': reward
        })

        ################################################################
        ### Debugging information
        # print(f"Step: {self.step_number} \n\
        #       STAs - aps: {self.agent_decision} \n\
        #       Queue sizes: {self.queue_sizes} \n\
        #       Delays: {[(self.simulator.sim_timeline - self.simulator.first_pos_timestamp[sta]) if self.simulator.first_pos_timestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['sta_number'])]} \n\
        #       Transmitted packets: {self.simulator.per_txop_sta_tx_packets} \n\
        #       Delay reward: {delay_reward} \n\
        #       Current worst delay: {self.simulator.sim_timeline - np.min(self.simulator.first_pos_timestamp)} \n\
        #       Timestamp: {self.simulator.sim_timeline} \n\
        #       Total reward: {reward} \n\
        # -----------------------------------------------------------------------------------------------------")

        # Store rewards
        self.reward_shaping.append(reward_shaping)
        self.long_term_reward.append(long_term_reward)
        self.reward.append(reward)

        return reward 

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def action_masks(self) -> List[Any]:
        """
        Updates the action masks according to the environment's state.
        """
        # Mask the actions that are not possible considering also the queue occupancy
        mask = self.mask.copy()   # mask is a boolean array indicating valid actions, at this point it already masks bad actions
        no_queues = [i for i in self.valid_indices[:-1] if np.all(self.queue_sizes[self.simulator.cgs_stas[i]] == 0)]   # indices of actions with no packets in all the participants in the group
        mask[no_queues] = 0

        # Ensure at least one valid action remains
        if not np.any(mask):
            raise ValueError("No valid actions available in the current state.")
    
        return mask.tolist()
    
    def _deployment_generator(self, seed=None):
        """
        Generates a new deployment.
        """
        # Generate a new deployment
        _, sta_matrix, self.sim_config['association'], channel_matrix = deployment_generator(self.sim_config, seed)
        
        if self.simulator.mobility_config:
            sta_mobility = generate_sta_mobility(
                self.sim_config['ap_matrix'],
                sta_matrix,
                self.sim_config['walls'],
                self.sim_config['grid_value'],
                self.sim_config['association'],
                self.sim_config['timestamp_to_stop'],
                self.simulator.mobility_config['ch_realization_duration'],
                self.simulator.mobility_config['speed'],
                min_dist_to_ap=1.0,
                max_attempts=30,
                rng=np.random.default_rng(seed),
                )
            # plot_mobility_trajectories(sta_mobility, self.sim_config['ap_matrix'], self.sim_config['association'], self.sim_config['walls'], self.sim_config['grid_value'])
        else:
            sta_mobility = None
        
        # mobility assignment
        self.simulator.sta_mobility = sta_mobility

        # Update channel matrix in the simulator
        self.simulator.channel_matrix = channel_matrix  # Channel matrix

        _, tx_power_matrix_temp, comb_ok = cg_creation_tpc(
            self.sim_config['association'],
            channel_matrix,
            self.sim_config['max_tx_power_dbm'],
            self.sim_config['nsc'],
            is_filtering=self.sim_config['filtering'], 
            tpc_method=self.sim_config['tpc_method'], # TPC Optimization method: None, 'PSO'
            cg_size=self.sim_config['cg_size']
        )

        self.simulator.tx_power_matrix = tx_power_matrix_temp
        self.simulator.comb_ok = comb_ok

    def association(self):
        n_aps = self.sim_config['ap_number']      # number of aps  
        n_stas = self.sim_config['sta_number']    # number of STAs

        assoc_ids = np.empty(n_stas, dtype=np.int32)

        for ap_id, sta_list in enumerate(self.sim_config['association']):
            for sta_id in sta_list:
                assoc_ids[sta_id] = ap_id

        return assoc_ids
    
    def plot_reward_trend(self):
        """
        Plots the reward trend over time.
        """

        pos_cum_reward = sum(r['reward'] for r in self.reward_records if r['reward'] > 0)
        neg_cum_reward = sum(r['reward'] for r in self.reward_records if r['reward'] < 0)
        print(f"\nCumulative Positive Reward: {pos_cum_reward:.3f}")
        print(f"Cumulative Negative Reward: {neg_cum_reward:.3f}")

        df = pd.DataFrame(self.reward_records)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # --- First subplot: delay trends ---
        axes[0].plot(df['step'], df['new_point'], label='New Delay', marker='o', markersize=4, alpha=0.7)
        axes[0].plot(df['step'], df['mean'], label='Rolling Mean', linestyle='--')
        axes[0].set_title("Delay Evolution Over Time")
        axes[0].set_ylabel("Delay")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Second subplot: reward ---
        axes[1].plot(df['step'], df['reward'], label='Reward', color='purple')
        axes[1].set_title("Reward Over Time")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Reward")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    
