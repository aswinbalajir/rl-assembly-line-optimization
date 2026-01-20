# assembly_line_env.py

import random
from typing import Tuple, Dict, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation_model import AssemblyLineSim, ORDER_BOOK_SIZE, MINS_IN_DAY

class AssemblyLineEnv(gym.Env):
    """
    A Gymnasium environment for an Assembly Line Digital Twin.
    
    This environment simulates a discrete manufacturing process where an RL agent
    manages order scheduling, line halts, and overtime allocation to optimize
    throughput and minimize lateness.

    Attributes:
        metadata (dict): Render modes supported by the environment.
        step_duration (int): Simulation time elapsed per environment step (minutes).
        max_episode_steps (int): Maximum steps per episode before truncation.
    """
    
    metadata = {'render_modes': ['human']}

    # --- Configuration Constants ---
    STEP_DURATION = 6            # Minutes per step
    MAX_STEPS = 2000             # Approx 1 week  of production at 6 min/step
    
    # Normalization Constants
    BUFFER_CAPACITY_NORM = 10.0  # Assumed max buffer size for normalization
    DUE_DATE_LOOKAHEAD = 720.0   # 12 hours window for normalizing due dates
    URGENT_THRESHOLD = 240.0     # 4 hours threshold for dynamic priority
    
    # Reward Weights (Default)
    # DEFAULT_REWARD_CONFIG = {
    #     'completion_high_prio': 20.0,
    #     'completion_low_prio': 5.0,
    #     'penalty_late_high_prio': -5.0, # Net result: 5.0 - 5.0 = 0 if late
    #     'penalty_wip': 0.1,
    #     'penalty_overtime': 2.0,
    #     'penalty_skip_order': 0.05,
    #     'penalty_backlog_urgent': 0.005,
    #     'penalty_backlog_late': 0.05
    # }
    
    DEFAULT_REWARD_CONFIG = {
            'completion_high_prio': 50.0,      # Big reward for success
            'completion_low_prio': 10.0,       # Keep steady
            'penalty_late_high_prio': -60.0,   # CRITICAL: Net result is -10 (50 - 60) if late. 
                                               # The agent will now fear lateness more than overtime costs.
            'penalty_wip': 0.2,                # Punish hoarding parts in buffers
            'penalty_overtime': 0.75,           # Reflect higher labor costs
            'penalty_skip_order': 0.1,
            'penalty_backlog_urgent': 0.2,     # Panic signal
            'penalty_backlog_late': 1.0        # severe bleeding if ignored
        }
    def __init__(
        self, 
        randomize: bool = False, 
        part_mix: Optional[Dict] = None, 
        priority_mix: Optional[Dict] = None, 
        fail_rate: Optional[float] = None,
        reward_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Assembly Line Environment.

        Args:
            randomize (bool): If True, enables domain randomization on reset.
            part_mix (dict, optional): Distribution of part types.
            priority_mix (dict, optional): Distribution of order priorities.
            fail_rate (float, optional): Probability of station failure.
            reward_config (dict, optional): Custom weights for reward shaping.
        """
        super().__init__()
        self.step_duration = self.STEP_DURATION
        self.max_episode_steps = self.MAX_STEPS
        
        self.randomize = randomize
        self.simulation = AssemblyLineSim(
            part_mix=part_mix, 
            priority_mix=priority_mix, 
            fail_rate=fail_rate
        )
        
        # Load reward configuration
        self.reward_config = self.DEFAULT_REWARD_CONFIG.copy()
        if reward_config:
            self.reward_config.update(reward_config)

        # Action Space:
        # 0: Part Selection (0 to ORDER_BOOK_SIZE-1)
        # 1: Line Control (0 = Active/Release, 1 = Halt)
        # 2: Overtime Control (0 = Off, 1 = On)
        self.action_space = spaces.MultiDiscrete([ORDER_BOOK_SIZE, 2, 2])

        # Observation Space
        # 2 Buffer levels + (ORDER_BOOK_SIZE * 3 features) + 3 Time features
        obs_size = 2 + (ORDER_BOOK_SIZE * 3) + 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.current_step = 0

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation vector from the simulation state.

        Returns:
            np.ndarray: Normalized state vector.
        """
        obs_data, _ = self.simulation.get_kpis_and_state()
        
        # 1. Buffer State Normalization
        # Using self.simulation.BUFFER_CAPACITY ensures dynamic scaling if sim changes
        capacity = self.simulation.BUFFER_CAPACITY
        b12_level_norm = obs_data["buffer_12_level"] / capacity
        b23_level_norm = obs_data["buffer_23_level"] / capacity
        
        obs_vector = [b12_level_norm, b23_level_norm]
        
        # 2. Order Book Feature Extraction
        current_time = self.simulation.env.now
        
        for i in range(ORDER_BOOK_SIZE):
            if i < len(obs_data["order_book"]):
                part = obs_data["order_book"][i]
                
                # Feature A: Part Type ID (Normalized)
                type_id_norm = part['config']['type_id'] / 2.0
                
                # Feature B: Dynamic Priority
                # If due date is within URGENT_THRESHOLD (4 hours), escalate priority
                due_in = part['due_date'] - current_time
                is_urgent = (part['priority'] == 1) or (due_in <= self.URGENT_THRESHOLD)
                effective_priority = 1 if is_urgent else 2
                
                # Map priority 1->-1.0, 2->1.0 for clearer separation
                priority_norm = -1.0 if effective_priority == 1 else 1.0
                
                # Feature C: Time to Due Date (Clipped [-1, 1])
                time_to_due_norm = np.clip(due_in / self.DUE_DATE_LOOKAHEAD, -1.0, 1.0)
                
                obs_vector.extend([type_id_norm, priority_norm, time_to_due_norm])
            else:
                # Padding for empty book slots
                obs_vector.extend([0.0, 0.0, 0.0])
        
        # 3. Cyclical Time Features
        time_of_day = current_time % MINS_IN_DAY
        day_of_week = (current_time // MINS_IN_DAY) % 7
        
        obs_vector.extend([
            np.sin(2 * np.pi * time_of_day / MINS_IN_DAY),
            np.cos(2 * np.pi * time_of_day / MINS_IN_DAY),
            day_of_week / 6.0
        ])
        
        return np.array(obs_vector, dtype=np.float32)

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to an initial state.
        
        Performs domain randomization if self.randomize is True.
        """
        super().reset(seed=seed)
        
        if self.randomize:
            self._apply_domain_randomization()

        self.simulation.setup_simulation()
        self.current_step = 0
        
        return self._get_obs(), {}

    def _apply_domain_randomization(self):
        """Randomizes simulation parameters to improve policy robustness."""
        # Randomize Priority Mix
        if random.random() < 0.3:
            # Scenario: High High-Priority influx
            self.simulation.PRIORITY_MIX = {'HIGH': 0.8, 'LOW': 0.2}
        else:
            # Scenario: Random fluctuation
            p = random.uniform(0.1, 0.5)
            self.simulation.PRIORITY_MIX = {'HIGH': p, 'LOW': 1.0 - p}
            
        # Randomize Processing Times (e.g., Machine wear causing variance)
        self.simulation.PART_CONFIGS['Type_B']['s2_time'] = random.uniform(22, 28)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one time step within the environment.

        Args:
            action (np.ndarray): Array containing [Part Choice, Halt Flag, Overtime Flag].

        Returns:
            observation (np.ndarray): New state.
            reward (float): Scalar reward signal.
            terminated (bool): Whether the episode ended naturally.
            truncated (bool): Whether the episode hit the time limit.
            info (dict): Auxiliary diagnostic information.
        """
        part_index, halt_signal, overtime_signal = action
        
        # --- Action Execution ---
        
        # Halt Logic: 0 = Active (Release), 1 = Halted (No Release)
        # Note: set_source_status logic in Sim depends on implementation. 
        # Assuming set_source_status(False) enables flow, set_source_status(True) stops it.
        # Adjusted to strictly follow provided logic: bool(halt_signal).
        self.simulation.set_source_status(bool(halt_signal))
        
        # Overtime Logic: 0 = Off, 1 = On
        self.simulation.set_overtime_status(bool(overtime_signal))
        
        # Release Part Logic
        # Only attempt to release a part if the line is NOT halted (halt_signal == 0)
        if not bool(halt_signal):
            self.simulation.release_part(part_index)

        # Advance Simulation
        self.simulation.run(duration=self.STEP_DURATION)

        # --- State & Reward Calculation ---
        obs_data, results = self.simulation.get_kpis_and_state()
        observation = self._get_obs()
        reward = self._calculate_reward(part_index, overtime_signal, results, observation)

        # --- Termination Logic ---
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.MAX_STEPS
        
        info = {
            'newly_completed_parts': results["newly_completed_parts"],
            'buffer_12_level': obs_data['buffer_12_level'],
            'buffer_23_level': obs_data['buffer_23_level'],
            'overtime_active': bool(overtime_signal)
        }
        
        return observation, float(reward), terminated, truncated, info

    def _calculate_reward(
        self, 
        part_choice: int, 
        overtime_active: int, 
        results: Dict, 
        observation: np.ndarray
    ) -> float:
        """Calculates the reward based on completed parts, operational costs, and backlog."""
        reward = 0.0
        cfg = self.reward_config

        # 1. Throughput Rewards
        for part in results["newly_completed_parts"]:
            if part['priority'] == 1:
                # High Priority Logic
                if part['is_late']:
                    # Assuming a late high prio part yields 0 net reward or slight negative
                    # Logic: Reward for finishing + Penalty for lateness
                    reward += (cfg['completion_high_prio'] + cfg['penalty_late_high_prio'])
                else:
                    reward += cfg['completion_high_prio']
            else:
                # Low Priority Logic
                reward += cfg['completion_low_prio']

        # 2. FIFO Adherence Nudge
        # Penalize selecting indices deeper in the order book (Index 0 is oldest)
        reward -= (part_choice * cfg['penalty_skip_order'])

        # 3. Operational Costs
        # WIP Penalty (Sum of normalized buffer levels from observation)
        wip_level = observation[0] + observation[1] 
        reward -= (wip_level * cfg['penalty_wip'])
        
        if overtime_active:
            current_time = self.simulation.env.now
            time_of_day = current_time % (24 * 60)
            day_of_week = int((current_time // (24 * 60)) % 7)
            
            # Define the Valid Window: 6:30 PM (1110m) to 10:30 PM (1350m)
            # And only on Mon-Sat (Day < 6)
            START_OT = 18.5 * 60
            END_OT = 22.5 * 60
            
            is_valid_window = (day_of_week < 6) and (START_OT <= time_of_day < END_OT)
            
            if is_valid_window:
                # Useful Overtime: Standard cost
                reward -= cfg['penalty_overtime']
            else:
                # Wasted Overtime (e.g., 3 AM): Heavy Penalty to stop spamming
                reward -= (cfg['penalty_overtime'] * 5.0)

        # 4. Backlog Pressure
        # Penalize based on the urgency of parts remaining in the order book
        current_time = self.simulation.env.now
        backlog_penalty = 0.0
        
        for part in self.simulation.order_book:
            time_left = part['due_date'] - current_time
            
            if time_left < 0:
                # Part is already late
                backlog_penalty += cfg['penalty_backlog_late']
            elif time_left < 120:
                # Part is becoming urgent (2 hours)
                backlog_penalty += cfg['penalty_backlog_urgent']
        
        reward -= backlog_penalty
        
        return reward