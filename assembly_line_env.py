# In assembly_line_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation_model import AssemblyLineSim, ORDER_BOOK_SIZE
import random
from simulation_model import AssemblyLineSim, ORDER_BOOK_SIZE, MINS_IN_DAY, MINS_IN_WEEK

class AssemblyLineEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, randomize=False, part_mix=None, priority_mix=None, fail_rate=None):
        super().__init__()
        self.randomize = randomize
        self.simulation = AssemblyLineSim(part_mix=part_mix, priority_mix=priority_mix, fail_rate=fail_rate)
        
        # MODIFIED: Action space now includes the overtime decision
        # [part_choice, flow_control, overtime_choice]
        self.action_space = spaces.MultiDiscrete([ORDER_BOOK_SIZE, 2, 2])

        # MODIFIED: Observation space now includes time of day and day of week
        # [buffer_lvls, order_book_features, time_of_day, day_of_week]
        obs_size = 2 + (ORDER_BOOK_SIZE * 3) + 3
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)
        
        self.step_duration = 60
        self.max_episode_steps = 1000
        self.current_step = 0

    def _get_obs(self):
        obs_data, _ = self.simulation.get_kpis_and_state()
        
        # 1. Normalize buffer levels (2 features)
        b12_level_norm = obs_data["buffer_12_level"] / self.simulation.BUFFER_CAPACITY
        b23_level_norm = obs_data["buffer_23_level"] / self.simulation.BUFFER_CAPACITY
        
        obs_vector = [b12_level_norm, b23_level_norm]
        
        # 2. Create feature vectors for the order book (30 features)
        for i in range(ORDER_BOOK_SIZE):
            if i < len(obs_data["order_book"]):
                part = obs_data["order_book"][i]
                type_id_norm = part['config']['type_id'] / 2.0
                priority_norm = (part['priority'] - 1.5) / 0.5
                time_to_due_norm = max(-1, (part['due_date'] - self.simulation.env.now) / 720.0)
                obs_vector.extend([type_id_norm, priority_norm, time_to_due_norm])
            else:
                # Pad with zeros if order book is smaller than its max size
                obs_vector.extend([0, 0, 0])
        
        # 3. Add time-based features (3 features)
        time_of_day = self.simulation.env.now % MINS_IN_DAY
        day_of_week = (self.simulation.env.now // MINS_IN_DAY) % 7
        
        time_of_day_sin = np.sin(2 * np.pi * time_of_day / MINS_IN_DAY)
        time_of_day_cos = np.cos(2 * np.pi * time_of_day / MINS_IN_DAY)
        day_of_week_norm = day_of_week / 6.0
        
        obs_vector.extend([time_of_day_sin, time_of_day_cos, day_of_week_norm])
        
        # Total features = 2 + 30 + 3 = 35
        return np.array(obs_vector, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- ADVANCED DOMAIN RANDOMIZATION ---
        if self.randomize:
            # Decide if this episode will be a "High Priority Rush" day
            rush_day_probability = 0.3 # chance of a crisis
            if random.random() < rush_day_probability:
                # This is a "crisis" day
                new_high_prio_mix = 0.8
            else:
                # This is a "normal" day
                new_high_prio_mix = random.uniform(0.1, 0.5)
            
            # Set the chosen priority mix for this episode
            self.simulation.PRIORITY_MIX = {'HIGH': new_high_prio_mix, 'LOW': 1.0 - new_high_prio_mix}
            
            # We can still randomize other things, like the bottleneck speed
            self.simulation.PART_CONFIGS['Type_B']['s2_time'] = random.uniform(22, 28)

        self.simulation.setup_simulation()
        self.current_step = 0
        observation = self._get_obs()
        info = {}
        return observation, info


    def step(self, action):
        part_choice, flow_choice, overtime_choice = action
        
        # --- Take Actions ---
        self.simulation.set_source_status(bool(flow_choice))
        self.simulation.set_overtime_status(bool(overtime_choice))
        if not bool(flow_choice):
            self.simulation.release_part(part_choice)

        # --- Run the Simulation ---
        self.simulation.run(duration=self.step_duration)

        # --- Get State and Results ---
        obs_data, results = self.simulation.get_kpis_and_state()
        
        # --- Build the Observation Vector ---
        b12_level_norm = obs_data["buffer_12_level"] / self.simulation.BUFFER_CAPACITY
        b23_level_norm = obs_data["buffer_23_level"] / self.simulation.BUFFER_CAPACITY
        obs_vector = [b12_level_norm, b23_level_norm]
        
        for i in range(ORDER_BOOK_SIZE):
            if i < len(obs_data["order_book"]):
                part = obs_data["order_book"][i]
                type_id_norm = part['config']['type_id'] / 2.0
                priority_norm = (part['priority'] - 1.5) / 0.5
                time_to_due_norm = max(-1, (part['due_date'] - self.simulation.env.now) / 720.0)
                obs_vector.extend([type_id_norm, priority_norm, time_to_due_norm])
            else:
                obs_vector.extend([0, 0, 0])
        
        # --- THIS IS THE MISSING PART IN YOUR CURRENT STEP FUNCTION ---
        time_of_day = self.simulation.env.now % MINS_IN_DAY
        day_of_week = (self.simulation.env.now // MINS_IN_DAY) % 7
        time_of_day_sin = np.sin(2 * np.pi * time_of_day / MINS_IN_DAY)
        time_of_day_cos = np.cos(2 * np.pi * time_of_day / MINS_IN_DAY)
        day_of_week_norm = day_of_week / 6.0
        obs_vector.extend([time_of_day_sin, time_of_day_cos, day_of_week_norm])
        # --- END OF MISSING PART ---

        observation = np.array(obs_vector, dtype=np.float32)

        # --- Calculate Reward ---
        reward = 0.0
        cycle_times_high = []
        cycle_times_low = []
        for part in results["newly_completed_parts"]:
            if part['priority'] == 1:
                if part['is_late']:
                    reward -= 10.0
                else:
                    reward += 25.0
                cycle_times_high.append(part['cycle_time'])
            else:
                reward += 5.0
                cycle_times_low.append(part['cycle_time'])
        
        wip_level = observation[0] + observation[1]
        reward -= wip_level * 0.1
        if bool(flow_choice): reward -= 1.0
        if bool(overtime_choice): reward -= 20.0
        final_reward = float(reward)

        # --- Finalize the Step ---
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {
            'cycle_times_high': cycle_times_high, 
            'cycle_times_low': cycle_times_low,
            'newly_completed_parts': results["newly_completed_parts"],
            'overtime_active': bool(overtime_choice)
        }
        
        return observation, final_reward, terminated, truncated, info