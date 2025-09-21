# In assembly_line_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation_model import AssemblyLineSim, ORDER_BOOK_SIZE

class AssemblyLineEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.MultiDiscrete([ORDER_BOOK_SIZE, 2])
        obs_size = 2 + (ORDER_BOOK_SIZE * 3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)
        self.simulation = AssemblyLineSim()
        self.step_duration = 60
        self.max_episode_steps = 1000
        self.current_step = 0

    def _get_obs(self):
        obs_data, _ = self.simulation.get_kpis_and_state()
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
        return np.array(obs_vector, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.setup_simulation()
        self.current_step = 0
        observation = self._get_obs()
        info = {}
        return observation, info


    def step(self, action):
        part_choice, flow_choice = action
        self.simulation.set_source_status(bool(flow_choice))
        if not bool(flow_choice):
            self.simulation.release_part(part_choice)

        self.simulation.run(duration=self.step_duration)

        obs_data, results = self.simulation.get_kpis_and_state()
        
        # This part of building the observation is correct
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
        observation = np.array(obs_vector, dtype=np.float32)

        # --- RE-BALANCED Value-Based Reward ---
        reward = 0.0
        cycle_times_high = []
        cycle_times_low = []
        for part in results["newly_completed_parts"]:
            if part['priority'] == 1: # HIGH priority
                if part['is_late']:
                    reward -= 5.0 # MODIFIED: Penalty is smaller, less terrifying
                else:
                    reward += 15.0 # NEW: Big reward for ON-TIME high-prio parts
                cycle_times_high.append(part['cycle_time'])
            else: # LOW priority
                reward += 2.0 # reward for low-prio
                cycle_times_low.append(part['cycle_time'])
        
        wip_level = obs_data["buffer_12_level"] / self.simulation.BUFFER_CAPACITY + obs_data["buffer_23_level"] / self.simulation.BUFFER_CAPACITY
        reward -= wip_level * 0.025 # Wip penality 

        if bool(flow_choice): reward -= 0.5# inaction penality 
        final_reward = float(reward)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {
            'cycle_times_high': cycle_times_high, 
            'cycle_times_low': cycle_times_low,
            'newly_completed_parts': results["newly_completed_parts"]
        }
        
        return observation, final_reward, terminated, truncated, info