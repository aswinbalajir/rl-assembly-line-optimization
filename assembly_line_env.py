# In assembly_line_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation_model import AssemblyLineSim # Import our simulation engine

class AssemblyLineEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()
        
        # --- Define Action and Observation Spaces ---
        # They must be gym.spaces objects
        
        # Example Action Space: Choose a priority rule
        # Action 0: Default rule (High prio > Low prio)
        # Action 1: Repaired parts get highest priority
        self.action_space = spaces.Discrete(2)

        # Example Observation Space: A vector of key metrics
        # [buffer12_level, buffer23_level, station2_utilization, repair_station_utilization]
        low_bounds = np.array([0, 0, 0, 0])
        high_bounds = np.array([10, 10, 1, 1]) # Buffer capacity is 10, utilization is 0-1
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # Initialize the simulation
        self.simulation = AssemblyLineSim()
        self.step_duration = 60 # Each step in the RL env simulates 60 minutes

    def _get_obs(self):
        kpis = self.simulation.get_kpis()
        # Important: The observation must be a NumPy array
        return np.array([
            kpis["buffer_12_level"],
            kpis["buffer_23_level"],
            kpis["station_2_utilization"],
            kpis["repair_station_utilization"]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Required by the latest gym API
        
        # Reset the simulation to its initial state
        self.simulation.setup_simulation()
        self.last_parts_completed = 0
        
        observation = self._get_obs()
        info = {} # You can pass extra info here if needed
        return observation, info

    def step(self, action):
        # --- 1. Take Action ---
        # The agent's action doesn't directly control the sim in this example.
        # It would be used to change rules, e.g., self.simulation.set_priority_rule(action)
        # For now, we'll keep it simple and the action won't do anything.
        # We will add its effect in the next phase.

        # --- 2. Run the Simulation for one step duration ---
        self.simulation.run(duration=self.step_duration)

        # --- 3. Get the New State (Observation) ---
        observation = self._get_obs()

        # --- 4. Calculate the Reward ---
        current_kpis = self.simulation.get_kpis()
        
        # Reward for new parts produced in this step
        newly_completed_parts = current_kpis["parts_completed_total"] - self.last_parts_completed
        self.last_parts_completed = current_kpis["parts_completed_total"]
        
        # Penalty for high WIP (work-in-progress)
        wip = current_kpis["buffer_12_level"] + current_kpis["buffer_23_level"]
        
        # The Reward Function
        reward = (newly_completed_parts * 10) - (wip * 0.5)

        # --- 5. Check for Termination ---
        # In our continuous simulation, it never truly "ends"
        terminated = False
        truncated = False
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        # For now, just print the KPIs to the console
        kpis = self.simulation.get_kpis()
        print(f"Time: {self.simulation.env.now:.2f} | WIP: {kpis['buffer_12_level'] + kpis['buffer_23_level']} | Completed: {kpis['parts_completed_total']}")

    def close(self):
        # Clean up any resources if needed
        pass