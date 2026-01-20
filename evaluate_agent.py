# evaluate_agent.py

import os
from typing import Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom Imports
from assembly_line_env import AssemblyLineEnv
from train_and_evaluate import run_simulation, generate_report, BaselineModel, SmartBaseline

# --- Constants ---
MODEL_PATH = "./best_model/best_model.zip"
EVALUATION_HOURS = 168  # 1 Week
EVALUATION_SEED = 42

# --- 1. Define Test Scenarios ---
TEST_SCENARIOS = {
    "Original": {
        "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15},
        "priority_mix": {'HIGH': 0.2, 'LOW': 0.8},
        "fail_rate": 0.08
    },
    "High Priority Rush": {
        "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15},
        "priority_mix": {'HIGH': 0.8, 'LOW': 0.2},
        "fail_rate": 0.08
    },
    "High Failure Rate": {
        "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15},
        "priority_mix": {'HIGH': 0.2, 'LOW': 0.8},
        "fail_rate": 0.20
    },
    "New Product Launch (Complex Parts)": {
        "part_mix": {'Type_A': 0.2, 'Type_B': 0.7, 'Type_C': 0.1},
        "priority_mix": {'HIGH': 0.2, 'LOW': 0.8},
        "fail_rate": 0.08
    }
}

class BaselineModel:
    """A heuristic baseline: Always release, never halt, never overtime."""
    def predict(self, obs, deterministic=True):
        # Action format: [Part_Index (0), Halt (0), Overtime (0)]
        # Reshaped to (1, 3) for VecEnv compatibility
        return np.array([[0, 0, 0]]), None 

def main():
    # --- 2. Load Models ---
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please run train_and_evaluate.py first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    trained_model = PPO.load(MODEL_PATH)
    baseline_model = BaselineModel()
    smart_baseline = SmartBaseline()
    # --- 3. Run Scenarios ---
    for name, params in TEST_SCENARIOS.items():
        print(f"\n\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"{'='*60}")
        
        # A. Create Environment for this specific scenario
        # We pass **params to unpack the dictionary into arguments
        # e.g., fail_rate=0.20
        eval_env_creator = lambda: AssemblyLineEnv(randomize=False, **params)
        eval_env = DummyVecEnv([eval_env_creator])

        # B. Calculate Steps Correctly
        # Access step_duration from the actual environment instance
        step_duration = eval_env.get_attr("step_duration", indices=0)[0]
        n_steps = int((EVALUATION_HOURS * 60) / step_duration)
        
        print(f"Simulation Duration: {EVALUATION_HOURS} Hours ({n_steps} Steps)")

        # C. Run Simulations
        print("-> Running Baseline...")
        baseline_results = run_simulation(
            eval_env, 
            baseline_model, 
            n_steps=n_steps, 
            seed=EVALUATION_SEED
        )
        print("-> Running Smart Baseline (High Prio First)...")
        res_smart = run_simulation(eval_env, smart_baseline, n_steps=n_steps, seed=EVALUATION_SEED)

        print("-> Running RL Agent...")
        agent_results = run_simulation(
            eval_env, 
            trained_model, 
            n_steps=n_steps, 
            seed=EVALUATION_SEED
        )

        # D. Generate Report
        # Combine scenario params with report meta-data
        report_params = params.copy()
        report_params.update({
            'seed': EVALUATION_SEED, 
            'steps': EVALUATION_HOURS # Passing hours for display
        })
        
        generate_report(baseline_results, res_smart, agent_results, report_params)

if __name__ == "__main__":
    main()