# In evaluate_agent.py

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from assembly_line_env import AssemblyLineEnv
from train_and_evaluate import run_simulation, generate_report # Re-use our functions
from simulation_model import AssemblyLineSim # Import for getting params

# --- 1. DEFINE YOUR TEST SCENARIOS ---
test_scenarios = {
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
    "New Product Launch (More Complex Parts)": {
        "part_mix": {'Type_A': 0.2, 'Type_B': 0.7, 'Type_C': 0.1},
        "priority_mix": {'HIGH': 0.2, 'LOW': 0.8},
        "fail_rate": 0.08
    }
}

# --- 2. SETUP MODELS ---
EVALUATION_STEPS = 168 # 1 week
EVALUATION_SEED = 42

# MODIFIED: The BaselineModel now returns a 3-part action
class BaselineModel:
    def predict(self, obs, deterministic=True):
        # Baseline: Always release part 0, never halt (0), never use overtime (0)
        return np.array([[0, 0, 0]]), None 
baseline_model = BaselineModel()

# Load the trained agent and the normalization stats
try:
    # Use the best model saved by the EvalCallback
    trained_model = PPO.load("./best_model/best_model.zip")
    vec_normalize_stats = "vec_normalize_stats.pkl"
except FileNotFoundError:
    print("ERROR: Model (./best_model/best_model.zip) or stats (vec_normalize_stats.pkl) not found.")
    print("Please run train_and_evaluate.py first to generate the model files.")
    exit()


# --- 3. RUN THE GAUNTLET ---
for name, params in test_scenarios.items():
    print(f"\n\n\n--- RUNNING TEST SCENARIO: {name} ---")
    
    # Create a fresh, configured environment for this test
    # Pass the scenario's parameters to the constructor
    eval_env_creator = lambda: AssemblyLineEnv(randomize=False, **params)
    eval_env = DummyVecEnv([eval_env_creator])
    eval_env = VecNormalize.load(vec_normalize_stats, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # Run both policies on this specific scenario
    baseline_results = run_simulation(eval_env, baseline_model, n_steps=EVALUATION_STEPS, seed=EVALUATION_SEED)
    agent_results = run_simulation(eval_env, trained_model, n_steps=EVALUATION_STEPS, seed=EVALUATION_SEED)

    # Generate a report for this specific scenario
    report_params = {**params, 'seed': EVALUATION_SEED, 'steps': EVALUATION_STEPS}
    generate_report(baseline_results, agent_results, report_params)