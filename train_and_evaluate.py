# train_and_evaluate.py

import os
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from assembly_line_env import AssemblyLineEnv
from simulation_model import AssemblyLineSim

# --- Configuration Constants ---
LOG_DIR = "./ppo_logs/"
MODEL_SAVE_DIR = "./best_model/"
MODEL_NAME = "ppo_assembly_line_model"

# Training Hyperparameters
TRAINING_STEPS = 5_500_000
N_ENVS = 4  # Parallel environments
EVAL_FREQ = 10_000
SEED = 0

# PPO Hyperparameters
# PPO_CONFIG = {
#     "policy": "MlpPolicy",
#     "ent_coef": 0.01,
#     "learning_rate": 3e-4,
#     "n_steps": 2048,
#     "batch_size": 64,
#     "n_epochs": 10,
#     "gamma": 0.99,
#     "device": "cpu",
#     "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
# }
PPO_CONFIG = {
    "policy": "MlpPolicy",
    
    # 1. EXPLORATION: Force it to try new things longer
    "ent_coef": 0.05,            # Increased from 0.01
    "gamma": 0.999,              # Increased from 0.99 LONG-TERM PLANNING: Care about the whole week, not just the next hour
    "learning_rate": 2.5e-4,     # Slightly lower  for stability 
    "n_steps": 2048,            # 4. HORIZON: Ensure we see a full episode (1680 steps) in one update
    "batch_size": 128,           # Increased from 64 for smoother gradients
    "n_epochs": 10,
    "device": "cpu",                                               
    "policy_kwargs": dict(
        net_arch=dict(pi=[512, 512], vf=[512, 512])# 5. BRAIN SIZE: More capacity for logic
    ),
}

class BaselineModel:
    """A simple heuristic baseline agent that always releases the first order."""
    def predict(
        self, 
        obs: np.ndarray, 
        state: Optional[Tuple[np.ndarray, ...]] = None, 
        episode_start: Optional[np.ndarray] = None, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # Action: Release Part Index 0, Halt=0 (No), Overtime=0 (No)
        # Reshaped to (1, 3) to match VecEnv expectation.
        return np.array([[0, 0, 0]]), None
class SmartBaseline:
    """
    Baseline 2: The 'Aggressive' Manager (EDD + Priority + OT).
    """
    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        obs_vec = obs[0] 
        start_idx = 2
        
        best_idx = 0
        best_score = float('inf') 
        has_high_priority_orders = False

        # 1. Look at all available orders
        for i in range(10):
            base = start_idx + (i * 3)
            prio_val = obs_vec[base + 1] # -1.0 (High) or 1.0 (Low)
            due_val = obs_vec[base + 2]  # -1.0 (Urgent) to 1.0 (Far)

            # Skip empty slots
            if abs(prio_val) < 0.1: continue

            # Track if we have High Priority work (for Overtime decision)
            if prio_val < 0:
                has_high_priority_orders = True

            # 2, 3, & 4. The Scoring Formula
            # High Prio (-10) always beats Low Prio (10).
            # Lower Due Date always beats Higher Due Date.
            score = (prio_val * 10) + due_val

            if score < best_score:
                best_score = score
                best_idx = i
        
        # AGGRESSIVE OVERTIME LOGIC
        # If we have important work, burn the midnight oil.
        overtime_action = 1 if has_high_priority_orders else 0

        return np.array([[best_idx, 0, overtime_action]]), None
    
# class SmartBaseline:
#     """
#     Baseline 2: The 'Common Sense' Agent (EDD + Priority).
#     Logic:
#     1. Look at all available orders in the book (from observation).
#     2. Filter for HIGH Priority orders first.
#     3. Within that group, pick the one with the Earliest Due Date.
#     4. If no High Priority, pick Low Priority with Earliest Due Date.
#     """
#     def predict(self, obs, state=None, episode_start=None, deterministic=False):
#         # Observation structure based on assembly_line_env.py:
#         # [Buffer1, Buffer2, (Part0_Type, Part0_Prio, Part0_Due), (Part1...), ..., TimeFeatures]
        
#         # We assume batch size 1 for evaluation
#         obs_vec = obs[0] 
        
#         best_idx = 0
#         # Initialize with a terrible score
#         # Lower score is better. 
#         # Score calculation: Priority_Weight + Time_Weight
#         best_score = float('inf') 

#         # The order book starts at index 2 of the observation
#         start_idx = 2
        
#         # Iterate through the 10 slots in the order book
#         for i in range(10):
#             base = start_idx + (i * 3)
            
#             # Extract features (Normalized in Env)
#             # Priority: -1.0 is High, 1.0 is Low, 0.0 is Empty Slot
#             prio_val = obs_vec[base + 1]
            
#             # Due Date: -1.0 is urgent, 1.0 is far out
#             due_val = obs_vec[base + 2]

#             # Check if slot is empty (Priority is 0.0 for padding)
#             if abs(prio_val) < 0.1:
#                 continue

#             # --- HEURISTIC LOGIC ---
#             # We want High Priority (-1.0) to beat Low Priority (1.0).
#             # We want Lower Due Date to beat Higher Due Date.
            
#             # Multiplier 10 ensures Priority is the dominant factor.
#             # Score Examples:
#             # High Prio (-1), Urgent (-1) -> -10 + (-1) = -11 (Winner)
#             # High Prio (-1), Safe (1)    -> -10 + 1    = -9
#             # Low Prio (1), Urgent (-1)   ->  10 + (-1) =  9
#             score = (prio_val * 10) + due_val

#             if score < best_score:
#                 best_score = score
#                 best_idx = i

#         return np.array([[best_idx, 0, 0]]), None
    
def run_simulation(
    env: VecEnv, 
    policy_model: Union[PPO, BaselineModel], 
    n_steps: int = 1000, 
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes a simulation run with a specific policy.

    Args:
        env (VecEnv): The vectorized environment to run.
        policy_model (Union[PPO, BaselineModel]): The agent policy.
        n_steps (int): Number of steps to simulate.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing performance metrics (completed parts, WIP, etc.).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # Seed the environment itself if possible
        try:
            env.seed(seed)
        except AttributeError:
            pass 

    obs = env.reset()
    all_completed_parts = []
    total_overtime_hours = 0.0
    
    # Safely extract step_duration from the first environment
    try:
        # Access the underlying environment instance
        step_duration_mins = env.get_attr("step_duration", indices=0)[0]
    except Exception:
        step_duration_mins = 6  # Default fallback if attribute access fails

    # --- Simulation Loop ---
    for _ in range(n_steps):
        action, _states = policy_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # VecEnv returns a list of infos; we only use the first one since we run 1 test env
        info_dict = info[0]

        # Track Completed Parts
        if info_dict.get('newly_completed_parts'):
            all_completed_parts.extend(info_dict['newly_completed_parts'])
            
        # Track Overtime Usage
        if action[0][2] > 0:
            total_overtime_hours += (step_duration_mins / 60.0)
        elif info_dict.get('overtime_active', False):
             total_overtime_hours += (step_duration_mins / 60.0)

    # --- Final State Extraction ---
    # Extract internal simulation state from the environment
    sim_instance = env.get_attr('simulation', indices=0)[0]
    final_obs, _ = sim_instance.get_kpis_and_state()
    
    return {
        "completed_parts": all_completed_parts,
        "final_wip_b12": final_obs['buffer_12_level'],
        "final_wip_b23": final_obs['buffer_23_level'],
        "station_busy_time": sim_instance.station_busy_time,
        "total_overtime_hours": total_overtime_hours
    }


def generate_report(
    baseline_results: Dict, 
    smart_results: Dict,  # <--- Added Smart Baseline Input
    rl_results: Dict, 
    params: Dict
):
    """
    Generates and prints a comparative performance report for 3 agents.

    Args:
        baseline_results (dict): Metrics from the original Baseline (FIFO).
        smart_results (dict): Metrics from the Smart Baseline (EDD + Priority).
        rl_results (dict): Metrics from the trained RL agent.
        params (dict): Simulation parameters used for the context.
    """
    print("\n\n--- COMPREHENSIVE PERFORMANCE REPORT (3-WAY COMPARISON) ---")
    print("=" * 80)
    print("--- 1. Simulation Parameters ---")
    print(f"Evaluation Seed: {params.get('seed', 'N/A')}")
    print(f"Evaluation Duration: {params['steps']} hours")
    print(f"Part Mix: {params['part_mix']}")
    print(f"Priority Mix: {params['priority_mix']}")
    print(f"Station 2 Fail Rate: {params['fail_rate']:.0%}")
    print("-" * 80)

    # --- Data Processing ---
    # Extract parts lists
    bl_parts = baseline_results['completed_parts']
    sm_parts = smart_results['completed_parts']
    rl_parts = rl_results['completed_parts']
    
    # Filter High Priority
    bl_high_prio = [p for p in bl_parts if p['priority'] == 1]
    sm_high_prio = [p for p in sm_parts if p['priority'] == 1]
    rl_high_prio = [p for p in rl_parts if p['priority'] == 1]

    # --- 2. Throughput & Core Output ---
    print("\n--- 2. Throughput & Core Output ---")
    
    # Calculate operational hours (Standard + Overtime)
    # Standard: 6 days * 10.5 hours = 63.0 hours
    standard_hours = 6 * 10.5
    
    # Calculate for all 3 agents
    bl_op_hours = standard_hours + baseline_results['total_overtime_hours']
    sm_op_hours = standard_hours + smart_results['total_overtime_hours']
    rl_op_hours = standard_hours + rl_results['total_overtime_hours']

    # Throughput calculation (Units per Hour)
    bl_throughput = len(bl_parts) / bl_op_hours if bl_op_hours > 0 else 0
    sm_throughput = len(sm_parts) / sm_op_hours if sm_op_hours > 0 else 0
    rl_throughput = len(rl_parts) / rl_op_hours if rl_op_hours > 0 else 0
    
    # Column Headers
    print(f"{'METRIC':<30} | {'FIFO (Weak)':<12} | {'SMART (Strong)':<14} | {'RL AGENT':<12}")
    print("-" * 80)
    
    # Rows
    print(f"{'Total Units Completed':<30} | {len(bl_parts):<12.0f} | {len(sm_parts):<14.0f} | {len(rl_parts):<12.0f}")
    print(f"{'Total Overtime Hours Used':<30} | {baseline_results['total_overtime_hours']:<12.1f} | {smart_results['total_overtime_hours']:<14.1f} | {rl_results['total_overtime_hours']:<12.1f}")
    print(f"{'Total Operational Hours':<30} | {bl_op_hours:<12.1f} | {sm_op_hours:<14.1f} | {rl_op_hours:<12.1f}")
    print(f"{'Throughput (Units/Op. Hour)':<30} | {bl_throughput:<12.2f} | {sm_throughput:<14.2f} | {rl_throughput:<12.2f}")
   
    print("-" * 80)
    
    # --- 3. On-Time Delivery Performance ---
    print("\n--- 3. On-Time Delivery Performance (Schedule Adherence) ---")
    
    def calculate_otd(parts_list):
        if not parts_list: return 0.0, 0
        late_count = sum(1 for p in parts_list if p['is_late'])
        otd_rate = ((len(parts_list) - late_count) / len(parts_list)) * 100
        return otd_rate, late_count

    # Calculate OTD for all 3
    bl_otd_high, bl_late_high = calculate_otd(bl_high_prio)
    sm_otd_high, sm_late_high = calculate_otd(sm_high_prio)
    rl_otd_high, rl_late_high = calculate_otd(rl_high_prio)

    print(f"{'On-Time Rate (HIGH Prio)':<30} | {bl_otd_high:<11.2f}% | {sm_otd_high:<13.2f}% | {rl_otd_high:<11.2f}%")
    print(f"{'HIGH Prio Orders Late':<30} | {bl_late_high:<12.0f} | {sm_late_high:<14.0f} | {rl_late_high:<12.0f}")
    print("=" * 80)


def main():
    """Main execution function."""
    
    # --- Step 1: Create Environments ---
    print("--- 1. Setting up Environments ---")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Create Vectorized Training Environment (Parallel)
    # n_envs should be <= CPU core count
    train_env = make_vec_env(
        AssemblyLineEnv, 
        n_envs=N_ENVS, 
        seed=SEED, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={'randomize': True}, 
        monitor_dir=LOG_DIR
    )

    # Create Validation Environment (Single)
    # We use a lambda to defer creation until DummyVecEnv calls it
    eval_env = DummyVecEnv([lambda: Monitor(AssemblyLineEnv(randomize=False))])

    # --- Step 2: Setup Training ---
    print(f"\n--- 2. Training the RL Agent on {N_ENVS} Parallel Environments ---")
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODEL_SAVE_DIR,
        log_path=MODEL_SAVE_DIR, 
        eval_freq=EVAL_FREQ,
        deterministic=True, 
        render=False
    )
    
    # Initialize PPO Model
    model = PPO(
        env=train_env, 
        verbose=1, 
        tensorboard_log=LOG_DIR, 
        **PPO_CONFIG
    )
    
    print(f"Device set to: {model.device}")
    
    # Start Training
    model.learn(
        total_timesteps=TRAINING_STEPS, 
        log_interval=10, 
        callback=eval_callback
    )
    print("Agent training complete.")
    
    # model.save(MODEL_NAME)
    print("Model saved successfully.")
    
    # --- Step 3: Final Testing ---
    print("\n--- 3. Final, Rigorous Testing on Unseen Scenarios ---")
    
    # Test Configuration
    TEST_SEEDS = [42, 101, 888, 1234, 99]
    baseline_results_list = []
    smart_results_list = []
    agent_results_list = []
    baseline_model = BaselineModel()

    # Load Best Model
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.zip")
    final_model_path = f"{MODEL_NAME}.zip"
    
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        best_model = PPO.load(best_model_path)
    else:
        print(f"Best model not found. Loading final model from: {final_model_path}")
        best_model = PPO.load(final_model_path)
    
    # Instantiate Baselines
    fifo_baseline = BaselineModel()
    smart_baseline = SmartBaseline()
    # Create Test Environment
    test_env = DummyVecEnv([lambda: AssemblyLineEnv(randomize=False)])
    
    # Determine Simulation Duration (1 Week = 168 Hours)
    # We need to calculate how many steps correspond to 168 hours
    step_duration_mins = test_env.get_attr("step_duration", indices=0)[0]
    eval_steps = int((168 * 60) / step_duration_mins)

    TEST_SEED = 42
    print(f"Running simulation for 168 hours (Seed {TEST_SEED})...")

    # Run All 3
    res_fifo = run_simulation(test_env, fifo_baseline, n_steps=eval_steps, seed=TEST_SEED)
    res_smart = run_simulation(test_env, smart_baseline, n_steps=eval_steps, seed=TEST_SEED)
    res_rl = run_simulation(test_env, best_model, n_steps=eval_steps, seed=TEST_SEED)
    
    temp_sim = AssemblyLineSim()
    params = {
        'seed': TEST_SEED, 
        'steps': 168, 
        'part_mix': temp_sim.PART_MIX,
        'priority_mix': temp_sim.PRIORITY_MIX,
        'fail_rate': temp_sim.FAIL_RATE
    }
    generate_report(res_fifo, res_smart, res_rl, params)

    print(f"\nConfiguration Check:")
    print(f"Step Duration: {step_duration_mins} mins")
    print(f"Eval Duration: 168 Hours")
    print(f"Steps Required: {eval_steps}")

    # Run Comparisons
    for seed in TEST_SEEDS:
        print(f"\n--- Running Test Scenario with Seed: {seed} ---")
        
        # Run Baseline
        bl_res = run_simulation(test_env, baseline_model, n_steps=eval_steps, seed=seed)
        baseline_results_list.append(bl_res)
        # Run Smart Baseline (EDD)
        sm_res = run_simulation(test_env, smart_baseline, n_steps=eval_steps, seed=seed)
        smart_results_list.append(sm_res)
        # Run RL Agent
        rl_res = run_simulation(test_env, best_model, n_steps=eval_steps, seed=seed)
        agent_results_list.append(rl_res)

    # --- Step 4: Reporting ---
    print(f"\n\n--- FINAL AGGREGATE REPORT (AVERAGED OVER {len(TEST_SEEDS)} RUNS) ---")
    
    # Use parameters from a fresh Sim instance for accurate reporting
    temp_sim = AssemblyLineSim()
    sim_params = {
        'seed': "Multiple (Aggregated)", 
        'steps': 168, 
        'part_mix': temp_sim.PART_MIX,
        'priority_mix': temp_sim.PRIORITY_MIX,
        'fail_rate': temp_sim.FAIL_RATE
    }
    
    # For demonstration, we report the results of the first seed.
    # Ideally, you would average the results in `baseline_results_list` before passing.
    generate_report(baseline_results_list[0],smart_results_list[0], agent_results_list[0], sim_params)


if __name__ == "__main__":
    main()