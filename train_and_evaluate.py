# In train_and_evaluate.py
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import statistics 
import random
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from assembly_line_env import AssemblyLineEnv
from simulation_model import AssemblyLineSim

def run_simulation(env, policy_model, n_steps=1000, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    obs = env.reset()
    all_completed_parts = []
    
    for _ in range(n_steps):
        action, _states = policy_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if info[0]['newly_completed_parts']:
            all_completed_parts.extend(info[0]['newly_completed_parts'])
        if done[0]:
            pass

    sim_instance = env.get_attr('simulation')[0]
    final_state, _ = sim_instance.get_kpis_and_state()
    
    return {
        "completed_parts": all_completed_parts,
        "final_wip_b12": final_state['buffer_12_level'],
        "final_wip_b23": final_state['buffer_23_level'],
        "station_busy_time": sim_instance.station_busy_time
    }

def generate_report(baseline_results, rl_results, params):
    print("\n\n--- COMPREHENSIVE PERFORMANCE REPORT ---")
    print("="*40)
    print("--- 1. Simulation Parameters ---")
    print(f"Evaluation Seed: {params['seed']}")
    print(f"Evaluation Duration: {params['steps']} hours ({params['steps']*60} mins)")
    print(f"Part Mix: {params['part_mix']}")
    print(f"Priority Mix: {params['prio_mix']}")
    print(f"Station 2 Fail Rate: {params['fail_rate']:.0%}")
    print("-"*40)

    bl_parts = baseline_results['completed_parts']
    rl_parts = rl_results['completed_parts']

    bl_high_prio = [p for p in bl_parts if p['priority'] == 1]
    bl_low_prio = [p for p in bl_parts if p['priority'] == 2]
    rl_high_prio = [p for p in rl_parts if p['priority'] == 1]
    rl_low_prio = [p for p in rl_parts if p['priority'] == 2]

    print("\n--- 2. Core Performance ---")
    bl_throughput = len(bl_parts) / params['steps']
    rl_throughput = len(rl_parts) / params['steps']
    improvement = ((rl_throughput - bl_throughput) / bl_throughput) * 100 if bl_throughput > 0 else float('inf')
    print(f"{'Throughput (Parts/Hour)':<25} | Baseline: {bl_throughput:6.2f} | RL Agent: {rl_throughput:6.2f} | Improvement: {improvement:6.2f}%")
    print(f"{'Total Parts Completed':<25} | Baseline: {len(bl_parts):6.0f} | RL Agent: {len(rl_parts):6.0f}")
    print("-"*40)
    
    print("\n--- 3. Order Fulfillment & Lateness ---")
    print(f"{'HIGH Prio Parts Completed':<25} | Baseline: {len(bl_high_prio):6.0f} | RL Agent: {len(rl_high_prio):6.0f}")
    print(f"{'LOW Prio Parts Completed':<25} | Baseline: {len(bl_low_prio):6.0f} | RL Agent: {len(rl_low_prio):6.0f}")
    bl_late_high = sum(1 for p in bl_high_prio if p['is_late'])
    rl_late_high = sum(1 for p in rl_high_prio if p['is_late'])
    print(f"{'HIGH Prio Parts LATE':<25} | Baseline: {bl_late_high:6.0f} | RL Agent: {rl_late_high:6.0f}")
    print("-"*40)
    
    print("\n--- 4. Operational Efficiency ---")
    bl_ct_high = statistics.mean([p['cycle_time'] for p in bl_high_prio]) if bl_high_prio else 0
    rl_ct_high = statistics.mean([p['cycle_time'] for p in rl_high_prio]) if rl_high_prio else 0
    bl_ct_low = statistics.mean([p['cycle_time'] for p in bl_low_prio]) if bl_low_prio else 0
    rl_ct_low = statistics.mean([p['cycle_time'] for p in rl_low_prio]) if rl_low_prio else 0
    print(f"{'Avg Cycle Time HIGH (mins)':<25} | Baseline: {bl_ct_high:6.2f} | RL Agent: {rl_ct_high:6.2f}")
    print(f"{'Avg Cycle Time LOW (mins)':<25} | Baseline: {bl_ct_low:6.2f} | RL Agent: {rl_ct_low:6.2f}")
    
    print("\nStation Utilization (%):")
    sim_duration_mins = params['steps'] * 60
    for station in baseline_results['station_busy_time']:
        bl_util = (baseline_results['station_busy_time'][station] / sim_duration_mins) * 100
        rl_util = (rl_results['station_busy_time'][station] / sim_duration_mins) * 100
        print(f"  {station.title():<23} | Baseline: {bl_util:6.2f}% | RL Agent: {rl_util:6.2f}%")
    print("-"*40)
    
    print("\n--- 5. Final System State ---")
    print(f"{'Final WIP (Buffer 1->2)':<25} | Baseline: {baseline_results['final_wip_b12']:6.0f} | RL Agent: {rl_results['final_wip_b12']:6.0f}")
    print(f"{'Final WIP (Buffer 2->3)':<25} | Baseline: {baseline_results['final_wip_b23']:6.0f} | RL Agent: {rl_results['final_wip_b23']:6.0f}")
    print("="*40)

if __name__ == "__main__":
    EVALUATION_STEPS = 168
    EVALUATION_SEED = 42
    
    print("--- 1. Setting up Environment ---")
    log_dir = "./ppo_logs/"
    os.makedirs(log_dir, exist_ok=True)
    env_creator = lambda: Monitor(AssemblyLineEnv())
    env = DummyVecEnv([env_creator])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    print("\n--- 2. Training the RL Agent (PPO) ---")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=5e-5,tensorboard_log=log_dir, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, device='cpu')

    model.learn(total_timesteps=400000, log_interval=10)
    
    print("✅ Agent training complete.")
    
    model.save("ppo_assembly_line_model")
    env.save("vec_normalize_stats.pkl")
    print("✅ Model and normalization stats saved.")
    
    eval_env_creator = lambda: AssemblyLineEnv()
    eval_env = DummyVecEnv([eval_env_creator])
    eval_env = VecNormalize.load("vec_normalize_stats.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    class BaselineModel:
        def predict(self, obs, deterministic=True):
            return np.array([[0, 0]]), None 
    
    baseline_model = BaselineModel()
    trained_model = PPO.load("ppo_assembly_line_model", env=eval_env)

    print("\n--- Running Baseline Simulation (with fixed seed) ---")
    baseline_results = run_simulation(eval_env, baseline_model, n_steps=EVALUATION_STEPS, seed=EVALUATION_SEED)
    print("✅ Baseline run complete.")

    print("\n--- Running RL Agent Simulation (with fixed seed) ---")
    rl_results = run_simulation(eval_env, trained_model, n_steps=EVALUATION_STEPS, seed=EVALUATION_SEED)
    print("✅ RL Agent evaluation complete.")

    sim_params = {
        'seed': EVALUATION_SEED, 'steps': EVALUATION_STEPS,
        'part_mix': AssemblyLineSim().PART_MIX,
        'prio_mix': AssemblyLineSim().PRIORITY_MIX,
        'fail_rate': AssemblyLineSim().FAIL_RATE
    }
    generate_report(baseline_results, rl_results, sim_params)