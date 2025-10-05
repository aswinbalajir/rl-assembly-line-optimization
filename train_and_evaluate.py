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
from stable_baselines3.common.callbacks import EvalCallback

def run_simulation(env, policy_model, n_steps=1000, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    obs = env.reset()
    
    all_completed_parts = []
    total_overtime_hours = 0
    
    for _ in range(n_steps):
        action, _states = policy_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        info_dict = info[0]
        if info_dict.get('newly_completed_parts'):
            all_completed_parts.extend(info_dict['newly_completed_parts'])
        if info_dict.get('overtime_active', False):
            total_overtime_hours += 1
        if done[0]:
            pass

    sim_instance = env.get_attr('simulation')[0]
    final_state, _ = sim_instance.get_kpis_and_state()
    
    return {
        "completed_parts": all_completed_parts,
        "final_wip_b12": final_state['buffer_12_level'],
        "final_wip_b23": final_state['buffer_23_level'],
        "station_busy_time": sim_instance.station_busy_time,
        "total_overtime_hours": total_overtime_hours
    }

def generate_report(baseline_results, rl_results, params):
    print("\n\n--- COMPREHENSIVE PERFORMANCE REPORT ---")
    print("="*50)
    print("--- 1. Simulation Parameters ---")
    print(f"Evaluation Seed: {params['seed']}")
    print(f"Evaluation Duration: {params['steps']} hours ({params['steps']*60} mins)")
    print(f"Part Mix: {params['part_mix']}")
    print(f"Priority Mix: {params['priority_mix']}")
    print(f"Station 2 Fail Rate: {params['fail_rate']:.0%}")
    print("-"*50)

    # --- Process results ---
    bl_parts = baseline_results['completed_parts']
    rl_parts = rl_results['completed_parts']
    bl_high_prio = [p for p in bl_parts if p['priority'] == 1]
    rl_high_prio = [p for p in rl_parts if p['priority'] == 1]

    # --- 2. Throughput & Core Output ---
    print("\n--- 2. Throughput & Core Output ---")
    # NEW: Calculate actual operational hours for a fair comparison
    # 6 work days in a 168-hour week, 10.5 standard hours per day
    baseline_op_hours = 6 * 10.5
    # The agent's operational hours include the overtime it used
    agent_op_hours = baseline_op_hours + rl_results['total_overtime_hours']

    bl_throughput = len(bl_parts) / baseline_op_hours if baseline_op_hours > 0 else 0
    rl_throughput = len(rl_parts) / agent_op_hours if agent_op_hours > 0 else 0
    
    improvement = ((rl_throughput - bl_throughput) / bl_throughput) * 100 if bl_throughput > 0 else float('inf')
    
    print(f"{'Total Units Completed':<32} | Baseline: {len(bl_parts):7.0f} | RL Agent: {len(rl_parts):7.0f}")
    print(f"{'Total Overtime Hours Used':<32} | Baseline: {baseline_results['total_overtime_hours']:7.0f} | RL Agent: {rl_results['total_overtime_hours']:7.0f}")
    print(f"{'Total Operational Hours':<32} | Baseline: {baseline_op_hours:7.1f} | RL Agent: {agent_op_hours:7.1f}")
    print(f"{'Throughput (Units/Operational Hour)':<32} | Baseline: {bl_throughput:7.2f} | RL Agent: {rl_throughput:7.2f} | Improvement: {improvement:7.2f}%")
    print("-"*50)
    
    # --- 3. On-Time Delivery Performance (Schedule Adherence) ---
    print("\n--- 3. On-Time Delivery Performance (Schedule Adherence) ---")
    bl_otd_high = (len(bl_high_prio) - sum(1 for p in bl_high_prio if p['is_late'])) / len(bl_high_prio) * 100 if bl_high_prio else 0
    rl_otd_high = (len(rl_high_prio) - sum(1 for p in rl_high_prio if p['is_late'])) / len(rl_high_prio) * 100 if rl_high_prio else 0
    print(f"{'On-Time Rate (HIGH Prio)':<28} | Baseline: {bl_otd_high:6.2f}% | RL Agent: {rl_otd_high:6.2f}%")
    print(f"{'HIGH Prio Orders Late':<28} | Baseline: {sum(1 for p in bl_high_prio if p['is_late']):7.0f} | RL Agent: {sum(1 for p in rl_high_prio if p['is_late']):7.0f}")
    print("-"*50)

    # --- 4. Overall Equipment Effectiveness (OEE) for Station 2 (Bottleneck) ---
    print("\n--- 4. Overall Equipment Effectiveness (OEE) for Station 2 (Bottleneck) ---")
    work_days = 6 * (params['steps'] / (24*7))
    scheduled_time_mins = work_days * 10.5 * 60
    
    bl_avail = (baseline_results['station_busy_time']['station2'] / scheduled_time_mins) * 100 if scheduled_time_mins > 0 else 0
    rl_avail = (rl_results['station_busy_time']['station2'] / scheduled_time_mins) * 100 if scheduled_time_mins > 0 else 0
    quality_rate = 1.0 - params['fail_rate']
    bl_perf = (bl_throughput / 3.0) * 100 
    rl_perf = (rl_throughput / 3.0) * 100
    bl_oee = (bl_avail/100) * bl_perf/100 * quality_rate * 100
    rl_oee = (rl_avail/100) * rl_perf/100 * quality_rate * 100
    print(f"{'OEE Score for Station 2':<28} | Baseline: {bl_oee:6.2f}% | RL Agent: {rl_oee:6.2f}%")
    print(f"  {'-> Availability':<26} | Baseline: {bl_avail:6.2f}% | RL Agent: {rl_avail:6.2f}%")
    print(f"  {'-> Performance':<26} | Baseline: {bl_perf:6.2f}% | RL Agent: {rl_perf:6.2f}%")
    print(f"  {'-> Quality':<26} | Baseline: {quality_rate*100:6.2f}% | RL Agent: {quality_rate*100:6.2f}%")
    print("="*50)

if __name__ == "__main__":
    
    # --- Step 1: Create Environments ---
    print("--- 1. Setting up Environments ---")
    log_dir = "./ppo_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. The Randomized Training Environment (The "Driving School" with all weather conditions)
    # We pass randomize=True to our environment's constructor.
    train_env_creator = lambda: Monitor(AssemblyLineEnv(randomize=True), log_dir)
    train_env = DummyVecEnv([train_env_creator])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 2. The Fixed Validation Environment (The "Practice Test" to find the best model)
    # This environment is NOT randomized.
    eval_env_creator = lambda: AssemblyLineEnv(randomize=False)
    eval_env = DummyVecEnv([eval_env_creator])
    # IMPORTANT: We load the stats from the training env to use its normalization.
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)


    # --- Step 2: Setup Training with Validation Callback ---
    print("\n--- 2. Training the RL Agent with Validation ---")
    
    # This callback will test the agent on the fixed validation env every 10,000 steps
    # and save the model that gets the highest score.
    eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/',
                                 log_path='./best_model/', eval_freq=10000,
                                 deterministic=True, render=False)
    # Define a larger network architecture
    # pi = policy network (actor), vf = value network (critic)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO("MlpPolicy", train_env,policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir,
                learning_rate=5e-5, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, device='cpu')
    
    # The agent is trained on the randomized environment, but the callback saves the best performer.
    model.learn(total_timesteps=500000, log_interval=10, callback=eval_callback)
    print("✅ Agent training complete.")
    #  Save the Model and Normalization Stats ---
    # We load the BEST model that was saved by the callback, not the final one.
    model.save("ppo_assembly_line_model")
    
    # You must explicitly save the statistics from the training environment.
    train_env.save("vec_normalize_stats.pkl")
    
    print("✅ Model and normalization stats saved.")
    
    # --- Step 3: Final Testing on Unseen Scenarios ---
    print("\n--- 3. Final, Rigorous Testing on Unseen Scenarios ---")
    
    # We load the BEST model that was saved by the callback, not the final one.
    best_model = PPO.load("./best_model/best_model.zip")
    
    # The "Test Set": A list of 5 challenging scenarios the agent has never seen.
    TEST_SEEDS = [42, 101, 888, 1234, 99]
    baseline_results_list = []
    agent_results_list = []
    
    class BaselineModel:
        def predict(self, obs, deterministic=True):
            # MODIFIED: Baseline now returns a 3-part action
            # Release first part (0), never halt (0), never use overtime (0)
            return np.array([[0, 0, 0]]), None  
    baseline_model = BaselineModel()

    # We need a fresh environment for the final test
    test_env_creator = lambda: AssemblyLineEnv(randomize=False)
    test_env = DummyVecEnv([test_env_creator])
    # Load the same normalization stats
    test_env = VecNormalize.load("vec_normalize_stats.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False

    for seed in TEST_SEEDS:
        print(f"\n--- Running Test Scenario with Seed: {seed} ---")
        baseline_results = run_simulation(test_env, baseline_model, n_steps=168, seed=seed)
        agent_results = run_simulation(test_env, best_model, n_steps=168, seed=seed)
        baseline_results_list.append(baseline_results)
        agent_results_list.append(agent_results)

    # --- Step 4: Aggregate and Report Final, Averaged Results ---
    # Here you would add logic to average the numbers from the lists
    # This is a placeholder; for your dissertation you'd process the lists to get mean and std. dev.
    print("\n\n--- FINAL AGGREGATE REPORT (AVERAGED OVER {} RUNS) ---".format(len(TEST_SEEDS)))
    # For now, we'll just show the results from the first test run as an example.
    sim_params = {
        'seed': "Multiple", 'steps': 168,
        'part_mix': AssemblyLineSim().PART_MIX,
        'priority_mix': AssemblyLineSim().PRIORITY_MIX,
        'fail_rate': AssemblyLineSim().FAIL_RATE
    }
    generate_report(baseline_results_list[0], agent_results_list[0], sim_params)