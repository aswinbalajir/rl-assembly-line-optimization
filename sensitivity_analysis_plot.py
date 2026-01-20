import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your existing classes
from assembly_line_env import AssemblyLineEnv
from train_and_evaluate import BaselineModel, SmartBaseline

# --- Configuration ---
MODEL_PATH = "./best_model/best_model.zip"
EVAL_DURATION_HOURS = 168  # 1 Week
SEEDS = [42, 101, 888, 999, 1234]  # Run 5 seeds to smooth the zig-zags
OT_COST_RANGE = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    print("Loading trained agent...")
    rl_model = PPO.load(MODEL_PATH)
    
    # --- 1. USE EXISTING AGENTS ---
    fifo = BaselineModel()  # The "Weak" Baseline (Never uses OT)
    smart = SmartBaseline() # The "Aggressive" Baseline (Always uses OT for High Prio)
    
    results_data = []

    print(f"\n--- RUNNING MECHANISM CHECK (Usage vs. Cost) ---")
    
    for ot_cost in OT_COST_RANGE:
        print(f"Testing Incentive: Penalty -{ot_cost}...")
        for seed in SEEDS:
            # Inject the specific cost into the environment
            custom_reward_config = {'penalty_overtime': ot_cost}
            env_creator = lambda: AssemblyLineEnv(randomize=False, reward_config=custom_reward_config)
            env = DummyVecEnv([env_creator])
            env.seed(seed) 
            
            step_duration = env.get_attr("step_duration", indices=0)[0]
            n_steps = int((EVAL_DURATION_HOURS * 60) / step_duration)

            # Compare all 3 Strategies
            agents = {
                "FIFO (Passive)": fifo,
                "Smart (Aggressive)": smart,
                "RL Agent (Adaptive)": rl_model
            }

            for agent_name, agent_model in agents.items():
                obs = env.reset()
                total_ot_hours = 0.0
                
                for _ in range(n_steps):
                    action, _ = agent_model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    
                    # --- TRACK USAGE (HOURS), NOT REWARD ---
                    # The 'overtime_active' flag in info tells us if OT was actually charged
                    if info[0].get('overtime_active'):
                        total_ot_hours += (step_duration / 60.0)

                results_data.append({
                    "Overtime Cost": ot_cost,
                    "Total OT Hours": total_ot_hours,
                    "Agent": agent_name
                })

    # --- PLOTTING ---
    print("\nGenerating Mechanism Proof Chart...")
    df = pd.DataFrame(results_data)
    
    sns.set_style("whitegrid") 
    plt.figure(figsize=(10, 6))
    
    # Colors: Red (Passive), Yellow (Aggressive), Blue (Smart/Adaptive)
    colors = {"FIFO (Passive)": "#e74c3c", "Smart (Aggressive)": "#f1c40f", "RL Agent (Adaptive)": "#2980b9"}

    # Use LINEPLOT to clearly show the behavior curves
    sns.lineplot(
        data=df, 
        x="Overtime Cost", 
        y="Total OT Hours", 
        hue="Agent",
        style="Agent",
        palette=colors,
        markers=True,
        dashes=False,
        linewidth=3
    )

    plt.title("Mechanism Check: Overtime Usage vs. Incentive", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Overtime Penalty Cost (Negative $)", fontsize=12, fontweight='bold')
    plt.ylabel("Overtime Hours Used (per Week)", fontsize=12, fontweight='bold')
    plt.legend(title="Strategy", fontsize=11)
    
    # Clean Layout
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    save_path = "mechanism_proof_usage.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()