import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from assembly_line_env import AssemblyLineEnv
from train_and_evaluate import BaselineModel, SmartBaseline

# --- Configuration ---
MODEL_PATH = "./best_model/best_model.zip"
EVAL_DURATION_HOURS = 168  # 1 Week
NUM_SEEDS = 20             # 20 seeds per scenario for a good "cloud"

# Define Scenarios
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
    "New Product (Complex)": {
        "part_mix": {'Type_A': 0.2, 'Type_B': 0.7, 'Type_C': 0.1},
        "priority_mix": {'HIGH': 0.2, 'LOW': 0.8},
        "fail_rate": 0.08
    }
}

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    print("Loading trained agent...")
    rl_model = PPO.load(MODEL_PATH)
    fifo = BaselineModel()
    smart = SmartBaseline()
    
    data = []
    
    temp_env = AssemblyLineEnv()
    step_duration = temp_env.step_duration
    n_steps = int((EVAL_DURATION_HOURS * 60) / step_duration)
    
    print("\n--- RUNNING SCENARIO PARETO ANALYSIS ---")

    for scenario_name, config in TEST_SCENARIOS.items():
        print(f"Testing: {scenario_name}...")
        
        agents = {
            "FIFO": fifo,
            "Smart (Aggressive)": smart,
            "RL Agent": rl_model
        }
        
        for agent_name, agent_model in agents.items():
            for seed in range(NUM_SEEDS):
                # Setup Environment
                env_creator = lambda: AssemblyLineEnv(
                    part_mix=config['part_mix'],
                    priority_mix=config['priority_mix'],
                    fail_rate=config['fail_rate'],
                    randomize=False
                )
                env = DummyVecEnv([env_creator])
                env.seed(42 + seed)
                
                obs = env.reset()
                total_ot = 0.0
                completed_parts = []
                
                for _ in range(n_steps):
                    action, _ = agent_model.predict(obs, deterministic=True)
                    obs, _, _, info = env.step(action)
                    
                    if info[0].get('overtime_active'):
                        total_ot += (step_duration / 60.0)
                    if info[0].get('newly_completed_parts'):
                        completed_parts.extend(info[0]['newly_completed_parts'])
                
                # Metrics
                high_prio = [p for p in completed_parts if p['priority'] == 1]
                if high_prio:
                    late = sum(1 for p in high_prio if p['is_late'])
                    otd = ((len(high_prio) - late) / len(high_prio)) * 100
                else:
                    otd = 0.0
                
                data.append({
                    "Agent": agent_name,
                    "Scenario": scenario_name,
                    "Overtime Hours": total_ot,
                    "High Prio OTD %": otd
                })

    # --- PLOTTING: FACETED PARETO CLOUDS ---
    print("Generating Faceted Pareto Plot...")
    df = pd.DataFrame(data)
    
    # Calculate Medians for the "Big Dot"
    df_med = df.groupby(["Agent", "Scenario"])[["Overtime Hours", "High Prio OTD %"]].median().reset_index()

    sns.set_style("whitegrid")
    
    # Define Colors
    palette = {"FIFO": "#e74c3c", "Smart (Aggressive)": "#f1c40f", "RL Agent": "#2ecc71"}
    
    # Initialize FacetGrid
    g = sns.FacetGrid(df, col="Scenario", hue="Agent", 
                      col_wrap=2, height=4, aspect=1.3,
                      palette=palette)
    
    # 1. Plot the "Cloud" (Small, Transparent Dots)
    g.map(sns.scatterplot, "Overtime Hours", "High Prio OTD %", alpha=0.3, s=40)
    
    # 2. Overlay the "Median" (Big, Solid Dots)
    # We loop through axes to manually add the medians on top of each subplot
    for ax, (scenario_name, sub_df) in zip(g.axes.flatten(), df_med.groupby("Scenario")):
        sns.scatterplot(data=sub_df, x="Overtime Hours", y="High Prio OTD %", 
                        hue="Agent", palette=palette, ax=ax, 
                        s=200, edgecolor='black', linewidth=1.5, legend=False, zorder=10)
        ax.set_title(scenario_name, fontweight='bold')

    # Formatting
    g.set_axis_labels("Cost: Overtime Hours", "Speed: High Prio OTD %")
    g.set(xlim=(-2, 170), ylim=(0, 105)) # Fixed scale for easy comparison
    for ax in g.axes.flatten():
        ax.axvspan(60, 170, color='red', alpha=0.05, zorder=0)
        ax.text(160, 5, "Excessive\nCost Zone", color='red', alpha=0.5, ha='right', fontsize=8)
    
    # Single Legend
    g.add_legend(title="Agent Strategy", fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    plt.tight_layout()
    save_path = "scenario_pareto_faceted.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()