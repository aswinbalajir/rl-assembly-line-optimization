# test_best_model.py (CORRECTED)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from assembly_line_env import AssemblyLineEnv
import numpy as np
import os

# 1. Load Environment (RAW - No Normalization)
# We removed VecNormalize because we fixed the rewards manually.
env = DummyVecEnv([lambda: AssemblyLineEnv(randomize=False)])

# 2. Load the Best Model
model_path = "ppo_assembly_line_model" # Or "./best_model/best_model.zip"
if not os.path.exists(model_path + ".zip"):
    model_path = "./best_model/best_model"

print(f"Loading model from: {model_path}")
model = PPO.load(model_path) 

print("--- Running Deterministic Evaluation ---")
obs = env.reset()
total_reward = 0

# Run for exactly 1000 steps (one full episode)
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True) 
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    
    if done[0]:
        print(f"Episode finished at step {i+1}")
        break

print(f"Final True Score: {total_reward}")

if total_reward > 300:
    print("✅ RESULT: PASSED. Ready for Dissertation/Production.")
else:
    print("❌ RESULT: FAILED. Model is still unstable.")