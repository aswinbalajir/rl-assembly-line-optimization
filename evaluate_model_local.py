"""Evaluate a saved PPO model and collect diagnostics.

Usage (PowerShell):
& .venv\Scripts\python.exe .\evaluate_model_local.py --model ./best_model/best_model.zip --vecstats vec_normalize_stats.pkl --episodes 20

Outputs per-episode metrics and a consolidated chosen-index histogram.
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from assembly_line_env import AssemblyLineEnv
from collections import Counter


def make_env():
    return AssemblyLineEnv(randomize=False)


def evaluate(model_path, vecstats_path, episodes=20, seeds=None):
    # Create eval env and load normalization
    eval_env = DummyVecEnv([lambda: Monitor(AssemblyLineEnv(randomize=False))])
    eval_env = VecNormalize.load(vecstats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = True

    model = PPO.load(model_path)

    chosen_index_counter = Counter()
    total_completed = 0
    total_late_high = 0.0
    total_late_low = 0.0
    total_skip_pen = 0.0
    total_steps = 0

    for ep in range(episodes):
        seed = None if seeds is None else seeds[ep % len(seeds)]
        if seed is not None:
            eval_env.seed(seed)
        obs, _ = eval_env.reset()
        ep_completed = 0
        ep_late_high = 0.0
        ep_late_low = 0.0
        ep_skip_pen = 0.0
        ep_steps = 0

        done = False
        while not done and ep_steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            info = info[0]
            # action may be array like [[i,f,o]] or [i,f,o]
            a = action[0] if isinstance(action, (list, tuple, np.ndarray)) and np.array(action).ndim>1 else action
            idx = int(np.array(a)[0])
            chosen_index_counter[idx] += 1
            ep_skip_pen += float(info.get('skip_penalty', 0.0))
            ep_late_high += float(info.get('late_minutes_high', 0.0))
            ep_late_low += float(info.get('late_minutes_low', 0.0))
            newly = info.get('newly_completed_parts', [])
            ep_completed += len(newly)
            ep_steps += 1
            done = bool(terminated) or bool(truncated)

        total_completed += ep_completed
        total_late_high += ep_late_high
        total_late_low += ep_late_low
        total_skip_pen += ep_skip_pen
        total_steps += ep_steps

        print(f"Episode {ep+1}/{episodes}: steps={ep_steps} completed={ep_completed} lateH={ep_late_high:.1f} lateL={ep_late_low:.1f} skip_pen={ep_skip_pen:.2f}")

    print("\n=== Summary ===")
    print(f"episodes={episodes} total_steps={total_steps} total_completed={total_completed}")
    print(f"avg_completed_per_ep={total_completed/episodes:.2f}")
    print(f"avg_skip_pen_per_step={total_skip_pen/total_steps:.4f}")
    print(f"avg_late_high_per_ep={total_late_high/episodes:.2f} minutes")
    print(f"avg_late_low_per_ep={total_late_low/episodes:.2f} minutes")
    print("Chosen-index distribution (counts):")
    for idx, cnt in sorted(chosen_index_counter.items()):
        print(f"  idx {idx}: {cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved PPO model zip')
    parser.add_argument('--vecstats', required=True, help='Path to vec normalize stats .pkl')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seeds', nargs='*', type=int, default=None)
    args = parser.parse_args()
    evaluate(args.model, args.vecstats, episodes=args.episodes, seeds=args.seeds)
