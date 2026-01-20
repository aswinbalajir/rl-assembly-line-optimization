from assembly_line_env import AssemblyLineEnv
import random

env = AssemblyLineEnv()
obs, info = env.reset()

steps = 200
sum_reward = 0.0
sum_skip = 0.0
sum_late_pen = 0.0
completed = 0
for i in range(steps):
    a = [random.randint(0,9), random.randint(0,1), random.randint(0,1)]
    obs, r, term, trunc, info = env.step(a)
    sum_reward += r
    sum_skip += info.get('skip_penalty',0.0)
    sum_late_pen += info.get('lateness_penalty',0.0)
    completed += len(info.get('newly_completed_parts',[]))

print(f'steps={steps} avg_reward={sum_reward/steps:.3f} avg_skip={sum_skip/steps:.3f} avg_late_pen={sum_late_pen/steps:.3f} completed_parts={completed}')
