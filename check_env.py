# In check_env.py

from stable_baselines3.common.env_checker import check_env
from assembly_line_env import AssemblyLineEnv

# Create an instance of your environment
env = AssemblyLineEnv()

# This will run a series of tests to make sure your environment is compatible
try:
    check_env(env)
    print(" Environment check passed!")
except Exception as e:
    print(" Environment check failed:")
    print(e)