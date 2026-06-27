"""Smoke test: can the Unity Reacher environment launch + step inside the container?"""
from unityagents import UnityEnvironment
import numpy as np

print(">>> creating UnityEnvironment (headless)...", flush=True)
env = UnityEnvironment(file_name="/app/Reacher_Linux_NoVis/Reacher.x86_64", no_graphics=True)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)
states = env_info.vector_observations
action_size = brain.vector_action_space_size
state_size = states.shape[1]

print(">>> CONNECTED OK", flush=True)
print("    num_agents :", num_agents)
print("    state_size :", state_size)
print("    action_size:", action_size)

actions = np.clip(np.random.randn(num_agents, action_size), -1, 1)
env_info = env.step(actions)[brain_name]
print(">>> stepped once; sample reward:", env_info.rewards[0], flush=True)

env.close()
print(">>> DONE", flush=True)
