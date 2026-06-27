"""
Headless DDPG training on the Unity Reacher (20-agent) environment.
Mirrors the Continuous_Control.ipynb logic; runs inside the linux/amd64 container.
"""
import numpy as np
import torch
from collections import deque

from unityagents import UnityEnvironment
from ddpg_agent import Agent

BINARY = "/app/Reacher_Linux_NoVis/Reacher.x86_64"


def main(n_episodes=200, max_t=1000, target_score=30.0):
    env = UnityEnvironment(file_name=BINARY, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    print("num_agents={} state_size={} action_size={}".format(num_agents, state_size, action_size), flush=True)

    agent = Agent(state_size=state_size, action_size=action_size,
                  num_agents=num_agents, random_seed=0)

    scores_deque = deque(maxlen=100)
    scores_all = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones, t)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        score = float(np.mean(scores))
        scores_deque.append(score)
        scores_all.append(score)
        avg = float(np.mean(scores_deque))
        print("Episode {:3d}\tAvg(100): {:6.2f}\tEpisode: {:6.2f}".format(i_episode, avg, score), flush=True)
        if avg >= target_score and len(scores_deque) >= 100:
            print("\nSOLVED in {} episodes!\tAvg(100): {:.2f}".format(i_episode, avg), flush=True)
            torch.save(agent.actor_local.state_dict(), "/app/checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "/app/checkpoint_critic.pth")
            break

    np.save("/app/scores.npy", np.array(scores_all))
    torch.save(agent.actor_local.state_dict(), "/app/checkpoint_actor.pth")
    torch.save(agent.critic_local.state_dict(), "/app/checkpoint_critic.pth")
    env.close()
    print("Training finished. Episodes run: {}".format(len(scores_all)), flush=True)


if __name__ == "__main__":
    import sys
    eps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n_episodes=eps)
