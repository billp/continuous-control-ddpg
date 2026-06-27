"""
Continue DDPG training from saved weights until the Reacher env is officially
solved (avg score >= 30 over 100 consecutive episodes). Seeds the rolling window
with the prior run's scores so the trailing average is continuous.
"""
import numpy as np
import torch
from collections import deque

from unityagents import UnityEnvironment
from ddpg_agent import Agent

BINARY = "/app/Reacher_Linux_NoVis/Reacher.x86_64"


def main(extra_episodes=120, max_t=1000, target=30.0):
    env = UnityEnvironment(file_name=BINARY, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    agent = Agent(state_size=state_size, action_size=action_size,
                  num_agents=num_agents, random_seed=0)

    # load saved weights into local AND target networks
    sd_a = torch.load("/app/checkpoint_actor.pth", map_location="cpu")
    sd_c = torch.load("/app/checkpoint_critic.pth", map_location="cpu")
    agent.actor_local.load_state_dict(sd_a);  agent.actor_target.load_state_dict(sd_a)
    agent.critic_local.load_state_dict(sd_c); agent.critic_target.load_state_dict(sd_c)
    print("loaded saved weights", flush=True)

    prior = list(np.load("/app/scores.npy"))
    scores_deque = deque(prior[-99:], maxlen=100)   # continuous trailing window
    scores_all = list(prior)
    start = len(prior)

    for k in range(1, extra_episodes + 1):
        i_episode = start + k
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
        if avg >= target:
            print("\n*** SOLVED at episode {} \tAvg(100): {:.2f} ***".format(i_episode, avg), flush=True)
            break

    np.save("/app/scores.npy", np.array(scores_all))
    torch.save(agent.actor_local.state_dict(), "/app/checkpoint_actor.pth")
    torch.save(agent.critic_local.state_dict(), "/app/checkpoint_critic.pth")
    env.close()
    print("done; total episodes: {}".format(len(scores_all)), flush=True)


if __name__ == "__main__":
    main()
