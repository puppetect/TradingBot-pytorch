import numpy as np
import torch
from lib import environ
import logging


def run_val(env, net, episodes=100, device='cpu', epsilon=0.02):
    stats = {'episode_rewards': [],
             'episode_steps': []}

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        total_step = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            state_value_v = net(obs_v)
            action_idx = state_value_v.max(1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            total_step += 1
            if done:
                break

        stats['episode_rewards'].append(total_reward)
        stats['episode_steps'].append(total_step)
        logging

    return {key: np.mean(vals) for key, vals in stats.items()}
