import numpy as np
import torch


class EpsilonGreedyAgent:
    def __init__(self, net, env, epsilon, device='cuda'):
        self.net = net
        self.env = env
        self.device = device
        self.epsilon = epsilon

    def __call__(self, state):
        if np.random.random() < self.epsilon:
            action_idx = self.env.action_space.sample()
        else:
            state_a = np.array(state, copy=False)
            state_v = torch.tensor(state_a).to(self.device)
            qval = self.net(state_v)
            action_idx = qval.max(1)[1].item()
        return action_idx
