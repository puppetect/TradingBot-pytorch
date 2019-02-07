import numpy as np
import torch
import torch.nn.functional as F


class EpsilonGreedyAgent:
    def __init__(self, net, epsilon, device='cpu'):
        self.net = net
        self.device = device
        self.epsilon = epsilon

    def __call__(self, states):  # states: (N, C, L)
        states_a = np.array(states, copy=False)
        states_v = torch.tensor(states_a).to(self.device)
        q_vals = self.net(states_v)  # (N, A)
        action_idx = q_vals.max(1)[1].data.cpu().numpy()  # (N,)
        mask = np.random.random(size=len(q_vals)) < self.epsilon
        action_idx[mask] = np.random.choice(q_vals.shape[1], sum(mask))
        return action_idx  # (N,)


class ProbabilityAgent:
    def __init__(self, net, apply_softmax=False, device='cpu'):
        self.net = net
        self.apply_softmax = apply_softmax
        self.device = device

    def __call__(self, states):  # state: (N, C, L)
        state_a = np.array(states, copy=False)
        state_v = torch.tensor(state_a).to(self.device)
        logits = self.net(state_v)  # (N, A)
        if self.apply_softmax:
            logits = F.softmax(logits, dim=1).data.cpu().numpy()
        action_idx = []
        for logit in logits:
            action_idx.append(np.random.choice(len(logit), p=logit))
        return np.array(action_idx)  # (N,)
