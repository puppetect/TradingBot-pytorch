import numpy as np
import torch
import torch.nn.functional as F


class BaseAgent:
    """
    Abstract Agent interface
    """

    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


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


class OUProcessAgent(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """

    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_theta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        state_a = np.array(states, dtype=np.float32)
        state_v = torch.tensor(state_a).to(self.device)
        mu_v = self.net(state_v)  # (N, A)
        actions = mu_v.data.cpu().numpy()  # (N, A)

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):  # for (A,) (A,) in (N, A), (N, A)
                if a_state is None:
                    a_state = np.zeros(shape=len(action), dtype=np.float32)
                a_state += self.ou_theta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=len(action))

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        return actions, new_a_states  # (N, A), (N, A)
