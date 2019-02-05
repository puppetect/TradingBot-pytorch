import torch
import torch.nn as nn
import numpy as np


def dqn_unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        last_states.append(np.array(exp.last_state, copy=False)
                           if exp.last_state is not None else np.array(exp.state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def dqn_loss(batch, net, tgt_net, gamma, device='cpu', double=True):
    states, actions, rewards, dones, last_states = dqn_unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    last_states_v = torch.tensor(last_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_v = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    if double:
        last_state_actions = net(last_states_v).max(1)[1]
        last_state_action_values = tgt_net(last_states_v).gather(1, last_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        last_state_action_values = tgt_net(last_states_v).max(1)[0]
    last_state_action_values[dones_v] = 0.0
    expected_state_action_values = last_state_action_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def policy_loss(batch, net, device='cpu'):
