import torch
import torch.nn.functional as F
import numpy as np

ENTROPY_BETA = 0.02


def unpack_batch(batch, device='cpu'):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        last_states.append(np.array(exp.last_state, copy=False)
                           if exp.last_state is not None else np.array(exp.state, copy=False))

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_v = torch.ByteTensor(dones).to(device)
    last_states_v = torch.tensor(last_states).to(device)
    return states_v, actions_v, rewards_v, dones_v, last_states_v


def dqn_loss(batch, net, tgt_net, gamma, double=True, device='cpu'):
    states_v, actions_v, rewards_v, dones_v, last_states_v = unpack_batch(batch, device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    if double:
        last_state_actions = net(last_states_v).max(1)[1]
        last_state_action_values = tgt_net(last_states_v).gather(1, last_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        last_state_action_values = tgt_net(last_states_v).max(1)[0]
    last_state_action_values[dones_v] = 0.0
    expected_state_action_values = last_state_action_values.detach() * gamma + rewards_v
    return F.mse_loss(state_action_values, expected_state_action_values)


def a2c_loss(batch, net, gamma, beta=ENTROPY_BETA, device='cpu'):
    states_v, actions_v, rewards_v, dones_v, last_states_v = unpack_batch(batch, device)

    last_vals_v = net(last_states_v)[1].squeeze(-1)
    rewards_v[~dones_v] += gamma * last_vals_v[~dones_v]
    ref_vals_v = rewards_v

    logits_v, vals_v = net(states_v)
    loss_val_v = F.mse_loss(vals_v.squeeze(-1), ref_vals_v)

    log_probs_v = F.log_softmax(logits_v, dim=1)
    adv_v = ref_vals_v - vals_v.squeeze(-1).detach()
    pg_v = adv_v * log_probs_v[range(len(states_v)), actions_v.squeeze(-1)]
    loss_policy_v = -pg_v.mean()

    probs_v = F.softmax(logits_v, dim=1)
    loss_entropy_v = beta * (probs_v * log_probs_v).sum(dim=1).mean()

    return loss_val_v, loss_policy_v, loss_entropy_v
