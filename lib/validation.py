import numpy as np
import torch
from lib import environ


def run_val(env, net, episodes=100, device='cpu', epsilon=0.02, commission=0.00025):
    episode_rewards, episode_steps, order_profits, order_steps = [], [], [], []

    for episode in range(episodes):
        obs = [env.reset()]
        total_reward = 0.0
        # total_steps = 0
        hold_price = None
        hold_steps = None

        while True:
            obs_v = torch.tensor(obs).to(device)
            state_value_v = net(obs_v)
            action_idx = state_value_v.max(1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            close_price = env.state._close()

            if action == environ.Actions.buy and hold_price is None:
                hold_price = close_price
                hold_steps = 0
            if action == environ.Actions.sell and hold_price is not None:
                end_price = close_price - hold_price - (close_price + hold_price) * commission
                profit = 100 * end_price / hold_price
                order_profits.append(profit)
                order_steps.append(hold_steps)
                hold_price = None
                hold_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            # total_steps += 1
            if hold_steps is not None:
                hold_steps += 1
            if done:
                if hold_price is not None:
                    end_price = close_price - hold_price - (close_price + hold_price) * commission
                    profit = 100 * end_price / hold_price
                    order_profits.append(profit)
                    order_steps.append(hold_steps)
                break

        episode_rewards.append(total_reward)
        # episode_steps.append(total_steps)

    return {'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'order_profits': order_profits,
            'order_steps': order_steps}
