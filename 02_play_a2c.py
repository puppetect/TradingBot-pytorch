import os
import argparse
import pandas as pd
from datetime import datetime
from lib import environ, models
from common import agent

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


EPSILON = 0.02
DEFAULT_COMMISSION = 0.00025
BARS_COUNT = 50
YEAR = 2018

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', type=int, default=YEAR, help='year of data to play (default=2018)')
    parser.add_argument('-m', '--model', type=str, required=True, help='model to play')
    parser.add_argument('--commission', type=float, default=DEFAULT_COMMISSION, help='commission size (default=0.00025)')
    parser.add_argument('--cuda', default=False, action='store_true', help='enable cuda')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda else 'cpu')

    play_data = (pd.read_csv('data/000001_prices_%d.csv' % args.year, index_col=0),
                 pd.read_csv('data/000001_factors_%d.csv' % args.year, index_col=0))
    env = environ.StockEnv(play_data, bars_count=BARS_COUNT, commission=args.commission, reset_on_sell=False, random_ofs_on_reset=False, reward_on_empty=False, play=True)
    net = models.A2CConv1d(env.observation_space.shape, env.action_space.n).to(device)
    agent = agent.ProbabilityAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    state_dict = torch.load(args.model, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    obs = env.reset()
    init_close = env.state._close()

    rewards = 0.0
    frame_idx = 0
    strategy_rewards = []
    benchmark_rewards = []

    while True:
        frame_idx += 1
        obs_v = torch.tensor([obs]).to(device)
        out_v = net(obs_v)[0]
        # print(out_v)
        action_idx = out_v.max(1)[1].item()
        # action_idx = agent(obs_v)
        obs, reward, done, _ = env.step(action_idx)
        rewards += reward

        strategy_rewards.append(rewards)

        cur_close = env.state._close()
        benchmark_reward = 100 * (cur_close - init_close) / init_close
        benchmark_rewards.append(benchmark_reward)
        if frame_idx % 100 == 0:
            print('%d: reward=%.3f, benchmark_reward=%.3f' % (frame_idx, rewards, benchmark_reward))
        if done:
            break

    file_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = os.path.join('plots', file_name)
    os.makedirs(save_dir, exist_ok=True)

    save_date = datetime.strftime(datetime.now(), '%m%d-%H%M')

    plt.plot(benchmark_rewards, linewidth=0.5, label='benchmark')
    plt.plot(strategy_rewards, linewidth=0.5, label='strategy')

    plt.title('%s Total Reward, comm %.3f' % (args.year, args.commission * 100))
    plt.ylabel('Reward, %')

    plt.legend()
    plt.savefig(os.path.join(save_dir, 'a2c-rewards-%s-comm_%.3f_%s.png' % (args.year, args.commission * 100, save_date)))
