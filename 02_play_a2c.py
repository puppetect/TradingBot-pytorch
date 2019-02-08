import os
import argparse
from datetime import datetime, date
import numpy as np
import pandas as pd
from lib import environ, models
from common import agent

import torch

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

EPSILON = 0.02
DEFAULT_COMMISSION = 0.00025
BARS_COUNT = 50
YEAR = 2018

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, default=YEAR, help='year of data to play, default=2018')
# parser.add_argument('-m', '--model', type=str, required=True, help='model to play')
parser.add_argument('--commission', type=float, default=DEFAULT_COMMISSION, help='commission size, default=0.00025')
parser.add_argument('--cuda', default=False, action='store_true', help='enable cuda')
args = parser.parse_args()

device = 'cuda' if args.cuda else 'cpu'

try:
    from lib import data
    play_data = data.read_csv(file_name='data/000001_%d.csv' % args.year)
except ModuleNotFoundError:
    play_data = (pd.read_csv('data/prices_%d.csv' % args.year, index_col=0),
                 pd.read_csv('data/factors_%d.csv' % args.year, index_col=0))

env = environ.StockEnv(play_data, bars_count=BARS_COUNT, commission=args.commission, reset_on_sell=False, random_ofs_on_reset=False)
net = models.A2CConv1d(env.observation_space.shape, env.action_space.n).to(device)
agent = agent.ProbabilityAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
save_path = os.path.join('saves', '02_train_a2c')
state_dict = torch.load(os.path.join(save_path, 'best_mean_reward-0.303.pth'), map_location=lambda storage, loc: storage)
net.load_state_dict(state_dict)

obs = env.reset()
start_price = env.state._close()
total_reward = 0.0
frame_idx = 0
rewards = []

while True:
    frame_idx += 1
    obs_v = torch.tensor([obs]).to(device)
    action_idx = agent(obs_v)

    obs, reward, done, _ = env.step(action_idx)
    total_reward += reward
    rewards.append(total_reward)
    if frame_idx % 100 == 0:
        print('%d: reward=%.3f' % (frame_idx, total_reward))
    if done:
        break

plt.clf()
plt.plot(rewards)
plt.title('%s Total Reward, comm %.3f' % (args.year, args.commission * 100))
plt.ylabel('Reward, %')
plt.savefig('dqn-rewards-%s-comm_%.3f.png' % (args.year, args.commission * 100))
