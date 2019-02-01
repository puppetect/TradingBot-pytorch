from lib import environ, models, validation
from common import agent, experience, helper
from datetime import datetime
import os
import time
import logging
import gym
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from common.writer import SummaryWriter


BATCH_SIZE = 32
BARS_COUNT = 100
REWARD_GROUPS = 100

GAMMA = 0.9
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
TARGET_NET_SYNC = 1000

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000
VALIDATION_EVERY_STEP = 100000

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_STEPS = 1000000


device = 'cuda'
run_local = False
datestr = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
save_path = os.path.join('saves', datestr)
os.makedirs(save_path, exist_ok=True)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',
                    handlers=[logging.FileHandler(os.path.join(save_path, 'console.log')),
                              logging.StreamHandler()])


try:
    from lib import data
    train_data = data.read_csv(file_name='data/000001_2017.csv')
    val_data = data.read_csv(file_name='data/000001_2018.csv')
except ModuleNotFoundError:
    train_data = (pd.read_csv('data/prices_2017.csv', index_col=0),
                  pd.read_csv('data/factors_2017.csv', index_col=0))
    val_data = (pd.read_csv('data/prices_2018.csv', index_col=0),
                pd.read_csv('data/factors_2018.csv', index_col=0))

env = environ.StockEnv(train_data, bars_count=BARS_COUNT, reset_on_close=True)
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
env_test = environ.StockEnv(train_data, bars_count=BARS_COUNT, reset_on_close=True)
env_test = gym.wrappers.TimeLimit(env_test, max_episode_steps=1000)
env_val = environ.StockEnv(val_data, bars_count=BARS_COUNT, reset_on_close=True)
env_val = gym.wrappers.TimeLimit(env_val, max_episode_steps=1000)

writer = SummaryWriter('runs')

net = models.DQNConv1d(env.observation_space.shape, env.action_space.n).to(device)
tgt_net = models.DQNConv1d(env.observation_space.shape, env.action_space.n).to(device)

agent = agent.EpsilonGreedyAgent(net, env, epsilon=EPSILON_START, device=device)
exp_source = experience.ExperienceSource(env, agent, GAMMA, steps_count=REWARD_STEPS)
buffer = experience.ExperienceBuffer(exp_source, REPLAY_SIZE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

total_reward = []
total_steps = []
reward_buf = []
steps_buf = []
frame_idx = 0
frame_prev = 0
start_time = ts = time.time()

eval_states = None
best_mean_val = None

while True:
    frame_idx += 1
    buffer.populate(1)
    agent.epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_STEPS)
    ep_reward, ep_steps = exp_source.pop_episode_result()
    if ep_reward:
        reward_buf.append(ep_reward)
        steps_buf.append(ep_steps)
        if len(reward_buf) < REWARD_GROUPS:
            continue
        reward = np.mean(reward_buf)
        steps = np.mean(steps_buf)
        reward_buf.clear()
        steps_buf.clear()
        total_reward.append(reward)
        total_steps.append(steps)
        speed = (frame_idx - frame_prev) / (time.time() - ts)
        frame_prev = frame_idx
        ts = time.time()
        mean_reward = np.mean(total_reward[-100:])
        mean_step = np.mean(total_steps[-100:])
        logger.info('%d done %d games, mean reward %.3f, mean step %d, epsilon %.2f, speed %.2f f/s' % (frame_idx, len(total_reward), mean_reward, mean_step, agent.epsilon, speed))
        writer.add_scalar('epsilon', agent.epsilon, frame_idx)
        writer.add_scalar('speed', speed, frame_idx)
        writer.add_scalar('reward', reward, frame_idx)
        writer.add_scalar('reward_100', mean_reward, frame_idx)
        writer.add_scalar('steps', steps, frame_idx)
        writer.add_scalar('steps_100', mean_step, frame_idx)

    if len(buffer) < REPLAY_INITIAL:
        continue

    if eval_states is None:
        logger.info('Initial buffer populated, start training')
        eval_states = buffer.sample(STATES_TO_EVALUATE)
        eval_states = np.array([np.array(exp.state, copy=False)
                                for exp in eval_states], copy=False)

    if frame_idx % EVAL_EVERY_STEP == 0:
        mean_vals = []
        for batch in np.array_split(eval_states, 64):
            states_v = torch.tensor(batch).to(device)
            action_values_v = net(states_v)
            best_action_values_v = action_values_v.max(1)[0]
            mean_vals.append(best_action_values_v.mean().item())
        mean_val = np.mean(mean_vals)
        writer.add_scalar('values_mean', mean_val, frame_idx)
        if best_mean_val is None or best_mean_val < mean_val:
            torch.save(net.state_dict(), os.path.join(save_path, 'best_mean_val.pt'))
            if best_mean_val is not None:
                logger.info('Best mean value updated %.3f -> %.3f'
                            % (best_mean_val, mean_val))
            best_mean_val = mean_val

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss = helper.calc_loss(batch, net, tgt_net, GAMMA**REWARD_STEPS, device=device)
    loss.backward()
    optimizer.step()

    if frame_idx % TARGET_NET_SYNC == 0:
        tgt_net.load_state_dict(net.state_dict())

    if frame_idx % VALIDATION_EVERY_STEP == 0:
        res = validation.run_val(env_test, net, device=device)
        for key, val in res.items():
            writer.add_scalar(key + '_test', val, frame_idx)
        res = validation.run_val(env_val, net, device=device)
        for key, val in res.items():
            writer.add_scalar(key + '_val', val, frame_idx)

    # 8h training time on google colab
    if time.time() - start_time > 28800:
        break

writer.close()
