from lib import data, environ, models
from common import agent, experience, loss
from datetime import datetime
import os
import time
import logging
import gym
import numpy as np
import torch
import torch.optim as optim

from tensorboardX import SummaryWriter


BATCH_SIZE = 32
BARS_COUNT = 100
DEFAULT_FILE = 'data/000001_2017.csv'

GAMMA = 0.9
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
TARGET_NET_SYNC = 1000

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_STEPS = 1000000

device = 'cuda'
datestr = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
save_path = os.path.join('saves', datestr)
os.makedirs(save_path, exist_ok=True)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',
                    handlers=[logging.FileHandler(os.path.join(save_path, 'console.log')),
                              logging.StreamHandler()])

stock_data = data.read_csv(file_name=DEFAULT_FILE)
env = environ.StockEnv(stock_data, bars_count=BARS_COUNT)
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
writer = SummaryWriter(comment='-stock-dqconv-')
net = models.DQNConv1d(env.observation_space.shape, env.action_space.n).to(device)
tgt_net = models.DQNConv1d(env.observation_space.shape, env.action_space.n).to(device)

agent = agent.EpsilonGreedyAgent(net, env, epsilon=EPSILON_START, device=device)
exp_source = experience.ExperienceSource(env, agent, GAMMA, steps_count=REWARD_STEPS)
buffer = experience.ExperienceBuffer(exp_source, REPLAY_SIZE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

total_rewards = []
frame_idx = 0
frame_prev = 0
ts = time.time()

eval_states = None
best_mean_reward = None

while True:
    frame_idx += 1
    buffer.populate(1)
    agent.epsilon = max(EPSILON_FINAL,
                        EPSILON_START - frame_idx / EPSILON_STEPS)
    reward = exp_source.pop_episode_reward()
    if reward:
        total_rewards.append(reward)
        speed = (frame_idx - frame_prev) / (time.time() - ts)
        frame_prev = frame_idx
        ts = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        logger.info('%d done %d games, mean reward %.3f, epsilon %.2f, speed %.2f f/s'
                    % (frame_idx, len(total_rewards), mean_reward, agent.epsilon, speed))
        writer.add_scalar('epsilon', agent.epsilon, frame_idx)
        writer.add_scalar('speed', speed, frame_idx)
        writer.add_scalar('mean_reward', mean_reward, frame_idx)
        writer.add_scalar('reward', reward, frame_idx)

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
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), os.path.join(save_path, 'mean_val-%.3f.data' % mean_val))
            if best_mean_reward is not None:
                logger.info('Best mean reward update %.3f -> %.3f'
                            % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss = loss.calc_loss(batch, net, tgt_net, GAMMA**REWARD_STEPS, device=device)
    loss.backward()
    optimizer.step()

    if frame_idx % TARGET_NET_SYNC == 0:
        tgt_net.load_state_dict(net.state_dict())

writer.close()
