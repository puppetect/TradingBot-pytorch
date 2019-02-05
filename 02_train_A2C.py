from lib import environ, models, validation
from common import agent, experience, helper
import os
import time
import logging
import argparse
import collections
import gym
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils

from common.writer import SummaryWriter

BATCH_SIZE = 32
BARS_COUNT = 50
REWARD_GROUPS = 100
STATS_GROUPS = 10


GAMMA = 0.99
ENTROPY_BETA = 0.01
REWARD_STEPS = 4
CLIP_GRAD = 0.1
LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000
VALIDATION_EVERY_STEP = 100000
CHECKPOINT_EVERY_STEP = 50000
GOOGLE_COLAB_MAX_STEP = 500000

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_STEPS = 1000000


def calc_qvals(rewards):
    sum = 0
    buf = []
    for reward in reversed(rewards):
        sum *= GAMMA
        sum += reward
        buf.append(sum)
    return list(reversed(buf))


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--cuda', default=False, action='store_true', help='enable cuda')
parser.add_argument('--colab', default=False, action='store_true', help='enable colab hosted runtime')
args = parser.parse_args()

device = torch.device('cuda' if args.cuda else 'cpu')

try:
    from lib import data
    train_data = data.read_csv(file_name='data/000001_2017.csv')
    val_data = data.read_csv(file_name='data/000001_2018.csv')
except ModuleNotFoundError:
    train_data = (pd.read_csv('data/prices_2017.csv', index_col=0),
                  pd.read_csv('data/factors_2017.csv', index_col=0))
    val_data = (pd.read_csv('data/prices_2018.csv', index_col=0),
                pd.read_csv('data/factors_2018.csv', index_col=0))

env = environ.StockEnv(train_data, bars_count=BARS_COUNT, reset_on_sell=True)
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
env_test = environ.StockEnv(train_data, bars_count=BARS_COUNT, reset_on_sell=True)
env_test = gym.wrappers.TimeLimit(env_test, max_episode_steps=1000)
env_val = environ.StockEnv(val_data, bars_count=BARS_COUNT, reset_on_sell=True)
env_val = gym.wrappers.TimeLimit(env_val, max_episode_steps=1000)

net = models.A2CConv1d(env.observation_space.shape, env.action_space.n).to(device)

agent = agent.ProbabilityAgent(net, env, apply_softmax=True, device=device)
exp_source = experience.ExperienceSource(env, agent, GAMMA)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

total_reward = []
total_steps = []
reward_buf = []
steps_buf = []
batch = []
frame_idx = 0
frame_prev = 0
ts = time.time()
stats = collections.defaultdict(list)
best_mean_reward = None

file_name = os.path.splitext(os.path.basename(__file__))[0]
save_path = os.path.join('saves', file_name)
os.makedirs(save_path, exist_ok=True)

if args.resume:
    print('==> Loading %s' % args.resume)
    checkpoint = torch.load(os.path.join(save_path, 'checkpoints', args.resume))
    total_reward = checkpoint['total_reward']
    total_steps = checkpoint['total_steps']
    frame_idx = checkpoint['frame_idx']
    best_mean_reward = checkpoint['best_mean_reward'],
    stats = checkpoint['stats'],
    net.load_state_dict(checkpoint['state_dict']),
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('==> Loaded %s' % args.resume)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',
                    handlers=[logging.FileHandler(os.path.join(save_path, 'console.log')),
                              logging.StreamHandler()])

writer = SummaryWriter(os.path.join('runs', file_name))


for exp in iter(exp_source):
    frame_idx += 1
    batch.append(exp)

    if len(batch) < BATCH_SIZE:
        continue

    optimizer.zero_grad()
    loss_val_v, loss_policy_v, loss_entropy_v = helper.a2c_loss(batch, net, GAMMA**REWARD_STEPS, ENTROPY_BETA, device=device)
    batch.clear()
    loss_policy_v.backward(retain_graph=True)
    grads = np.concatenate([p.grad.data.cpu().numpy().flatten() for p in net.parameters() if p.grad is not None])
    loss_v = loss_entropy_v + loss_val_v
    loss_v.backward()
    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    optimizer.step()
    loss_v += loss_policy_v

    stats['loss_value'].append(loss_val_v)
    stats['loss_policy'].append(loss_policy_v)
    stats['loss_entropy'].append(loss_entropy_v)
    stats['loss'].append(loss_v)
    stats['grad_l2'].append(np.sqrt(np.mean(np.square(grads))))
    stats['grad_max'].append(np.max(np.abs(grads)))
    stats['grad_var'].append(np.var(grads))
    for stat in stats:
        if len(stat) >= STATS_GROUPS:
            writer.add_scalar(stat, np.mean(stats[stat]), frame_idx)
            stats[stat].clear()

    ep_reward, ep_steps = exp_source.pop_episode_result()
    if ep_reward:
        reward_buf.append(ep_reward)
        steps_buf.append(ep_steps)
        if len(reward_buf) == REWARD_GROUPS:
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
            logging.info('%d done %d games, mean reward %.3f, mean step %d, epsilon %.2f, speed %.2f f/s' % (frame_idx, len(total_reward), mean_reward, mean_step, agent.epsilon, speed))
            writer.add_scalar('epsilon', agent.epsilon, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)
            writer.add_scalar('reward_100', mean_reward, frame_idx)
            writer.add_scalar('steps', steps, frame_idx)
            writer.add_scalar('steps_100', mean_step, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), os.path.join(save_path, 'best_mean_reward.pth'))
                if best_mean_reward is not None:
                    logging.info('Best mean value updated %.3f -> %.3f'
                                 % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward

    if frame_idx % VALIDATION_EVERY_STEP == 0:
        res = validation.run_val(env_test, net, device=device)
        logging.info('%d test done, reward %.3f, step %d' % (frame_idx, res['episode_rewards'], res['episode_steps']))
        for key, val in res.items():
            writer.add_scalar(key + '_test', val, frame_idx)
        res = validation.run_val(env_val, net, device=device)
        logging.info('%d validation done, reward %.3f, step %d' % (frame_idx, res['episode_rewards'], res['episode_steps']))
        for key, val in res.items():
            writer.add_scalar(key + '_val', val, frame_idx)

    if frame_idx % CHECKPOINT_EVERY_STEP == 0:
        checkpoint = {'frame_idx': frame_idx,
                      'state_dict': net.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'total_reward': total_reward,
                      'total_steps': total_steps,
                      'best_mean_reward': best_mean_reward,
                      'stats': stats}
        torch.save(checkpoint, os.path.join(save_path, 'checkpoints', 'checkpoint-%d.pth' % frame_idx))
        print('==> checkpoint saved at frame %d' % frame_idx)

    # workaround Colab's time limit
    if args.colab:
        if frame_idx % GOOGLE_COLAB_MAX_STEP == 0:
            break

writer.close()
