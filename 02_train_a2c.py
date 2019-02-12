from lib import environ, models
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
REWARD_GROUPS = 10
STATS_GROUPS = 10


GAMMA = 0.99
ENTROPY_BETA = 0.01
REWARD_STEPS = 4
CLIP_GRAD = 0.1
LEARNING_RATE = 0.0001

CHECKPOINT_EVERY_STEP = 1000000
GOOGLE_COLAB_MAX_STEP = 5000000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', default=2018, type=int, help='year of data to train (default: 2018')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--cuda', default=False, action='store_true', help='enable cuda')
    parser.add_argument('--colab', default=False, action='store_true', help='enable colab hosted runtime')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda else 'cpu')

    try:
        from lib import data
        train_data = data.load_data(year=args.year)
    except ModuleNotFoundError:
        # workaround that Ta-lib cannot be installed on Colab
        train_data = (pd.read_csv('data/000001_prices_%d.csv' % args.year, index_col=0),
                      pd.read_csv('data/000001_factors_%d.csv' % args.year, index_col=0))

    env = environ.StockEnv(train_data, bars_count=BARS_COUNT, commission=0.0, reset_on_sell=True, reward_on_empty=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    net = models.A2CConv1d(env.observation_space.shape, env.action_space.n).to(device)

    agent = agent.ProbabilityAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = experience.ExperienceSource(env, agent, GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

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
        best_mean_reward = float(checkpoint['best_mean_reward']),
        net.load_state_dict(checkpoint['state_dict']),
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('==> Loaded %s' % args.resume)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_path, 'console.log')),
                                  logging.StreamHandler()])

    writer = SummaryWriter(os.path.join('runs', file_name))

    for exp in exp_source:
        frame_idx += 1
        batch.append(exp)

        if len(batch) < BATCH_SIZE:
            continue

        optimizer.zero_grad()
        loss_val_v, loss_policy_v, loss_entropy_v = helper.a2c_loss(batch, net, GAMMA**REWARD_STEPS, ENTROPY_BETA, device)
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
        stats['loss_total'].append(loss_v)
        stats['grad_l2'].append(np.sqrt(np.mean(np.square(grads))))
        stats['grad_max'].append(np.max(np.abs(grads)))
        stats['grad_var'].append(np.var(grads))
        for stat in stats:
            if len(stat) >= STATS_GROUPS:
                writer.add_scalar(stat, torch.mean(torch.stack(stats[stat])).item(), frame_idx)
                stats[stat].clear()

        ep_reward, ep_steps = exp_source.pop_episode_result()
        if ep_reward:
            logging.info('%d done, Episode reward: %.4f, Episode step: %d' % (frame_idx, ep_reward, ep_steps))
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
                logging.info('%d done %d games, mean reward %.3f, mean step %d, speed %.2f f/s' % (frame_idx, len(total_reward), mean_reward, mean_step, speed))
                writer.add_scalar('speed', speed, frame_idx)
                writer.add_scalar('reward', reward, frame_idx)
                writer.add_scalar('reward_100', mean_reward, frame_idx)
                writer.add_scalar('steps', steps, frame_idx)
                writer.add_scalar('steps_100', mean_step, frame_idx)
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), os.path.join(save_path, 'best_mean_reward.pth'))
                    try:
                        if best_mean_reward is not None:
                            logging.info('Best mean reward updated %.3f -> %.3f'
                                         % (best_mean_reward, mean_reward))
                    except Exception as e:
                        print(e)
                        pass
                    finally:
                        best_mean_reward = mean_reward

        if frame_idx % CHECKPOINT_EVERY_STEP == 0:
            checkpoint = {'frame_idx': frame_idx,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'total_reward': total_reward,
                          'total_steps': total_steps,
                          'best_mean_reward': best_mean_reward,
                          }
            os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
            torch.save(checkpoint, os.path.join(save_path, 'checkpoints', 'checkpoint-%d.pth' % frame_idx))
            print('==> checkpoint saved at frame %d' % frame_idx)

        # workaround Colab's time limit
        if args.colab:
            if frame_idx % GOOGLE_COLAB_MAX_STEP == 0:
                break

    writer.close()
