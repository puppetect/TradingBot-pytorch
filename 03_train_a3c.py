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
import torch.multiprocessing as mp

from common.writer import SummaryWriter


GRAD_BATCH = 64
TRAIN_BATCH = 2

BARS_COUNT = 50
REWARD_GROUPS = 50
STATS_GROUPS = 10


GAMMA = 0.99
ENTROPY_BETA = 0.01
REWARD_STEPS = 4
CLIP_GRAD = 0.1
LEARNING_RATE = 0.0001

PROCESSES_COUNT = 4
CHECKPOINT_EVERY_STEP = 50000
GOOGLE_COLAB_MAX_STEP = 1000000

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_STEPS = 1000000

EpisodeReward = collections.namedtuple('EpisodeReward', 'reward, steps, frame_idx')


def worker(net, device, train_queue, reward_queue, proc_idx):
    try:
        from lib import data
        train_data = data.read_csv(file_name='data/000001_2018.csv')
    except ModuleNotFoundError:
        train_data = (pd.read_csv('data/prices_2018.csv', index_col=0),
                      pd.read_csv('data/factors_2018.csv', index_col=0))
    env = environ.StockEnv(train_data, bars_count=BARS_COUNT, reset_on_sell=True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    agt = agent.ProbabilityAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = experience.ExperienceSource(env, agt, GAMMA, steps_count=REWARD_STEPS)

    batch = []
    frame_idx = 0

    stats = collections.defaultdict(list)

    file_name = os.path.splitext(os.path.basename(__file__))[0]
    proc_name = 'worker_' + '%d' % proc_idx
    writer = SummaryWriter(os.path.join('runs', file_name, proc_name))

    for exp in exp_source:
        frame_idx += 1
        batch.append(exp)
        if len(batch) < GRAD_BATCH:
            continue

        net.zero_grad()
        loss_val_v, loss_policy_v, loss_entropy_v = helper.a2c_loss(batch, net, GAMMA**REWARD_STEPS, ENTROPY_BETA, device)
        batch.clear()
        loss_v = loss_entropy_v + loss_val_v + loss_policy_v
        loss_v.backward()
        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        grads = [param.grad.data.cpu().numpy() if param.grad is not None else None
                 for param in net.parameters()]
        train_queue.put(grads)

        stats['loss_value'].append(loss_val_v)
        stats['loss_policy'].append(loss_policy_v)
        stats['loss_entropy'].append(loss_entropy_v)
        stats['loss_total'].append(loss_v)
        for stat in stats:
            if len(stat) >= STATS_GROUPS:
                writer.add_scalar(stat, torch.mean(torch.stack(stats[stat])).item(), frame_idx)
                stats[stat].clear()

        ep_reward, ep_steps = exp_source.pop_episode_result()
        if ep_reward:
            print('worker_%d: %d done, Episode reward: %.4f, Episode step: %d' % (proc_idx, frame_idx, ep_reward, ep_steps))
            reward_queue.put(EpisodeReward(reward=ep_reward, steps=ep_steps, frame_idx=frame_idx))
    writer.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--cuda', default=False, action='store_true', help='enable cuda')
    parser.add_argument('--colab', default=False, action='store_true', help='enable colab hosted runtime')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda else 'cpu')

    try:
        from lib import data
        train_data = data.read_csv(file_name='data/000001_2018.csv')
    except ModuleNotFoundError:
        train_data = (pd.read_csv('data/prices_2018.csv', index_col=0),
                      pd.read_csv('data/factors_2018.csv', index_col=0))

    env = environ.StockEnv(train_data, bars_count=BARS_COUNT, reset_on_sell=True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    net = models.A2CConv1d(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    file_name = os.path.splitext(os.path.basename(__file__))[0]
    save_path = os.path.join('saves', file_name)
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_path, 'console.log')),
                                  logging.StreamHandler()])
    writer = SummaryWriter(os.path.join('runs', file_name, 'main'))

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    reward_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=worker, args=(net, device, train_queue, reward_queue, proc_idx))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    total_reward = []
    total_steps = []
    reward_buf = []
    steps_buf = []
    grad_buf = None
    step_idx = 0
    frame_idx = 0
    frame_prev = 0
    ts = time.time()
    best_mean_reward = None

    try:
        while True:
            train_entry = train_queue.get()
            reward_entry = reward_queue.get()
            if reward_entry:
                reward_buf.append(reward_entry.reward)
                steps_buf.append(reward_entry.steps)
                frame_idx += reward_entry.frame_idx
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
                    logging.info('%d done, mean reward %.3f, mean step %d, speed %d f/s' % (frame_idx, mean_reward, mean_step, speed))
                    writer.add_scalar('speed', speed, frame_idx)
                    writer.add_scalar('reward', reward, frame_idx)
                    writer.add_scalar('reward_100', mean_reward, frame_idx)
                    writer.add_scalar('steps', steps, frame_idx)
                    writer.add_scalar('steps_100', mean_step, frame_idx)
                    if best_mean_reward is None or best_mean_reward < mean_reward:
                        torch.save(net.state_dict(), os.path.join(save_path, 'best_mean_reward-%.3f.pth') % mean_reward)
                        if best_mean_reward is not None:
                            logging.info('Best mean value updated %.3f -> %.3f'
                                         % (best_mean_reward, mean_reward))
                        best_mean_reward = mean_reward

            step_idx += 1

            if grad_buf is None:
                grad_buf = train_entry
            else:
                for tgt_grad, grad in zip(grad_buf, train_entry):
                    tgt_grad += grad

            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buf):
                    param.grad = torch.FloatTensor(grad).to(device)

                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buf = None

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()


# if frame_idx % CHECKPOINT_EVERY_STEP == 0:
#     checkpoint = {'frame_idx': frame_idx,
#                   'state_dict': net.state_dict(),
#                   'optimizer': optimizer.state_dict(),
#                   'total_reward': total_reward,
#                   'total_steps': total_steps,
#                   'best_mean_reward': best_mean_reward,
#                   'stats': stats}
#     os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
#     torch.save(checkpoint, os.path.join(save_path, 'checkpoints', 'checkpoint-%d.pth' % frame_idx))
#     print('==> checkpoint saved at frame %d' % frame_idx)

# # workaround Colab's time limit
# if args.colab:
#     if frame_idx % GOOGLE_COLAB_MAX_STEP == 0:
#         break


# if args.resume:
#     print('==> Loading %s' % args.resume)
#     checkpoint = torch.load(os.path.join(save_path, 'checkpoints', args.resume))
#     total_reward = checkpoint['total_reward']
#     total_steps = checkpoint['total_steps']
#     frame_idx = checkpoint['frame_idx']
#     best_mean_reward = checkpoint['best_mean_reward'],
#     stats = checkpoint['stats'],
#     net.load_state_dict(checkpoint['state_dict']),
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print('==> Loaded %s' % args.resume)
