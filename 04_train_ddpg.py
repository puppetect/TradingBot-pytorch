from lib import environ, models
from common import agent, experience
import os
import time
import logging
import argparse
import collections
import gym
import gym.wrappers
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from common.writer import SummaryWriter
from common.helper import unpack_batch, TargetNet


BATCH_SIZE = 32
BARS_COUNT = 50
REWARD_GROUPS = 10
STATS_GROUPS = 10

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

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

    env = environ.StockEnv(train_data, bars_count=BARS_COUNT, commission=0.0, reset_on_sell=False, reward_on_empty=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_act_net = TargetNet(act_net)
    crt_net = models.DDPGCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_crt_net = TargetNet(crt_net)

    agt = agent.OUProcessAgent(act_net, device=device)
    exp_source = experience.ExperienceSource(env, agt, GAMMA, steps_count=REWARD_STEPS)
    buffer = experience.ExperienceBuffer(exp_source, REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE, eps=1e-3)

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
    file_name = file_name.split('_')[-1]
    save_path = os.path.join('saves', file_name)
    os.makedirs(save_path, exist_ok=True)

    # if args.resume:
    #     print('==> Loading %s' % args.resume)
    #     checkpoint = torch.load(os.path.join(save_path, 'checkpoints', args.resume))
    #     total_reward = checkpoint['total_reward']
    #     total_steps = checkpoint['total_steps']
    #     frame_idx = checkpoint['frame_idx']
    #     best_mean_reward = float(checkpoint['best_mean_reward']),
    #     net.load_state_dict(checkpoint['state_dict']),
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print('==> Loaded %s' % args.resume)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_path, 'console.log')),
                                  logging.StreamHandler()])

    writer = SummaryWriter(os.path.join('runs', file_name))

    for exp in exp_source:
        frame_idx += 1
        buffer.populate(1)
        batch.append(exp)

        if len(batch) < BATCH_SIZE:
            continue

        states_v, actions_v, rewards_v, dones_v, last_states_v = unpack_batch(batch, device)
        crt_opt.zero_grad()
        q_v = crt_net(states_v, actions_v)
        last_act_v = tgt_act_net.target_model(last_states_v)
        q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
        q_last_v[dones_v] = 0.0
        q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA ** REWARD_STEPS
        critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
        critic_loss_v.backward()
        crt_opt.step()

        # train actor
        act_opt.zero_grad()
        cur_actions_v = act_net(states_v)
        actor_loss_v = -crt_net(states_v, cur_actions_v)
        actor_loss_v = actor_loss_v.mean()
        actor_loss_v.backward()
        act_opt.step()

        tgt_act_net.alpha_sync(alpha=1 - 1e-3)
        tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

        stats['loss_critic'].append(critic_loss_v)
        stats['critic_ref'].append(q_ref_v.mean())
        stats['loss_actor'].append(actor_loss_v)
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
                    torch.save(act_net.state_dict(), os.path.join(save_path, 'best_mean_reward.pth'))
                    try:
                        if best_mean_reward is not None:
                            logging.info('Best mean reward updated %.3f -> %.3f'
                                         % (best_mean_reward, mean_reward))
                    except Exception as e:
                        print(e)
                        pass
                    finally:
                        best_mean_reward = mean_reward

        # if frame_idx % CHECKPOINT_EVERY_STEP == 0:
        #     checkpoint = {'frame_idx': frame_idx,
        #                   'state_dict': net.state_dict(),
        #                   'optimizer': optimizer.state_dict(),
        #                   'total_reward': total_reward,
        #                   'total_steps': total_steps,
        #                   'best_mean_reward': best_mean_reward,
        #                   }
        #     os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
        #     torch.save(checkpoint, os.path.join(save_path, 'checkpoints', 'checkpoint-%d.pth' % frame_idx))
        #     print('==> checkpoint saved at frame %d' % frame_idx)

        # workaround Colab's time limit
        if args.colab:
            if frame_idx % GOOGLE_COLAB_MAX_STEP == 0:
                break

    writer.close()
