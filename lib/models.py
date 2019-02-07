import numpy as np
import torch
import torch.nn as nn


class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super().__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),  # (N, L) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 512),  # (N, 512) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 1)  # (N, 512) -> (N, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),  # (N, L) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 512),  # (N, 512) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, actions_n)  # (N, 512) -> (N, A)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)  # (N, A)


class DQNConv1d(nn.Module):
    """Dueling DQN
    :math::
    \text{Q}(s,a) = \text{V}(s)+\text{A}(s,a) - \frac{1}{N}\textstyle\sum_{k}\text{A}(s,k)
    """

    def __init__(self, obs_shape, actions_n):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(obs_shape[0], 128, 5),  # (N, C, L) -> (N, 128, L-4)
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),  # (N, 128, L-4) -> (N, 128, L-8)
            nn.ReLU()
        )

        out_size = self.get_conv_out(obs_shape)

        self.val = nn.Sequential(
            nn.Linear(out_size, 512),  # (N, 128*(L-8)) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 1)  # (N, 512) -> (N, 1)
        )

        self.adv = nn.Sequential(
            nn.Linear(out_size, 512),  # (N, 128*(L-8)) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, actions_n)  # (N, 512) -> (N, A)
        )

    def get_conv_out(self, shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.shape))

    def forward(self, x):
        conv_out = self.conv(x).view(x.shape[0], -1)
        val = self.val(conv_out)
        adv = self.adv(conv_out)
        return val + adv - torch.mean(adv, dim=1, keepdim=True)  # (N, A)


class A2CConv1d(nn.Module):
    """Dueling DQN
    :math::
    \text{Q}(s,a) = \text{V}(s)+\text{A}(s,a) - \frac{1}{N}\textstyle\sum_{k}\text{A}(s,k)
    """

    def __init__(self, obs_shape, actions_n):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(obs_shape[0], 128, 5),  # (N, C, L) -> (N, 128, L-4)
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),  # (N, 128, L-4) -> (N, 128, L-8)
            nn.ReLU()
        )

        out_size = self.get_conv_out(obs_shape)

        self.policy = nn.Sequential(
            nn.Linear(out_size, 512),  # (N, 128*(L-8)) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, actions_n)  # (N, 512) -> (N, A)
        )

        self.value = nn.Sequential(
            nn.Linear(out_size, 512),  # (N, 128*(L-8)) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 1)  # (N, 512) -> (N,)
        )

    def get_conv_out(self, shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.shape))

    def forward(self, x):
        conv_out = self.conv(x).view(x.shape[0], -1)
        return self.policy(conv_out), self.value(conv_out)  # (N, A), (N,)
