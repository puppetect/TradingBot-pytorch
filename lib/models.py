import numpy as np
import torch
import torch.nn as nn


class DQNConv1d(nn.Module):
    """Dueling DQN
    :math::
    \text{Q}(s,a) = \text{V}(s)+\text{A}(s,a) - \frac{1}{N}\textstyle\sum_{k}\text{A}(s,k)
    """

    def __init__(self, obs_shape, actions_n):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(obs_shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU()
        )

        out_size = self.get_conv_out(obs_shape)

        self.val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def get_conv_out(self, shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.shape))

    def forward(self, x):
        conv_out = self.conv(x).view(x.shape[0], -1)
        val = self.val(conv_out)
        adv = self.adv(conv_out)
        return val + adv - torch.mean(adv, dim=1, keepdim=True)
