import numpy as np
import os
import enum
import gym

DEFAULT_BARS_COUNT = 100
DEFAULT_COMMISSION = 0.00025


class Actions(enum.Enum):
    skip = 0
    buy = 1
    sell = 2


class State:
    def __init__(self, bars_count, reset_on_close, commission):
        self.bars_count = bars_count
        self.reset_on_close = reset_on_close
        self.commission = commission

    def reset(self, prices, factors, offset):
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.prices = prices
        self.factors = factors
        self.offset = offset

    @property
    def shape(self):
        return (58, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        for idx in range(len(self.factors.columns)):
            res[idx] = self.factors.iloc[self.offset - self.bars_count:self.offset, idx]
        return res

    def step(self, action):
        reward = 0
        done = False
        close = self._close()
        if action == Actions.buy and not self.have_position:
            reward -= self.commission
            self.have_position = True
        if action == Actions.sell and self.have_position:
            reward -= self.commission
            done |= self.reset_on_close
            self.have_position = False

        self.offset += 1
        tmr_close = self._close()
        if self.have_position:
            reward += 100 * (tmr_close - close) / close
        done |= self.offset >= len(self.prices) - 1
        return reward, done

    def _close(self):
        return self.prices.ix[self.offset, 'close']


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT, reset_on_close=True,
                 commission=DEFAULT_COMMISSION, random_ofs_on_reset=True):
        self.prices = prices[0]
        self.factors = prices[1]
        self.state = State(bars_count, reset_on_close, commission)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=self.state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset

    def reset(self):
        bars = self.state.bars_count
        if self.random_ofs_on_reset:
            offset = np.random.choice(len(self.prices) - bars * 10) + bars
        else:
            offset = bars
        self.state.reset(self.prices, self.factors, offset)
        return self.state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self.state.step(action)
        obs = self.state.encode()
        info = {'offset': self.state.offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass
