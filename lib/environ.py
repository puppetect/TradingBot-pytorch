import numpy as np
import pandas as pd
import enum
import gym

BETA = 1


class Actions(enum.Enum):
    hold = 0
    empty = 1


class State:
    def __init__(self, bars_count, commission, reset_on_sell, reward_on_empty, play):
        assert isinstance(bars_count, int)
        assert isinstance(commission, float)
        assert isinstance(reset_on_sell, bool)
        assert isinstance(reward_on_empty, bool)
        assert bars_count > 0
        self.bars_count = bars_count
        assert commission >= 0.0
        self.commission = commission
        self.reset_on_sell = reset_on_sell
        self.reward_on_empty = reward_on_empty
        self.play = play

    def reset(self, prices, factors, offset):
        assert isinstance(prices, pd.DataFrame)
        assert isinstance(factors, pd.DataFrame)
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.prices = prices
        self.factors = factors
        self.offset = offset

    @property
    def shape(self):
        return (21, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        fac_count = len(self.factors.columns)
        for idx in range(fac_count):
            res[idx] = self.factors.iloc[self.offset - self.bars_count:self.offset, idx]
        if self.have_position:
            res[fac_count] = 1.0
            res[fac_count + 1] = (self._close() - self.open_price) / self.open_price
        return res

    def step(self, action):
        assert isinstance(action, Actions)
        reward = 0
        done = False
        close = self._close()
        if action == Actions.hold and not self.have_position:
            reward -= self.commission * 100
            self.have_position = True
            self.open_price = close
        if action == Actions.empty and self.have_position:
            reward -= self.commission * 100
            done |= self.reset_on_sell
            self.have_position = False
            self.open_price = 0.0

        self.offset += 1
        next_close = self._close()
        if self.have_position:
            if self.play:
                reward += 100 * (next_close - close) / close
            else:
                reward += max(-10, min(10, (100 * (next_close - close) / close)**3))
        if self.reward_on_empty and not self.have_position:
            reward -= BETA * 100 * (next_close - close) / close
        done |= self.offset >= len(self.prices) - 1
        info = {'have_position': int(self.have_position)}
        return reward, done, info

    def _close(self):
        return self.prices.ix[self.offset, 'close']


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=100, commission=0.00025, reset_on_sell=True,
                 random_ofs_on_reset=True, reward_on_empty=False, play=False):
        assert isinstance(prices, tuple)
        self.prices, self.factors = prices
        self.state = State(bars_count, commission, reset_on_sell, reward_on_empty, play)
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
        reward, done, info = self.state.step(action)
        obs = self.state.encode()
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass
