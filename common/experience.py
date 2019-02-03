import collections
import numpy as np


Step = collections.namedtuple('Step', 'state, action, reward, done')
Experience = collections.namedtuple('Experience', 'state, action, reward, last_state')


class ExperienceSource:
    """Experience source using single environment"""

    def __init__(self, env, agent, gamma, steps_count=2):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.episode_reward = None
        self.episode_step = None

    def __iter__(self):
        state = self.env.reset()
        exp = collections.deque(maxlen=self.steps_count)
        total_reward = 0.0
        total_step = 0

        while True:
            action_idx = self.agent([state])
            next_state, reward, done, _ = self.env.step(action_idx)
            total_reward += reward
            total_step += 1
            step = Step(state=state, action=action_idx,
                        reward=reward, done=done)
            exp.append(step)
            if len(exp) == self.steps_count:
                last_state = next_state if not done else None
                sum_reward = 0.0
                for e in reversed(exp):
                    sum_reward *= self.gamma
                    sum_reward += e.reward
                yield Experience(state=exp[0].state, action=exp[0].action,
                                 reward=sum_reward, last_state=last_state)
            state = next_state
            if done:
                self.episode_reward = total_reward
                self.episode_step = total_step
                total_reward = 0.0
                total_step = 0
                state = self.env.reset()
                exp.clear()

    def pop_episode_result(self):
        res = (self.episode_reward, self.episode_step)
        if res:
            self.episode_reward = None
            self.episode_step = None
        return res


class ExperienceBuffer:
    def __init__(self, source, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.source = iter(source)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def populate(self, exp_count):
        for _ in range(exp_count):
            entry = next(self.source)
            self.append(entry)
