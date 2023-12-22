import gym
import numpy as np


class MountainCarWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)

        # You may modify these values here.

        return next_obs, reward, done, info


class MaxEpisodeLengthWrapper(gym.Wrapper):
    # An example of applying a maximum of episode length to environment through wrappers.
    def __init__(self, env, max_episode_length):
        super().__init__(env)
        self._env = env
        self._max_episode_length = max_episode_length
        self._step = None

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        next_obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._max_episode_length:
            done = True
            self.step = None
        return next_obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()