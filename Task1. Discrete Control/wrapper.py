import gym
import numpy as np


class CliffWalkingWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)

        # You may modify these values here.

        return next_obs, reward, done, info
