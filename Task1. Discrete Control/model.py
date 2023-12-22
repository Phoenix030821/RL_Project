import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class Agent(object):

    def __init__(self, obs_shape, action_shape, device):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        # More variations to initialize

    def choose_action(self, obs):
        'Choose an action based on observation input'
        # In CliffWalking, 4 discrete actions are possible: 0 up, 1 right, 2 down, 3 left

    def update(self, step):
        'Update the parameter'

    def save(self, model_dir, step):
        'Save the model'

    def load(self, model_dir, step):
        'Load the model'
