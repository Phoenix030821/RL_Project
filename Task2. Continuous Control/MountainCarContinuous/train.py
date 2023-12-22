import gym
import argparse
from model import Agent
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='MountainCarContinuous-v0')
    parser.add_argument('--train_eps', default=200)
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = gym.make(args.task)
    # You may choose proper wrappers for the environment to modify the original action, observation, reward, etc.
    # from wrapper import MountainCarWrapper
    # env = MountainCarWrapper(env)

    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    # You need to implement Agent in model.py in order to interact with the environment.
    agent = Agent()
    rewards = []
    # You may choose to use tensorboard to record your results, instead of maintaining a 'rewards' list.
    # You can refer to logger.py for useful methods.
    # L = Logger(args.work_dir, use_tb=args.save_tb)
    for i_ep in range(args.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, args.train_eps,ep_reward))


if __name__ == '__main__':
    main()