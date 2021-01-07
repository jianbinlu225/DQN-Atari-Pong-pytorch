import torch
import gym
import torch.nn as nn
import numpy as np
import collections
import cv2
import time

from agent import Agent
# from q_network import QNetwork
from wrappers import make_env

ENV_NAME='PongNoFrameskip-v4'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'



if __name__ == '__main__':

    env = make_env(ENV_NAME)

    dqn_model = Agent(env.observation_space.shape, env.action_space.n)

    total_reward = 0
    for i in range(10):
        t_reward = play(env, dqn_model, DEVICE)
        print(f'Episodes: {i}, Reward: {t_reward}.')
        total_reward += t_reward
    
    avg_reward = total_reward/10
    print(f"Average reward: {avg_reward}")
