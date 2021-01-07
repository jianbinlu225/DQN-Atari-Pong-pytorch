import torch
import gym
import torch.nn as nn
import numpy as np
import collections
import cv2
import time

from agent import Agent
from q_network import QNetwork
from wrappers import make_env

ENV_NAME='PongNoFrameskip-v4'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'



def load_model(path, input_shape, n_actions):
    dqn = QNetwork(input_shape,n_actions).to(DEVICE)
    dqn.load_state_dict(torch.load(path))

    return dqn


def play(env, model, device="cpu"):
    observation = env.reset()
    total_reward = 0
    while (True):
        env.render()
        # action = env.action_space.sample() # your agent here (this takes random actions)

        observation_a = np.array([observation], copy=False)
        observation_v = torch.tensor(observation_a).to(device)
        q_vals_v = model(observation_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())

        observation, reward, done, info = env.step(action)

        total_reward += reward

        if done:
            return total_reward

if __name__ == '__main__':

    env = make_env(ENV_NAME)

    dqn_model = load_model("./solved_rb_10k.dat", env.observation_space.shape, env.action_space.n)

    total_reward = 0
    for i in range(10):
        t_reward = play(env, dqn_model, DEVICE)
        print(f'Episodes: {i}, Reward: {t_reward}.')
        total_reward += t_reward
    
    avg_reward = total_reward/10
    print(f"Average reward: {avg_reward}")
