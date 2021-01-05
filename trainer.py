import time
import collections
import torch
import numpy as np
import torch.nn as nn

from q_network import QNetwork
from replay_buffer import ReplayBuffer
from wrappers import make_env
from agent import Agent
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, config):
        self.conf = config
        self.env = make_env(self.conf.environmentName)
        self.device = config.device
        self.model = QNetwork(self.env.observation_space.shape,
                              self.env.action_space.n).to(self.device)
        self.target_model = QNetwork(
            self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.buffer = ReplayBuffer(capacity=config.replayBufferCapacity)
        self.agent = Agent(env=self.env, exp_buffer=self.buffer)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.conf.learningRate)
        self.writer = SummaryWriter(self.conf.log_dir)
        self._reset()

    def _reset(self):
        self.frame_per_second = 0
        self.best_reward = None
        self.prev_frame_idx = 0
        self.total_rewards = []
        self.prev_time = time.time()

    def train(self):
        step = 0

        while True:
            step += 1
            epsilon = self._get_epsilon(step)
            reward, state_v, action = self.agent.play_step(
                net=self.model, epsilon=epsilon, device=self.device)
            mean_reward = self.mean_reward(reward)
            self._report(step, reward, mean_reward, epsilon)
            if self.replay_buffer_is_not_full(step):
                continue
           
            self.train_network_sgd(target)
            self.sync_target_network(step)
            
            if self.check_point(reward, mean_reward):
                break

    def _get_epsilon(self, step):
        """Linearly decays the learning rate until the frame_idx
         reaches the last frame decay, afterwards
          it will always returns a constant value of last_learning_rate"""
        start = self.conf.epsilonStart
        final = self.conf.epsilonFinal
        last_decay_frame_idx = self.conf.epsilonLastDecayStep
        return max(final, start - (start - final) * step / last_decay_frame_idx)

    def _report(self, step, reward, mean_reward, epsilon):
        if reward is not None:
            speed = (step - self.prev_frame_idx) / \
                (time.time() - self.prev_time)
            print(f'{step} steps, episodes:{len(self.total_rewards)}, Reward: {mean_reward:.2f}, Epsilon:{epsilon:.2f}, Speed:{speed:.2f}')
            self.prev_frame_idx = step
            self.prev_time = time.time()
            episode = len(self.total_rewards)
            self.writer.add_scalar('speed', speed, episode)
            self.writer.add_scalar('epsilon', epsilon, episode)
            self.writer.add_scalar('reward', reward, episode)
            self.writer.add_scalar('m_reward', mean_reward, episode)

    def replay_buffer_is_not_full(self, step):
        return len(self.buffer) < self.conf.replayBufferStart

    def train_network_sgd(self, target):
        self.optimizer.zero_grad()
        loss = self.calculate_loss(target)
        loss.backward()
        self.optimizer.step()

    def calculate_loss(self):
        states, actions, rewards, dones, next_states = self.buffer.sample(
            self.conf.batchSize)
        states_v = torch.tensor(np.array(
            states, copy=False)).to(self.device)
        next_states_v = torch.tensor(np.array(
            next_states, copy=False)).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)
        state_action_values = self.model(states_v).gather(
            1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_values_next = self.model(next_states_v)
            best_actions = np.argmax(q_values_next)
            # todo: 打印type->得到
            next_state_values = self.target_model(next_states_v)      
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = self.conf.discountFactor*next_states_values[] \
            +rewards_v
        return nn.MSELoss()(state_action_values,
                            expected_state_action_values)

    def sync_target_network(self, step):
        if step % self.conf.syncNetworkFrequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def check_point(self, reward, mean_reward):
        if reward is None:
            return False
        if self.best_reward is None or self.best_reward < reward:
            print(f'New best reward: {self.best_reward} --> {reward}')
            self.best_reward = reward
            torch.save(self.model.state_dict(),
                       f'{self.conf.environmentName}_{self.best_reward}.dat')

        if mean_reward > self.conf.rewardBound:
            torch.save(self.model.state_dict(), "solved.dat")
            print('Solved!')
            self.writer.close()
            return True

        return False

    def mean_reward(self, reward):
        if reward is not None:
            self.total_rewards.append(reward)
        return np.mean(self.total_rewards[-100:])
