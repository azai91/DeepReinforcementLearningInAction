"""
Vanilla Policy Gradient Network

python policy_gradient_net.py --render
"""

import gym
import numpy as np
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch import nn
from torch import FloatTensor


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(4, 150)
        self.fc2 = nn.Linear(150, 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x


class PolicyNet():
    def __init__(self, gamma=0.99, lr=0.001, render=False):
        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.render = render

    def train(self, env, episode_length=200):
        """
        Collects one episode of training data and then updates parameters
        """
        obs = env.reset()
        act_probs = []
        rewards = []
        discounted_rewards = []
        for i in range(episode_length):
            if self.render: env.render()
            act_prob = self.evaluate(obs)
            action_taken = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())
            obs, reward, done, _ = env.step(action_taken)
            act_probs.append(act_prob[action_taken])
            rewards.append(reward)
            if done:
                break

        for i1 in range(len(rewards)):
            discount = 1
            future_reward = 0
            for i2 in range(i1, len(rewards)):
                future_reward += discount * rewards[i2]
                discount *= self.gamma
            discounted_rewards.append(future_reward)

        loss = self.update(act_probs, discounted_rewards)
        return loss, sum(rewards)

    def update(self, act_probs, discounted_rewards):
        """
        Updates parameters with policy gradient algorithm
        """
        self.optimizer.zero_grad()
        act_probs = torch.stack(act_probs)
        drewards = FloatTensor(discounted_rewards)
        loss = -torch.sum(drewards * torch.log(act_probs))
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, x):
        x = torch.from_numpy(x).float()
        return self.net(x)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--episode_length', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    env = gym.make('CartPole-v1')
    model = PolicyNet(gamma=args.gamma, render=args.render, lr=args.lr)

    for epoch_num in range(args.epochs):
        model.train(env, episode_length=args.episode_length)
