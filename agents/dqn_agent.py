import numpy as np
import random
import math
from collections import namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from buffers.dqn_replay_memory import ReplayMemory
from models.dqn_model import DQN


class DQNAgent:
    def __init__(self, env, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005, lr=1e-4):
        # environment
        self.env = env

        # constants
        self.batch_size = batch_size  # the number of transitions sampled from the replay buffer
        self.gamma = gamma  # the discount factor
        self.eps_start = eps_start  # the starting value of epsilon
        self.eps_end = eps_end  # the final value of epsilon
        self.eps_decay = eps_decay  # controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tau = tau  # the update rate of the target network

        # torch, networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get values for policy/target net dims
        n_actions = self.env.action_space.n
        observation, _ = self.env.reset()
        n_observations = len(observation)
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        # replay memory
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(10000, self.Transition)

        # counters, recording
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state, eps_threshold):
        sample = random.random()
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                math.exp(-1. * self.steps_done / self.eps_decay)
                action = self.select_action(state, eps_threshold)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    # Save the agent's model or parameters to a file
    def save_model(self, filepath):
        print(f"saving model to {filepath}")
        torch.save(self.policy_net.state_dict(), filepath)

    # Load the agent's model or parameters from a file
    def load_model(self, filepath):
        print(f"loading model {filepath}")
        self.policy_net.load_state_dict(torch.load(filepath))
