import numpy as np
import random
import math
from collections import namedtuple
from itertools import count

import torch
import torch.optim as optim
import torch.nn as nn

from buffers.dqn_replay_memory import ReplayMemory
from models.dqn_model import DQN
from models.dqn_resnet_model import ResNet
from agents.agent import Agent
from utils.plotting import plot_rewards


class DQNAgent(Agent):
    """
    Source for most code before it was rearranged/changed, and details:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    def __init__(self, env, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05,
                 eps_decay=1000, tau=0.005, lr=1e-4, replay_mem_size=10000, model_type="fc"):
        super().__init__(env)
        # constants
        self.batch_size = batch_size  # the number of transitions sampled from the replay buffer
        self.gamma = gamma  # the discount factor
        self.eps_start = eps_start  # the starting value of epsilon
        self.eps_end = eps_end  # the final value of epsilon
        self.eps_decay = eps_decay  # controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tau = tau  # the update rate of the target network

        # torch, networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        observation, _ = self.env.reset()
        n_actions = self.env.action_space.n

        if model_type == "fc":
            # get values for policy/target net dims
            n_observations = len(observation)
            self.policy_net = DQN(n_observations, n_actions).to(self.device)
            self.target_net = DQN(n_observations, n_actions).to(self.device)
        elif model_type == "resnet":
            self.policy_net = ResNet(2, observation.shape, n_actions).to(self.device)
            self.target_net = ResNet(2, observation.shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        # replay memory
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(replay_mem_size, self.Transition)

    def select_action(self, obs: np.ndarray, action_mask=None) -> int:
        return int(self.select_action_with_eps(obs, self.eps_end, action_mask))

    def select_action_with_eps(self, obs: np.ndarray, eps_threshold, action_mask=None) -> int:
        sample = random.random()
        if sample > eps_threshold:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                if action_mask is not None:
                    # if we have an action mask, only consider those actions
                    # so get which index in action_mask? the one with the highest value, as it was taken from policy
                    return int(action_mask[self.policy_net(obs)[:, action_mask].max(1)[1].view(1, 1).item()])
                else:
                    return self.policy_net(obs).max(1)[1].view(1, 1).item()
        else:
            # cast to int because sample() returns np.int64, for consistency
            if action_mask is not None:
                return int(np.random.choice(action_mask, 1)[0])
            else:
                return int(self.env.action_space.sample())

    def train(self, epochs: int, early_stopping_rounds: int = -1, early_stopping_threshold: float = 200.0,
              show_progress: bool = False, print_progress: int = 0) -> list:
        steps_done = 0
        epoch_rewards = []
        for i_episode in range(epochs):
            # Initialize the environment and get it's state
            total_reward = 0
            observation, info = self.env.reset()
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                current_eps = self.eps_end + (self.eps_start - self.eps_end) * \
                                math.exp(-1. * steps_done / self.eps_decay)
                action = self.select_action_with_eps(observation, current_eps)
                action_tensor = torch.tensor([[action]], device=self.device, dtype=torch.long)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                steps_done += 1
                total_reward += reward
                reward_tensor = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_observation_tensor = None
                else:
                    next_observation_tensor = torch.tensor(next_observation,
                                                           dtype=torch.float32,
                                                           device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(observation_tensor, action_tensor, next_observation_tensor, reward_tensor)

                # Move to the next state
                observation = next_observation
                observation_tensor = next_observation_tensor

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
                    epoch_rewards.append(total_reward)
                    # if it is positive, we want to stop early (when it reaches some threshold)
                    if early_stopping_rounds and len(epoch_rewards) >= early_stopping_rounds:
                        if (sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds) > early_stopping_threshold:
                            return epoch_rewards

                    if show_progress:
                        plot_rewards(epoch_rewards)
                    if print_progress:
                        if i_episode % print_progress == 0:
                            print(total_reward)
                    break
            # for testing only!
            # if i_episode % 100 == 0:
            #     print(current_eps)
        return epoch_rewards

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

    def eval(self, num_episodes: int) -> list:
        return super().eval(num_episodes)

    def investigate_model_outputs(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(obs).detach().numpy()[0]

    # Save the agent's model or parameters to a file
    def save_model(self, filepath):
        print(f"saving model to {filepath}.pt")
        filepath_with_extension = filepath + ".pt"
        torch.save(self.policy_net.state_dict(), filepath_with_extension)

    # Load the agent's model or parameters from a file
    def load_model(self, filepath):
        print(f"loading model {filepath}")
        self.policy_net.load_state_dict(torch.load(filepath))
