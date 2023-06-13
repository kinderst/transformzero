from collections import deque
import random
import torch


class ReplayMemory(object):

    def __init__(self, capacity, transition_param, device):
        self.memory = deque([], maxlen=capacity)
        self.Transition = transition_param
        self.device = device

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        # add this because other replay memory takes multimodal, and this take unimodal
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state,
                                  dtype=torch.float32,
                                  device=self.device).unsqueeze(0) if next_state is not None else None
        self.memory.append(self.Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            # Return an empty list if there are not enough transitions in the memory
            return []
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
