from collections import deque
import random


class ReplayMemory(object):

    def __init__(self, capacity, transition_param):
        self.memory = deque([], maxlen=capacity)
        self.Transition = transition_param

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            # Return an empty list if there are not enough transitions in the memory
            return []
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
