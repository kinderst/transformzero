import random
from buffers.dqn_replay_memory import ReplayMemory


class MultimodalReplayMemory(ReplayMemory):
    def push(self, state, action, next_state, next_action_mask, reward):
        """Save a transition"""
        self.memory.append(self.Transition(state, action, next_state, next_action_mask, reward))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            # Return an empty list if there are not enough transitions in the memory
            return []
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
