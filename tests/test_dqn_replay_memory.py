import unittest
from collections import namedtuple
from buffers.dqn_replay_memory import ReplayMemory

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class TestReplayMemory(unittest.TestCase):

    def test_push_and_sample(self):
        memory = ReplayMemory(100, Transition)
        transition = Transition(1, 2, 3, 4)
        memory.push(*transition)  # Use * to unpack the transition elements
        self.assertEqual(len(memory), 1)

        sampled_transitions = memory.sample(1)
        self.assertEqual(len(sampled_transitions), 1)
        self.assertEqual(sampled_transitions[0], transition)

    def test_capacity(self):
        memory = ReplayMemory(2, Transition)
        transitions = [
            Transition(1, 2, 3, 4),
            Transition(5, 6, 7, 8),
            Transition(9, 10, 11, 12)
        ]

        for transition in transitions:
            memory.push(*transition)  # Use * to unpack the transition elements

        self.assertEqual(len(memory), 2)
        sampled_transitions = memory.sample(2)
        self.assertIn(transitions[1], sampled_transitions)
        self.assertIn(transitions[2], sampled_transitions)

    def test_empty_memory(self):
        memory = ReplayMemory(100, Transition)
        self.assertEqual(len(memory), 0)

        sampled_transitions = memory.sample(1)
        self.assertEqual(len(sampled_transitions), 0)

    def test_memory_size(self):
        capacity = 100
        memory = ReplayMemory(capacity, Transition)
        transitions = [
            Transition(i, i+1, i+2, i+3) for i in range(capacity + 10)
        ]

        for transition in transitions:
            memory.push(*transition)  # Use * to unpack the transition elements

        self.assertEqual(len(memory), capacity)
        sampled_transitions = memory.sample(capacity)
        self.assertEqual(len(sampled_transitions), capacity)

    def test_transition_attributes(self):
        memory = ReplayMemory(100, Transition)
        transition = Transition(1, 2, 3, 4)
        memory.push(*transition)  # Use * to unpack the transition elements

        sampled_transitions = memory.sample(1)
        sampled_transition = sampled_transitions[0]

        self.assertEqual(sampled_transition.state, transition.state)
        self.assertEqual(sampled_transition.action, transition.action)
        self.assertEqual(sampled_transition.next_state, transition.next_state)
        self.assertEqual(sampled_transition.reward, transition.reward)


if __name__ == '__main__':
    unittest.main()
