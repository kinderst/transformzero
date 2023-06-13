import numpy as np
import unittest
from buffers.ppo_buffer import Buffer


class BufferTests(unittest.TestCase):

    def test_store(self) -> None:
        buffer = Buffer(observation_dimensions=4, size=10)
        observation = np.array([1, 2, 3, 4], dtype=np.float32)
        action = 2
        reward = 0.5
        value = 1.0
        logprobability = -0.1

        buffer.store(observation, action, reward, value, logprobability)

        self.assertTrue(np.array_equal(buffer.observation_buffer[0], observation))
        self.assertEqual(buffer.action_buffer[0], action)
        self.assertEqual(buffer.reward_buffer[0], reward)
        self.assertEqual(buffer.value_buffer[0], value)
        # note, for some reason have to check almost equals here, floating point precision on -0.1?
        self.assertAlmostEqual(buffer.logprobability_buffer[0], logprobability, places=7)

    def test_finish_trajectory(self) -> None:
        buffer = Buffer(observation_dimensions=4, size=10)
        buffer.pointer = 5
        buffer.trajectory_start_index = 2
        last_value = 0.5

        buffer.finish_trajectory(last_value)

        expected_advantage = np.array([0.43784744, 0.4655475, 0.495, 0.0, 0.0])
        expected_return = np.array([0.4851495, 0.49005, 0.495, 0.0, 0.0])

        self.assertTrue(np.allclose(buffer.advantage_buffer[2:7], expected_advantage, atol=1e-7))
        self.assertTrue(np.allclose(buffer.return_buffer[2:7], expected_return, atol=1e-7))

    def test_get(self) -> None:
        buffer = Buffer(observation_dimensions=4, size=10)
        initial_advantage = np.array([0.2, 0.25, 0.3, 0.35])
        buffer.advantage_buffer = np.array([0.2, 0.25, 0.3, 0.35])
        buffer.observation_buffer = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        buffer.action_buffer = np.array([1, 2, 3, 4])
        buffer.return_buffer = np.array([0.5, 0.7, 1.0, 1.2])
        buffer.logprobability_buffer = np.array([-0.1, -0.2, -0.3, -0.4])

        expected_observation = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        expected_action = np.array([1, 2, 3, 4])
        # it is normalized
        expected_advantage = (initial_advantage - np.mean(initial_advantage)) / np.std(initial_advantage)
        expected_return = np.array([0.5, 0.7, 1.0, 1.2])
        expected_logprobability = np.array([-0.1, -0.2, -0.3, -0.4])

        observation, action, advantage, ret, logprob = buffer.get()

        self.assertTrue(np.array_equal(observation, expected_observation))
        self.assertTrue(np.array_equal(action, expected_action))
        self.assertTrue(np.allclose(advantage, expected_advantage))
        self.assertTrue(np.array_equal(ret, expected_return))
        self.assertTrue(np.allclose(logprob, expected_logprobability))


if __name__ == '__main__':
    unittest.main()
