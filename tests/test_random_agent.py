import unittest
import numpy as np
import gymnasium as gym

from agents.random_agent import RandomAgent


class RandomAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        env = gym.make("CartPole-v1")
        self.agent = RandomAgent(env)

    def test_select_action(self) -> None:
        # Ensure the selected action is within the valid range
        obs, _ = self.agent.env.reset()
        action = self.agent.select_action(obs)
        self.assertTrue(0 <= action < self.agent.n_actions)

    def test_train(self) -> None:
        # Ensure train() returns an empty list
        epochs = 10
        early_stopping_rounds = 3
        early_stopping_threshold = 0.1
        result = self.agent.train(epochs, early_stopping_rounds, early_stopping_threshold)
        self.assertEqual(result, [])

    def test_eval(self) -> None:
        num_episodes = 5
        result = self.agent.eval(num_episodes)

        self.assertEqual(len(result), num_episodes)  # Check if the list has the correct length

        for value in result:
            # Check if each value is positive and less than max because random should not get max...
            self.assertTrue(0 <= value < 500)

    def test_investigate_model_outputs(self) -> None:
        # Ensure investigate_model_outputs() returns an empty array
        obs, _ = self.agent.env.reset()
        result = self.agent.investigate_model_outputs(obs).tolist()
        self.assertListEqual(result, [])


if __name__ == "__main__":
    unittest.main()
