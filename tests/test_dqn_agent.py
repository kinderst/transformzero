import unittest
import numpy as np
from agents.dqn_agent import DQNAgent
import gymnasium as gym


class DQNAgentTests(unittest.TestCase):
    def setUp(self):
        # Initialize any necessary objects or variables for the tests
        self.cartpole_env = gym.make("CartPole-v1")

    def test_select_action(self):
        agent = DQNAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        obs = np.arange(1, n_observations+1)
        action = agent.select_action(obs)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, agent.env.action_space.n)

    def test_select_action_with_eps(self):
        agent = DQNAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        obs = np.arange(1, n_observations+1)
        eps_threshold = 0.1
        action = agent.select_action_with_eps(obs, eps_threshold)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, agent.env.action_space.n)

    def test_train(self):
        agent = DQNAgent(self.cartpole_env)
        epochs = 3
        episode_rewards = agent.train(epochs)
        self.assertIsInstance(episode_rewards, list)
        self.assertEqual(len(episode_rewards), epochs)

    def test_eval(self):
        agent = DQNAgent(self.cartpole_env)
        num_episodes = 10
        episode_rewards = agent.eval(num_episodes)
        self.assertIsInstance(episode_rewards, list)
        self.assertEqual(len(episode_rewards), num_episodes)
        for reward in episode_rewards:
            self.assertIsInstance(reward, (int, float))

    def test_investigate_model_outputs(self):
        agent = DQNAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        obs = np.arange(1, n_observations+1)
        outputs = agent.investigate_model_outputs(obs)
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (agent.env.action_space.n,))

    def test_train_and_eval_cartpole(self):
        # Tests to see if model can converge on cartpole
        agent = DQNAgent(self.cartpole_env)
        early_stopping_rounds = 25
        early_stopping_threshold = 475.0
        eval_threshold = 450.0
        has_converged = False
        for _ in range(3):
            epoch_rewards = agent.train(650,
                                        early_stopping_rounds=early_stopping_rounds,
                                        early_stopping_threshold=early_stopping_threshold,
                                        show_progress=False)
            if (sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds) > early_stopping_threshold:
                has_converged = True
                break

        self.assertTrue(has_converged)

        eval_results = agent.eval(early_stopping_rounds)
        avg_eval = sum(eval_results) / len(eval_results)
        self.assertGreaterEqual(avg_eval, eval_threshold)


if __name__ == '__main__':
    unittest.main()
