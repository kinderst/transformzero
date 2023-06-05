import unittest
import numpy as np
from agents.ppo_agent import PPOAgent
import gymnasium as gym
import tensorflow as tf


class PPOAgentTests(unittest.TestCase):
    def setUp(self):
        self.cartpole_env = gym.make("CartPole-v1")

    def test_select_action(self):
        agent = PPOAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        num_tests = 10
        for i in range(num_tests):
            obs = np.arange(i, n_observations+i) / num_tests
            action = agent.select_action(obs)
            self.assertIsInstance(action, int)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, agent.env.action_space.n)

    def test_select_action_with_logits(self):
        agent = PPOAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        num_tests = 10
        for i in range(num_tests):
            obs = np.arange(i, n_observations+i) / num_tests
            logits, action = agent.select_action_with_logits(obs.reshape(1, -1))
            self.assertIsInstance(logits, tf.Tensor)
            self.assertIsInstance(action, tf.Tensor)
            self.assertIsInstance(action[0].numpy(), np.int64)
            self.assertGreaterEqual(action[0].numpy(), 0)
            self.assertLess(action[0].numpy(), agent.env.action_space.n)

    def test_train(self):
        agent = PPOAgent(self.cartpole_env)
        epochs = 3
        episode_rewards = agent.train(epochs)
        self.assertIsInstance(episode_rewards, list)
        self.assertEqual(len(episode_rewards), epochs)

    def test_eval(self):
        agent = PPOAgent(self.cartpole_env)
        num_episodes = 10
        episode_rewards = agent.eval(num_episodes)
        self.assertIsInstance(episode_rewards, list)
        self.assertEqual(len(episode_rewards), num_episodes)
        for reward in episode_rewards:
            self.assertIsInstance(reward, (int, float))

    def test_investigate_model_outputs(self):
        agent = PPOAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        obs = np.arange(1, n_observations+1)
        outputs = agent.investigate_model_outputs(obs)
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (agent.env.action_space.n,))

    def test_train_and_eval_cartpole(self):
        # Tests to see if model can converge on cartpole
        agent = PPOAgent(self.cartpole_env)
        early_stopping_rounds = 7
        early_stopping_threshold = 475.0
        eval_threshold = 450.0
        has_converged = False
        for i in range(3):
            epoch_rewards = agent.train(40,
                                        early_stopping_rounds=early_stopping_rounds,
                                        early_stopping_threshold=early_stopping_threshold,
                                        show_progress=False)
            avg_train = sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds
            print(f"ppo cartpole training average: {avg_train} for attempt: {i}")
            if avg_train > early_stopping_threshold:
                has_converged = True
                break

        self.assertTrue(has_converged)

        eval_results = agent.eval(early_stopping_rounds * 100)
        avg_eval = sum(eval_results) / len(eval_results)
        print("ppo cartpole eval avg: ", avg_eval)
        self.assertGreaterEqual(avg_eval, eval_threshold)


if __name__ == '__main__':
    unittest.main()
