import unittest
import numpy as np
from agents.ppo_agent import PPOAgent
import gymnasium as gym
import tensorflow as tf


class PPOAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cartpole_env = gym.make("CartPole-v1")

    def test_select_action(self) -> None:
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

    def test_select_action_with_logits(self) -> None:
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

    def test_train(self) -> None:
        agent = PPOAgent(self.cartpole_env)
        epochs = 3
        episode_rewards = agent.train(epochs)
        self.assertIsInstance(episode_rewards, list)
        self.assertEqual(len(episode_rewards), epochs)

    def test_eval(self) -> None:
        agent = PPOAgent(self.cartpole_env)
        num_episodes = 10
        episode_rewards = agent.eval(num_episodes)
        self.assertIsInstance(episode_rewards, list)
        self.assertEqual(len(episode_rewards), num_episodes)
        for reward in episode_rewards:
            self.assertIsInstance(reward, (int, float))

    def test_investigate_model_outputs(self) -> None:
        agent = PPOAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        obs = np.arange(1, n_observations+1)
        outputs = agent.investigate_model_outputs(obs)
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (agent.env.action_space.n,))

    def test_train_and_eval_cartpole(self) -> None:
        # Tests to see if model can converge on cartpole
        agent = PPOAgent(self.cartpole_env)
        early_stopping_rounds = 7
        early_stopping_threshold = 480.0
        eval_threshold = 450.0
        eval_rounds = 50
        has_converged = False
        train_results = []
        # Because training is a noisy process than can fail say 25% of the time at most
        # in my experience, we do it 5 times, cause 0.25^5=0.001 or 0.1%, so 1/1000 times
        # it fails, and when that 1/1000 times happen we can investigate and try rerun
        # and if it continues to fail, then investigate for errors
        for i in range(5):
            epoch_rewards = agent.train(40,
                                        early_stopping_rounds=early_stopping_rounds,
                                        early_stopping_threshold=early_stopping_threshold,
                                        show_progress=False)
            avg_train = sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds
            train_results.append(avg_train)
            if avg_train > early_stopping_threshold:
                has_converged = True
                break
        print(f"ppo cartpole training average: ", train_results)
        self.assertTrue(has_converged)

        eval_passed = False
        eval_results = []
        for i in range(5):
            eval_rewards = agent.eval(eval_rounds)
            avg_eval = sum(eval_rewards) / len(eval_rewards)
            eval_results.append(avg_eval)
            if avg_eval > eval_threshold:
                eval_passed = True
                break

        print("ppo cartpole eval avg: ", eval_results)
        self.assertTrue(eval_passed)


if __name__ == '__main__':
    unittest.main()
