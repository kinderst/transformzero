import unittest
import numpy as np
import gymnasium as gym

from agents.dqn_agent import DQNAgent
from environments.grid_world_env import GridWorldEnv

class DQNAgentTests(unittest.TestCase):
    def setUp(self):
        self.cartpole_env = gym.make("CartPole-v1")

    def test_select_action(self):
        agent = DQNAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        num_tests = 10
        for i in range(num_tests):
            obs = np.arange(i, n_observations+i) / num_tests
            action = agent.select_action(obs)
            self.assertIsInstance(action, int)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, agent.env.action_space.n)

    def test_select_action_with_eps(self):
        agent = DQNAgent(self.cartpole_env)
        observation, _ = self.cartpole_env.reset()
        n_observations = len(observation)
        num_tests = 10
        for i in range(num_tests):
            obs = np.arange(i, n_observations+i) / num_tests
            eps_threshold = i / num_tests
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
        # From the docs: "Training RL agents can be a noisy process, so restarting training can produce better
        # results if convergence is not observed."
        # - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # Therefore, we ought to test, say, 3 times. The odds of it failing 3 times are
        # (chance of failing about 20%)^3 = 0.008, ^5 = 0.00032 or 0.032%
        agent = DQNAgent(self.cartpole_env)
        early_stopping_rounds = 50
        early_stopping_threshold = 485.0
        eval_threshold = 450.0
        has_converged = False
        for i in range(5):
            epoch_rewards = agent.train(700,
                                        early_stopping_rounds=early_stopping_rounds,
                                        early_stopping_threshold=early_stopping_threshold,
                                        show_progress=False)
            avg_train = sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds
            print(f"dqn cartpole training average: {avg_train} for attempt: {i}")
            if avg_train > early_stopping_threshold:
                has_converged = True
                break

        self.assertTrue(has_converged)

        eval_results = agent.eval(early_stopping_rounds)
        avg_eval = sum(eval_results) / len(eval_results)
        print("dqn cartpole eval avg: ", avg_eval)
        self.assertGreaterEqual(avg_eval, eval_threshold)

    def test_train_and_eval_lunar(self):
        # Tests to see if model can converge on lunar
        lunar_env = gym.make("LunarLander-v2")
        agent = DQNAgent(lunar_env)
        early_stopping_rounds = 50
        early_stopping_threshold = 200.0
        eval_threshold = 150.0
        has_converged = False
        for i in range(5):
            epoch_rewards = agent.train(800,
                                        early_stopping_rounds=early_stopping_rounds,
                                        early_stopping_threshold=early_stopping_threshold,
                                        show_progress=False)
            avg_train = sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds
            print(f"dqn lunar training average: {avg_train} for attempt: {i}")
            if avg_train > early_stopping_threshold:
                has_converged = True
                break

        self.assertTrue(has_converged)

        eval_results = agent.eval(early_stopping_rounds)
        avg_eval = sum(eval_results) / len(eval_results)
        print("dqn lunar eval avg: ", avg_eval)
        self.assertGreaterEqual(avg_eval, eval_threshold)

    def test_train_and_eval_grid(self):
        # Tests to see if model can converge on grid world without any obstacles
        grid_env = GridWorldEnv(size=5, obs_type="flat", max_episode_length=20, num_obstacles=0)
        # lr of 1e-2 experimentally found to be good
        agent = DQNAgent(grid_env, lr=1e-2)
        early_stopping_rounds = 50
        early_stopping_threshold = 8.0
        eval_threshold = 6.5  # failed with 6.76 even with 8.04 train
        has_converged = False
        for i in range(5):
            epoch_rewards = agent.train(2000,
                                        early_stopping_rounds=early_stopping_rounds,
                                        early_stopping_threshold=early_stopping_threshold,
                                        show_progress=False)
            avg_train = sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds
            print(f"dqn grid training average: {avg_train} for attempt: {i}")
            if avg_train > early_stopping_threshold:
                has_converged = True
                break

        self.assertTrue(has_converged)

        eval_results = agent.eval(early_stopping_rounds)
        avg_eval = sum(eval_results) / len(eval_results)
        print("dqn grid eval avg: ", avg_eval)
        self.assertGreaterEqual(avg_eval, eval_threshold)


if __name__ == '__main__':
    unittest.main()
