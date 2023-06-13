import unittest
from typing import Tuple
import numpy as np
import gymnasium as gym

from agents.dqn_agent import DQNAgent
from environments.grid_world_env import GridWorldEnv


class DQNAgentTests(unittest.TestCase):
    def loop_train_and_eval(self, agent, early_stopping_rounds, early_stopping_threshold, eval_threshold, name_string,
                            train_epochs=1000, train_loops=3, eval_loops=3):
        has_converged = False
        avg_train_results = []

        eval_passed = False
        avg_eval_results = []
        for i in range(train_loops):
            epoch_rewards = agent.train(epochs=train_epochs,
                                        early_stopping_rounds=early_stopping_rounds,
                                        early_stopping_threshold=early_stopping_threshold,
                                        show_progress=False)
            avg_train = sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds
            avg_train_results.append(avg_train)
            if avg_train > early_stopping_threshold:
                has_converged = True
                break

        if has_converged:
            for i in range(eval_loops):
                eval_results = agent.eval(early_stopping_rounds)
                avg_eval = sum(eval_results) / len(eval_results)
                avg_eval_results.append(avg_eval)
                if avg_eval > eval_threshold:
                    eval_passed = True
                    break

        print(f"dqn {name_string} train results: {avg_train_results}, eval results: {avg_eval_results}")
        self.assertTrue(has_converged)
        self.assertTrue(eval_passed)

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
        self.loop_train_and_eval(agent, early_stopping_rounds=100, early_stopping_threshold=485.0, eval_threshold=450.0,
                                 name_string="fc cartpole",
                                 train_epochs=2500, train_loops=3, eval_loops=3)

    def test_train_and_eval_lunar(self):
        # Tests to see if model can converge on lunar
        lunar_env = gym.make("LunarLander-v2")
        agent = DQNAgent(lunar_env)
        self.loop_train_and_eval(agent, early_stopping_rounds=100, early_stopping_threshold=200.0, eval_threshold=150.0,
                                 name_string="fc lunar",
                                 train_epochs=2500, train_loops=3, eval_loops=3)

    def test_train_and_eval_gridnone_fc(self):
        # Tests to see if model can converge on grid world without any obstacles
        grid_env = GridWorldEnv(size=5, obs_type="flat", max_episode_length=20, num_obstacles=0)
        # lr of 1e-2 experimentally found to be good
        agent = DQNAgent(grid_env, lr=1e-2, model_type="fc")
        self.loop_train_and_eval(agent, early_stopping_rounds=100, early_stopping_threshold=7.5, eval_threshold=6.0,
                                 name_string="fc gridnone",
                                 train_epochs=2500, train_loops=3, eval_loops=3)

    def test_train_and_eval_gridnone_resnet(self):
        grid_env = GridWorldEnv(size=5, obs_type="img", max_episode_length=20, num_obstacles=0)
        agent = DQNAgent(grid_env, lr=1e-2, model_type="resnet", eps_decay=1250)
        self.loop_train_and_eval(agent, early_stopping_rounds=100, early_stopping_threshold=7.5, eval_threshold=6.0,
                                 name_string="resnet gridnone",
                                 train_epochs=2500, train_loops=3, eval_loops=3)

    def test_train_and_eval_gridone_resnet(self):
        grid_env = GridWorldEnv(size=5, obs_type="img", max_episode_length=20, num_obstacles=1)
        agent = DQNAgent(grid_env, lr=5e-3, model_type="resnet", eps_decay=1500)
        self.loop_train_and_eval(agent, early_stopping_rounds=100, early_stopping_threshold=5.75, eval_threshold=3.5,
                                 name_string="resnet gridone",
                                 train_epochs=2500, train_loops=3, eval_loops=3)

    def test_train_and_eval_gridtwo_resnet(self):
        grid_env = GridWorldEnv(size=5, obs_type="img", max_episode_length=20, num_obstacles=2)
        agent = DQNAgent(grid_env, lr=5e-3, model_type="resnet", eps_decay=2000)
        self.loop_train_and_eval(agent, early_stopping_rounds=100, early_stopping_threshold=5.5, eval_threshold=2.0,
                                 name_string="resnet gridtwo",
                                 train_epochs=3000, train_loops=3, eval_loops=5)


if __name__ == '__main__':
    unittest.main()
