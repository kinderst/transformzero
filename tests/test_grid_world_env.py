import unittest
import numpy as np
import gymnasium as gym

from environments.grid_world_env import GridWorldEnv


class GridWorldEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv(render_mode=None)

    def tearDown(self):
        self.env.close()

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (self.env.size * self.env.size * 3,))
        self.assertIsInstance(info, dict)

    def test_step(self):
        # reset is assumed to be called before steps taken
        _, _ = self.env.reset()
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = self.env.step(action)
        self.assertEqual(next_obs.shape, (self.env.size * self.env.size * 3,))
        self.assertIsInstance(reward, int)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(next_info, dict)

    def test_render(self):
        env = GridWorldEnv(render_mode="rgb_array")
        env.reset()
        rgb_array = env.render()
        self.assertEqual(rgb_array.shape, (self.env.window_size, self.env.window_size, 3))

    def test_close(self):
        self.env.close()

    def test_observation_types(self):
        obs_types = ["flat", "img", "dict"]
        for obs_type in obs_types:
            with self.subTest(obs_type=obs_type):
                env = GridWorldEnv(render_mode=None, obs_type=obs_type)

                obs, info = env.reset()
                action = env.action_space.sample()
                next_obs, _, _, _, _ = env.step(action)

                if obs_type == "flat":
                    self.assertIsInstance(obs, np.ndarray)
                    self.assertEqual(obs.shape, (self.env.size * self.env.size * 3,))

                    self.assertIsInstance(next_obs, np.ndarray)
                    self.assertEqual(next_obs.shape, (self.env.size * self.env.size * 3,))

                elif obs_type == "img":
                    self.assertIsInstance(obs, np.ndarray)
                    self.assertEqual(obs.shape, (3, self.env.size, self.env.size))

                    self.assertIsInstance(next_obs, np.ndarray)
                    self.assertEqual(next_obs.shape, (3, self.env.size, self.env.size))
                else:
                    self.assertIsInstance(obs, dict)
                    self.assertIsInstance(next_obs, dict)

                env.close()

    def test_random_obstacle_addition(self):
        _, _ = self.env.reset()
        self.env.add_random_obstacles(2)
        self.assertEqual(len(self.env._obstacles), 2)
        obstacle_matrix = self.env._get_obstacle_matrix()
        self.assertEqual(np.count_nonzero(obstacle_matrix), 2)


if __name__ == "__main__":
    unittest.main()
