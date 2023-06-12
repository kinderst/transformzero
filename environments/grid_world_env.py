import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    Source for most code before it was rearranged/changed, and details:
    https://www.gymlibrary.dev/content/environment_creation/

    Obstacles and other things added.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, obs_type="flat", max_episode_length=20, num_obstacles=0):
        self._current_step = None
        self._target_location = None
        self._agent_location = None
        self._obstacles = []
        self.num_obstacles = num_obstacles
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.obs_type = obs_type
        self.max_episode_length = max_episode_length

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        if obs_type == "flat":
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(size*size,), dtype=np.float32)
        elif obs_type == "img":
            self.observation_space = spaces.MultiBinary([size, size, 3])
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "obstacles": spaces.MultiBinary((size, size)),
                }
            )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.obs_type == "flat":
            return self._get_flat_obs()
        elif self.obs_type == "img":
            return self._get_img_obs()
        else:
            return {
                "agent": self._agent_location,
                "target": self._target_location,
                "obstacles": self._get_obstacle_matrix(),
            }

    def _get_flat_obs(self):
        # all_plane = self._get_obstacle_matrix()
        # # agent is arbitrarily denoted as -0.5 (closer to 0, closer to target?)
        # all_plane[self._agent_location[1]][self._agent_location[0]] = -0.5
        # # target is arbitrarily denoted as -1.0 (opposite of target?)
        # all_plane[self._target_location[1]][self._target_location[0]] = -1
        # # flatten to 1D
        # return all_plane.flatten()
        return self._get_img_obs().reshape(-1)

    def _get_img_obs(self):
        # plane of 0's, 1 where agent is
        agent_plane = np.zeros((self.size, self.size))
        agent_plane[self._agent_location[1]][self._agent_location[0]] = 1
        # plane of 0's, 1 where target is
        target_plane = np.zeros((self.size, self.size))
        target_plane[self._target_location[1]][self._target_location[0]] = 1
        # plane of 0's, 1's where obstacle(s) are
        obstacle_plane = self._get_obstacle_matrix()
        # Stack to 3D obs
        return np.stack((agent_plane, target_plane, obstacle_plane))

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _get_obstacle_matrix(self):
        obstacle_matrix = np.zeros((self.size, self.size))
        for obstacle in self._obstacles:
            obstacle_matrix[obstacle[1], obstacle[0]] = 1
        return obstacle_matrix

    def add_random_obstacles(self, num_obstacles):
        self._obstacles = []
        for _ in range(num_obstacles):
            obstacle = self.np_random.integers(0, self.size, size=2, dtype=int)
            while np.array_equal(obstacle, self._agent_location) or np.array_equal(obstacle, self._target_location) or \
                    any((obstacle == obs).all() for obs in self._obstacles):
                obstacle = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._obstacles.append(obstacle)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # add random number of obstacles between 0 and size-1 (if it was size, could potentially block path)
        # though even with 3, it could form a barrier around goal in corner, or 4 could barricade goal
        # around side...
        self.add_random_obstacles(self.num_obstacles)

        observation = self._get_obs()
        info = self._get_info()
        
        self._current_step = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if the new location is occupied by an obstacle
        if any((new_location == obs).all() for obs in self._obstacles):
            # If it is, don't update the agent's location
            new_location = self._agent_location

        # Update the agent's location
        self._agent_location = new_location

        # An episode is done if the agent has reached the target, or 50 steps
        terminated = np.array_equal(self._agent_location, self._target_location)
        # if the agent ends, they get the reward, -1 for stepping
        reward = 10 if terminated else -1

        self._current_step += 1
        # but also terminate if agent reached max num steps, just don't want to give reward
        if self._current_step > self.max_episode_length:
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
                ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
            )

        # Draw obstacles
        for obstacle in self._obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obstacle,
                    (pix_square_size, pix_square_size),
                    ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
