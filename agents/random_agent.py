import numpy as np
import random

from agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n

    def select_action(self, obs: np.ndarray) -> int:
        return random.randrange(0, self.n_actions)

    def train(self, epochs: int, early_stopping_rounds: int, early_stopping_threshold: float, show_progress: bool = False) -> list:
        return []

    def eval(self, num_episodes: int) -> list:
        return super().eval(num_episodes)

    def investigate_model_outputs(self, obs: np.ndarray) -> np.ndarray:
        return np.array([])

    def save_model(self, filepath: str) -> None:
        return

    def load_model(self, filepath: str) -> None:
        return
