import numpy as np


class Agent:
    """Reinforcement Learning Agents.

    Attributes:
        env (gymnasium): The environment for the agent.

    """

    def __init__(self, env):
        """Initialize a Rectangle instance.

        Args:
            env (gymnasium): The environment for the agent.

        """

        self.env = env

    def select_action(self, obs: np.ndarray) -> int:
        """Selects an action based on an observation

        Args:
            obs (np.ndarray): A flat, 1D numpy array for the observation

        Returns:
            int: Action taken, as defined by environment.

        """

        raise NotImplementedError("Agents must implement the select_action method")

    def train(self, epochs: int, early_stopping_rounds: int, early_stopping_threshold: float, show_progress: bool = False) -> list:
        """Trains the agent

        Args:
            epochs (int): Number of epochs to perform training.
            early_stopping_rounds (int): Exits training if amount of last reward rounds above threshold
            early_stopping_threshold (float): Value which is considered (sometimes nearly) solved
            show_progress (bool): Whether to display any output for training progress.

        Returns:
            list: List of average scores per episode for the epochs

        """

        raise NotImplementedError("Agents must implement the train method")

    def investigate_model_outputs(self, obs: np.ndarray) -> np.ndarray:
        """Trains the agent

        Args:
            obs (np.ndarray): A flat, 1D numpy array for the observation

        Returns:
            np.ndarray: Outputs from the agent's model for the given observation

        """

        raise NotImplementedError("Agents must implement the investigate_model_outputs method")

    def save_model(self, filepath: str) -> None:
        """Saves the agent's model(s)

        Args:
            filepath (str): File path location to save model(s).

        """

        raise NotImplementedError("Agents must implement the save_model method")

    def load_model(self, filepath: str) -> None:
        """Loads the agent's model(s)

        Args:
            filepath (str): File path location to load model(s).

        """

        raise NotImplementedError("Agents must implement the load_model method")
