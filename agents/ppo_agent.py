import numpy as np
import tensorflow as tf
from tensorflow import keras

from agents.agent import Agent
from buffers.ppo_buffer import Buffer
from models.ppo_model import mlp
from utils.plotting import plot_rewards


class PPOAgent(Agent):
    """
    Source for most code before it was rearranged/changed, and details:
    https://keras.io/examples/rl/ppo_cartpole/
    """
    def __init__(self, env, steps_per_epoch=4000, gamma=0.99, clip_ratio=0.2, train_policy_iters=80,
                 train_value_iters=80, lam=0.97, target_kl=0.01, hidden_sizes=(64, 64),
                 policy_learning_rate=3e-4, value_function_learning_rate=1e-3):
        super().__init__(env)
        # Hyperparameters of the PPO algorithm
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_policy_iterations = train_policy_iters
        self.train_value_iterations = train_value_iters
        self.lam = lam
        self.target_kl = target_kl

        # env info
        observation_dimensions = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Initialize the buffer
        self.buffer = Buffer(observation_dimensions, self.steps_per_epoch)

        # Initialize the actor and the critic as keras models
        observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
        logits = mlp(observation_input, list(hidden_sizes) + [self.num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=observation_input, outputs=logits)
        value = tf.squeeze(
            mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=observation_input, outputs=value)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    # Sample action from actor, basic functionally
    def select_action(self, obs: np.ndarray) -> int:
        logits = self.actor(obs.reshape(1, -1))
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return int(action[0].numpy())

    # Sample action from actor, for efficient use (obs must be shaped 1,-1 going in)
    @tf.function
    def select_action_with_logits(self, obs: np.ndarray):
        logits = self.actor(obs)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    def train(self, epochs: int, early_stopping_rounds: int = -1,
              early_stopping_threshold: float = 200.0, show_progress: bool = False) -> list:
        # Initialize the observation, episode return and episode length
        observation, info = self.env.reset()
        episode_return, episode_length = 0, 0
        epoch_rewards = []

        # Iterate over the number of epochs
        for epoch in range(epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            num_episodes = 0

            # Iterate over the steps of each epoch
            for t in range(self.steps_per_epoch):
                # Get the logits, action, and take one step in the environment
                observation = observation.reshape(1, -1)
                logits, action = self.select_action_with_logits(observation)
                observation_new, reward, terminated, truncated, info = self.env.step(action[0].numpy())
                done = terminated or truncated
                episode_return += reward
                episode_length += 1

                # Get the value and log-probability of the action
                value_t = self.critic(observation)
                logprobability_t = self.logprobabilities(logits, action)

                # Store obs, act, rew, v_t, logp_pi_t
                self.buffer.store(observation, action, reward, value_t, logprobability_t)

                # Update the observation
                observation = observation_new

                # Finish trajectory if reached to a terminal state
                terminal = done
                if terminal or (t == self.steps_per_epoch - 1):
                    last_value = 0 if done else self.critic(observation.reshape(1, -1))
                    self.buffer.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, info = self.env.reset()
                    episode_return, episode_length = 0, 0

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(self.train_policy_iterations):
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

            # Update the value function
            for _ in range(self.train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)

            epoch_rewards.append(sum_return / num_episodes)

            if early_stopping_rounds:
                if (sum(epoch_rewards[-early_stopping_rounds:]) / early_stopping_rounds) > early_stopping_threshold:
                    return epoch_rewards

            if show_progress:
                # Plotting doesn't work locally, maybe too many resources used by Keras on CPU?
                # just use original printing
                # plot_rewards(epoch_rewards)
                # Print mean return and length for each epoch
                print(
                    f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
                )
        # return the average episode rewards per epoch
        return epoch_rewards

    def eval(self, num_episodes: int) -> list:
        return super().eval(num_episodes)

    def investigate_model_outputs(self, obs: np.ndarray) -> np.ndarray:
        logits = self.actor(obs.reshape(1, -1))
        # action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits.numpy()[0]

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    # Train the policy by maximizing the PPO-Clip objective
    @tf.function
    def train_policy(
            self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
                )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def save_model(self, filepath: str) -> None:
        print(f"saving model to {filepath}.h5")
        filepath_with_extension = filepath + ".h5"
        self.actor.save_weights(filepath_with_extension)

    def load_model(self, filepath: str) -> None:
        print(f"loading model {filepath}")
        self.actor.load_weights(filepath)
