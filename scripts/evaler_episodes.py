import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from itertools import count

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environments.grid_world_env import GridWorldEnv


def eval_step_episode(env_name, agent_name, test_episodes, weights_path):
    # Initialize environment
    if env_name == "cartpole":
        env = gym.make("CartPole-v1")
    elif env_name == "lunar":
        env = gym.make("LunarLander-v2")
    elif env_name == "grid":
        env = GridWorldEnv()
    else:
        print('err bad env name')
        return

    # Initialize agent
    if agent_name == "dqn":
        agent = DQNAgent(env)
    elif agent_name == "ppo":
        agent = PPOAgent(env)
    else:
        print(f"No agents with the name: {str(agent_name)} found, exiting...")
        return

    if weights_path != "initial":
        # Load the trained agent
        agent.load_model(weights_path)

    reward_results = agent.eval(test_episodes)
    print("Test episode reward results:")
    print(reward_results)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train My DRL Agent")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--agent", type=str, default="dqn", help="Agent name")
    parser.add_argument("--episodes", type=int, default=3, help="Path to weights")
    parser.add_argument("--weights", type=str, default="initial", help="Path to weights")
    args = parser.parse_args()

    # Run the training script
    eval_step_episode(args.env, args.agent, args.episodes, args.weights)
