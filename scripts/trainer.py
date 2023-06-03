import argparse
from agents.dqn_agent import DQNAgent
from environments.grid_world_env import GridWorldEnv
# from environments.my_environment import MyEnvironment
import gymnasium as gym
import matplotlib.pyplot as plt
import time


def train_agent(env_name, agent_name, num_episodes):
    # Initialize environment
    if env_name == "cartpole":
        env = gym.make("CartPole-v1")
    elif env_name == "lunar":
        env = gym.make("LunarLander-v2")
    elif env_name == "grid":
        env = gym.make(GridWorldEnv)
    else:
        print(f"No envs with name: {str(env_name)} found, exiting...")
        return

    # Initialize agent
    if agent_name == "dqn":
        agent = DQNAgent(env)
    elif agent_name == "ppo":
        return
    else:
        print(f"No agents with the name: {str(agent_name)} found, exiting...")
        return

    plt.ion()

    agent.train(num_episodes)

    print("training complete")

    plt.ioff()

    # Save the trained agent weights
    agent.save_model("dqn_policy_" + str(env_name) + "_" + str(int(time.time())) + ".pt")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train My DRL Agent")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--agent", type=str, default="dqn", help="Agent name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    args = parser.parse_args()

    # Run the training script
    train_agent(args.env, args.agent, args.episodes)
