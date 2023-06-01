import argparse
from agents.dqn_agent import DQNAgent
# from environments.my_environment import MyEnvironment
import gymnasium as gym
import matplotlib.pyplot as plt
import torch


def train_agent(env_name, agent_name, num_episodes):
    # Initialize environment
    env = gym.make(env_name)

    # Initialize agent
    if agent_name == "DQNAgent":
        agent = DQNAgent(env)
    elif agent_name == "PPOAgent":
        return
    else:
        print(f"No agents with the name: {str(agent_name)} found, exiting...")
        return

    plt.ion()

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 600

    agent.train(num_episodes)

    print("training complete")

    plt.ioff()

    # Save the trained agent
    agent.save_model("dqn_policy.pt")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train My DRL Agent")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--agent", type=str, default="DQNAgent", help="Agent name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    args = parser.parse_args()

    # Run the training script
    train_agent(args.env, args.agent, args.episodes)
