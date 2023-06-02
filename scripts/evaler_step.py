import argparse
from agents.dqn_agent import DQNAgent
# from environments.my_environment import MyEnvironment
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from itertools import count


def eval_step_episode(env_name, agent_name, weights_path):
    # Initialize environment
    if env_name == "cartpole":
        env = gym.make("CartPole-v1", render_mode="human")
    elif env_name == "lunar":
        env = gym.make("LunarLander-v2", render_mode="human")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent
    if agent_name == "dqn":
        agent = DQNAgent(env)
    elif agent_name == "ppo":
        return
    else:
        print(f"No agents with the name: {str(agent_name)} found, exiting...")
        return

    # Load the trained agent
    agent.load_model(weights_path)

    done = False
    observation, info = env.reset()
    while not done:
        # convert obs to tensor
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action = agent.select_action(observation, 0.05)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train My DRL Agent")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--agent", type=str, default="dqn", help="Agent name")
    parser.add_argument("--weights", type=str, default="dqn_policy.pt", help="Path to weights")
    args = parser.parse_args()

    # Run the training script
    eval_step_episode(args.env, args.agent, args.weights)
