import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from itertools import count

from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.dqn_multimodal_agent import MultimodalDQNAgent
from agents.ppo_agent import PPOAgent
from environments.grid_world_env import GridWorldEnv
from environments.solitaire_env import SolitaireWorldEnv


def eval_step_episode(env_name, agent_name, weights_path):
    # Initialize environment
    if env_name == "cartpole":
        env = gym.make("CartPole-v1", render_mode="human")
    elif env_name == "lunar":
        env = gym.make("LunarLander-v2", render_mode="human")
    elif env_name == "grid":
        env = GridWorldEnv(render_mode="human", num_obstacles=0, obs_type="multiimg")
    elif env_name == "solitaire":
        env = SolitaireWorldEnv(obs_type="img")
    elif env_name == "solitairemulti":
        env = SolitaireWorldEnv(render_mode="human", obs_type="multiimg")
    else:
        print('err bad env name')
        return

    # Initialize agent
    if agent_name == "rand":
        agent = RandomAgent(env)
    elif agent_name == "dqn":
        agent = DQNAgent(env, model_type="resnet")
    elif agent_name == "multidqn":
        print('using multidqn')
        agent = MultimodalDQNAgent(env, model_type="multires")
    elif agent_name == "ppo":
        agent = PPOAgent(env)
    else:
        print(f"No agents with the name: {str(agent_name)} found, exiting...")
        return

    if weights_path != "initial":
        # Load the trained agent
        agent.load_model(weights_path)

    done = False
    observation, info = env.reset()
    while not done:
        print("press any button to take step")
        _ = input()

        action_mask = info['action_mask'] if 'action_mask' in info else None
        print("action mask: ", action_mask)

        print("observation is: ", observation)
        model_outputs = agent.investigate_model_outputs(observation)
        # print("model outputs: ", model_outputs)

        action = agent.select_action(observation, action_mask)
        print("action taken: ", action)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train My DRL Agent")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--agent", type=str, default="dqn", help="Agent name")
    parser.add_argument("--weights", type=str, default="initial", help="Path to weights")
    args = parser.parse_args()

    # Run the training script
    eval_step_episode(args.env, args.agent, args.weights)
