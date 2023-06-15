import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import time

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.dqn_multimodal_agent import MultimodalDQNAgent
from environments.grid_world_env import GridWorldEnv
from environments.solitaire_env import SolitaireWorldEnv
from utils.plotting import plot_rewards


def train_agent(env_name, agent_name, num_epochs):
    dqn_lr = 1e-4
    dqn_eps_decay = 1000
    dqn_use_action_mask = False
    dqn_tau = 0.005

    # Initialize environment
    if env_name == "cartpole":
        env = gym.make("CartPole-v1")
        early_stopping_threshold = 475.0
    elif env_name == "lunar":
        env = gym.make("LunarLander-v2")
        early_stopping_threshold = 200.0
    elif env_name == "gridnone":
        env = GridWorldEnv(num_obstacles=0, obs_type="multiimg")
        early_stopping_threshold = 8.0
        dqn_lr = 1e-2
        dqn_eps_decay = 1500
    elif env_name == "gridone":
        env = GridWorldEnv(num_obstacles=1, obs_type="img")
        early_stopping_threshold = 7.0
        dqn_lr = 1e-2
        dqn_eps_decay = 2000
    elif env_name == "gridtwo":
        env = GridWorldEnv(num_obstacles=2, obs_type="img")
        early_stopping_threshold = 6.0
        dqn_lr = 1e-2
        dqn_eps_decay = 2500
    elif env_name == "solitaireimg":
        env = SolitaireWorldEnv(obs_type="img")
        early_stopping_threshold = 500.0
        dqn_lr = 1e-6
        dqn_eps_decay = 5000
        dqn_use_action_mask = True
    elif env_name == "solitairemulti":
        env = SolitaireWorldEnv(obs_type="multiimg")
        early_stopping_threshold = 50.0
        dqn_lr = 1e-2
        dqn_eps_decay = 12500
        dqn_tau = 0.05
        dqn_use_action_mask = True
    else:
        print(f"No envs with name: {str(env_name)} found, exiting...")
        return

    # Initialize agent
    if agent_name == "dqn":
        agent = DQNAgent(env, lr=dqn_lr, model_type="resnet", eps_decay=dqn_eps_decay, eps_end=0.1, gamma=0.99)
        early_stopping_rounds = 50
    elif agent_name == "multidqn":
        agent = MultimodalDQNAgent(env, lr=dqn_lr, model_type="multires",
                                   eps_decay=dqn_eps_decay, eps_end=0.1, gamma=0.9, use_action_mask=dqn_use_action_mask)
        early_stopping_rounds = 50
    elif agent_name == "ppo":
        agent = PPOAgent(env)
        early_stopping_rounds = 3
    else:
        print(f"No agents with the name: {str(agent_name)} found, exiting...")
        return

    plt.ion()

    print("starting training...")

    epoch_rewards = agent.train(num_epochs,
                                early_stopping_rounds=early_stopping_rounds,
                                early_stopping_threshold=early_stopping_threshold,
                                show_progress=False,
                                print_progress=True,
                                save_checkpoints=50)

    print("training complete")

    # Save the trained agent weights
    agent.save_model("model_" + agent_name + "_" + str(env_name) + "_" + str(int(time.time())))

    plot_rewards(epoch_rewards, show_result=True)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train My DRL Agent")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--agent", type=str, default="dqn", help="Agent name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training episodes")
    args = parser.parse_args()

    # Run the training script
    train_agent(args.env, args.agent, args.epochs)
