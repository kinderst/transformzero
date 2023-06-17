import numpy as np
import random
import torch


def get_random_episode_transitions(env, n_episodes, seq_len, device, use_one_hot=True):
    n_actions = env.action_space.n
    data_arr = []
    for i in range(n_episodes):
        # reset episode
        obs, info = env.reset()
        done = False
        curr_seq_num = 0
        stacked_states = None
        stacked_next_states = None
        while not done:
            action = random.randrange(n_actions)
            if use_one_hot:
                action_one_hot = [0.] * n_actions
                action_one_hot[action] = 1.
                obs_and_action = np.concatenate((obs, np.array(action_one_hot)))
            else:
                obs_and_action = np.concatenate((obs, np.array([action])))

            state_action_tensor = torch.Tensor(obs_and_action).to(device)

            obs, reward, done, truncated, info = env.step(action)

            next_state_tensor = torch.Tensor(obs).to(device)

            if stacked_states is None:
                stacked_states = state_action_tensor.unsqueeze(0)
                stacked_next_states = next_state_tensor.unsqueeze(0)
            else:
                stacked_states = torch.cat([stacked_states, state_action_tensor.unsqueeze(0)], dim=0)
                stacked_next_states = torch.cat([stacked_next_states, next_state_tensor.unsqueeze(0)], dim=0)

            if curr_seq_num == seq_len:
                data_arr.append((stacked_states, stacked_next_states))
                stacked_states = None
                stacked_next_states = None
                curr_seq_num = 0
            else:
                curr_seq_num += 1
    return data_arr


def get_random_episode_masked(env, n_episodes, seq_len, device):
    n_actions = env.action_space.n
    data_arr = []
    for i in range(n_episodes):
        # reset episode
        obs, info = env.reset()
        done = False

        current_timestep = 0

        timestep_x_arr = torch.full((seq_len, len(obs) + 1), 0.0, dtype=torch.float32)

        while not done:
            # every step we have mostly masked y's, because we ALWAYS only focus on one target
            # for an input (generally the last, but we also sometimes need to learn earlier ones)
            timestep_y_arr = torch.full((seq_len, len(obs)), float('-inf'))
            action = random.randrange(n_actions)

            obs_and_action = np.concatenate((obs, np.array([action])))

            state_action_tensor = torch.Tensor(obs_and_action).to(device)

            obs, reward, done, truncated, info = env.step(action)

            next_state_tensor = torch.Tensor(obs).to(device)
            if current_timestep >= seq_len:
                timestep_x_arr = timestep_x_arr[1:, :]
                timestep_x_arr = torch.cat((timestep_x_arr, state_action_tensor.unsqueeze(0)))
                timestep_y_arr[-1] = next_state_tensor
            else:
                timestep_x_arr[current_timestep] = state_action_tensor
                timestep_y_arr[current_timestep] = next_state_tensor

            data_arr.append((timestep_x_arr.clone(), timestep_y_arr))
            current_timestep += 1

    return data_arr
