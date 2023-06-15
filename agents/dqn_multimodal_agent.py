import numpy as np
import torch
import torch.nn as nn
import random

from agents.dqn_agent import DQNAgent
from buffers.dqn_multimodal_replay_memory import MultimodalReplayMemory


class MultimodalDQNAgent(DQNAgent):
    def __init__(self, env, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05,
                 eps_decay=1000, tau=0.005, lr=1e-4, replay_mem_size=10000, model_type="multires", use_action_mask=False):
        super().__init__(env, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, replay_mem_size, model_type, use_action_mask)
        # replay memory
        self.memory = MultimodalReplayMemory(replay_mem_size, self.Transition, self.device)
        obs, info = env.reset()
        self.modalities = obs.keys()

    def select_action_with_eps(self, obs: dict, eps_threshold, action_mask=None) -> int:
        self.policy_net.eval()
        sample = random.random()
        if sample > eps_threshold:
            # obs_one = torch.tensor(obs['imgone'], dtype=torch.float32, device=self.device).unsqueeze(0)
            # obs_two = torch.tensor(obs['imgtwo'], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                if action_mask is not None:
                    # if we have an action mask, only consider those actions
                    # so get which index in action_mask? the one with the highest value, as it was taken from policy
                    return int(action_mask[self.policy_net([obs])[:, action_mask].max(1)[1].view(1, 1).item()])
                else:
                    return self.policy_net([obs]).max(1)[1].view(1, 1).item()
        else:
            # cast to int because sample() returns np.int64, for consistency
            if action_mask is not None:
                return int(np.random.choice(action_mask, 1)[0])
            else:
                return int(self.env.action_space.sample())

    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        self.policy_net.train()

        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states_batch = []
        for s in batch.next_state:
            if s is not None:
                non_final_next_states_batch.append({modality: s[modality] for modality in self.modalities})

        state_batch = []
        for s in batch.state:
            state_batch.append({modality: s[modality] for modality in self.modalities})

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        if self.use_action_mask:
            # Pad the mask arrays to the same length
            max_len = max(len(row_mask) for row_mask in batch.next_action_mask if row_mask is not None)
            # note: instead of padding with zero, could also just pad with row_mask[0] which is some (first) action
            # in the mask, because when we take max anyway, doesn't matter if it is duplicated. But 0 works for
            # solitaire
            next_action_mask_batch = [np.append(row_mask, [row_mask[0]] * (max_len - len(row_mask))) for row_mask in batch.next_action_mask if row_mask is not None]
            # next_action_mask_batch = [np.append(row_mask, [0] * (max_len - len(row_mask))) for row_mask in batch.next_action_mask if row_mask is not None]
            next_action_mask_batch = torch.tensor(np.array(next_action_mask_batch), dtype=torch.long)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if self.use_action_mask:
                # https://discuss.pytorch.org/t/selecting-from-a-2d-tensor-with-rows-of-column-indexes/167717
                next_state_values[non_final_mask] = self.target_net(non_final_next_states_batch)[torch.arange(len(non_final_next_states_batch)).unsqueeze(1), next_action_mask_batch].max(1)[0]
            else:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
        self.optimizer.step()

    def investigate_model_outputs(self, obs: dict) -> np.ndarray:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net([obs]).detach().cpu().numpy()[0]
