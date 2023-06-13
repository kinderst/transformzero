import numpy as np
import torch
import random

from agents.dqn_agent import DQNAgent


class MultimodalDQNAgent(DQNAgent):
    def select_action_with_eps(self, obs: np.ndarray, eps_threshold, action_mask=None) -> int:
        sample = random.random()
        if sample > eps_threshold:
            obs_one = torch.tensor(obs['imgone'], dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_two = torch.tensor(obs['imgtwo'], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                if action_mask is not None:
                    # if we have an action mask, only consider those actions
                    # so get which index in action_mask? the one with the highest value, as it was taken from policy
                    return int(action_mask[self.policy_net(obs_one,
                                                           obs_two)[:, action_mask].max(1)[1].view(1, 1).item()])
                else:
                    return self.policy_net(obs_one,
                                           obs_two).max(1)[1].view(1, 1).item()
        else:
            # cast to int because sample() returns np.int64, for consistency
            if action_mask is not None:
                return int(np.random.choice(action_mask, 1)[0])
            else:
                return int(self.env.action_space.sample())
