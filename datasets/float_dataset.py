import random
import torch
from torch.utils.data import Dataset


class FloatDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Input: [0.1 1.1 2.1 3.1 4.1] -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, in_dim=3):
        assert split in {'train', 'test', 'val'}
        self.split = split
        self.length = length
        self.in_dim = in_dim

    def __len__(self):
        return 10000  # ...

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):
        state_actions = []
        current_dim_one = random.uniform(-3, 3)
        current_dim_two = random.uniform(-3, 3)
        current_action = float(random.randint(0, 3))
        state_actions.append([current_dim_one, current_dim_two, current_action])
        # minus 1 because we already have the first step
        for i in range(5):
            if current_action == 0.0:
                current_dim_one -= 0.5
            elif current_action == 1.0:
                current_dim_two -= 0.5
            elif current_action == 2.0:
                current_dim_one += 0.5
            elif current_action == 3.0:
                current_dim_two += 0.5

            current_action = float(random.randint(0, 3))
            state_actions.append([current_dim_one, current_dim_two, current_action])

        x = torch.tensor(state_actions[:-1])
        y = torch.tensor(state_actions[1:])[:, :-1]

        return x, y
