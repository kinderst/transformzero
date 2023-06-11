from torch.utils.data import Dataset


class LunarDataset(Dataset):
    def __init__(self, data, length=11):
        self.length = length
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
