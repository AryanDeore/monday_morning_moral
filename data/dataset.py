# Dataset class to convert the entire token tensor to a input, target pair

from torch.utils.data import Dataset

class TokenDataset(Dataset):
    """
    Converts the entire token tensor to a input, target pair
    """
    def __init__(self, token_tensor, context_length, stride):
        # token_tensor: shape [500M] - all tokens
        # context_length: window size (256 or 1024)
        # stride: how much to skip between windows. we dont want duplicates.

        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_tensor) - context_length, stride):
            input_window = token_tensor[i:i+context_length]
            target_window = token_tensor[i+1:i+context_length+1]
            self.input_ids.append(input_window)
            self.target_ids.append(target_window)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]