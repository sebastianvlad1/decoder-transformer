from torch.utils.data import Dataset
import torch

class GPTDataset(Dataset):
    def __init__(self, processed_dataset):
        self.inputs = processed_dataset["train"]["inputs"]
        self.targets = processed_dataset["train"]["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx]),
            "targets": torch.tensor(self.targets[idx]),
        }
