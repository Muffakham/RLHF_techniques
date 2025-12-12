import torch
import json
from torch.utils.data import Dataset, DataLoader

class RewardIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, tokenizer):
        self.filename = filename
        self.tokenizer = tokenizer

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                yield json.loads(line)

    def __getitem__(self, idx):
        # IterableDatasets typically do not implement __getitem__
        # for direct indexing. This method is a placeholder if needed
        # for compatibility with some DataLoader features that might
        # indirectly call it, but the primary mode of access is __iter__.
        # For this specific use case with DataLoader, __iter__ is sufficient.
        raise NotImplementedError("__getitem__ not implemented for RewardIterableDataset")

def collate_fn(batch):
    return batch
