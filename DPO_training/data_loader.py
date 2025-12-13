
import json
from typing import List, Dict, Any
from torch.utils.data import Dataset

class PairwisePreferenceDataset(Dataset):
    """
    Expects list of dicts with keys: 'prompt', 'chosen', 'rejected'
    """
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():                     # skip empty lines
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Returns a dict with lists; trainer will tokenize using the tokenizer so we keep raw strings here.
    """
    prompts = [item['prompt'] for item in batch]
    chosen = [p + "\n" + item['chosen'] for p, item in zip(prompts, batch)]
    rejected = [p + "\n" + item['rejected'] for p, item in zip(prompts, batch)]
    return {"prompt": prompts, "chosen": chosen, "rejected": rejected}
