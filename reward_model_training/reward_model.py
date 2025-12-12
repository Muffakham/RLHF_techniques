import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class RewardModel(nn.Module):
    def __init__(self, base_model_id):
        super().__init__()
        # Initialize as a sequence classifier with 1 label (scalar score)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            num_labels=1,
            torch_dtype=torch.float32 # Use torch_dtype for AutoModelForSequenceClassification
        )
        # Ensure pad token is set
        if self.model.config.pad_token_id is None:
             self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1) # [batch_size]
