
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from . import config

def get_batch_logps(model, sequences, attention_mask, prompt_lens):
    """
    Calculates the log probabilities of the generated text, masking out the prompt.
    """
    # 1. Forward pass (Input Mask: Attend to Prompt + Response, ignore Pad)
    outputs = model(sequences, attention_mask=attention_mask)

    # 2. Shift logits and labels (Next-token prediction)
    # Logits match the output of the model (predictions)
    # Labels match the input tokens shifted by 1 (ground truth)
    logits = outputs.logits[:, :-1, :]
    labels = sequences[:, 1:]

    # 3. Get log probs of the actual tokens chosen
    # Shape: [Batch, Seq_Len-1]
    per_token_logps = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(2)

    # 4. Create the Loss Mask (Grading Mask: Grade only Response, ignore Prompt & Pad)
    loss_mask = attention_mask[:, 1:].clone()

    for i, start_idx in enumerate(prompt_lens):
        # Zero out the prompt tokens (start_idx - 1 because of shifting)
        loss_mask[i, :start_idx - 1] = 0.0

    # Return per-token log probs and the mask (for averaging later)
    return per_token_logps, loss_mask

def load_reward_model():
    print(f"Loading reward model from {config.REWARD_MODEL_ID}...")
    reward_tokenizer = AutoTokenizer.from_pretrained(config.REWARD_MODEL_ID)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(config.REWARD_MODEL_ID, num_labels=1)
        reward_model.load_state_dict(torch.load(config.REWARD_MODEL_PATH, map_location=config.DEVICE), strict=False) # strict=False just in case
    except Exception as e:
        print(f"Warning: Could not load state dict directly into AutoModel: {e}")
        print("Initializing fresh model for demonstration.")
        reward_model = AutoModelForSequenceClassification.from_pretrained(config.REWARD_MODEL_ID, num_labels=1)

    reward_model.to(config.DEVICE)
    reward_model.eval()
    return reward_model, reward_tokenizer