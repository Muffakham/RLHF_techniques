import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import modules from our project structure
from reward_model_training.data_loader import RewardIterableDataset, collate_fn
from reward_model_training.reward_model import RewardModel
import reward_model_training.config as config

# --- Configuration (using config.py) ---
MODEL_ID = config.MODEL_ID
DEVICE = config.DEVICE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
MAX_LENGTH = config.MAX_LENGTH
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH # New: Model save path

# --- Tokenizer Initialization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Data Loading ---
dataset = RewardIterableDataset(config.DATASET_PATH, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- Model Initialization ---
reward_model = RewardModel(MODEL_ID).to(DEVICE)
reward_model.train()

# --- Optimizer ---
optimizer = optim.AdamW(reward_model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
print("Starting training...")
for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Prepare chosen and rejected inputs
        prompts = [item['prompt'] for item in batch]
        chosen_texts = [p + item['chosen'] for p, item in zip(prompts, batch)]
        rejected_texts = [p + item['rejected'] for p, item in zip(prompts, batch)]

        # Tokenize
        chosen_enc = tokenizer(chosen_texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        rejected_enc = tokenizer(rejected_texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)

        # Forward pass to get scores
        chosen_rewards = reward_model(chosen_enc.input_ids, chosen_enc.attention_mask)
        rejected_rewards = reward_model(rejected_enc.input_ids, rejected_enc.attention_mask)

        # --- Bradley-Terry Loss ---
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

print("\nTraining complete.")

# --- Save the trained model ---
print(f"Saving reward model to {MODEL_SAVE_PATH}")
torch.save(reward_model.state_dict(), MODEL_SAVE_PATH)
print("Model saved successfully.")

# --- Quick Verification ---
print("\n--- Verification: Scoring a sample ---")
reward_model.eval()

# Load just one sample for verification manually
with open("financial_rewards.jsonl", 'r') as f:
    sample = json.loads(f.readline())
prompt = sample['prompt']
chosen = prompt + sample['chosen']
rejected = prompt + sample['rejected']

print(f"Prompt: {prompt}")
print(f"Chosen: {chosen}")
print(f"Rejected: {rejected}")

with torch.no_grad():
    c_enc = tokenizer(chosen, return_tensors='pt').to(DEVICE)
    r_enc = tokenizer(rejected, return_tensors='pt').to(DEVICE)

    c_score = reward_model(c_enc.input_ids, c_enc.attention_mask).item()
    r_score = reward_model(r_enc.input_ids, r_enc.attention_mask).item()

print(f"Chosen Score:   {c_score:.4f}")
print(f"Rejected Score: {r_score:.4f}")

if c_score > r_score:
    print("SUCCESS: Chosen response scored higher.")
else:
    print("FAILURE: Rejected response scored higher (expected with very little training).")
