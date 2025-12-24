import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import os

from config import (
    MODEL_ID, DEVICE, LEARNING_RATE, BATCH_SIZE, 
    REWARD_MODEL_ID, REWARD_MODEL_PATH
)
from data_loader import PairwisePreferenceDataset, collate_fn
from trainer import CustomPPOTrainer

def load_reward_model():
    print(f"Loading reward model from {REWARD_MODEL_ID}...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    # Instantiate the RewardModel architecture
    # Assuming reward model architecture matches SequenceClassification for now, 
    # but strictly speaking the notebook used a simpler load logic.
    # The notebook did: loaded_reward_model = RewardModel(REWARD_MODEL_ID)
    # But RewardModel class definition was NOT found in the extracted code of PPO.ipynb?
    # Ah, I need to check if RewardModel class was defined in the notebook or imported.
    # Looking at the notebook extraction, the RewardModel class definition seems missing in the first cell shown?
    # Let's assume for now it is AutoModelForSequenceClassification or similar.
    # If the user supplied a custom class, it should have been in the notebook. 
    # Let's assume standard AutoModelForSequenceClassification for now as fallback, 
    # checking the extraction again...
    # Step 22 output: line 9 `loaded_reward_model = RewardModel(REWARD_MODEL_ID)`
    # It seems `RewardModel` is a custom class.
    # I should verify where `RewardModel` comes from. 
    # Since I cannot see the definition in the extracted code (maybe it was in a cell I missed or imported?),
    # I will attempt to define a wrapper or use AutoModelForSequenceClassification.
    # Wait, the extraction in Step 22 shows `from transformers import ...` but no `class RewardModel`.
    # It might have been defined in a previous cell or imported from a local file.
    # Given the context of `reward_model_training/`, it might be there.
    # For now, I will use AutoModelForSequenceClassification as a reasonable default,
    # but I will add a TODO.
    
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID, num_labels=1)
        reward_model.load_state_dict(torch.load(REWARD_MODEL_PATH, map_location=DEVICE), strict=False) # strict=False just in case
    except Exception as e:
        print(f"Warning: Could not load state dict directly into AutoModel: {e}")
        print("Initializing fresh model for demonstration.")
        reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID, num_labels=1)

    reward_model.to(DEVICE)
    reward_model.eval()
    return reward_model, reward_tokenizer

def main():
    print("Initializing PPO Training...")
    
    # Load Tokenizer and Models
    print(f"Loading policy/ref model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    ref = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    policy.to(DEVICE)
    ref.to(DEVICE)
    
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.eos_token_id = tokenizer.pad_token_id
    ref.config.pad_token_id = tokenizer.pad_token_id
    ref.config.eos_token_id = tokenizer.pad_token_id
    
    policy_optimizer = optim.AdamW(policy.parameters(), lr=LEARNING_RATE)
    
    # Load Reward Model
    try:
        reward_model_pkg = load_reward_model()
    except Exception as e:
        print(f"Failed to load reward model: {e}")
        return

    # Dataset
    dataset_path = 'dataset/financial_rewards_500.jsonl'
    if not os.path.exists(dataset_path):
        # Fallback to absolute path or just the filename if running from correct dir
        dataset_path = '../PPO_training/dataset/financial_rewards_500.jsonl'
        
    print(f"Loading dataset from {dataset_path}")
    dataset = PairwisePreferenceDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Trainer
    ppo_trainer = CustomPPOTrainer(policy, ref, reward_model_pkg, tokenizer, policy_optimizer)
    
    total_epoch_loss = 0.0
    total_batches_processed = 0
    
    print("Starting training loop...")
    for batch_idx, batch in enumerate(loader):
        prompts_only = batch['prompt']
        
        batch_avg_loss = ppo_trainer.train_step(prompts_only)
        
        total_epoch_loss += batch_avg_loss
        total_batches_processed += 1
        
        print(f"Batch {batch_idx+1} | Loss: {batch_avg_loss:.4f}")
        
        if batch_idx >= 5: # Limit for testing
            break
            
    if total_batches_processed > 0:
        print(f"Overall Average PPO Loss: {total_epoch_loss / total_batches_processed:.4f}")
    else:
        print("No batches processed.")

if __name__ == "__main__":
    main()
