
import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from . import config
from .data_loader import get_dataloader
from .trainer import CustomGRPOTrainer
import os
from .utils import load_reward_model

def main():
    print("Starting GRPO Training...")
    
    # Initialize Tokenizer and Models
    print(f"Loading Policy Model: {config.POLICY_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(config.POLICY_MODEL_ID)
    policy.to(config.DEVICE)
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.eos_token_id = tokenizer.pad_token_id
    
    print("Loading Reference Model...")
    ref = AutoModelForCausalLM.from_pretrained(config.POLICY_MODEL_ID) # Ref is usually a copy of init policy
    ref.to(config.DEVICE)
    ref.config.pad_token_id = tokenizer.pad_token_id
    ref.config.eos_token_id = tokenizer.pad_token_id

    # Optimizer
    policy_optimizer = optim.AdamW(policy.parameters(), lr=config.LEARNING_RATE)

    # Initialize Reward Model
   

    print(f"Loading Reward Model: {config.REWARD_MODEL_ID}")
    try:
        reward_model, reward_tokenizer = load_reward_model()
    except Exception as e:
        print(f"Could not load reward model {config.REWARD_MODEL_ID}: {e}")
        print("Using a dummy reward model for demonstration purposes if needed, or fail.")

        raise e

    # Data Loader
    # Assuming dataset is in the parent directory or relative path
    dataset_path = 'financial_rewards_500.jsonl'
    if not os.path.exists(dataset_path):
        # Check standard locations
        # GRPO_training/grpo_modular/train.py relative to PPO_training/dataset/
        # base: RLHF_techniques/
        # file: RLHF_techniques/PPO_training/dataset/financial_rewards_500.jsonl
        
        # Try finding it relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__)) # .../GRPO_training/grpo_modular
        rlhf_root = os.path.dirname(os.path.dirname(base_dir)) # .../RLHF_techniques
        
        potential_paths = [
            os.path.join(rlhf_root, 'PPO_training', 'dataset', 'financial_rewards_500.jsonl'),
            os.path.join(rlhf_root, 'financial_rewards_500.jsonl'),
            'financial_rewards_500.jsonl'
        ]
        
        for p in potential_paths:
            if os.path.exists(p):
                dataset_path = p
                break
    
    print(f"Loading data from {dataset_path}")
    # If file doesn't exist, we can't create loader typically.
    if os.path.exists(dataset_path):
        loader = get_dataloader(dataset_path, batch_size=5)
    else:
        loader = []
        print("Skipping data loading as file not found.")

    # Trainer
    ppo_trainer = CustomGRPOTrainer(
        policy_model=policy,
        ref_model=ref,
        reward_model=(reward_model, reward_tokenizer),
        tokenizer=tokenizer,
        policy_optimizer=policy_optimizer
    )

    # Training Loop
    total_epoch_loss = 0.0
    total_batches_processed = 0

    print("Starting Training Loop...")
    for batch_idx, batch in enumerate(loader):
        prompts_only = batch['prompt']

        batch_avg_loss = ppo_trainer.train_step(prompts_only)

        total_epoch_loss += batch_avg_loss
        total_batches_processed += 1

        print(f"\nBatch {batch_idx+1} completed. Average PPO Loss for this batch: {batch_avg_loss:.4f}\n")
        
        # Train only on first few batches for demonstration if needed, or remove break for full training
        if batch_idx >= 1: 
            break

    if total_batches_processed > 0:
        avg_overall_training_loss = total_epoch_loss / total_batches_processed
        print(f"\nTraining completed. Overall Average PPO Loss: {avg_overall_training_loss:.4f}")
    else:
        print("\nNo batches processed.")

if __name__ == "__main__":
    main()
