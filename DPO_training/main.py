
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from .data_loader import PairwisePreferenceDataset, collate_fn
from .trainer import CustomDPOTrainer
import DPO_training.config as config

if __name__ == "__main__":
    print(f"Loading tokenizer and model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # TinyLlama or Qwen or whatever you chose. 
    # Note: For real training, you might want to load weights from an SFT checkpoint.
    policy = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
    ref = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)

    print(f"Loading dataset from: {config.DATASET_PATH}")
    dataset = PairwisePreferenceDataset(config.DATASET_PATH)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    print("Initializing Trainer...")
    trainer = CustomDPOTrainer(
        policy_model=policy,
        ref_model=ref,
        tokenizer=tokenizer,
        lr=config.LEARNING_RATE,
        beta=config.BETA,
        device=config.DEVICE,
        max_length=config.MAX_LENGTH
    )

    print("Starting DPO Training...")
    avg_loss = trainer.train_epoch(loader, clip_norm=config.GRAD_CLIP_NORM, log_every=config.BATCH_LOG_EVERY)
    print("Avg epoch loss:", avg_loss)
