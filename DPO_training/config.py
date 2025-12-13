
import torch

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 1e-6
MAX_LENGTH = 512
GRAD_CLIP_NORM = 1.0
BATCH_LOG_EVERY = 10
BETA = 1.0  # The beta parameter for DPO loss

# Data paths
DATASET_PATH = "financial_rewards.jsonl" # Assuming dataset is in the root or accessible path
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
