
import torch

# Model configuration
POLICY_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reward Model
REWARD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_PATH = "reward_model.pt"

# Dataset
DATASET_PATH = "financial_rewards_500.jsonl"


# Training hyperparameters
LEARNING_RATE = 1e-6 # From cell 14. Cell 4 had 1e-5 but cell 14 is the one used in trainer.
MAX_LENGTH = 256
MAX_GRAD_NORM = 1.0

# GRPO Hyperparameters
GROUP_SIZE = 4        # G: Number of samples per prompt
CLIP_EPS = 0.2        # PPO Clipping Epsilon
KL_BETA = 0.04        # KL Penalty Coefficient
GRPO_EPOCHS = 1       # Inner optimization loops

# Generation Parameters
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
