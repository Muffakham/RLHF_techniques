import torch

# Model Configurations
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_ID = "microsoft/deberta-v3-small"
REWARD_MODEL_PATH = 'reward_model.pt'

# Training Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
MAX_LENGTH = 256
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 10
PPO_EPOCHS = 5

# PPO Specific Parameters
CLIP_EPS = 0.2
KL_BETA = 0.0
