import torch

MODEL_ID = "microsoft/deberta-v3-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
EPOCHS = 3
BATCH_SIZE = 10
MAX_LENGTH = 128
MODEL_SAVE_PATH = "reward_model.pt"
DATASET_PATH = "dataset/financial_rewards.jsonl"
