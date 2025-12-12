# Reward Model Training Project

This project implements a reward model using a pre-trained transformer model (`microsoft/deberta-v3-small`) to differentiate between chosen and rejected responses based on a dataset of financial questions and answers.

## Project Structure
- `requirements.txt`: Lists the Python dependencies.
- `README.md`: Project documentation.
- `config.py`: Centralizes all configuration parameters for the training process.
- `data_loader.py`: Contains the `RewardIterableDataset` and `collate_fn` for loading and processing data.
- `reward_model.py`: Defines the `RewardModel` architecture.
- `train_reward_model.py`: Script to train the reward model and save the trained model.

## Setup and Installation
1. Clone this repository (if applicable).
2. Create a virtual environment (optional but recommended):
   `python -m venv venv`
   `source venv/bin/activate` (on Linux/macOS)
   `.\venv\Scripts\activate` (on Windows)
3. Install the required packages:
   `pip install -r requirements.txt`

## Configuration
All training parameters are defined in `config.py`, including `MODEL_ID`, `DEVICE`, `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`, `MAX_LENGTH`, and `MODEL_SAVE_PATH`.

## Data
The model is trained on a JSONL file named `financial_rewards.jsonl`, which contains entries with `prompt`, `chosen`, and `rejected` fields.

## Training
To train the reward model, run the `train_reward_model.py` script:
`python train_reward_model.py`

The trained model will be saved to the path specified in `config.MODEL_SAVE_PATH` (default: `reward_model.pt`).

## Model Details
- **Base Model**: `microsoft/deberta-v3-small` (configured in `config.py`)
- **Loss Function**: Bradley-Terry loss
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5 (configured in `config.py`)
- **Epochs**: 3 (configured in `config.py`)
- **Batch Size**: 10 (configured in `config.py`)
- **Max Length**: 128 (configured in `config.py`)
