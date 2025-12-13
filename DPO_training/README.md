
# DPO Training Module

This module implements **Direct Preference Optimization (DPO)** for aligning language models with human preferences. 

## Structure

*   `config.py`: Contains hyperparameters (Learning Rate, Batch Size, etc.) and model/dataset paths.
*   `data_loader.py`: Handles loading of pairwise preference datasets (`prompt`, `chosen`, `rejected`).
*   `utils.py`: Helper functions, primarily `get_batch_logps` for calculating log probabilities of response tokens.
*   `trainer.py`: The `CustomDPOTrainer` class which implements the DPO loss function and training loop.
*   `main.py`: The entry point to run the training.

## Usage

1.  Ensure you have the requirements in a virtual environment:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the training script (make sure you are in the root `RLHF_techniques` directory):
    ```bash
    python3 -m DPO_training.main
    ```

## Configuration

Modify `DPO_training/config.py` to change:
*   `MODEL_NAME`: The base model to train (e.g., `Qwen/Qwen2.5-0.5B-Instruct`).
*   `DATASET_PATH`: Path to your `.jsonl` preference dataset.
*   `LEARNING_RATE`, `MAX_LENGTH`, `BETA`, etc.

## How it Works

DPO avoids training a separate Reward Model. Instead, it directly optimizes the policy model to increase the likelihood of "chosen" responses relative to "rejected" ones, while staying close to a "reference" model (usually the original base model) to prevent degradation.

The loss function is:
`Loss = -log(sigmoid(beta * (log(policy_chosen/ref_chosen) - log(policy_rejected/ref_rejected))))`
