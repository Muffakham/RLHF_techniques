# GRPO Training (Group Relative Policy Optimization)

This directory contains a modular implementation of **GRPO (Group Relative Policy Optimization)**, a memory-efficient Reinforcement Learning from Human Feedback (RLHF) algorithm.

## 🧠 What is GRPO?

**GRPO** stands for **Group Relative Policy Optimization**. It is designed to optimize language models based on rewards without needing a separate "Critic" or "Value" model (like in PPO), which saves significant memory and computational resources.

### The Core Concept in Simple Terms

In standard RLHF (like PPO), you have:
1.  **Actor (Policy):** Generates text.
2.  **Critic (Value):** Estimates "how good" the text is to calculate advantages.

In **GRPO**:
1.  **Actor (Policy):** Generates *multiple* outputs (a "group") for the *same* prompt.
2.  **No Critic:** Instead of a learned value model, GRPO uses the **average reward of the group** as the baseline.

### How It Works (Example)

Imagine the model is given the prompt: *"Explain inflation."*

1.  **Generate a Group:** The model generates 4 different answers (Group Size = 4).
    *   *Answer A*
    *   *Answer B*
    *   *Answer C*
    *   *Answer D*

2.  **Score Them:** A Reward Model scores each answer:
    *   Answer A: **0.9** (Great)
    *   Answer B: **0.5** (Okay)
    *   Answer C: **0.4** (Bad)
    *   Answer D: **0.6** (Okay)

3.  **Calculate Baseline (Average):**
    *   Average Reward = (0.9 + 0.5 + 0.4 + 0.6) / 4 = **0.6**

4.  **Calculate Advantage:** How much better/worse is each answer compared to the average?
    *   **Answer A:** 0.9 - 0.6 = **+0.3** (Positive signal -> "Do this more!")
    *   **Answer B:** 0.5 - 0.6 = **-0.1** (Negative signal -> "Do this less.")
    *   **Answer C:** 0.4 - 0.6 = **-0.2** (Strong negative signal -> "Definitely stop doing this.")
    *   **Answer D:** 0.6 - 0.6 = **0.0** (Neutral)

*(Note: In practice, we also divide by the standard deviation to normalize the advantages.)*

## 📂 Project Structure

- `config.py`: Configuration for models and hyperparameters (learning rate, group size, etc.).
- `data_loader.py`: Handles loading the preference dataset (`financial_rewards_500.jsonl`).
- `trainer.py`: Implements the `CustomGRPOTrainer` class with the GRPO logic.
- `train.py`: The main entry point to start training.
- `utils.py`: Utility functions for log probability calculations.

## 🚀 How to Run

1.  **Install Dependencies:**
    Ensure you have `torch`, `transformers`, and `accelerate` installed.

2.  **Configure:**
    Edit `config.py` to set your desired model IDs and hyperparameters.

3.  **Train:**
    Run the training script:
    ```bash
    python train.py
    ```

## 🛠 Key Hyperparameters (`config.py`)

- `GROUP_SIZE` (Default: 4): How many outputs to generate per prompt.
- `GRPO_EPOCHS` (Default: 1): How many times to train on the generated group data.
- `KL_BETA` (Default: 0.04): Penalty to keep the model from diverging too far from the base model.
