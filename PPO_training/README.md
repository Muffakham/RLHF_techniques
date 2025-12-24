# PPO Training Module for RLHF

This directory contains a modularized implementation of **Proximal Policy Optimization (PPO)** for Reinforcement Learning from Human Feedback (RLHF). This method aligns a language model with human preferences by optimizing it against a reward model while constraining deviation from a reference model.

## What is PPO?
**Proximal Policy Optimization (PPO)** is a policy gradient algorithm introduced by OpenAI. In the context of RLHF, it serves as the reinforcement learning phase where the language model (the policy) is fine-tuned to maximize rewards given by a separate Reward Model.

Key components of PPO in RLHF:
1.  **Policy Model**: The LLM being trained (Actor).
2.  **Reference Model**: A frozen copy of the original LLM to ensure the policy doesn't drift too far (KL Divergence constraint).
3.  **Reward Model**: A model that assigns a scalar score to the policy's outputs.
4.  **Value/Critic Model**: Often used to estimate expected future rewards, though in this simplified implementation, we calculate advantages directly from rewards and KL penalties.

The "Proximal" part refers to the **clipped objective function**, which prevents the new policy from changing too drastically from the old policy in a single update step, ensuring training stability.

## Utility Functions (`utils.py`)
A set of mathematical and tensor manipulation functions essential for the PPO loss calculation.

### `get_batch_logps(model, input_ids, attention_mask, response_start_indices)`
Calculates the log probabilities of the generated response tokens.
-   **Inputs**: The model, full sequence input IDs, attention mask, and the start indices of the generated response.
-   **Logic**:
    1.  Runs the model to get logits.
    2.  Shifts logits and input IDs by one position (standard AutoRegressive training setup).
    3.  Computes `log_softmax` to get log probabilities.
    4.  Applies a mask to select only the log probabilities corresponding to the **response** (ignoring the prompt and padding).
    5.  Sums the log probabilities for the response tokens.
-   **Output**: The total log probability for each response in the batch.

### `calculate_kl_and_advantage(...)`
Computes the KL divergence penalty and the advantage estimates.
-   **Inputs**: Rewards from Reward Model, log probs from Policy and Reference models.
-   **Logic**:
    1.  **KL Divergence**: Measures how different the Policy's token distribution is from the Reference Model.
        `KL = log_prob_policy - log_prob_ref`
    2.  **Total Reward**: Combines the explicit reward with the KL penalty.
        `R_total = Reward - (beta * KL)`
    3.  **Advantage**: Normalizes the total reward to determine how much "better" (or worse) a specific action was compared to the average.
-   **Output**: KL divergence values and normalized advantages.

### `calculate_ppo_loss(...)`
Computes the standard PPO clipped loss.
-   **Inputs**: New and old log probs, advantages.
-   **Logic**:
    1.  Calculates the probability ratio `r(t) = exp(new_log_probs - old_log_probs)`.
    2.  Computes the unclipped objective: `ratio * advantage`.
    3.  Computes the clipped objective: `clamp(ratio, 1-eps, 1+eps) * advantage`.
    4.  Takes the minimum of the two (pessimistic bound) and negates it (since we want to maximize).
-   **Output**: scalar loss value and the ratio (for logging).

## Training Loop (`trainer.py`)
The `CustomPPOTrainer` orchestrates the RLAL process. The `train_step` function performs one full PPO cycle:

1.  **Rollout (Generation)**:
    -   The Policy Model generates responses for a batch of prompts.
    -   We record the log probabilities of these exact sequences under the *current* Policy.

2.  **Reward Computation**:
    -   The generated text is passed to the **Reward Model** to get a scalar score (e.g., how helpful/harmless it is).

3.  **Reference Log Probs**:
    -   The same generated sequences are passed through the frozen **Reference Model** to assume baseline probabilities.

4.  **Advantage Estimation**:
    -   We compare the Policy's log probs vs. the Reference's log probs to calculate the KL penalty.
    -   `Advantage = Reward - KL_Penalty` (normalized).

5.  **PPO Optimization (Inner Loop)**:
    -   We run `PPO_EPOCHS` (e.g., 5) of gradient descent updates on the Policy Model.
    -   In each epoch, we recalculate log probs for the generated sequences (since the Policy works changes).
    -   We compute the PPO Loss (clipped ratio of new/old probs * advantage).
    -   Backpropagate and update the Policy Model weights.

## Usage
To start training:
```bash
python train.py
```
Ensure you have set up `config.py` with the correct model paths and hyperparameters.
