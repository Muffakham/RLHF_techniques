# Financial Domain RLHF Techniques 🚀

This repository acts as a comprehensive sandbox for exploring and modularizing various **Reinforcement Learning from Human Feedback (RLHF)** techniques, specifically tailored for aligning Large Language Models (LLMs) in the **Financial Domain**.

The project breaks down complex alignment algorithms into understandable, modular components, making it easier to experiment with and understand the core mechanics of modern LLM alignment.

## 📚 Techniques Explored

### 1. Reward Modeling (RM)
**Location:** [`reward_model_training/`](reward_model_training/)

Before we can align a model, we need a way to measure "quality" automatically. A Reward Model is trained to mimic human preferences.

*   **Concept**: It acts as a classifier or regressor that takes a prompt and a response and outputs a scalar score (e.g., 4.5/5).
*   **Method**: We use the **Bradley-Terry Model**. The model is given pairs of (Chosen, Rejected) responses and trained to assign a higher score to the Chosen response.
*   **Loss Function**:
    $$ Loss = -\log(\sigma(r_{chosen} - r_{rejected})) $$
    *Where $\sigma$ is the sigmoid function.*
*   **Example**:
    *   *Prompt*: "Is this stock overvalued?"
    *   *Chosen*: "It depends on metrics X, Y, Z..." (Score: 0.9)
    *   *Rejected*: "Buy now!" (Score: 0.1)
    *   The model learns to maximize the gap (0.9 - 0.1).

### 2. Proximal Policy Optimization (PPO)
**Location:** [`PPO_training/`](PPO_training/)

The standard industry approach for RLHF (used by ChatGPT, etc.).

*   **Concept**: An **Actor-Critic** method. The "Actor" (LLM) generates text, and the "Critic" (Value Model) helps estimates expected future rewards to reduce variance.
*   **Key Mechanic**: **Clipping**. PPO limits how much the policy can change in a single update step. This prevents catastrophic forgetting where the model learns something new but destroys its language abilities.
*   **KL Divergence Penalty**: We add a penalty if the model drifts too far from the original "Reference Model" to ensure fluent text.
*   **Example**:
    *   The model initially wants to yell "BUY!" to get a high reward.
    *   PPO compares this valid but risky move against its previous policy.
    *   If the probability shift is too massive, the update is "clipped" (ignored).

### 3. Direct Preference Optimization (DPO)
**Location:** [`DPO_training/`](DPO_training/)

A more efficient alternative to PPO that bypasses the need for a separate Reward Model.

*   **Concept**: PPO is complex and unstable. DPO answers the question: "Can we optimize the policy directly using the preference data?"
*   **Method**: It mathematically derives a loss function where the language model *itself* acts as the reward model implicitly.
*   **Advantages**:
    *   No separate Reward Model training needed.
    *   No sampling (generation) required during training (faster).
    *   More stable than PPO.
*   **Example**:
    *   Instead of `Generate -> Score -> Update`, DPO simply takes the dataset batch `(Prompt, Chosen, Rejected)` and pushes the probability of `Chosen` up and `Rejected` down relative to the reference model.

### 4. Group Relative Policy Optimization (GRPO)
**Location:** [`GRPO_training/`](GRPO_training/)

A memory-efficient method recently popularized by DeepSeek-Math and other reasoning models.

*   **Concept**: Traditional PPO needs a "Critic" model which is as huge as the Actor (doubling memory usage). GRPO removes the Critic.
*   **Method**:
    1.  Generate a **Group** of outputs (e.g., 4 responses) for the same prompt.
    2.  Score them all.
    3.  Use the **Group Average** as the baseline.
*   **Why it helps**: If a prompt is hard, all scores might be low (e.g., 0.1, 0.2, 0.15, 0.1). Standard RL might penalize all. GRPO sees that 0.2 is *relatively* great compared to the 0.13 average, so it rewards it.
*   **Example**:
    *   *Prompt*: "Explain derivatives."
    *   *Outputs*: A (Score 0.8), B (Score 0.7), C (Score 0.9), D (Score 0.8).
    *   *Average*: 0.8.
    *   *Advantage*: C is +0.1 above average (Reinforce). B is -0.1 below average (Penalize).

## 📂 Project Structure

```
RLHF_techniques/
├── reward_model_training/  # Training the judge (Deberta)
├── PPO_training/           # The classic reinforcement learning loop
├── DPO_training/           # The efficient, direct optimization method
├── GRPO_training/          # The memory-efficient, group-based method
├── dataset/                # Shared datasets (financial_rewards.jsonl)
└── requirements.txt        # Shared dependencies
```

## 🛠 Quick Start

### Prerequisites
Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running a Module
Navigate to the specific folder and run the training script.

**Reward Model:**
```bash
cd reward_model_training
python train_reward_model.py
```

**DPO:**
```bash
# From root
python -m DPO_training.main
```

**GRPO:**
```bash
cd GRPO_training
python train.py
```
