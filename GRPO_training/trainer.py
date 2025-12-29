
import torch
import torch.nn.utils as nn_utils
from . import config
from .utils import get_batch_logps

class CustomGRPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, tokenizer, policy_optimizer, debug=True):
        self.policy_model = policy_model
        self.ref_model = ref_model
        # Reward model unpack if it's a tuple (model, tokenizer)
        if isinstance(reward_model, tuple):
             self.reward_model, self.reward_tokenizer = reward_model
        else:
             self.reward_model = reward_model
             self.reward_tokenizer = tokenizer # Fallback if not provided specifically

        self.tokenizer = tokenizer
        self.optimizer = policy_optimizer
        self.debug = debug # Debug flag

        # --- GRPO Hyperparameters ---
        self.group_size = config.GROUP_SIZE
        self.clip_eps = config.CLIP_EPS
        self.kl_beta = config.KL_BETA
        self.grpo_epochs = config.GRPO_EPOCHS

        # Freeze Reference Model to save memory/compute
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_rewards(self, texts):
        """
        Computes the reward (Scalar) for each sequence in the batch.
        Assumes reward_model returns a single logit/score per sequence.
        """
        inputs = self.reward_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(config.DEVICE)

        if self.debug:
            print(f"  [DEBUG] Reward Model Input IDs shape: {inputs.input_ids.shape}, dtype: {inputs.input_ids.dtype}")

        with torch.no_grad():
            # Adjust based on your specific reward model architecture
            # If classifier, you might need outputs.logits[:, 1] for the positive class
            rewards = self.reward_model(inputs.input_ids, inputs.attention_mask)
            # print(rewards) # Reduced verbosity

            # Ensure output is flattened [Batch_Size]
            if rewards.dim() > 1:
                rewards = rewards.squeeze()

        if self.debug:
            print(f"  [DEBUG] Computed Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
        return rewards

    def _expand_prompts(self, prompts):
        """
        Duplicates prompts to create the group batch.
        [A, B] -> [A, A, A, A, B, B, B, B] (if group_size=4)
        """
        expanded_prompts = [p for p in prompts for _ in range(self.group_size)]
        if self.debug:
            print(f"  [DEBUG] Expanded prompts count: {len(expanded_prompts)}")
        return expanded_prompts

    def _rollout_responses(self, prompts):
        # 1. Expand Inputs
        group_prompts = self._expand_prompts(prompts)
        inputs = self.tokenizer(group_prompts, return_tensors='pt', padding=True).to(config.DEVICE)

        if self.debug:
            print(f"  [DEBUG] Rollout Input IDs shape: {inputs.input_ids.shape}, dtype: {inputs.input_ids.dtype}")

        # Determine where the prompt ends (for masking later)
        # We calculate this BEFORE generation so we know exactly where the answer starts
        prompt_lens = inputs.attention_mask.sum(dim=1).tolist()
        if self.debug:
            print(f"  [DEBUG] Prompt lengths (first 5): {prompt_lens[:5]}")

        # 2. Generate G samples per prompt
        with torch.no_grad():
            sequence = self.policy_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=config.MAX_NEW_TOKENS,   # Adjust based on task
                do_sample=True,      # Essential for diversity in GRPO
                temperature=config.TEMPERATURE,     # Standard for RLHF
                pad_token_id=self.tokenizer.pad_token_id
            )

        if self.debug:
            print(f"  [DEBUG] Generated Sequence shape: {sequence.shape}, dtype: {sequence.dtype}")

        generated_texts = self.tokenizer.batch_decode(sequence, skip_special_tokens=True)
        if self.debug:
            print(f"  [DEBUG] Generated texts (first): {generated_texts[0][:100]}...")

        # 3. Calculate Log Probs of the Generated Text (Old Policy)
        # Proper attention mask for the full sequence
        seq_attention_mask = (sequence != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            old_log_probs, final_mask = get_batch_logps(
                self.policy_model, sequence, seq_attention_mask, prompt_lens
            )

        if self.debug:
            print(f"  [DEBUG] Old Log Probs shape: {old_log_probs.shape}, dtype: {old_log_probs.dtype}")
            print(f"  [DEBUG] Final Mask shape: {final_mask.shape}, dtype: {final_mask.dtype}")

        return sequence, generated_texts, prompt_lens, old_log_probs, final_mask, seq_attention_mask

    def _calculate_group_advantage(self, rewards, num_unique_prompts):
        """
        Computes the relative advantage of each sample compared to its group mean.
        """
        # Reshape to [Num_Prompts, Group_Size]
        # e.g., if batch=8 and group=4, reshape to [2, 4]
        rewards_reshaped = rewards.view(num_unique_prompts, self.group_size)

        # Calculate stats per group
        group_means = rewards_reshaped.mean(dim=1, keepdim=True)
        group_stds = rewards_reshaped.std(dim=1, keepdim=True)

        # Normalize: (R - Mean) / Std
        # Add small epsilon to std to prevent division by zero
        advantages = (rewards_reshaped - group_means) / (group_stds + 1e-8)

        # Flatten back to [Batch_Size]
        flattened_advantages = advantages.view(-1)
        if self.debug:
            print(f"  [DEBUG] Calculated Advantages shape: {flattened_advantages.shape}, dtype: {flattened_advantages.dtype}")
        return flattened_advantages

    def _calculate_grpo_loss(self, new_log_probs, old_log_probs, advantages, token_kl, final_mask):
        """
        Calculates the surrogate loss with KL penalty.
        """
        # 1. Calculate Ratio (New / Old)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 2. Broadcast Scalar Advantage to Vector Sequence
        # advantages: [Batch] -> [Batch, 1]
        # This repeats the scalar advantage across all tokens in that sequence
        adv_expanded = advantages.unsqueeze(1)

        # 3. Standard PPO Clipping
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_expanded

        # 4. Combine Objective with KL Penalty
        # We subtract Beta * KL. Since we want to MAXIMIZE objective, this minimizes KL.
        # token_kl is [Batch, Length], clipped_obj is [Batch, Length]
        objective = torch.min(surr1, surr2) - (self.kl_beta * token_kl)

        # 5. Mask and Average
        # We only want to train on the Response tokens (masked by final_mask)
        # Negative sign because PyTorch optimizers minimize loss
        loss = -(objective * final_mask).sum() / final_mask.sum()

        if self.debug:
            print(f"  [DEBUG] GRPO Loss: {loss.item():.4f}")
        return loss

    def train_step(self, prompts):
        if self.debug:
            print(f"[DEBUG] Starting train_step for batch with {len(prompts)} unique prompts.")
        try:
            # --- 1. Experience Collection (Rollout) ---
            if self.debug:
                print("  [DEBUG] --- Rolling out responses ---")
            sequence, generated_texts, prompt_lens, old_log_probs, final_mask, seq_attention_mask = self._rollout_responses(prompts)

            # --- 2. Scoring ---
            if self.debug:
                print("  [DEBUG] --- Computing rewards ---")
            rewards = self.compute_rewards(generated_texts) # [Batch]

            # --- 3. Advantage Calculation (Group Relative) ---
            if self.debug:
                print("  [DEBUG] --- Calculating advantages ---")
            advantages = self._calculate_group_advantage(rewards, len(prompts)) # [Batch]

            # --- 4. Reference Model Log Probs (for KL) ---
            if self.debug:
                print("  [DEBUG] --- Calculating reference model log probs ---")
            with torch.no_grad():
                ref_log_probs, _ = get_batch_logps(
                    self.ref_model, sequence, seq_attention_mask, prompt_lens
                )
            if self.debug:
                print(f"  [DEBUG] Reference Log Probs shape: {ref_log_probs.shape}, dtype: {ref_log_probs.dtype}")

            # Calculate KL per token (Vector)
            # Approx KL: log_p_model - log_p_ref
            token_kl = old_log_probs - ref_log_probs
            if self.debug:
                print(f"  [DEBUG] Token KL shape: {token_kl.shape}, dtype: {token_kl.dtype}")

            # --- 5. Optimization Loop ---
            if self.debug:
                print("  [DEBUG] --- Starting optimization loop ---")
            batch_loss = 0.0

            for i in range(self.grpo_epochs):
                if self.debug:
                    print(f"  [DEBUG]   Optimization Epoch {i+1}/{self.grpo_epochs}")
                # Recalculate new log probs (with gradients enabled)
                new_log_probs, _ = get_batch_logps(
                    self.policy_model, sequence, seq_attention_mask, prompt_lens
                )
                if self.debug:
                    print(f"  [DEBUG]   New Log Probs shape: {new_log_probs.shape}, dtype: {new_log_probs.dtype}")

                loss = self._calculate_grpo_loss(
                    new_log_probs,
                    old_log_probs,
                    advantages,
                    token_kl,     # Vector KL
                    final_mask
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: Loss is NaN/Inf. Skipping batch.")
                    return 0.0

                self.optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(self.policy_model.parameters(), config.MAX_GRAD_NORM)
                self.optimizer.step()

                batch_loss += loss.item()
            if self.debug:
                print("  [DEBUG] --- Optimization loop finished ---")
            return batch_loss / self.grpo_epochs

        except Exception as e:
            print(f"Error in train_step: {e}")
            import traceback; traceback.print_exc()
            self.optimizer.zero_grad()
            return 0.0
