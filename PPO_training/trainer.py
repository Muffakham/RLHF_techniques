import torch
import traceback
from .config import DEVICE, MAX_LENGTH, MAX_GRAD_NORM, PPO_EPOCHS, CLIP_EPS, KL_BETA
from .utils import get_batch_logps, calculate_kl_and_advantage, calculate_ppo_loss

class CustomPPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, tokenizer, policy_optimizer):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model, self.reward_tokenizer = reward_model
        self.tokenizer = tokenizer
        self.optimizer = policy_optimizer
        self.clip_eps = CLIP_EPS
        self.kl_beta = KL_BETA
        self.debug = False
        self.ppo_epoch = PPO_EPOCHS

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_rewards(self, texts):
        inputs = self.reward_tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            rewards = self.reward_model(inputs.input_ids, inputs.attention_mask)
        return rewards

    def _rollout_responses(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
        # Calculate prompt_lens BEFORE it is used
        prompt_lens = (inputs.attention_mask.sum(dim=1)).tolist()

        with torch.no_grad():
            sequence = self.policy_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                do_sample=True,
            )
            # Now old_policy_log_probs and final_mask are computed correctly
            old_policy_log_probs, final_mask = self._compute_log_probs_and_mask(self.policy_model, sequence, prompt_lens)

        generated_texts = self.tokenizer.batch_decode(sequence, skip_special_tokens=True)
        return sequence, generated_texts, prompt_lens, old_policy_log_probs, final_mask

    def _compute_log_probs_and_mask(self, model, sequence, prompt_lens):
        attention_mask = (sequence != self.tokenizer.pad_token_id).long()
        log_probs, final_mask = get_batch_logps(model, sequence, attention_mask, prompt_lens)
        return log_probs, final_mask

    def _ppo_optimization_loop(self, sequence, prompt_lens, policy_model_old_log_probs, advantages, final_mask, rewards, kl_div):
        total_ppo_loop_loss = 0.0
        last_ratio = None # To store the ratio from the last PPO step for printing

        for i in range(self.ppo_epoch):
            # Calculate PPO Loss
            policy_model_new_log_probs, _ = self._compute_log_probs_and_mask(self.policy_model, sequence, prompt_lens)
            loss, ratio = calculate_ppo_loss(policy_model_new_log_probs, policy_model_old_log_probs, advantages, final_mask, self.clip_eps)
            last_ratio = ratio # Store for later printing

            # Logging for each PPO epoch
            if self.debug:
              print(f"  PPO Epoch {i+1}/{self.ppo_epoch}")
              print(f"    advantages: {advantages.shape}, ratio: {ratio.shape}, mask: {final_mask.shape}")
              print(f"    rewards mean/std: {rewards.mean().item():.4f}/{rewards.std().item():.4f}")
              print(f"    kl_seq mean/std: {kl_div.mean().item():.4f}/{kl_div.std().item():.4f}")
              print(f"    advantages mean/std: {advantages.mean().item():.4f}/{advantages.std().item():.4f}")
              print(f"    ratio mean/std: {ratio.mean().item():.4f}/{ratio.std().item():.4f}")


            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: PPO Loss is NaN/Inf during optimization. Returning 0.0 for this batch.")
                return 0.0, None # Return 0 loss and None for ratio if invalid

            # Model updates
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
            total_ppo_loop_loss += loss.item()
            if self.debug:
              print(f"    ppo loss = {loss.item():.4f}")

        return total_ppo_loop_loss / self.ppo_epoch if self.ppo_epoch > 0 else 0.0, last_ratio


    def train_step(self, prompts):
        try:
            # 1. Rollout
            sequence, generated_texts, prompt_lens, policy_model_old_log_probs, final_mask = self._rollout_responses(prompts)

            # 2a. Compute Rewards
            rewards = self.compute_rewards(generated_texts)

            # 2b. Calculate log_probs for ref model
            with torch.no_grad():
                ref_log_probs, _ = self._compute_log_probs_and_mask(self.ref_model, sequence, prompt_lens)

            # 3. Calculate Advantage
            kl_div, advantages = calculate_kl_and_advantage(rewards, policy_model_old_log_probs, ref_log_probs, final_mask, self.kl_beta)

            # Perform PPO optimization steps
            batch_avg_ppo_loss, last_ratio = self._ppo_optimization_loop(sequence, prompt_lens, policy_model_old_log_probs, advantages, final_mask, rewards, kl_div)

            if torch.isnan(torch.tensor(batch_avg_ppo_loss)) or torch.isinf(torch.tensor(batch_avg_ppo_loss)):
                print("Warning: Batch average PPO Loss is NaN/Inf. Skipping batch and returning 0.0.")
                return 0.0

            return batch_avg_ppo_loss # Return the average loss for this batch (averaged over PPO_EPOCHS inner loops)

        except Exception as e:
            print(f"Error in train_step: {e}")
            traceback.print_exc() # Print full traceback for debugging
            self.optimizer.zero_grad()
            return 0.0
