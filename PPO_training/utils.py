import torch
import torch.nn.functional as F
from typing import List, Tuple

def get_batch_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_start_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns sum of log probabilities over response tokens for each batch element.
    input_ids: LongTensor [B, L]
    attention_mask: LongTensor [B, L]
    response_start_indices: list[int] length B; index (0-based) of first response token in unshifted input_ids
    """
    model_output = model(input_ids=input_ids, attention_mask=attention_mask)
    # logits: [B, L, V]
    logits = model_output.logits

    # Shift logits and targets for causal LM: predict token t from logits at t-1
    logits = logits[:, :-1, :].float()  # [B, L-1, V] as float32 for numerical stability
    targets = input_ids[:, 1:]           # [B, L-1]
    attn_shifted = attention_mask[:, 1:] # [B, L-1]

    # log-softmax over vocab
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L-1, V]

    # gather token log probs
    token_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

    # Build response mask in the full (unshifted) space, then shift it to align with token_log_probs
    B, L = input_ids.shape
    full_indices = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)  # [B, L]
    start_tensor = torch.tensor(response_start_indices, device=input_ids.device).unsqueeze(1)  # [B, 1]

    response_mask_full = (full_indices >= start_tensor).float()  # 1 for response tokens, 0 otherwise; shape [B, L]

    # Shift mask to align with token_log_probs/targets (since they are input_ids[:,1:] etc.)
    response_mask_shifted = response_mask_full[:, 1:]  # [B, L-1]

    # Combine with attention mask (only keep non-padding tokens)
    final_mask = (response_mask_shifted * attn_shifted).to(token_log_probs.dtype)  # float

    # Zero-out non-response positions (keeping negative log_probs for true tokens)
    masked_token_log_probs = token_log_probs * final_mask  # [B, L-1]

    # Sum across sequence length to produce a scalar log-prob per example
    batch_logps = masked_token_log_probs  # [B]

    return batch_logps, final_mask  # dtype: float32 on device

def calculate_kl_and_advantage(rewards: torch.Tensor, 
                               policy_model_old_log_probs: torch.Tensor, 
                               ref_log_probs: torch.Tensor, 
                               final_mask: torch.Tensor,
                               kl_beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate KL divergence and advantages.
    """
    token_kl_div = policy_model_old_log_probs - ref_log_probs
    kl_div = (token_kl_div * final_mask).sum(dim=1) / final_mask.sum(dim=1)
    R_total = rewards - (kl_beta * kl_div)

    if R_total.std() < 1e-8:
        advantages = R_total - R_total.mean()
    else:
        advantages = (R_total - R_total.mean()) / (R_total.std() + 1e-8)
    return kl_div, advantages.unsqueeze(1)

def calculate_ppo_loss(policy_model_new_log_probs: torch.Tensor, 
                       policy_model_old_log_probs: torch.Tensor, 
                       advantages: torch.Tensor, 
                       final_mask: torch.Tensor,
                       clip_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the PPO loss.
    """
    ratio = torch.exp(policy_model_new_log_probs - policy_model_old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -(torch.min(surr1, surr2) * final_mask).sum()
    loss /= final_mask.sum()
    return loss, ratio
