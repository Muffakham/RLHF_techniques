
import torch
import torch.nn.functional as F
from typing import List, Tuple

def shift_for_causal_lm(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shifts logits (t) and targets (t+1) for standard Causal LM training/loss calculation.
    Returns calculated (logits_shifted, targets_shifted, mask_shifted).
    """
    # [B, L-1, V] as float32 for numerical stability
    logits_shifted = logits[:, :-1, :].float()
    # [B, L-1]
    targets_shifted = input_ids[:, 1:]
    # [B, L-1]
    mask_shifted = attention_mask[:, 1:]
    
    return logits_shifted, targets_shifted, mask_shifted

def create_response_mask(input_ids: torch.Tensor, response_start_indices: List[int]) -> torch.Tensor:
    """
    Creates a mask for response tokens only.
    Returns response_mask_shifted [B, L-1] (shifted to align with logits).
    """
    B, L = input_ids.shape
    full_indices = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)  # [B, L]
    start_tensor = torch.tensor(response_start_indices, device=input_ids.device).unsqueeze(1)  # [B, 1]

    response_mask_full = (full_indices >= start_tensor).float()  # 1 for response tokens, 0 otherwise; shape [B, L]
    
    # Shift mask to align with token_log_probs/targets (since they are input_ids[:,1:] etc.)
    response_mask_shifted = response_mask_full[:, 1:]  # [B, L-1]
    
    return response_mask_shifted

def compute_token_logps(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes standard CrossEntropy-style log probabilities for each token.
    logits: [B, L-1, V] (already shifted)
    targets: [B, L-1]   (already shifted)
    Returns: [B, L-1]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs

def get_batch_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_start_indices: List[int]) -> torch.Tensor:
    """
    Computes the total log probability of the completion (response) given the prompt.
    
    Steps:
    1. Forward pass to get logits.
    2. Shift logits and targets for causal prediction.
    3. Compute raw log probabilities for every token.
    4. Create a mask that is 1 ONLY for the response tokens (ignoring prompt and padding).
    5. Sum up the log probabilities for the response tokens.
    """
    # 1. Forward Pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits # [B, L, V]

    # 2. Shift for Causal LM (predict t using t-1)
    logits_shifted, targets_shifted, attn_shifted = shift_for_causal_lm(logits, input_ids, attention_mask)

    # 3. Compute Per-Token Log Probs
    token_log_probs = compute_token_logps(logits_shifted, targets_shifted)

    # 4. Create Response Mask (Mask out prompt and padding)
    response_mask_shifted = create_response_mask(input_ids, response_start_indices)
    
    # Combined Mask: Must be in response region AND not be padding
    final_mask = (response_mask_shifted * attn_shifted).float()

    # 5. Apply Mask & Sum
    masked_log_probs = token_log_probs * final_mask
    summed_log_probs = masked_log_probs.sum(dim=-1) # [B]

    return summed_log_probs
