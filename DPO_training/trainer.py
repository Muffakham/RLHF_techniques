
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import traceback
from .utils import get_batch_logps
import reward_model_training.config as config  # Reuse global config if needed, but mainly use passed args

class CustomDPOTrainer:
    def __init__(
        self,
        policy_model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        lr: float,
        beta: float,
        device: torch.device,
        max_length: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.max_length = max_length
        self.beta = beta

        # freeze ref model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # optimizer
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(self.policy_model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader, clip_norm: float, scheduler=None, log_every=10):
        self.policy_model.train()
        total_loss = 0.0
        total_batches = 0

        for step, batch in enumerate(dataloader):
            try:
                prompts = batch["prompt"]
                chosen_texts = batch["chosen"]
                rejected_texts = batch["rejected"]

                # Tokenize chosen/rejected using same settings so prompt length computation is consistent
                chosen_enc = self.tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
                rejected_enc = self.tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
                prompt_enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)

                # Compute response start indices from prompt_enc attention masks (unshifted)
                # This is the index (0-based) of the first response token within each sequence.
                response_start_indices = (prompt_enc.attention_mask.sum(dim=1)).tolist()  # list of ints

                # Get policy log probs
                policy_chosen_logps = get_batch_logps(self.policy_model, chosen_enc.input_ids, chosen_enc.attention_mask, response_start_indices)
                policy_rejected_logps = get_batch_logps(self.policy_model, rejected_enc.input_ids, rejected_enc.attention_mask, response_start_indices)

                # Get reference log probs (no grad)
                with torch.no_grad():
                    ref_chosen_logps = get_batch_logps(self.ref_model, chosen_enc.input_ids, chosen_enc.attention_mask, response_start_indices)
                    ref_rejected_logps = get_batch_logps(self.ref_model, rejected_enc.input_ids, rejected_enc.attention_mask, response_start_indices)

                # Compute log ratios
                chosen_log_ratio = policy_chosen_logps - ref_chosen_logps   # [B]
                rejected_log_ratio = policy_rejected_logps - ref_rejected_logps  # [B]

                logits = self.beta * (chosen_log_ratio - rejected_log_ratio)  # [B]
                # Numerically stable: convert to float32
                loss = -F.logsigmoid(logits.to(torch.float32)).mean()

                # Sanity checks
                if torch.isnan(loss) or torch.isinf(loss):
                    # print diagnostics
                    print("NaN/Inf loss detected. Diagnostics:")
                    print("policy_chosen_logps:", policy_chosen_logps)
                    print("policy_rejected_logps:", policy_rejected_logps)
                    print("ref_chosen_logps:", ref_chosen_logps)
                    print("ref_rejected_logps:", ref_rejected_logps)
                    raise ValueError("Loss is NaN or Inf")

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                # optional grad clip
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), clip_norm)
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                total_batches += 1

                if (step + 1) % log_every == 0:
                    avg = total_loss / max(1, total_batches)
                    print(f"Step {step+1} | avg loss: {avg:.6f}")

            except Exception as e:
                print("Exception during training step:", e)
                traceback.print_exc()
                # don't crash full training; skip this batch
                continue

        avg_loss = total_loss / max(1, total_batches)
        return avg_loss
