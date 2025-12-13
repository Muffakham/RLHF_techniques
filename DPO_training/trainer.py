
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

    def prepare_batch(self, batch):
        prompts = batch["prompt"]
        chosen_texts = batch["chosen"]
        rejected_texts = batch["rejected"]

        # Tokenize
        chosen_enc = self.tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
        rejected_enc = self.tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
        prompt_enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)

        # Compute prompt lengths/start indices
        response_start_indices = (prompt_enc.attention_mask.sum(dim=1)).tolist()
        
        return chosen_enc, rejected_enc, response_start_indices

    def compute_dpo_loss(self, chosen_enc, rejected_enc, response_start_indices):
        """Computes the DPO loss for a single batch."""
        
        # 1. Policy Log probabilities
        policy_chosen_logps = get_batch_logps(self.policy_model, chosen_enc.input_ids, chosen_enc.attention_mask, response_start_indices)
        policy_rejected_logps = get_batch_logps(self.policy_model, rejected_enc.input_ids, rejected_enc.attention_mask, response_start_indices)

        # 2. Reference Log probabilities (no_grad)
        with torch.no_grad():
            ref_chosen_logps = get_batch_logps(self.ref_model, chosen_enc.input_ids, chosen_enc.attention_mask, response_start_indices)
            ref_rejected_logps = get_batch_logps(self.ref_model, rejected_enc.input_ids, rejected_enc.attention_mask, response_start_indices)

        # 3. DPO Loss Calculation
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits.to(torch.float32)).mean()
        
        return loss

    def update_model(self, loss, clip_norm, scheduler):
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), clip_norm)
            
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def train_epoch(self, dataloader: DataLoader, clip_norm: float, scheduler=None, log_every=10):
        self.policy_model.train()
        total_loss = 0.0
        total_batches = 0

        for step, batch in enumerate(dataloader):
            try:
                # 1. Prepare Data
                chosen_enc, rejected_enc, response_start_indices = self.prepare_batch(batch)
                
                # 2. Compute Loss
                loss = self.compute_dpo_loss(chosen_enc, rejected_enc, response_start_indices)
                
                # 3. Optimize
                self.update_model(loss, clip_norm, scheduler)

                total_loss += loss.item()
                total_batches += 1

                if (step + 1) % log_every == 0:
                    avg = total_loss / max(1, total_batches)
                    print(f"Step {step+1} | avg loss: {avg:.6f}")

            except Exception as e:
                print(f"Exception at step {step}: {e}")
                traceback.print_exc()
                continue

        avg_loss = total_loss / max(1, total_batches)
        return avg_loss
