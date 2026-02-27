"""
Training utilities.
Example submission.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import sys
from tqdm import tqdm

_parent = str(Path(__file__).parent.parent)
if sys.path[0] != _parent:
    sys.path.insert(0, _parent)

from part3.nn_utils import cross_entropy, gradient_clipping


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    batch_size: int = 8
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False  # Set to True when initializing TrainingConfig to use AMP
    patience: Optional[int] = None


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, compute_loss_fn: Optional[Callable] = None):
        ### FASTER IMPLEMENTATION
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        
        # Cache device type once (used in every forward pass)
        self.device_type = 'cuda' if 'cuda' in config.device else 'cpu'
        
        # Move to device, compile, then assign
        # Note: compile AFTER .to(device), BEFORE training
        model = model.to(config.device)
        if hasattr(torch, "compile") and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            
            if major >= 7:
                print(f"🚀 CUDA Compute Capability {major}.{minor} detected. Enabling torch.compile!")
                self.model = torch.compile(model)
            else:
                print(f"⚠️ CUDA Compute Capability {major}.{minor} detected. Skipping torch.compile (requires >= 7.0).")
                self.model = model  # <--- Important: Fallback assignment
        else:
            print("⚠️ CUDA not available or torch.compile missing. Skipping compilation.")
            self.model = model  # <--- Important: Fallback assignment
        
        # Weight decay exclusion (unchanged, this is correct)
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params,   'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Use fused AdamW if on CUDA — faster kernel, fewer launches
        use_fused = self.device_type == 'cuda'
        self.optimizer = AdamW(optim_groups, lr=config.learning_rate, fused=use_fused)
        
        # Scheduler (unchanged)
        total_steps = len(train_dataloader) * config.num_epochs
        if config.warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, 
                              total_iters=config.warmup_steps)
            main = CosineAnnealingLR(self.optimizer, 
                                      T_max=total_steps - config.warmup_steps,
                                      eta_min=config.learning_rate * 0.1)
            self.scheduler = SequentialLR(self.optimizer, [warmup, main], 
                                           milestones=[config.warmup_steps])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps,
                                                eta_min=config.learning_rate * 0.1)
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        
        # self.model = model.to(config.device)
        # self.config = config
        # self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader
        # self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        
        # # 1. Initialize GradScaler for AMP
        # self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
        # # self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
        
        # # ==========================================
        # # 2. NEW: Weight Decay Exclusion
        # # ==========================================
        # # Create a dictionary of all parameters that require gradients
        # param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        
        # # Any parameter that is 2D or higher (like Linear weights or Embedding weights) gets decayed
        # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        
        # # Any parameter that is 1D (like biases, LayerNorm weights/biases) does NOT get decayed
        # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # optim_groups = [
        #     {'params': decay_params, 'weight_decay': config.weight_decay},
        #     {'params': nodecay_params, 'weight_decay': 0.0}
        # ]
        
        # # Initialize AdamW with our parameter groups
        # self.optimizer = AdamW(optim_groups, lr=config.learning_rate)
        # # ==========================================
        
        # total_steps = len(train_dataloader) * config.num_epochs

        # # ==========================================
        # # 3. NEW: Cosine Decay with floor (eta_min)
        # # ==========================================
        # if config.warmup_steps > 0:
        #     warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
            
        #     # Decay down to 10% of maximum learning rate instead of 0
        #     main = CosineAnnealingLR(
        #         self.optimizer, 
        #         T_max=total_steps - config.warmup_steps,
        #         eta_min=config.learning_rate * 0.1
        #     )
            
        #     self.scheduler = SequentialLR(self.optimizer, [warmup, main], milestones=[config.warmup_steps])
        # else:
        #     # Also apply eta_min here in case warmup is disabled
        #     self.scheduler = CosineAnnealingLR(
        #         self.optimizer, 
        #         T_max=total_steps, 
        #         eta_min=config.learning_rate * 0.1
        #     )
            
        # self.global_step = 0
        # self.best_val_loss = float("inf")
        # self.train_losses = []
        # self.val_losses = []
    
    def _default_lm_loss(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        logits = model(input_ids)
        batch_size, seq_len, vocab_size = logits.shape
        return cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
    
    def train_epoch(self) -> float:
        ### FASTER IMPLEMENTATION
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.config.device)
        num_batches = 0
        
        for batch in tqdm(self.train_dataloader):
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                loss = self.compute_loss_fn(batch, self.model)
            
            loss.backward()
            gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.detach()
            num_batches += 1
            self.global_step += 1
            
        return (total_loss / num_batches).item() if num_batches > 0 else 0.0
        
        # self.model.train()
        # total_loss = 0.0
        # num_batches = 0
        
        # # Extract base device type for autocast (e.g., 'cuda:0' -> 'cuda')
        # device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        
        # for batch in tqdm(self.train_dataloader):
        #     self.optimizer.zero_grad()
            
        #     # 2. Use autocast for the forward pass and loss computation
        #     with torch.autocast(device_type=device_type, enabled=self.config.use_amp, dtype=torch.float16):
        #         loss = self.compute_loss_fn(batch, self.model)
            
        #     # 3. Scale loss and backward
        #     self.scaler.scale(loss).backward()
            
        #     # 4. Unscale BEFORE gradient clipping so the norm is calculated on actual gradient values
        #     self.scaler.unscale_(self.optimizer)
        #     gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
            
        #     # 5. Step the scaler and update
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
            
        #     self.scheduler.step()
        #     total_loss += loss.item()
        #     num_batches += 1
        #     self.global_step += 1
            
        # return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_dataloader is None:
            return 0.0
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.config.device)
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader):
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                loss = self.compute_loss_fn(batch, self.model)
            total_loss += loss.detach()
            num_batches += 1
    
        return (total_loss / num_batches).item() if num_batches > 0 else 0.0
        # if self.val_dataloader is None:
        #     return 0.0
        # self.model.eval()
        # total_loss = 0.0
        # num_batches = 0
        
        # device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        
        # for batch in tqdm(self.val_dataloader):
        #     # Also use autocast during evaluation for faster inference
        #     with torch.autocast(device_type=device_type, enabled=self.config.use_amp, dtype=torch.float16):
        #         loss = self.compute_loss_fn(batch, self.model)
        #     total_loss += loss.item()
        #     num_batches += 1
        # return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self) -> Dict[str, Any]:
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            if self.val_dataloader:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
        return {"train_losses": self.train_losses, "val_losses": self.val_losses}


def compute_qa_loss(batch: Dict[str, torch.Tensor], model: nn.Module, device: str = "cuda") -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    print("input_ids:", input_ids.shape)
    print("labels:", labels.shape)
    logits = model(input_ids, attention_mask)
    print("logits:", logits.shape)
    return cross_entropy(logits, labels)


def create_qa_loss_fn(device: str = "cuda") -> Callable:
    return lambda batch, model: compute_qa_loss(batch, model, device)