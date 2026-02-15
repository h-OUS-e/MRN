"""
Training loop for TRM with adaptive halting.

The core idea: we maintain a persistent batch buffer and model state across
training steps. When a sequence halts (got it right or hit max steps), we
swap in a fresh problem from the dataloader and reset just that slot's state.
Sequences that haven't halted keep thinking on the same problem.

One training step:
    1. Reset halted slots (state, halting, batch data)
    2. Forward pass
    3. Compute loss
    4. Backward + optimizer step
    5. Update halting decisions
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict

from models.trm import TRM
from models.criterion import TRMCriterion
from models.halting_controller import HaltingController


def swap_batch_data(
    current_batch: Dict[str, torch.Tensor],
    new_batch: Dict[str, torch.Tensor],
    halted_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    For halted slots, replace current data with fresh data from the dataloader.
    Non-halted slots keep their current problem.
    
    Matches old code:
        new_current_data = {k: torch.where(carry.halted.view((-1,)+(1,)*(v.ndim-1)), batch[k], v) ...}
    """
    out = {}
    for k, current_v in current_batch.items():
        new_v = new_batch[k]
        # Reshape mask to broadcast: (B,) -> (B, 1, 1, ...) to match tensor dims
        mask = halted_mask.view((-1,) + (1,) * (current_v.ndim - 1))
        out[k] = torch.where(mask, new_v, current_v)
    return out


def train(
    model: TRM,
    dataloader: DataLoader,
    criterion: TRMCriterion,
    halting_controller: HaltingController,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
):
    model.train()
    data_iter = iter(dataloader)
    
    # ---- Initialization ----
    # Get first batch to determine shapes and fill the buffer
    first_batch = next(data_iter)
    first_batch = {k: v.to(device) for k, v in first_batch.items()}
    batch_size = first_batch["inputs"].shape[0]
    
    # Initialize persistent state (all slots start as "halted" so they get
    # filled with real data on the first step)
    state = model.trm_state_manager.init_state(first_batch["inputs"])
    halting_state = halting_controller.init_halting_state(batch_size, device)
    # halting_state.halted is all True initially ^
    
    current_batch = {k: torch.empty_like(v) for k, v in first_batch.items()}
    pending_batch = first_batch  # First real data ready to be swapped in

    # ---- Training Loop ----
    for step in range(num_steps):
        
        halted_mask = halting_controller.get_halted_mask(halting_state)
        
        # --- Step 1: Reset halted slots ---
        # For halted sequences: reset model state, reset halting flags, swap in new data
        model.trm_state_manager.reset_state(state, halted_mask)
        halting_state = halting_controller.reset_halting_state(halting_state, halted_mask)
        current_batch = swap_batch_data(current_batch, pending_batch, halted_mask)
        
        # Pre-fetch next batch for the NEXT reset (so dataloader isn't blocking mid-step)
        try:
            pending_batch = next(data_iter)
            pending_batch = {k: v.to(device) for k, v in pending_batch.items()}
        except StopIteration:
            data_iter = iter(dataloader)
            pending_batch = next(data_iter)
            pending_batch = {k: v.to(device) for k, v in pending_batch.items()}
        
        # --- Step 2: Forward pass ---
        new_state, outputs = model(state, current_batch)
        
        # --- Step 3: Compute loss ---
        loss, info = criterion(outputs, current_batch["labels"], new_state)
        
        # --- Step 4: Backward + optimizer ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Step 5: Update halting decisions ---
        with torch.no_grad():
            halting_state = halting_controller.update_halting_state(
                halting_state, new_state, outputs, info, training=True,
            )
        
        # --- Step 6: Carry state forward ---
        state = new_state  # state.y and state.z are already detached in TRM.forward()
        
        # --- Logging ---
        if step % 100 == 0:
            print(
                f"step {step:>5d} | "
                f"loss {info['loss/lm'].item():.4f} | "
                f"halt_loss {info['loss/halt'].item():.4f} | "
                f"seq_acc {info['metric/seq_acc'].item():.3f} | "
                f"tok_acc {info['metric/token_acc'].item():.3f} | "
                f"halted {halting_state.halted.sum().item()}/{batch_size}"
            )