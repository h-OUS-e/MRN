"""Code from DTE-DynamicTrainingEngine at https://github.com/windows7lover/DTE-DynamicTrainingEngine/tree/7b7cb4a3a97c4b085a33100725dd4f6d7d9fc929"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import torch

from models.trm import TRMState, TRMOutputs


@dataclass
class HaltingState:
    halted: torch.Tensor # bool[B]
    min_halt_steps: torch.Tensor # int[B]

    def to(self, device: torch.device) -> "HaltingState":
        """Simple method to cast items to gpu."""
        return HaltingState(
            halted=self.halted.to(device),
            min_halt_steps=self.min_halt_steps.to(device),
        )
        


class HaltingController:
    """
    Halting policy for ACT.

    Stores:
      - max_steps
      - exploration parameters
      - per-sample minimum halt step (for exploration)
      - dataset metadata (kept for possible future use)
    """    
    def __init__(
        self,
        max_steps: int,
        exploration_enabled: bool = True,
        exploration_prob: float = 0.0,
        exploration_min_steps: int = 2,
        device: str = "cpu",
        metadata: Optional[Any] = None,
    ):
        self.min_steps = exploration_min_steps
        self.max_steps = max_steps
        self.exploration_enabled = exploration_enabled
        self.exploration_prob = exploration_prob
        self.metadata = metadata
        
    def init_halting_state(self, batch_size: int, device: torch.device) -> HaltingState:
        """
        Initialize halting_state for a new container.
        By default:
          - all samples start as halted=True (to replace empty data)
          - min_halt_steps = 0
        """
        halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        min_halt_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        return HaltingState(halted=halted, min_halt_steps=min_halt_steps)
    
    
    def reset_halting_state(self, halting_state: HaltingState, reset_mask: torch.Tensor) -> HaltingState:
        """
        Reset halting halting_state for rows in reset_mask.
        - halted -> False
        - min_halt_steps -> 0, then re-sampled for exploration if enabled
        """
        # Find batch indices where mask is True (sequences have halted/finshed)
        idx = reset_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return halting_state

        halted = halting_state.halted.clone()
        min_halt_steps = halting_state.min_halt_steps.clone()

        # Reset halting flags for newly assigned sequences.
        halted.index_fill_(0, idx, False)
        min_halt_steps.index_fill_(0, idx, 0)

        # Reset exploration constraints for newly assigned sequences.
        # Basically a coin-flip to decide if the new problem gets an 
        # exploration constraint to increase min halting steps.
        if self.exploration_enabled and self.exploration_prob > 0:
            explore_mask = torch.rand(idx.numel(), device=idx.device) < self.exploration_prob
            if explore_mask.any():
                chosen = idx[explore_mask]
                min_halt_steps[chosen] = torch.randint(
                    low=self.min_steps,
                    high=self.max_steps + 1,
                    size=(explore_mask.sum(),),
                    device=idx.device,
                    dtype=torch.int32,
                )

        return HaltingState(halted=halted, min_halt_steps=min_halt_steps)
    
    
    def update_halting_state(
        self,
        halting_state: HaltingState,
        model_state: TRMState,
        outputs: TRMOutputs,
        loss_info: dict,
        *,
        training: bool,
    ) -> HaltingState:
        """
        Halting Policy. Every update step:
          - reads model outputs + loss_info + steps
          - returns a NEW HaltingState
          
        The model can only halt early when it's actually getting the right answer. 
        This prevents it from learning to "give up" on hard problems; it must keep
        thinking until it either gets it right or runs out of steps.
        """
        assert outputs is not None

        # Check if any items have exceeded max reasoning steps
        reached_limit = model_state.steps >= self.max_steps

        # In eval: only hard cap at max_steps
        if not training:
            halted = reached_limit
            return HaltingState(halted=halted, min_halt_steps=halting_state.min_halt_steps)

        # Three signals to decide when to halt:
        
        # i. Is the model's current answer correct
        # seq_correct is precomputed by the loss criterion (boolean [B])
        seq_correct = loss_info["halt/seq_correct"]
        if seq_correct.dtype != torch.bool:
            seq_correct = seq_correct > 0.5
            
        # ii. Does the model q_head want to halt?
        model_halt = outputs.q_halt > 0
        
        # iii. Has it run enough steps to satisfy the exploration constraint of min reasoning steps, if any
        pass_exploration = (model_state.steps >= halting_state.min_halt_steps)

        halted = reached_limit | (model_halt & seq_correct & pass_exploration)
        return HaltingState(halted=halted, min_halt_steps=halting_state.min_halt_steps)

    def get_halted_mask(self, halting_state: HaltingState) -> torch.Tensor:
        return halting_state.halted