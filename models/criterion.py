from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trm import TRMOutputs, TRMState


# ------------------------------------------------------------
# Stablemax helpers 
# ------------------------------------------------------------
def s(x, eps=1e-30):
    """
    Piecewise mapping replacing exp() in softmax Always positive,
    overflow-free.
    """
    return torch.where(x < 0, 1 / (1 - x + eps), x + 1)


def log_stablemax(x, dim=-1):
    """
    Log-probabilities using stablemax normalization instead of log-softmax. 
    Numerically stable alternative to log_softmax.
    """
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, valid_mask):
    """
    Cross-entropy loss using stablemax instead of softmax. 
    Masks out invalid tokens (padding/ignore) via valid_mask.
    """
    logprobs = log_stablemax(logits.to(torch.float64))

    labels = labels.to(torch.int64)
    safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))

    gathered = torch.gather(logprobs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    return -torch.where(valid_mask, gathered, torch.zeros_like(gathered))


# The only diff between softmax CE and stablemax CE is log_softmax and log_stablemax
# def softmax_cross_entropy2(logits, labels, valid_mask):
#     """Standard cross-entropy loss with valid_mask instead of ignore_index."""
#     logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
#     labels = labels.to(torch.int64)
#     safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
#     gathered = torch.gather(logprobs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
#     return -torch.where(valid_mask, gathered, torch.zeros_like(gathered))


class TRMCriterion(nn.Module):
    """
    Unified loss + metrics + halting-signals.

    forward() returns:
      - loss : scalar Tensor
      - info : flat dict containing:
            loss/*
            halt/*
            metric/*
    """

    def __init__(self, loss_type: str, pad_id: int = -100):
        """
        pad_id is set to -100 (IGNORE_LABEL_ID), used both for masking 
        and F.cross_entropy's ignore_index. That is because we don't want
        masked items to contribute to the loss.
        """
        super().__init__()
        self.pad_id = pad_id

        # normalize alias names
        lt = loss_type.lower()
        self.loss_type = {
            "stablemax": "stablemax",
            "stablemax_cross_entropy": "stablemax",
            "softmax": "softmax",
            "softmax_cross_entropy": "softmax",
        }.get(lt, lt)
        
        
    def forward(self, output: TRMOutputs, labels, state: TRMState):
        """
        Args:
            output: TRMOutputs(logits, q_halt, ...)
            labels: Tensor[B, L]
            state: TRMState (unused except to show consistency / future use)

        Returns:
            loss: scalar
            info: dict[str, Tensor]
        """
        logits = output.logits          # [B, L, V]
        q_halt_logits = output.q_halt   # [B] or [B,1]
        
        # Mask out padding tokens so they don't contribute to the loss.
        # Real tokens → True, padding → False. Not all tokens in the output
        # sequence contribute to the answer. The attention architecture simply
        # process all positions in parallel and output a hidden state for every
        # position. out_head layer maps each hidden state to vocab logits.
        valid_mask = labels != self.pad_id # which tokens are real vs padding to ignore
        
        B, L, V = logits.shape
        
        # reshape for CE
        flat_logits = logits.reshape(B * L, V)
        flat_labels = labels.reshape(B * L)
        flat_valid = valid_mask.reshape(B * L) 

        # Calculate loss for valid outputs
        if self.loss_type == "stablemax":
            loss = stablemax_cross_entropy(flat_logits, flat_labels, flat_valid)
        else:
            loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=self.pad_id,
                reduction="none",
            )
        
        # unflatten from [B*L,] -> [B,L]
        per_token_loss = loss.reshape(B, L) 
        
        # count how many valid (non-masked) tokens per sequence
        token_count = valid_mask.sum(-1).clamp(min=1)
        
        # average loss per valid sequence, then sum across the batch. 
        # This way short answers and long answers contribute equally
        lm_loss = (per_token_loss.sum(-1) / token_count).sum()

        # For each position, pick the token with the highest logit
        preds = torch.argmax(logits, dim=-1) # [B, L, vocab_size] -> [B, L]
        
        # Mask to check which prediction (token) matches label AND position is not padding
        token_correct = (preds == labels) & valid_mask
        
        # Mask to check if ALL valid tokens in a sequence are correct (B,)
        # This means we only accept answers that are 100% correct for halting,
        # otherwise they are wrong (even if close enough to answer).
        # TODO: what if we train it at 99% instead of 100%
        seq_correct = (token_correct.sum(-1) == token_count)
        
        # halting loss
        q_h = q_halt_logits.squeeze(-1)
        halt_loss = F.binary_cross_entropy_with_logits(
            q_h, seq_correct.float(), reduction="sum"
        )
        
        # Halting loss: trains Q-head to predict whether the full sequence is correct (binary yes/no).
        # This is combined with LM loss (which gives partial credit per token) at 0.5 weight
        # since getting the answer right matters more than knowing when to stop.
        total_loss = lm_loss + 0.5 * halt_loss # TODO: what if we increase weight of halt loss
        
        # metrics
        token_acc = (
            token_correct.float().sum() / valid_mask.float().sum()
            if valid_mask.any()
            else torch.tensor(0.0)
        )
        seq_acc = seq_correct.float().mean()
        
        info = {
            "loss/lm": lm_loss.detach(),
            "loss/halt": halt_loss.detach(),
            "loss/total": total_loss.detach(),
            "halt/seq_correct": seq_correct,
            "halt/token_correct": token_correct,
            "metric/seq_acc": seq_acc.detach(),
            "metric/token_acc": token_acc.detach(),
        }

        return total_loss, info
