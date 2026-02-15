import torch
from torch import nn

from pydantic import BaseModel
from dataclasses import dataclass
from typing import List
import math

from models.layers import Attention, SwiGLU, rms_norm, CastedLinear, RotaryEmbedding, CastedEmbedding, CosSin
from models.stat_utils import trunc_normal_init_


class TRMConfig(BaseModel):
    # TODO: fill out important config!
    batch_size: int
    seq_len: int
    vocab_size: int
    
    T : int = 3  # Number of total recursions over the RecursiveBlock (H_cycles in original repo)
    n: int = 6 # Number of recursions in the RecursiveBlock (L_cycles in original repo)
    
    hidden_size: int = 128
    num_heads: int = 4
    mlp_expansion: int = 4
    attn_block_num: int = 2
    
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0    
    pos_encoding: str = "rope" # or 'learned', else none !! 'learned' hasn't been implemented yet
    forward_dtype: str = 'bfloat16'
    
    # Puzzle embeddings: set puzzle_emb_ndim > 0 to enable
    puzzle_emb_ndim: int = 0 # embedding dimension per puzzle (0 = disabled)
    puzzle_emb_len: int = 16 # number of sequence positions the puzzle emb occupies
    num_puzzle_identifiers: int = 0 # total number of distinct puzzles in the dataset
    
    
class TRMEncoderConfig(BaseModel):
    vocab_size: int
    hidden_size: int
    forward_dtype: str
    
    # Puzzle embedding fields
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    num_puzzle_identifiers: int = 0
    batch_size: int = 1
    
@dataclass
class TRMStateManagerConfig:
    seq_len: int
    hidden_size: int
    forward_dtype: str
    puzzle_emb_len: int = 0  # added: state tensors are [B, seq_len + puzzle_emb_len, D]
    
    
#---------------------------------------------------------------------------
# INTERNAL STATES + OUTPUT STRUCTURES
#---------------------------------------------------------------------------
@dataclass
class TRMOutputs:
    logits: torch.Tensor
    q_halt: torch.Tensor
    q_continue: torch.Tensor

    def reset(self, mask: torch.Tensor | None):
        if mask is None:
            self.logits = None
            self.q_halt = None
            self.q_continue = None
            return

        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return

        if self.logits is not None:
            self.logits[idx] = 0.0
        if self.q_halt is not None:
            self.q_halt[idx] = 0.0
        if self.q_continue is not None:
            self.q_continue[idx] = 0.0

    def to(self, device: torch.device) -> "TRMOutputs":
        return TRMOutputs(
            logits=self.logits.to(device) if self.logits is not None else None,
            q_halt=self.q_halt.to(device) if self.q_halt is not None else None,
            q_continue=self.q_continue.to(device) if self.q_continue is not None else None,
        )

@dataclass
class TRMState:
    y: torch.Tensor # model output prediction
    z: torch.Tensor # latent vector (what the paper calls 'reasoning')
    steps: torch.Tensor
    
    def to(self, device: torch.device) -> "TRMState":
        """Simple method to cast items to GPU with .to('cuda')"""
        return TRMState(
            y=self.y.to(device),
            z=self.z.to(device),
            steps=self.steps.to(device),
        )
    
# @dataclass
# class TRMState:
#     latent_state: TRMLatentState
#     steps: torch.Tensor
#     halted: torch.Tensor # revise
#     current_data: Dict[str, torch.Tensor] # revise



class TRMStateManager(nn.Module):
    def __init__(self, config: TRMStateManagerConfig):
        super().__init__()
        dtype = getattr(torch, config.forward_dtype)
        self.seq_len = config.seq_len
        self.hidden_size = config.hidden_size
        
        # Initial states of latent vector 'z' and output 'y'. 
        # These are unlearnable/fixed initialization buffers (not trained by optimizer cuz not added to model's parameters)
        # Saved with model but gradients don't flow through them
        y_init = trunc_normal_init_(torch.empty(1, 1, config.hidden_size, dtype=dtype), std=1)
        z_init = trunc_normal_init_(torch.empty(1, 1, config.hidden_size, dtype=dtype), std=1)
        self.register_buffer("y_init", y_init, persistent=True)
        self.register_buffer("z_init", z_init, persistent=True)       
   
        
    def init_state(self, inputs: torch.Tensor) -> TRMState:
        """
        Initialize state for an entire batch. Creates new TRMState with y and z 
        initialized from fixed buffers, and step counters set to 0.
        
        Args:
            inputs: Input tensor of shape (B, ...) to determine batch size
            
        Returns:
            New TRMState object for the entire batch
        """
        B = inputs.shape[0] # batch_size
        L = self.seq_len
        D = self.hidden_size
        
        # creating init values based on initalized y and z states in the class
        # detach is unnecessary since clone keeps independent storage
        y = self.y_init.expand(B, L, D).clone()
        z = self.z_init.expand(B, L, D).clone()
        
        # Initial steps are 0. Each item from the batch has its own 'step'
        steps = torch.zeros(B, dtype=torch.int32, device=inputs.device)
        
        return TRMState(y=y, z=z, steps=steps)


    def reset_state(self, state: TRMState, mask: torch.Tensor):
        """
        Selectively reset specific sequences within existing batch state.
        
        Only sequences where mask=True are reset to initial values. Other
        sequences retain their current memory state. Updates are done in-place
        on the existing state object.
        
        Args:
            state: Existing TRMState object to modify
            mask: Boolean tensor of shape (B,) indicating which sequences to reset
            
        Returns:
            Modified TRMState object
        """
        # Find batch indices where mask is True
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return state # Nothing to reset
        
        # Prepare fresh initialization values for sequences being reset
        B = idx.numel()
        L = self.seq_len
        D = self.hidden_size        
        y = self.y_init.expand(B, L, D).clone()     
        z = self.z_init.expand(B, L, D).clone()
        
        # In-place updates: Overwrites only rows (the selected batches) at positions idx
        state.y.index_copy_(0, idx, y)
        state.z.index_copy_(0, idx, z)
        state.steps.index_fill_(0, idx, 0)
        
        # Equivalent to this line of code in original code:
        # z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H), where z_H is y, H_init is y_init, z_H is z, reset_flag is mask
        
        return state
        
        

#---------------------------------------------------------------------------
#                            NN OBJECT CLASSES
#---------------------------------------------------------------------------    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_expansion: float, rms_norm_eps: float):
        super().__init__()        
        
        # Defining building blocks of TRM
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size//num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=mlp_expansion
        )
        self.rms_norm_eps = rms_norm_eps
        
    def forward(self, x: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        """
        # B, L, D = x.shape
        B = Batch size (number of samples processed in parallel)
        L = Length (sequence length, i.e., number of tokens)
        D = Dimension (hidden size, i.e., feature dimension per token)
        """
        # 1. Self-Attention Pass
        out = self.self_attn(x, cos_sin=cos_sin) # cos_sin is for RoPE
        x = rms_norm(out + x, variance_epsilon=self.rms_norm_eps)
        
        # 2. MLP Pass
        out = self.mlp(x)
        x = rms_norm(out + x, variance_epsilon=self.rms_norm_eps)
        
        return x


class AttentionBlockStack(nn.Module):
    """
    Stack of Attention Blocks. OG coded calls it 'Reasoning Module'.
    """
    def __init__(self, layers: List[AttentionBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor, input_injection: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        x = x + input_injection # TODO: should we put this inside the loop for each layer maybe?
        for layer in self.layers:
            x = layer(x=x, cos_sin=cos_sin)
        return x
    
    
class TRMEncoder(nn.Module):
    def __init__(self, config: TRMEncoderConfig):
        super().__init__()
        # A trick to maintain variance in output activation regardless of 
        # hidden size by dividing by embed_scale, and multiplying it again 
        # in the forward pass.
        self.scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.scale
        
        dtype = getattr(torch, config.forward_dtype)        
        
        # Embed inputs
        self.embed = CastedEmbedding(
            config.vocab_size, 
            config.hidden_size, 
            init_std=embed_init_std, 
            cast_to=dtype
        )
        
    def forward(self, input_ids: torch.Tensor):
        e = self.embed(input_ids.int())
        # TODO: puzzle embeddings had a separate treatment, idk what they mean and if important, check!
        return self.scale * e
    
    
class TRM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = TRMConfig(**config)
        self.config = cfg
        self.forward_dtype = getattr(torch, cfg.forward_dtype)

        # Create Encoder and init input embeddings
        self.trm_encoder = TRMEncoder(TRMEncoderConfig(**config))
        
        # Create a state manager to track intermediary steps and initialize y and z
        self.trm_state_manager = TRMStateManager(TRMStateManagerConfig(**config))        
        
        # Core TRM model (which is a stack of custom attention blocks that are called recursively)
        attn_blocks = [
            AttentionBlock(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                mlp_expansion=cfg.mlp_expansion,
                rms_norm_eps=cfg.rms_norm_eps
            ) for _i in range(cfg.attn_block_num)
        ]
        
        self.trm_core_model = AttentionBlockStack(attn_blocks)
        
        # TRM heads for final calculations
        self.out_head = CastedLinear(cfg.hidden_size, cfg.vocab_size, bias=False) # lm_head in OG code
        self.q_head = CastedLinear(cfg.hidden_size, 2, bias=True)
        
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore            
        
        # RoPE embeddings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=cfg.hidden_size // cfg.num_heads,
                max_position_embeddings=cfg.seq_len, # + self.puzzle_emb_len,
                base=cfg.rope_theta)
        else:
            self.rotary_emb = None
  
    # TODO: check if we still need empty_state and reset_state now that we have TRMStateManager
    # def empty_state(self, batch_size: int):
    #     return TRMState(
    #         y=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
    #         z=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
    #     )
        
    # def reset_state(self, reset_flag: torch.Tensor, state: TRMState):
    #     return TRMState(
    #         y=torch.where(reset_flag.view(-1, 1, 1), self.H_init, state.y),
    #         z=torch.where(reset_flag.view(-1, 1, 1), self.L_init, state.z),
    #     )    
        
    def _latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int, cos_sin: CosSin):
        """
        Core of the TRM paper. Shows how the same model is used to predict
        latent vector 'z' given original input 'x' and intermediate output 'y'; and
        finally the final output 'y' given the final latent vector 'z'.
        """
        input_injection = x + y
        for i in range(n):
            z = self.trm_core_model(z, input_injection, cos_sin=cos_sin)
        y = self.trm_core_model(y, z, cos_sin=cos_sin) # Test here RoPE as none? hmm
        
        return y, z
    
    def forward(self, state: TRMState, batch) -> tuple[TRMState, TRMOutputs]:
        """
        Args:
            carry.y [batch, seq_len + puzzle_emb_len, hidden_size]
            carry.z [batch, seq_len + puzzle_emb_len, hidden_size]
        """
        # RoPE params
        cos_sin = self.rotary_emb() if self.rotary_emb is not None else None
        steps = state.steps + 1
        
        # Input encoding
        # TODO: understand batches in input embeddings? how they related to puzzle?
        input_embeddings = self.trm_encoder(batch["inputs"])
        
        # TODO: find a better word for state maybe
        # Forward iterations
        y = state.y
        z = state.z
        
        # Deep recursion
        # Recursting T-1 times to improve output y and and latent z (no gradients needed)
        with torch.no_grad():
            for i in range(self.config.T - 1):
                y, z = self._latent_recursion(x=input_embeddings, y=y, z=z, n=self.config.n, cos_sin=cos_sin)
            
        # Final recursive pass with 1 grad
        y, z = self._latent_recursion(x=input_embeddings, y=y, z=z, n=self.config.n, cos_sin=cos_sin)
        
        new_state = TRMState(
            y=y.detach().clone(), 
            z=z.detach().clone(), 
            steps=steps.detach().clone()
        )  # y and z are [B, seq_len + puzzle_embed_len, hidden_size]
        
        logits = self.out_head(y) #[:, self.puzzle_embed_len:] # [B, seq_len, vocab_size]
        q_logits = self.q_head(y[:, 0]).float()  # Q-head; uses the first puzzle_emb position. [B, 2]
        
        #[...,0] for halt action, [...,1] for continue action
        return new_state, TRMOutputs(logits=logits, q_halt=q_logits[...,0], q_continue=q_logits[...,1])



# class TRMWrapper(nn.Module):
#     """ACT Wrapper to train TRM with halt/continue actions."""
#     def __init__(self, config: dict):
#         super().__init__()
#         self.config = TRMConfig(**config)
#         self.model = TRM(config)
        
#     def forward(self, state: TRMState, input: torch.Tensor):
        
#         # Update state and remove halted sequences
#         new_state = self.model.trm_state_manager.reset_state(state=state, mask=x.halt)