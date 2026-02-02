import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

import einops
from typing import Tuple
from stat_utils import trunc_normal_init_



#---------------------------------------------------------------------------
#                            Data Classes
#---------------------------------------------------------------------------
CosSin = Tuple[torch.Tensor, torch.Tensor]


#---------------------------------------------------------------------------
#                            Common Functions
#---------------------------------------------------------------------------
# Functions for SwiGLU
def _find_multiple(a, b):
    return (-(a // -b)) * b

# Functions for RoPE
def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


#---------------------------------------------------------------------------
#                            ML Object Classes (layers)
#---------------------------------------------------------------------------
class CastedLinear(nn.Module):
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class Attention(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        head_dim, 
        num_heads, 
        num_key_value_heads, 
        causal=False
    ):
        super().__init__()
        
        # Saving initial class attributes
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        
        # Defining projection layers for Q, K, V (Group Head Attention or GHA)
        #    Q needs num_heads * head_dim (full heads for queries)
        #    K needs num_kv_heads * head_dim (can be fewer heads)
        #    V needs num_kv_heads * head_dim (can be fewer heads)
        #    Total: num_heads + 2 * num_kv_heads
        out_features_size = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        self.qkv_proj = CastedLinear(self.hidden_size, out_features_size, bias=False)
        
        # Output projection of the attention output back to hidden size (model dimension)
        self.output_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos_sin:CosSin) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(x) # Attention projection
        
        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            
        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # (Batch, Seq, Heads, Dim) → (Batch, Heads, Seq, Dim)
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D') # Reshape back: (Batch, Heads, Seq, Dim) → (Batch, Seq, Heads, Dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # Flatten heads: (B, S, H, D) → (B, S, H*D)
      
        return self.output_proj(attn_output)

    
class SwiGLU(nn.Module):
    """
    An activation function combining SiLU and GLU mechanisms. Research shows
    that SwiGLU can enhance model performance by allowing for more complex
    interactions between input features, compared to ReLU or standard GLU for transformer MLPs
    """
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        intermediate_dim = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, intermediate_dim * 2, bias=False)
        self.down_proj    = CastedLinear(intermediate_dim, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
    

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

