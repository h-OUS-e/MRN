"""
Efficient Rotary Positional Embeddings (RoPE) module from https://blog.eleuther.ai/rotary-embeddings/.
@misc{rope-eleutherai,
  title = {Rotary Embeddings: A Relative Revolution},
  author = {Biderman, Stella and Black, Sid and Foster, Charles and Gao, Leo and Hallahan, Eric and He, Horace and Wang, Ben and Wang, Phil},
  howpublished = url{blog.eleuther.ai/},
  note = {[Online; accessed ]},
  year = {2021}
}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        # If the sequence length hasn't changed, reuse the previously calculated 
        # sines/cosines to save computation speed.
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            # This is where we compute the angle for every position in the sequence
            # against every frequency in the embedding dimension.
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [Batch, SeqLen, Heads, HeadDim]
    # cos, sin: [SeqLen, 1, 1, HeadDim] (broadcastable)

    # We need to slice cos/sin to matching sequence length of q/k
    # (In case cache is larger than current sequence)
    # cos = cos[:q.shape[0], ...]
    # sin = sin[:q.shape[0], ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RoPE3D(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.dim_split = self._compute_dim_split(dim)

        # Create frequency bands for each plane
        for i, d in enumerate(self.dim_split):
            inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
            self.register_buffer(f"inv_freq_{i}", inv_freq)

        self.cache = {}

    def _compute_dim_split(self, dim):
        base_split = dim // 3
        remainder = dim % 3
        return [base_split + remainder, base_split, base_split]

    def forward(self, x, seq_dim=1, aux_positions=None):
        seq_len = x.shape[seq_dim]
        device = x.device

        # Default auxiliary positions to zeros
        if aux_positions is None:
            aux1 = torch.zeros(seq_len, device=device)
            aux2 = torch.zeros(seq_len, device=device)
        else:
            aux1, aux2 = aux_positions

        # Create cache key
        cache_key = (seq_len, tuple(aux1.cpu().tolist()), tuple(aux2.cpu().tolist()))

        if cache_key not in self.cache:
            # Primary position
            t_primary = torch.arange(seq_len, device=device).type_as(self.inv_freq_0)
            t_aux1 = aux1.type_as(self.inv_freq_1)
            t_aux2 = aux2.type_as(self.inv_freq_2)

            # Compute frequencies for each plane
            freqs_list = []
            for i, t_pos in enumerate([t_primary, t_aux1, t_aux2]):
                inv_freq = getattr(self, f"inv_freq_{i}")
                freqs = torch.einsum("i,j->ij", t_pos, inv_freq)
                freqs = torch.cat((freqs, freqs), dim=-1)
                freqs_list.append(freqs)

            # Concatenate all frequency embeddings
            emb = torch.cat(freqs_list, dim=-1)

            cos_cached = emb.cos()[:, None, None, :]
            sin_cached = emb.sin()[:, None, None, :]

            self.cache[cache_key] = (cos_cached, sin_cached)

        return self.cache[cache_key]


@torch.jit.script
def apply_rotary_pos_emb_3d(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def default_aux_positions(x, block_size: int = 32):
    seq_len = x.shape[1]
    device = x.device
    positions = torch.arange(seq_len, device=device).float()
    aux1 = positions // block_size
    aux2 = positions % block_size
    return (aux1, aux2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=512, n_head=12, max_len=1024, use_rope=True, causal=False, rope_type='2d', aux_positions_fn=None):
        super().__init__()
        assert d_model % n_head == 0
        self.head_dim = d_model // n_head
        self.n_head = n_head
        self.use_rope = use_rope
        self.causal = causal
        self.rope_type = rope_type
        self.aux_positions_fn = aux_positions_fn

        # Projections for Q, K, V
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        # Initialize RoPE
        # NOTE: RoPE acts on the *head dimension*, not the model dimension
        if use_rope:
            if rope_type == '3d':
                self.rope = RoPE3D(self.head_dim)
            else:
                self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (SeqLen), Channels (Dim)

        # 1. Calculate Q, K, V
        # Result shape: [B, T, 3 * C]
        qkv = self.c_attn(x)
        
        # Split and reshape to [B, T, n_head, head_dim]
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # 2. Apply RoPE if enabled
        if self.use_rope:
            if self.rope_type == '3d':
                aux_pos = self.aux_positions_fn(x) if self.aux_positions_fn else None
                cos, sin = self.rope(q, seq_dim=1, aux_positions=aux_pos)
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                q, k = apply_rotary_pos_emb_3d(q, k, cos, sin)
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
            else:
                cos, sin = self.rope(q, seq_dim=1)
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)

        # 4. Standard Attention (Scaled Dot Product)
        # Transpose for attention calculation: [B, Heads, T, Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention calculation
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        # Apply causal mask if enabled
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(causal_mask, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v # [B, Heads, T, Dim]

        # Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# --- Simple Transformer Wrapper ---

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Note: No nn.PositionalEmbedding here! RoPE handles it.
        self.attn = CausalSelfAttention(d_model, n_head)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        x = self.token_embedding(idx)
        x = self.attn(x)
        return x

# Usage
model = SimpleTransformer(vocab_size=1000, d_model=64, n_head=4)
dummy_input = torch.randint(0, 1000, (1, 10)) # Batch 1, Seq 10
output = model(dummy_input)
print("Output shape:", output.shape)