"""
Step 4: Curriculum Learning

Phase 1: Train MLP to be a perfect full adder (using hardcoded correct pairs)
Phase 2: Freeze MLP, train only the Transformer to learn reordering

Hypothesis: With a frozen, perfect MLP, the Transformer gets a clean learning signal
and should learn a more generalizable reordering strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# ============== Vocabulary ==============
TOK = {
    "0": 0,
    "1": 1,
    "+": 2,
    ">": 3,
    "#": 4,
}
TOK_INV = {v: k for k, v in TOK.items()}

# ============== Data Generation ==============

def generate_paired_batch_for_mlp(batch_size, num_bits, device='cpu'):
    """Generate correctly paired data for MLP pretraining."""
    max_val = (2 ** num_bits) - 1
    
    paired_inputs = []
    sum_targets = []
    
    for _ in range(batch_size):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        c = a + b
        
        a_bits = [(a >> i) & 1 for i in range(num_bits)]
        b_bits = [(b >> i) & 1 for i in range(num_bits)]
        c_bits = [(c >> i) & 1 for i in range(num_bits + 1)]
        
        pairs = list(zip(a_bits, b_bits))
        paired_inputs.append(pairs)
        sum_targets.append(c_bits)
    
    paired_inputs = torch.tensor(paired_inputs, dtype=torch.float32, device=device)
    sum_targets = torch.tensor(sum_targets, dtype=torch.float32, device=device)
    
    return paired_inputs, sum_targets


def generate_mixed_length_batch(batch_size, min_bits, max_bits, max_bits_arch, device='cpu'):
    """Generate batch with mixed lengths for transformer training."""
    max_seq_len = max_bits_arch + 1 + max_bits_arch + 1
    
    input_tokens = []
    sum_targets = []
    bit_lengths = []
    paired_targets = []
    
    for _ in range(batch_size):
        num_bits = random.randint(min_bits, max_bits)
        bit_lengths.append(num_bits)
        
        max_val = (2 ** num_bits) - 1
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        c = a + b
        
        a_str = format(a, f'0{num_bits}b')
        b_str = format(b, f'0{num_bits}b')
        
        tokens = [TOK[ch] for ch in a_str] + [TOK["+"]] + [TOK[ch] for ch in b_str] + [TOK[">"]]
        while len(tokens) < max_seq_len:
            tokens.append(TOK["#"])
        input_tokens.append(tokens)
        
        c_bits = [(c >> i) & 1 for i in range(num_bits + 1)]
        while len(c_bits) < max_bits_arch + 1:
            c_bits.append(0)
        sum_targets.append(c_bits)
        
        a_bits_lsb = [int(a_str[-(i+1)]) for i in range(num_bits)]
        b_bits_lsb = [int(b_str[-(i+1)]) for i in range(num_bits)]
        pairs = list(zip(a_bits_lsb, b_bits_lsb))
        while len(pairs) < max_bits_arch:
            pairs.append((0, 0))
        paired_targets.append(pairs)
    
    input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=device)
    sum_targets = torch.tensor(sum_targets, dtype=torch.float32, device=device)
    bit_lengths = torch.tensor(bit_lengths, dtype=torch.long, device=device)
    paired_targets = torch.tensor(paired_targets, dtype=torch.float32, device=device)
    
    masks = torch.zeros(batch_size, max_bits_arch + 1, device=device)
    for i, length in enumerate(bit_lengths):
        masks[i, :length + 1] = 1
    
    pair_masks = torch.zeros(batch_size, max_bits_arch, device=device)
    for i, length in enumerate(bit_lengths):
        pair_masks[i, :length] = 1
    
    return input_tokens, sum_targets, paired_targets, bit_lengths, masks, pair_masks


def generate_fixed_length_batch(batch_size, num_bits, max_bits_arch, device='cpu'):
    """Generate batch with fixed length for evaluation."""
    max_seq_len = max_bits_arch + 1 + max_bits_arch + 1
    
    input_tokens = []
    sum_targets = []
    paired_targets = []
    
    max_val = (2 ** num_bits) - 1
    
    for _ in range(batch_size):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        c = a + b
        
        a_str = format(a, f'0{num_bits}b')
        b_str = format(b, f'0{num_bits}b')
        
        tokens = [TOK[ch] for ch in a_str] + [TOK["+"]] + [TOK[ch] for ch in b_str] + [TOK[">"]]
        while len(tokens) < max_seq_len:
            tokens.append(TOK["#"])
        input_tokens.append(tokens)
        
        c_bits = [(c >> i) & 1 for i in range(num_bits + 1)]
        while len(c_bits) < max_bits_arch + 1:
            c_bits.append(0)
        sum_targets.append(c_bits)
        
        a_bits_lsb = [int(a_str[-(i+1)]) for i in range(num_bits)]
        b_bits_lsb = [int(b_str[-(i+1)]) for i in range(num_bits)]
        pairs = list(zip(a_bits_lsb, b_bits_lsb))
        while len(pairs) < max_bits_arch:
            pairs.append((0, 0))
        paired_targets.append(pairs)
    
    input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=device)
    sum_targets = torch.tensor(sum_targets, dtype=torch.float32, device=device)
    paired_targets = torch.tensor(paired_targets, dtype=torch.float32, device=device)
    
    masks = torch.zeros(batch_size, max_bits_arch + 1, device=device)
    masks[:, :num_bits + 1] = 1
    
    pair_masks = torch.zeros(batch_size, max_bits_arch, device=device)
    pair_masks[:, :num_bits] = 1
    
    return input_tokens, sum_targets, paired_targets, masks, pair_masks


# ============== Model Components ==============

class AdditionMLP(nn.Module):
    """MLP that learns to be a full adder."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
    
    def forward(self, a_bit, b_bit, carry_in):
        x = torch.stack([a_bit, b_bit, carry_in], dim=-1)
        out = self.net(x)
        return out[..., 0], out[..., 1]
    
    def forward_loop(self, pairs, num_bits):
        """Run the full adder loop over pairs."""
        B = pairs.shape[0]
        device = pairs.device
        carry = torch.zeros(B, device=device)
        sum_bits = []
        
        for i in range(num_bits):
            sum_bit, carry = self.forward(pairs[:, i, 0], pairs[:, i, 1], carry)
            sum_bits.append(sum_bit)
        sum_bits.append(carry)
        
        return torch.stack(sum_bits, dim=1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        if mask is not None:
            attn = attn.masked_fill(~mask.bool().unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class ReorderingTransformer(nn.Module):
    """Transformer that outputs reordered bit pairs."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_pairs, dropout=0.1):
        super().__init__()
        self.max_pairs = max_pairs
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output_queries = nn.Parameter(torch.randn(max_pairs * 2, d_model) * 0.02)
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, tokens, input_mask=None):
        B = tokens.shape[0]
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, input_mask)
        x = self.ln_f(x)
        
        queries = self.output_queries.unsqueeze(0).expand(B, -1, -1)
        attn = torch.bmm(queries, x.transpose(1, 2))
        if input_mask is not None:
            attn = attn.masked_fill(~input_mask.bool().unsqueeze(1), float('-inf'))
        attn = F.softmax(attn / math.sqrt(self.d_model), dim=-1)
        selected = torch.bmm(attn, x)
        bits = torch.sigmoid(self.output_proj(selected).squeeze(-1))
        return bits.reshape(B, self.max_pairs, 2), attn


class FullModel(nn.Module):
    """Combined model with separate MLP and Transformer."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_bits, mlp_hidden=64, dropout=0.1):
        super().__init__()
        self.max_bits = max_bits
        self.transformer = ReorderingTransformer(
            vocab_size, d_model, n_heads, n_layers, max_bits, dropout
        )
        self.mlp = AdditionMLP(hidden_dim=mlp_hidden)
        
    def forward(self, tokens, input_mask=None, num_bits=None):
        pairs, attn = self.transformer(tokens, input_mask)
        num_bits = num_bits or self.max_bits
        
        B, device = tokens.shape[0], tokens.device
        carry = torch.zeros(B, device=device)
        sum_bits = []
        
        for i in range(num_bits):
            sum_bit, carry = self.mlp(pairs[:, i, 0], pairs[:, i, 1], carry)
            sum_bits.append(sum_bit)
        sum_bits.append(carry)
        
        while len(sum_bits) < self.max_bits + 1:
            sum_bits.append(torch.zeros(B, device=device))
        
        return torch.stack(sum_bits, dim=1), pairs, attn


# ============== Training ==============

def verify_mlp_truth_table(mlp, device):
    """Verify the MLP learned the correct full adder."""
    truth_table = {
        (0, 0, 0): (0, 0), (0, 0, 1): (1, 0), (0, 1, 0): (1, 0), (0, 1, 1): (0, 1),
        (1, 0, 0): (1, 0), (1, 0, 1): (0, 1), (1, 1, 0): (0, 1), (1, 1, 1): (1, 1),
    }
    
    mlp.eval()
    all_correct = True
    
    with torch.no_grad():
        for (a, b, c), (exp_s, exp_c) in truth_table.items():
            pred_s, pred_c = mlp(
                torch.tensor([float(a)], device=device),
                torch.tensor([float(b)], device=device),
                torch.tensor([float(c)], device=device)
            )
            pred_s_bit = int(pred_s.item() > 0.5)
            pred_c_bit = int(pred_c.item() > 0.5)
            if pred_s_bit != exp_s or pred_c_bit != exp_c:
                all_correct = False
    
    return all_correct


def train_phase1_mlp(device, num_steps=5000):
    """Phase 1: Train MLP to be a perfect full adder."""
    print("=" * 70)
    print("PHASE 1: Training MLP to be a perfect full adder")
    print("=" * 70)
    
    mlp = AdditionMLP(hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    for step in range(num_steps):
        # Train on various bit lengths
        num_bits = random.randint(2, 8)
        pairs, targets = generate_paired_batch_for_mlp(128, num_bits, device)
        
        sum_pred = mlp.forward_loop(pairs, num_bits)
        
        # Pad targets if needed
        if targets.shape[1] < sum_pred.shape[1]:
            pad = torch.zeros(targets.shape[0], sum_pred.shape[1] - targets.shape[1], device=device)
            targets = torch.cat([targets, pad], dim=1)
        
        loss = F.binary_cross_entropy(sum_pred[:, :num_bits+1], targets[:, :num_bits+1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            pred_bits = (sum_pred[:, :num_bits+1] > 0.5).float()
            acc = (pred_bits == targets[:, :num_bits+1]).all(dim=1).float().mean().item()
            print(f"  Step {step:5d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")
    
    # Verify
    is_correct = verify_mlp_truth_table(mlp, device)
    print(f"\n  MLP Truth Table: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    # Test on various lengths
    print("\n  MLP Generalization:")
    mlp.eval()
    with torch.no_grad():
        for test_bits in [4, 8, 16, 32]:
            pairs, targets = generate_paired_batch_for_mlp(200, test_bits, device)
            sum_pred = mlp.forward_loop(pairs, test_bits)
            pred_bits = (sum_pred[:, :test_bits+1] > 0.5).float()
            acc = (pred_bits == targets).all(dim=1).float().mean().item()
            print(f"    {test_bits:2d}-bit: {acc:.2%}")
    
    return mlp


def train_phase2_transformer(mlp, device, num_steps=20000):
    """Phase 2: Freeze MLP, train Transformer to learn reordering."""
    print("\n" + "=" * 70)
    print("PHASE 2: Training Transformer (MLP frozen)")
    print("=" * 70)
    
    # Config
    min_bits_train, max_bits_train = 2, 6
    max_bits_arch = 16
    d_model, n_heads, n_layers = 128, 4, 4
    
    # Create full model and copy pretrained MLP
    model = FullModel(
        len(TOK), d_model, n_heads, n_layers, max_bits_arch, 
        mlp_hidden=64, dropout=0.1
    ).to(device)
    
    # Copy MLP weights and FREEZE
    model.mlp.load_state_dict(mlp.state_dict())
    for param in model.mlp.parameters():
        param.requires_grad = False
    
    # Only optimize transformer
    optimizer = torch.optim.Adam(model.transformer.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params: {frozen_params:,}")
    print("-" * 70)
    
    steps_log, loss_log = [], []
    acc_history = {b: [] for b in range(min_bits_train, max_bits_train + 1)}
    
    for step in range(num_steps):
        model.train()
        
        tokens, sum_targets, paired_targets, bit_lengths, sum_masks, pair_masks = \
            generate_mixed_length_batch(128, min_bits_train, max_bits_train, max_bits_arch, device)
        
        input_mask = (tokens != TOK["#"]).float()
        sum_pred, pairs_pred, _ = model(tokens, input_mask)
        
        # Only sum loss (MLP is frozen, so pair supervision not needed)
        sum_loss = (F.binary_cross_entropy(sum_pred, sum_targets, reduction='none') * sum_masks).sum() / sum_masks.sum()
        
        # Optional: pair supervision to guide learning (even with frozen MLP)
        pair_loss = (F.binary_cross_entropy(pairs_pred, paired_targets, reduction='none') * pair_masks.unsqueeze(-1)).sum() / (pair_masks.sum() * 2 + 1e-8)
        
        loss = sum_loss + 0.5 * pair_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 500 == 0 or step == num_steps - 1:
            model.eval()
            steps_log.append(step)
            loss_log.append(loss.item())
            
            accs = []
            with torch.no_grad():
                for test_bits in range(min_bits_train, max_bits_train + 1):
                    tokens_eval, sum_targets_eval, _, sum_masks_eval, _ = \
                        generate_fixed_length_batch(100, test_bits, max_bits_arch, device)
                    input_mask_eval = (tokens_eval != TOK["#"]).float()
                    sum_pred_eval, _, _ = model(tokens_eval, input_mask_eval, num_bits=test_bits)
                    
                    pred_bits = (sum_pred_eval > 0.5).float()
                    correct = ((pred_bits == sum_targets_eval) | (sum_masks_eval == 0)).all(dim=1)
                    acc = correct.float().mean().item()
                    accs.append(acc)
                    acc_history[test_bits].append(acc)
            
            acc_str = " | ".join([f"{b}b:{a:.0%}" for b, a in zip(range(min_bits_train, max_bits_train+1), accs)])
            print(f"  Step {step:5d} | Loss: {loss.item():.4f} | {acc_str}")
    
    return model, steps_log, loss_log, acc_history, min_bits_train, max_bits_train, max_bits_arch


def evaluate_and_plot(model, steps_log, loss_log, acc_history, min_bits_train, max_bits_train, max_bits_arch, device):
    """Final evaluation and plotting."""
    print("\n" + "=" * 70)
    print("LENGTH GENERALIZATION TEST")
    print("=" * 70)
    
    model.eval()
    test_lengths = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
    results = []
    
    with torch.no_grad():
        for test_bits in test_lengths:
            tokens, sum_targets, _, sum_masks, _ = \
                generate_fixed_length_batch(500, test_bits, max_bits_arch, device)
            input_mask = (tokens != TOK["#"]).float()
            sum_pred, _, _ = model(tokens, input_mask, num_bits=test_bits)
            
            pred_bits = (sum_pred > 0.5).float()
            correct = ((pred_bits == sum_targets) | (sum_masks == 0)).all(dim=1)
            acc = correct.float().mean().item()
            results.append((test_bits, acc))
            
            marker = "←(train)" if min_bits_train <= test_bits <= max_bits_train else "(OOD)"
            print(f"  {test_bits:2d}-bit: {acc:6.2%} {marker}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(steps_log, loss_log, 'b-', lw=1.5)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Phase 2: Training Loss (Transformer only)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training accuracy
    colors = plt.cm.viridis(np.linspace(0, 1, max_bits_train - min_bits_train + 1))
    for i, b in enumerate(range(min_bits_train, max_bits_train + 1)):
        axes[0, 1].plot(steps_log, acc_history[b], color=colors[i], label=f'{b}-bit', lw=1.5)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy by Length')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Generalization
    bits, accs = zip(*results)
    bar_colors = ['green' if min_bits_train <= b <= max_bits_train else 'steelblue' for b in bits]
    bars = axes[1, 0].bar(range(len(bits)), accs, color=bar_colors, edgecolor='black')
    axes[1, 0].set_xticks(range(len(bits)))
    axes[1, 0].set_xticklabels([str(b) for b in bits])
    axes[1, 0].set_xlabel('Number of Bits')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title(f'Length Generalization (trained {min_bits_train}-{max_bits_train}, MLP frozen)')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].axhline(1.0, color='gray', ls='--', alpha=0.5)
    for bar, acc in zip(bars, accs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{acc:.0%}', ha='center', fontsize=9)
    
    # Attention
    with torch.no_grad():
        tokens, _, _, _, _ = generate_fixed_length_batch(1, 6, max_bits_arch, device)
        input_mask = (tokens != TOK["#"]).float()
        _, _, attn = model(tokens, input_mask)
        
        seq_len = int(input_mask[0].sum().item())
        attn_np = attn[0, :12, :seq_len].cpu().numpy()
        
        input_labels = [TOK_INV[t] for t in tokens[0, :seq_len].tolist()]
        output_labels = [f'a{i}' if j==0 else f'b{i}' for i in range(6) for j in range(2)]
        
        im = axes[1, 1].imshow(attn_np, cmap='Blues', aspect='auto')
        axes[1, 1].set_xticks(range(len(input_labels)))
        axes[1, 1].set_xticklabels(input_labels)
        axes[1, 1].set_yticks(range(len(output_labels)))
        axes[1, 1].set_yticklabels(output_labels)
        axes[1, 1].set_xlabel('Input Position')
        axes[1, 1].set_ylabel('Output Query')
        axes[1, 1].set_title('Attention Pattern (6-bit example)')
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.suptitle('Step 4: Curriculum Learning (Pretrained MLP + Transformer)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('step4_curriculum.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to 'step4_curriculum.png'")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Phase 1: Train MLP
    mlp = train_phase1_mlp(device, num_steps=5000)
    
    # Phase 2: Train Transformer with frozen MLP
    model, steps_log, loss_log, acc_history, min_bits, max_bits, max_bits_arch = \
        train_phase2_transformer(mlp, device, num_steps=20000)
    
    # Evaluate
    evaluate_and_plot(model, steps_log, loss_log, acc_history, min_bits, max_bits, max_bits_arch, device)


if __name__ == "__main__":
    main()