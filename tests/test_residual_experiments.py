import time
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------
# Config (fast on GPU)
# ----------------------------
@dataclass
class CFG:
    seed: int = 0
    vocab: int = 64
    d: int = 128
    heads: int = 4
    ff_mult: int = 4

    T: int = 64
    delay_k: int = 12
    batch: int = 256

    steps: int = 600       # per variant
    lr: float = 3e-4

    recursions: int = 6      # repeats same block R times (tiny recursive)
    dropout: float = 0.0

    log_every: int = 25
    eval_batches: int = 10

    alpha_min: float = 0.8   # identity decay floor

cfg = CFG()

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

torch.manual_seed(cfg.seed)
random.seed(cfg.seed)

# ----------------------------
# Toy data: Delayed Copy + Noise
# ----------------------------
def sample_batch(batch, T, vocab, k, device):
    x = torch.randint(1, vocab, (batch, T), device=device)
    # distractors
    mask = (torch.rand(batch, T, device=device) < 0.10)
    x = torch.where(mask, torch.randint(1, vocab, (batch, T), device=device), x)

    y = torch.zeros(batch, T, dtype=torch.long, device=device)
    y[:, k:] = x[:, :-k]
    return x, y

# ----------------------------
# Residual variants
# ----------------------------
def orthogonal_component(u, h, eps=1e-8):
    # u_perp = u - proj_h(u)
    dot = (u * h).sum(dim=-1, keepdim=True)                    # [B,T,1]
    denom = (h * h).sum(dim=-1, keepdim=True).clamp_min(eps)   # [B,T,1]
    return u - (dot / denom) * h

def alpha_schedule(r, R, alpha_min=0.2):
    # linear decay 1.0 -> alpha_min over recursion steps
    if R <= 1:
        return 1.0
    return 1.0 - (r / (R - 1)) * (1.0 - alpha_min)

# ----------------------------
# Model (tiny recursive transformer-ish block)
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.g

class TinyBlock(nn.Module):
    def __init__(self, d, heads, ff_mult, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.GELU(),
            nn.Linear(ff_mult * d, d),
        )
    def forward(self, h):
        B, T, D = h.shape
        causal = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        a, _ = self.attn(h, h, h, attn_mask=causal)
        return self.ff(h + a)  # returns update candidate F(h)

class RecursiveModel(nn.Module):
    def __init__(self, vocab, d, heads, ff_mult, R, T, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        self.block = TinyBlock(d, heads, ff_mult, dropout)
        self.norm = RMSNorm(d)
        self.out = nn.Linear(d, vocab)
        self.R = R

    def forward(self, x, variant="baseline", alpha_min=0.2):
        h = self.emb(x) + self.pos[:, :x.size(1), :]

        for r in range(self.R):
            u = self.block(h)  # F(h)

            if variant == "baseline":
                h = self.norm(h + u)

            elif variant == "decay":
                a = alpha_schedule(r, self.R, alpha_min=alpha_min)
                h = self.norm(a * h + u)

            elif variant == "orthogonal":
                u_perp = orthogonal_component(u, h)
                h = self.norm(h + u_perp)

            elif variant == "both":
                a = alpha_schedule(r, self.R, alpha_min=alpha_min)
                u_perp = orthogonal_component(u, h)
                h = self.norm(a * h + u_perp)

            else:
                raise ValueError("unknown variant")

        return self.out(h)

@torch.no_grad()
def eval_acc(model, variant):
    model.eval()
    correct, total = 0, 0
    for _ in range(cfg.eval_batches):
        x, y = sample_batch(cfg.batch, cfg.T, cfg.vocab, cfg.delay_k, device)
        logits = model(x, variant=variant, alpha_min=cfg.alpha_min)
        pred = logits.argmax(dim=-1)

        mask = torch.zeros_like(y).bool()
        mask[:, cfg.delay_k:] = True  # ignore first k positions
        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()

    model.train()
    return correct / max(1, total)

def train_variant(variant):
    model = RecursiveModel(cfg.vocab, cfg.d, cfg.heads, cfg.ff_mult, cfg.recursions, cfg.T, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # logs
    xs, loss_log, acc_log = [], [], []
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        x, y = sample_batch(cfg.batch, cfg.T, cfg.vocab, cfg.delay_k, device)
        logits = model(x, variant=variant, alpha_min=cfg.alpha_min)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            acc = eval_acc(model, variant)
            xs.append(step)
            loss_log.append(loss.item())
            acc_log.append(acc)
            print(f"{variant:10s} step {step:4d}/{cfg.steps}  loss {loss.item():.3f}  acc {acc*100:.1f}%")

    dt = time.time() - t0
    print(f"FINAL {variant:10s}  acc {acc_log[-1]*100:.1f}%  time {dt:.1f}s")
    return {"steps": xs, "loss": loss_log, "acc": acc_log}

# ----------------------------
# Run all variants
# ----------------------------
results = {}
for v in ["baseline", "decay", "orthogonal", "both"]:
    print("\n===", v, "===")
    results[v] = train_variant(v)

# ----------------------------
# Plots (matplotlib only; no seaborn; no explicit colors)
# ----------------------------
plt.figure()
for v in results:
    plt.plot(results[v]["steps"], results[v]["loss"], label=v)
plt.xlabel("training step")
plt.ylabel("train loss")
plt.title(f"Toy: loss vs step (device={device})")
plt.legend()
plt.show()

plt.figure()
for v in results:
    plt.plot(results[v]["steps"], [a * 100 for a in results[v]["acc"]], label=v)
plt.xlabel("training step")
plt.ylabel("test accuracy (%)")
plt.title(f"Toy: accuracy vs step (device={device})")
plt.legend()
plt.show()

print({v: round(results[v]["acc"][-1] * 100, 2) for v in results})
