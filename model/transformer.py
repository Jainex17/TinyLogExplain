import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Self-Attention ----------
class SelfAttention(nn.Module):
    def __init__(self, n_embed, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = n_embed // n_heads

        self.key   = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)

        self.proj = nn.Linear(n_embed, n_embed)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

# ---------- Feed Forward ----------
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)

# ---------- Transformer Block ----------
class Block(nn.Module):
    def __init__(self, n_embed, n_heads, block_size):
        super().__init__()
        self.sa = SelfAttention(n_embed, n_heads, block_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ---------- Language Model ----------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb   = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads, block_size) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
