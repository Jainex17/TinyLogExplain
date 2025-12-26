import torch
from transformer import TinyTransformer

# -------- CONFIG --------
device = "mps" if torch.backends.mps.is_available() else "cpu"

block_size = 256
n_layers = 4
n_heads = 4
n_embed = 128
# -----------------------


# -------- LOAD DATA --------
with open("../data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join([itos[i] for i in l])
# ---------------------------


# -------- LOAD MODEL --------
model = TinyTransformer(
    vocab_size=vocab_size,
    n_embed=n_embed,
    n_heads=n_heads,
    n_layers=n_layers,
    block_size=block_size
).to(device)

model.load_state_dict(
    torch.load("outputs/checkpoints/tiny_react_model.pt", map_location=device)
)
model.eval()
# ---------------------------


# -------- GENERATE --------
def generate(idx, max_new_tokens, temperature=0.7):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)
    return idx


context = torch.zeros((1, 1), dtype=torch.long).to(device)
out = generate(context, max_new_tokens=500)

print(decode(out[0].tolist()))
