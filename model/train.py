import os
import torch
from transformer import TinyTransformer

# ---------------- CONFIG ----------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

batch_size = 16
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4

n_layers = 4
n_heads = 4
n_embed = 128
# ----------------------------------------


# ---------------- DATA ----------------
with open("../data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_src = train_data if split == "train" else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
# ----------------------------------------


# ---------------- MODEL ----------------
model = TinyTransformer(
    vocab_size=vocab_size,
    n_embed=n_embed,
    n_heads=n_heads,
    n_layers=n_layers,
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# ----------------------------------------


# ---------------- TRAIN ----------------
for step in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step} | loss {loss.item():.4f}")

os.makedirs("outputs/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "outputs/checkpoints/tiny_react_model.pt")
print("Training complete, model saved. let's goooo")
