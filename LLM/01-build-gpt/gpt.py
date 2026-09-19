import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
eval_interval = 500  # how often to estimate loss
eval_iters = 200  # batches per split each time we do
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  # Apple Silicon GPU
    if torch.backends.mps.is_available()
    else "cpu"
)
# bf16 autocast: matmuls run in bf16, weights stay fp32. Same exponent range as
# fp32, so unlike fp16 it needs no GradScaler.
amp_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu")
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# vocab
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # eval phase
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with amp_ctx:
                _logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # back to training phase
    return out


class Head(nn.Module):
    """on head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _B, T, C = x.shape
        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        # compute attention score, aka affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, H) @ (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # slicing is to broadcast a shorter sequence against the full (block_size, block_size) mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of attention in parallel"""

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])  # each one learning in parallel and independently
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate over channel dim. Each head is (B, T, head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, E)

        out = self.dropout(self.proj(out))  # (B, T, E)
        return out


class FeedForward(nn.Module):
    """a simple MLPlinear layer with non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transfomer block: communcation followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # keeps the channel size consistent
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        # Layer normalization (mean 0, std 1) of the rows (every example is normalized) across C. Per-token transformation.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # adding a residual path
        x = x + self.ffwd(self.ln2(x))  # adding a residual path
        return x


# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) auto broadcasting to the batch size of the pos_emb
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to have the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _loss = self(idx_cond)
            # focus on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel().to(device)
# compile fuses the many small per-head ops into fewer GPU kernels; the first
# step (and the first eval, which recompiles for eval mode) is slow while it traces
m = torch.compile(model)

# fused: on MPS the default AdamW is a Python loop over each parameter tensor;
# fused does the whole update in GPU kernels
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch data
    xb, yb = get_batch("train")

    # forward pass
    with amp_ctx:
        logits, loss = m(xb, yb)

    # backward pass and optimization step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with a '0' token, which is a '\n'

# generate from the uncompiled model: T grows 1..block_size, so the compiled
# one would recompile for each new sequence length
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
