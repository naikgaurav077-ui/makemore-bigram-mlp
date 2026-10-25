"""Bigram character model two ways: (1) counting, (2) a single-layer neural net.
Both optimise the same thing: maximum likelihood = minimise average negative log-likelihood (NLL)."""
import torch
import torch.nn.functional as F
from data import words, stoi, itos, vocab_size

g = torch.Generator().manual_seed(2147483647)

# ---------- 1. counting ----------
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for w in words:
    cs = ["."] + list(w) + ["."]
    for a, b in zip(cs, cs[1:]):
        N[stoi[a], stoi[b]] += 1

P = (N + 1).float()                       # +1 = model smoothing (avoids zero probabilities)
P /= P.sum(1, keepdim=True)

log_likelihood, n = 0.0, 0
for w in words:
    cs = ["."] + list(w) + ["."]
    for a, b in zip(cs, cs[1:]):
        log_likelihood += torch.log(P[stoi[a], stoi[b]])
        n += 1
print(f"[counting] average NLL: {-log_likelihood / n:.4f}")

# ---------- 2. neural net (one linear layer + softmax) ----------
xs, ys = [], []
for w in words:
    cs = ["."] + list(w) + ["."]
    for a, b in zip(cs, cs[1:]):
        xs.append(stoi[a]); ys.append(stoi[b])
xs, ys = torch.tensor(xs), torch.tensor(ys)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

for k in range(200):
    xenc = F.one_hot(xs, num_classes=vocab_size).float()
    logits = xenc @ W                                   # log-counts
    probs = logits.exp() / logits.exp().sum(1, keepdim=True)   # softmax
    loss = -probs[torch.arange(len(ys)), ys].log().mean() + 0.01 * (W ** 2).mean()
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
    if k % 50 == 0 or k == 199:
        print(f"[neural]   step {k:3d} loss {loss.item():.4f}")

# ---------- sampling ----------
print("\nsampled names:")
for _ in range(10):
    out, ix = [], 0
    while True:
        ix = torch.multinomial(P[ix], 1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("".join(out))
