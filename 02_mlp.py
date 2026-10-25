"""MLP language model (Bengio-style): char embeddings -> hidden layers with BatchNorm + tanh.
Includes train/dev/test split, sane weight init, and an activation/gradient health check."""
import torch
import matplotlib.pyplot as plt
from data import vocab_size, split_data
from layers import Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, Sequential
from train_utils import train, evaluate, sample

g = torch.Generator().manual_seed(2147483647)
block_size, n_embd, n_hidden = 3, 10, 100
train_set, dev_set, test_set = split_data(block_size)

model = Sequential([
    Embedding(vocab_size, n_embd, g), FlattenConsecutive(block_size),
    Linear(n_embd * block_size, n_hidden, bias=False, gen=g), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False, gen=g), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, gen=g),
])
with torch.no_grad():
    model.layers[-1].weight *= 0.1          # make the last layer less confident at init
print("parameters:", sum(p.numel() for p in model.parameters()))

train(model, train_set, gen=g)
evaluate(model, {"train": train_set, "dev": dev_set, "test": test_set})

# ---- diagnostics: are any tanh neurons saturated (dead)? ----
plt.figure(figsize=(10, 4))
for i, l in enumerate(model.layers[:-1]):
    if isinstance(l, Tanh):
        t = l.out
        sat = (t.abs() > 0.97).float().mean() * 100
        h, edges = torch.histogram(t, density=True, bins=20)
        plt.plot(edges[:-1].detach(), h.detach(), label=f"layer {i} (saturated {sat:.1f}%)")
plt.legend(); plt.title("tanh activation distribution"); plt.savefig("activations.png", dpi=120)
print("saved activations.png")

print("\nsamples:")
sample(model, block_size, gen=g)
