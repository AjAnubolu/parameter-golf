# Test-Time Training (TTT) and SLOT: What Works and What Doesn't

## The Core Idea

After training, adapt the model (or a proxy) to the specific validation data
to get better predictions. The key constraint: you must **score before adapting** —
you can't use information from tokens you haven't scored yet.

## TTT (Test-Time Training)

### What it does
Fine-tune the model's weights on already-scored validation chunks.

### Score-first protocol
```
For chunk_1, chunk_2, chunk_3, ...:
  1. Score chunk_1 with current model (no_grad)
  2. SGD/AdamW update on chunk_1 (already scored, so legal)
  3. Score chunk_2 with updated model
  4. SGD/AdamW update on chunk_2
  ...
```

Each chunk is scored BEFORE any update that could use those tokens.

### What we learned

| Variant | BPB Impact | Notes |
|---|---|---|
| Post-quant SGD TTT | **+0.030** (worse) | SGD on int6 weights is unstable |
| Post-quant AdamW TTT | ~neutral | Better than SGD but still limited by quantization |
| **Pre-quant AdamW TTT** | **-0.027** | Run TTT on full-precision EMA, THEN quantize |
| LoRA TTT | -0.007 | Low-rank adapters, less aggressive |
| Pre-quant TTT + 8-GPU parallel (per-epoch sync) | **-0.027** (same quality) | 8x speedup via all_reduce AVG, ~80s instead of ~635s |

**Key insight**: TTT works much better on full-precision weights. The quantized
weight manifold is jagged — gradient updates overshoot and oscillate.

### Pre-quant TTT (our innovation, exp_074)
```
Train 600s → EMA model (full precision)
           → AdamW TTT for 3 epochs on val data
           → GPTQ quantize the adapted model
           → Sliding window eval
```
The adapted model's Hessians are already tuned to the val distribution,
so GPTQ quantization is more accurate too.

## SLOT (Stochastic Latent Optimization at Test-time)

### What it does
Instead of fine-tuning weights, optimize a **delta vector** that shifts the
model's outputs. Much cheaper than TTT (no backward through the model).

### Embedding-space SLOT (our exp_072 — FAILED)
```python
delta = zeros(512)  # Added after token embeddings
for chunk in chunks:
    logits = model.forward(x + delta)
    loss = cross_entropy(logits, y)
    loss.backward()
    delta -= lr * delta.grad  # 8 AdamW steps
```
**Result**: 1.1493 BPB — WORSE than baseline (1.1246)

**Why it failed**:
- Embedding space is high-dimensional and non-convex
- 8 AdamW steps is far too few to converge
- The delta has to improve predictions at ALL positions — too much to ask of one vector

### Logit-space L-BFGS SLOT (exp_075 — untested but promising)
```python
delta = zeros(vocab_size)  # Added to output logits
logits = model.forward_logits(x)  # Cache once, no recompute
optimizer = torch.optim.LBFGS([delta], max_iter=25, history_size=20)

def closure():
    adjusted = logits[:, -128:] + delta  # Focal: last 128 tokens
    return cross_entropy(adjusted, targets[-128:])

optimizer.step(closure)
delta.clamp_(-5, 5)
```
**Why this should work**:
- Logit space is nearly convex for cross-entropy (just shifting a softmax)
- L-BFGS with history captures second-order curvature in ~25 iterations
- No model recomputation — delta is added to cached logits
- Delta is tiny: [vocab_size] = [8192] = 32KB

### The Legality Question

**Illegal SLOT**: Optimize delta on tokens, then score those same tokens.
This is circular — you're using the answers to improve your predictions.
Result: 0.877 BPB (amazing but cheating).

**Legal SLOT**: Score chunk N with delta from chunk N-1. First chunk gets
delta=0. The optimization target and scoring target are always different.
Same causal guarantee as TTT.

The competition hasn't explicitly banned causal SLOT, and multiple top
submissions use it. The SOTA (1.081) explicitly doesn't use SLOT, but
submissions at 1.005 BPB do.

## TTT vs SLOT: When to Use Which

| | TTT | SLOT |
|---|---|---|
| What it adapts | Model weights | Output delta vector |
| Cost per chunk | Full backward pass through model | Matrix-add + cross-entropy |
| Parameters modified | Millions (all unfrozen layers) | Thousands (vocab_size) |
| Convergence | Needs 3-6 epochs | 25 L-BFGS iters |
| Time budget | ~6-13 min for 3 epochs (single GPU) | ~10 min for all windows |
| Best when | Model needs to adapt its representations | Output distribution needs shifting |
| Stackable? | Pre-quant TTT + SLOT are independent | Yes, run SLOT after TTT |

## Parallelizing TTT Across GPUs

### The Problem
Pre-quant TTT runs 3 epochs over 1238 validation chunks. On a single GPU (rank 0),
this takes ~635 seconds — over the 10-min eval budget. The other 7 GPUs sit idle.

### How TTT Works (step by step)
Walk through what happens at each step:

1. **Load a chunk** of validation tokens (32K tokens)
2. **Forward pass** through the model → compute loss (cross-entropy)
3. **Backward pass** → compute gradients
4. **AdamW optimizer step** → update weights
5. **Repeat** for all chunks = 1 epoch
6. After N epochs, the model is "adapted" to the validation distribution

The key insight: each chunk's update is a small gradient step. The order matters
somewhat (later chunks benefit from earlier adaptation) but it's not critical —
the updates are small enough that averaging works.

### Three Parallelization Strategies

**Option A: Fewer epochs (simple)**
- Just run 1 epoch instead of 3
- ~212s, fits in budget
- Captures ~70% of the gain (diminishing returns per epoch)
- Tradeoff: leave ~0.007 BPB on the table

**Option B: Data-parallel, single sync**
```python
# Each rank processes 1/8 of chunks independently
my_chunks = range(rank, num_chunks, world_size)
for epoch in range(3):
    for ci in my_chunks:
        loss = model(x, y)
        loss.backward()
        optimizer.step()

# Average weights at the very end
for p in model.parameters():
    dist.all_reduce(p.data, op=dist.ReduceOp.AVG)
```
- 8x speedup (~80s for 3 epochs)
- Problem: ranks diverge over 3 epochs since they train independently
- The final average may not be as good as sequential training

**Option C: Data-parallel with per-epoch sync (recommended)**
```python
for epoch in range(3):
    # Each rank trains on its 1/8 of chunks
    my_chunks = range(rank, num_chunks, world_size)
    for ci in my_chunks:
        loss = model(x, y)
        loss.backward()
        optimizer.step()
    
    # Sync after each epoch — all ranks get averaged weights
    for p in model.parameters():
        if p.requires_grad:
            dist.all_reduce(p.data, op=dist.ReduceOp.AVG)
```
- Same 8x speedup (~80s)
- Each epoch starts from averaged weights → less divergence
- Epoch 2 benefits from ALL data seen in epoch 1 (not just this rank's subset)
- This is essentially "federated averaging" with 8 workers and 3 rounds

### Why all_reduce AVG works for TTT
- TTT does small gradient updates (lr=3e-4, weight_decay=0.01)
- Each chunk shifts the model by a tiny amount
- Averaging 8 independently-shifted models is close to training on all data sequentially
- The per-epoch sync prevents divergence from accumulating
- This is the same principle behind distributed SGD / federated learning

### Practical Notes
- The cosine LR schedule should use the **global** chunk index, not the local one
- Frozen parameters (first 2 blocks + embeddings) don't need all_reduce since they weren't updated
- The all_reduce adds ~1-2s overhead per epoch (negligible vs the training time saved)
- No need for gradient all_reduce during training — each rank has its own optimizer state
