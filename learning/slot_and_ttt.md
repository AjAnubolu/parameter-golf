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
| Time budget | ~6-13 min for 3 epochs | ~10 min for all windows |
| Best when | Model needs to adapt its representations | Output distribution needs shifting |
| Stackable? | Pre-quant TTT + SLOT are independent | Yes, run SLOT after TTT |
