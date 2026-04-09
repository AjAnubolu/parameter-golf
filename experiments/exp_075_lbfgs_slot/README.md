# exp_075_lbfgs_slot — L-BFGS Causal SLOT (READY, untested)

**Hypothesis**: SLOT in **logit space** with **L-BFGS** (25 iters, history=20) should work far better than our failed embedding-space AdamW SLOT (exp_072).

**Source**: [PR #1350](https://github.com/openai/parameter-golf/pull/1350) reports 1.0046 BPB using this approach.

## Why logit space + L-BFGS

- **Logit space** is nearly convex for the per-token CE loss — the loss
  surface is just shifting a softmax, so second-order optimization
  converges in few iterations.
- **L-BFGS with history=20** captures curvature via stored gradient
  pairs; far more powerful than 8 AdamW steps (what exp_072 tried).
- **Small delta**: `[vocab_size]` = 1024 floats = 4 KB, trivially small.
  Broadcast across all positions.
- **No model re-forward**: the delta is added to pre-computed logits,
  so each L-BFGS iteration is just a matrix-add + cross-entropy.

## Causality

For each sliding window:
1. **Score** the window with the current delta (carried over from previous window)
2. **Optimize** the delta on the focal region (last 128 tokens of this window)
3. Clamp delta to ±5
4. Carry delta to next window (warm-start)

This matches the score-before-adapt pattern of legal TTT. First window
uses delta=0. Only already-scored positions contribute to the optimization
objective.

## Running

```bash
LBFGS_SLOT_ENABLED=1 LBFGS_SLOT_ITERS=25 LBFGS_SLOT_HISTORY=20 \
  LBFGS_SLOT_FOCAL=128 LBFGS_SLOT_CLAMP=5.0 LBFGS_SLOT_LR=1.0 \
  GPTQ_ENABLED=1 GPTQ_N_BATCHES=64 TTT_ENABLED=0 EVAL_STRIDE=64 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Result

PR #1350 reports 1.0046 BPB on top of the full SOTA stack. On our
weaker baseline we should see a large improvement but not necessarily
reach 1.005. Targeting ~1.05-1.09 BPB.

## Legality Note

This is the "Causal SLOT" variant discussed in PR #1350's compliance
section. The key guarantee: the delta used to score position `t` only
depends on optimization over positions `< t`. This matches Issue #677's
four conditions for legal test-time adaptation.
