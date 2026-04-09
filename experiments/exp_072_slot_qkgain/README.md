# exp_072_slot_qkgain — Score-First SLOT (FAILED, 1.1493 BPB)

**Result**: 1.1493 BPB — **worse** than sliding window baseline (1.1246) and naive eval (1.1479).

## Hypothesis

SLOT (Stochastic Latent Optimization at Test-time) with a **score-first** ordering should be legal and should improve eval BPB:

1. Score chunk N with delta from chunk N-1 (first chunk: delta=0)
2. Then optimize delta on chunk N (for use on chunk N+1)

This follows the same causal information flow as legal TTT.

## Implementation

- Delta in **embedding space**: 512-dim vector added after token embedding + smear
- 8 AdamW steps per chunk, LR 0.01
- Non-overlapping chunks (~59 batches, ~83 s eval)
- QK-Gain bumped from 1.5 → 4.0 (free −0.006 BPB from PR #1125)

## Why it failed

Embedding-space deltas optimized with AdamW for only 8 steps underfit.
The top SLOT submissions ([PR #1350](https://github.com/openai/parameter-golf/pull/1350))
instead optimize a **logit-space** delta with **L-BFGS (25 iters, history=20)**,
getting ~1.005 BPB — a fundamentally different approach.

See `exp_075_lbfgs_slot/` for the L-BFGS logit-space version.

## Running

```bash
GPTQ_ENABLED=1 GPTQ_N_BATCHES=64 SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.01 \
  SLOT_BATCH_SEQS=64 TTT_ENABLED=0 EVAL_STRIDE=64 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
