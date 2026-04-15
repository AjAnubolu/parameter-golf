# exp_031_submission — Validated SOTA (1.1191 BPB)

**Result**: 1.1191 BPB (3-seed mean) on 8×H100 SXM, 600s training.

## Stack

- 11L × 512d, 8 heads, 4 KV heads, MLP 3×
- **Full Hessian GPTQ int6** with Cholesky error compensation
  - Column-sequential quantization
  - Hessians collected from 64 calibration batches
  - 5-percentile clip search for per-row scales
- **SGD TTT** (test-time training): 3 epochs, 32K chunks, freeze first 2 blocks
- LeakyReLU(0.5)² activation
- XSA on last 4 layers
- BigramHash (1536 buckets × 128d)
- EMA(0.997) + SWA(every 50)
- Parallel Muon optimizer (async reduce-scatter → Newton-Schulz → all-gather)
- Late QAT with STE (threshold 0.15)
- lzma compression (preset=9)

## Running

```bash
GPTQ_ENABLED=1 GPTQ_N_BATCHES=64 TTT_ENABLED=1 EVAL_STRIDE=64 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires FA3 (Hopper kernels) for ~86 ms/step on H100 SXM.

## Artifact

15.96–15.99 MB across seeds (well under 16 MB limit).

## Provenance

Built from the 2026-03-23 LeakyReLU + Legal TTT + Parallel Muon baseline,
adding Full Hessian GPTQ on top. This is the script we'd submit as a
record PR if we stopped iterating today.
