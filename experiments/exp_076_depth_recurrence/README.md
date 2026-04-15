# exp_076_depth_recurrence — MLP-only Depth Recurrence (READY, untested)

**Hypothesis**: Sharing MLP weights across a subset of layers gives more
effective depth for the same parameter budget. **But only MLP weights**,
not attention — attention sharing was our previous failure mode.

**Source**: [PR #1334](https://github.com/openai/parameter-golf/pull/1334)
and [PR #1344](https://github.com/openai/parameter-golf/pull/1344) report
1.089-1.092 BPB with this technique.

## Architecture Change

Physical layers: 11. Unique MLP weight sets: 9.

```
Layer:         0  1  2  3  4  5  6  7  8  9  10
MLP bank idx:  0  1  2  3  3  3  4  5  6  7  8
               └──unique──┘ └shared┘ └──unique──┘
```

Layers 3, 4, 5 all use the same MLP up/down weights. Attention weights
(`qo_bank`, `kv_bank`), norms, scales, and residual gates remain fully
unique per layer.

## Why this works (now)

Previous attempt failed because we shared **everything**, and quantization
error amplified 900× over 3 recurrence cycles. The fix:

1. **Share only MLP, keep attention unique** — attention provides the
   per-layer "discrimination" that prevents error amplification
2. **Higher weight decay** (0.09 vs 0.04) — regularizes the shared MLP
   against overfitting to any one layer's statistics
3. **GPTQ quantization** — Full Hessian GPTQ handles shared weights
   correctly (accumulates Hessians from all users of the weight)

## Parameter Savings

2 × (up_dim × model_dim + model_dim × up_dim) × 4 bytes
= 2 × (1536 × 512 + 512 × 1536) × 4
≈ 6.3 MB

This budget can be reinvested into:
- Wider MLP (3× → 4×)
- Larger BigramHash (3072 → 4096)
- More unique layers
- Higher-precision quantization (int6 → mixed int6/int8 for critical layers)

## GPTQ Integration

Shared MLP weights need careful Hessian handling:
- Forward pass collects Hessians under the shared key `mlp_up_bank[3]`
  from all three layers (3, 4, 5) that use it
- GPTQ quantizes the weight once using the combined Hessian
- `_rebank_state_dict` deduplicates shared entries when constructing
  the export bank

## Running

```bash
DEPTH_RECURRENCE_ENABLED=1 DEPTH_RECURRENCE_LAYERS=3,4,5 \
  MUON_WD=0.09 ADAM_WD=0.09 \
  GPTQ_ENABLED=1 GPTQ_N_BATCHES=64 TTT_ENABLED=0 EVAL_STRIDE=64 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Result

Targeting ~1.10-1.11 BPB. Should be close to the merged SOTA (1.1147).
