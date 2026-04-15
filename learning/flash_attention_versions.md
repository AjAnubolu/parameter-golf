# Flash Attention: FA2 vs FA3 and Why It Matters

## What Flash Attention Does

Standard attention materializes the full [seq_len, seq_len] attention matrix in HBM:
```
Attention = softmax(Q @ K^T / sqrt(d)) @ V
```
At seq_len=2048, that's a 2048×2048 = 4M element matrix per head. FlashAttention
tiles this computation so the attention matrix never leaves SRAM (shared memory),
reducing HBM traffic from O(N²) to O(N).

## FA2 vs FA3

| | FA2 | FA3 |
|---|---|---|
| **GPU support** | SM 80+ (A100, H100, 5090, everything) | **SM 90 only** (H100 Hopper) |
| **Speed on H100** | ~100 ms/step (our benchmark) | ~86 ms/step |
| **Package** | `flash-attn` on PyPI | `flash_attn_interface` (Hopper branch) |
| **Install** | `pip install flash-attn` | Build from source: `pip install flash-attn --no-build-isolation` (~12 min) |
| **Import** | `from flash_attn import flash_attn_func` | `from flash_attn_interface import flash_attn_func` |

FA3 is ~15% faster on H100 because it uses Hopper-specific features:
- **TMA (Tensor Memory Accelerator)**: hardware unit for async memory copies
- **WGMMA**: warp-group matrix multiply-accumulate instructions
- **Cluster-level execution**: coordinate across multiple SMs

## Why 86ms vs 100ms Matters

In the Parameter Golf competition, training is capped at 600 seconds:
- At 100 ms/step (FA2): ~6,000 steps
- At 86 ms/step (FA3): ~6,900 steps
- **900 extra steps ≈ 0.005-0.01 BPB improvement**

At the competition frontier (~1.08 BPB), every 0.005 matters.

## The FA3 Fallback Pattern

Our scripts use this pattern for development portability:
```python
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    FLASH_3 = True
except ImportError:
    FLASH_3 = False
    try:
        from flash_attn import flash_attn_func as flash_attn_2_func
        FLASH_2 = True
    except ImportError:
        FLASH_2 = False

# In the attention forward:
if FLASH_3:
    out = flash_attn_3_func(q, k, v, causal=True)
elif FLASH_2:
    out = flash_attn_2_func(q, k, v, causal=True)
else:
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

This lets us develop on RTX 5090 (SM 120, no FA3) and run on H100 (SM 90, FA3).
The SDPA fallback is slowest but works everywhere.

## SM Architecture Quick Reference

| GPU | SM Version | FA3 Support |
|---|---|---|
| A100 | SM 80 | No |
| H100 SXM | SM 90 | **Yes** |
| H200 | SM 90 | Yes (but different BPB, not competition-valid) |
| B200 | SM 100 | No (needs FA4?) |
| RTX 4090 | SM 89 | No |
| RTX 5090 | SM 120 | No |

## Build Tips

FA3 build from source takes ~12 minutes and compiles ~450 CUDA kernels.
On vast.ai / cloud instances:
```bash
# Clone and build from Hopper branch
pip install flash-attn --no-build-isolation

# Or if that fails, build manually:
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
cd /tmp/flash-attention/hopper
pip install . --no-build-isolation
```

Pre-built wheels sometimes exist but are CUDA-version-specific:
```bash
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```
