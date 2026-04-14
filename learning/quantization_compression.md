# Quantization & Compression for 16MB Models

## The Budget Problem

Parameter Golf: fit the best model in 16,000,000 bytes (code + compressed weights).

A typical model has ~27-36M parameters at bf16 (2 bytes each) = 54-72 MB.
We need 4-5x compression. The stack:
```
bf16 weights (72 MB) → int6 quantization (27 MB) → lzma/brotli compression (16 MB)
```

## Quantization: Reducing Bits Per Weight

### Uniform quantization (baseline)
```python
# Round each weight independently
scale = max(abs(row)) / (2^(bits-1) - 1)
quantized = round(weight / scale)
dequantized = quantized * scale
```

| Bits | Values | Max val | BPB gap vs fp16 |
|---|---|---|---|
| int8 | 256 | 127 | ~0.001 |
| int6 | 64 | 31 | ~0.005 |
| int5 | 32 | 15 | ~0.02 |
| int4 | 16 | 7 | ~0.065 (too much) |

int6 is the sweet spot: good enough quality, small enough to fit.

### GPTQ: Hessian-compensated quantization
Instead of rounding each weight independently, use H = X^T X (Hessian) to
compensate remaining weights for each rounding error. Reduces the int6 gap
from ~0.05 to ~0.005 BPB. See `gptq_debugging.md` for the full story.

### SDClip: Principled scale selection
Instead of searching clip percentiles, use statistics:
```python
scale = k * std(row) / clip_range  # k = 12.85 for int6
```
This is the rate-distortion optimal clip for Gaussian-distributed weights.
Slightly better than percentile search and much simpler.

### int8 for embeddings
Embeddings have different statistics than attention/MLP weights. Using int8
(127 values) instead of int6 (31 values) for the embedding table costs more
bytes but the quality gain is worth it, especially with larger vocabularies.

```
SP8192 embedding table:
  int6: 8192 × 512 × 6/8 = 3.1 MB
  int8: 8192 × 512 × 1 = 4.0 MB (+0.9 MB, but much better quality)
```

## Compression: Reducing Bytes Further

After quantization, weights are stored as integers. Compression exploits the
statistical structure of these integers.

### lzma (what we used)
- Preset 9 (max compression)
- Slow to compress, fast to decompress
- Good compression ratio
- Python stdlib (`import lzma`)

### Brotli (what SOTA uses)
- Quality 11 (max)
- Generally ~5% better compression than lzma on quantized weights
- Requires `pip install brotli`

### Byte shuffling (SOTA trick)
Before compression, rearrange bytes to group similar values together:
```python
def byte_shuffle(data):
    # Transpose from [w0_b0, w0_b1, w1_b0, w1_b1, ...]
    # to [w0_b0, w1_b0, ..., w0_b1, w1_b1, ...]
    # This groups MSBs together and LSBs together
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bytes_per_weight).T.tobytes()
```
This can improve compression by 2-5% because the high bytes of weights
are more predictable than the low bytes.

## The Compression-Quantization Interaction

A key SOTA insight: the compressed size depends on **entropy of the quantized values**,
not just the bitwidth. A wider clip range means fewer distinct values are actually used,
which lowers entropy and compresses better:

```
int5 with wide clip → many weights round to 0 → low entropy → compresses small
int6 with narrow clip → values spread across all 64 bins → high entropy → larger
```

This means int5 can sometimes compress **smaller** than int4 if the clip range
is chosen correctly. The relationship between quantization and compression is
non-trivial.

## Artifact Size Budget (16 MB)

Typical breakdown for SOTA:
```
Code (lzma-wrapped Python):     ~17 KB
Quantized attention/MLP (int6): ~11 MB
Quantized embeddings (int8):    ~4 MB
Metadata/headers:               ~50 KB
─────────────────────────────────────
Total:                          ~15.9 MB
```

The 16 MB limit is TIGHT. Going from int6 → int8 on MLP weights would
cost ~5 MB and blow the budget. Going from SP1024 → SP8192 adds ~3 MB
to the embedding table. Every architectural choice affects the budget.
