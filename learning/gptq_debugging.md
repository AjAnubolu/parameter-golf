# GPTQ Debugging: From 4.71 BPB to 0.005 Gap

## What is GPTQ?

GPTQ (GPT Quantization) compresses model weights from 16-bit to 6-bit (or lower)
to fit within a size budget. It's "smart" quantization — instead of rounding each
weight independently, it uses the Hessian matrix (H = X^T X) to compensate remaining
weights for each rounding error.

```
For each column j of weight matrix W:
  1. Round w_j to nearest quantized value
  2. Compute rounding error: err = w_j - round(w_j)
  3. Compensate remaining columns: W[:, j+1:] -= err * H_inv[j, j+1:] / H_inv[j,j]
```

This is why it's called "column-sequential" — it processes one column at a time,
and each column's error gets distributed across all remaining columns.

## The Bug Hunt (4 bugs, 4 hours)

We implemented GPTQ and got **4.71 BPB** — catastrophically bad (baseline is ~1.15).
An autonomous debugging agent found 4 independent bugs:

### Bug 1: Double-squaring the Hessian

```python
# BUG: _accumulate_hessian received pre-computed H = X^T @ X
# but then did H = H.T @ H, giving (X^T X)^2
def _accumulate_hessian(self, H_accumulated, new_H):
    H_accumulated += new_H.T @ new_H  # WRONG: new_H is already X^T @ X
    
# FIX:
def _accumulate_hessian(self, H_accumulated, new_H):
    H_accumulated += new_H  # Just accumulate directly
```

**Impact**: The curvature matrix was wrong in every direction, so compensation
pushed weights the wrong way. This alone accounted for ~2 BPB of the gap.

### Bug 2: bf16 autocast precision loss

```python
# BUG: Hessian matmuls ran under torch.autocast(dtype=bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    H = activations.T @ activations  # bf16 matmul loses precision

# FIX:
with torch.autocast("cuda", enabled=False):
    H = activations.float().T @ activations.float()  # fp32
```

**Impact**: bf16 has only 7 bits of mantissa. For Hessian computation where you're
summing thousands of outer products, the accumulated error is devastating. The
Hessian diagonal (which determines per-column scaling) was off by 10-30%.

### Bug 3: Suboptimal quantization scales

```python
# BUG: Used simple row_max / 31 as the quantization scale
scale = W.abs().max(dim=1).values / 31

# FIX: Search 5 clip percentiles, pick the one with lowest MSE
for percentile in [0.9, 0.95, 0.99, 0.995, 1.0]:
    clip = torch.quantile(W.abs(), percentile, dim=1)
    scale = clip / 31
    mse = ((W - round(W/scale)*scale) ** 2).mean(dim=1)
    # Keep the scale with lowest MSE per row
```

**Impact**: Outlier weights dominated the scale, wasting dynamic range.
Clip-search gives ~0.005 BPB improvement. The SOTA later replaced this with
SDClip (`scale = k * std(row)`) which is even better.

### Bug 4: Unwanted stride-64 eval

```python
# BUG: When EVAL_STRIDE=0, code still ran sliding window with stride=64
if args.eval_stride > 0:
    stride = args.eval_stride  
else:
    stride = 64  # Should have been: stride = seq_len (no overlap)
```

**Impact**: Not a quantization bug per se, but it made our eval numbers
inconsistent with ablation baselines, confusing the debugging process.

## The Hessian Intuition

Why does H = X^T X matter? Think of it as "which directions in weight space
the model cares about most." If a weight column is multiplied by large
activations (high H diagonal), rounding error there hurts a lot. GPTQ uses
H^{-1} to figure out: "if I round this column, how much should I adjust the
others to minimize total output change?"

Getting H wrong (bug 1, 2) is like navigating with a broken compass — every
compensation step makes things worse instead of better.

## Later Innovation: SDClip

The SOTA replaced our percentile clip search with:
```python
scale = clip_sigmas * std(row) / clip_range  # clip_sigmas = 12.85
```

This is principled rate-distortion optimization: the clip range that minimizes
expected quantization error for a Gaussian-distributed weight row is proportional
to its standard deviation.
