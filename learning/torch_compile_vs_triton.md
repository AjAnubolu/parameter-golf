# torch.compile Already Does Your Fusions

## What We Tried

We wrote custom Triton kernels to fuse common operations:
- `fused_rmsnorm_scale`: RMSNorm × scale factor
- `fused_leaky_relu_sq`: leaky_relu(x, 0.5).square()
- `fused_residual_mix`: mix[0]*x + mix[1]*x0
- `fused_scale_add`: x + scale*y
- **Megakernel MLP**: Up-GEMM → LeakyReLU² → Down-GEMM (two GEMMs fused)

## Benchmark Results (RTX 5090, dim=512)

| Kernel | PyTorch | Custom Triton | Speedup |
|---|---|---|---|
| RMSNorm + Scale | 8.2 µs | 9.3 µs | **0.88x (slower)** |
| LeakyReLU² | 6.4 µs | 16.3 µs | **0.39x** |
| ResidualMix | 13.3 µs | 15.6 µs | **0.85x** |
| ScaleAdd | 7.6 µs | 10.3 µs | **0.73x** |
| Megakernel MLP | 37.1 µs | 154.9 µs | **0.24x** |
| MLP vs torch.compile | 42.6 µs | 154.9 µs | **0.27x** |

Every single custom kernel was **slower**.

## Why

### 1. torch.compile(fullgraph=True) already fuses elementwise ops

The model uses:
```python
model = torch.compile(model, dynamic=False, fullgraph=True)
```

Inductor (torch.compile's backend) traces the computation graph and
automatically fuses elementwise operations into optimized CUDA kernels.
Our hand-written Triton kernels competed against Inductor-generated code,
not against individual PyTorch kernels.

### 2. Tensors are too small at dim=512

At batch=4, seq=128, dim=512:
- Total elements: 262,144 (1 MB at fp32)
- This fits in L2 cache on modern GPUs
- Kernel launch overhead (~5-10 µs) dominates actual compute time
- Custom kernels add Python → Triton → CUDA dispatch overhead

### 3. cuBLAS dominates at small GEMM sizes

The megakernel MLP fuses two GEMMs (512→2048 and 2048→512) to avoid writing
the 2048-wide intermediate to HBM. But at these sizes:
- The intermediate is ~2 MB — fits in L2 cache anyway
- cuBLAS has hand-tuned assembly for small matrices
- Our Triton tiling loop has overhead that exceeds the HBM savings

## When Custom Kernels DO Help

Custom kernels win when:
1. **torch.compile can't see the fusion** (across module boundaries, dynamic shapes)
2. **Tensors are large enough** that kernel launch overhead is amortized (dim≥2048)
3. **The fusion is non-trivial** (FlashAttention fuses softmax into matmul — torch.compile can't do this)
4. **You're fusing across a GEMM boundary** AND the matrices are large enough

FlashAttention is the canonical example: it fuses QK^T → softmax → V multiplication
into a single kernel that tiles in SRAM. This is impossible for torch.compile because
it requires custom tiling logic. At our scale (dim=512, head_dim=64), even FA3 is
mostly helpful for its memory efficiency, not raw speed.

## The Real Optimization Opportunities

What would actually speed up training at dim=512:
- **Pre-cast weight banks once per step** (avoid 44 dtype casts per forward pass)
- **Fuse EMA update** into a single `_foreach_lerp_` (70 kernel launches → 1)
- **Pin CPU buffers** for async data loading
- **SWA on GPU** instead of GPU→CPU→GPU roundtrip

These are all about reducing kernel launch overhead and unnecessary memory traffic,
not about fusing compute operations (which torch.compile handles).

## Key Lesson

Before writing custom kernels, check:
1. Is torch.compile already active? If yes, your elementwise fusions are redundant.
2. Are the tensors large enough? At dim<1024, launch overhead dominates.
3. Is cuBLAS being used for GEMMs? It has assembly-level optimizations you can't beat in Triton at small sizes.

Profile first, optimize second. `torch.profiler` or `nsight systems` will show
you exactly where time is spent.
