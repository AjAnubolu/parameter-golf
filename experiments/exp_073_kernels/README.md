# exp_073_kernels — Custom Triton Kernel Exploration (NEGATIVE RESULT)

**Result**: Custom kernels were slower than `torch.compile` on the relevant tensor sizes. Not integrated into the SOTA.

## What we tried

### `fused_kernels.py` — elementwise fusions
- `fused_rmsnorm_scale` — RMSNorm × ln_scale_factor
- `fused_leaky_relu_sq` — leaky_relu(x, 0.5).square() with custom backward
- `fused_residual_mix` — mix[0] × x + mix[1] × x0
- `fused_scale_add` — x + scale × y

### `megakernel_mlp.py` — fused double-GEMM
- Fused: **Up-GEMM → LeakyReLU² → Down-GEMM** in a single Triton kernel
- Tiles the MLP intermediate (mlp_dim=2048) to stay in SRAM
- Never writes the [M, 2048] intermediate tensor to HBM

## Benchmarks (RTX 5090)

| Kernel | Unfused | Fused | Speedup |
|---|---|---|---|
| RMSNorm + Scale | 8.2 µs | 9.3 µs | 0.88× |
| LeakyReLU² | 6.4 µs | 16.3 µs | 0.39× |
| ResidualMix | 13.3 µs | 15.6 µs | 0.85× |
| ScaleAdd | 7.6 µs | 10.3 µs | 0.73× |
| **Megakernel MLP** | 37.1 µs | 154.9 µs | **0.24×** |
| Megakernel MLP vs `torch.compile` | 42.6 µs | 154.9 µs | **0.27×** |

## Why they lost

1. **`torch.compile(fullgraph=True)` already fuses elementwise ops.** Our hand-written fusions competed against Inductor-generated code, not against individual PyTorch kernels.
2. **Tensors are too small.** At `dim=512`, memory bandwidth is cheap; kernel launch overhead dominates.
3. **cuBLAS dominates at small GEMM sizes.** The double-GEMM megakernel is ~4× slower than cuBLAS because the tiling loop overhead exceeds the HBM savings.
4. **Correctness was fine.** `fused_leaky_relu_sq` had 0 error (fwd + bwd), megakernel had 0.5% relative error (bf16 double-GEMM accumulation).

## What might actually help

Real wins at this scale would come from:
- Reducing HBM traffic from the **fp32 → bf16 weight bank casts** on every block forward
- Fusing the **EMA update loop** (~70 kernel launches per step)
- Handling the **small attention head GEMMs** (head_dim=64) which don't saturate tensor cores

Not from naive elementwise Triton fusions when `torch.compile` is already active.
