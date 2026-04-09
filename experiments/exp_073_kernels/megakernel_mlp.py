"""
Megakernel: Fused MLP for Parameter Golf transformer.

Two-pass approach:
  Pass 1 (lightweight Triton): RMSNorm * ln_scale (fused, avoids intermediate)
  Pass 2 (Triton megakernel):  Up-GEMM → LeakyReLU² → Down-GEMM
    Keeps the [B*T, mlp_dim] intermediate in SRAM, never writes to HBM.

The key win: the intermediate activation after up-projection is 4x wider (2048 vs 512).
By fusing the two GEMMs through the activation, we save ~2MB of HBM traffic per block call.

Architecture: model_dim=512, mlp_dim=2048 (4x expansion)
Activation: leaky_relu(x, neg_slope=0.5).square()
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
import torch.nn.functional as F
import time


# ============================================================
# Megakernel: Up-GEMM → LeakyReLU² → Down-GEMM
# ============================================================
# Strategy: tile over MLP_D dimension. For each tile:
#   1. Compute partial up-projection [BLOCK_M, BLOCK_K] by loading input rows
#      and weight tiles
#   2. Apply LeakyReLU² in registers
#   3. Accumulate into down-projection output [BLOCK_M, BLOCK_N]
#
# Grid: (M // BLOCK_M, D // BLOCK_N)
# Each program computes a [BLOCK_M, BLOCK_N] tile of the output.

@triton.jit
def _fused_double_gemm_kernel(
    # [M, D] input (already RMSNormed)
    X_ptr,
    # [MLP_D, D] up weight
    UpW_ptr,
    # [D, MLP_D] down weight
    DownW_ptr,
    # [M, D] output
    Out_ptr,
    # Dimensions
    M, D: tl.constexpr, MLP_D: tl.constexpr,
    # Strides
    stride_xm, stride_xd: tl.constexpr,
    stride_um, stride_ud: tl.constexpr,
    stride_dm, stride_dd: tl.constexpr,
    stride_om, stride_od: tl.constexpr,
    # Activation
    neg_slope: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,   # rows
    BLOCK_N: tl.constexpr,   # output cols (tile of D)
    BLOCK_K: tl.constexpr,   # intermediate cols (tile of MLP_D)
    BLOCK_D: tl.constexpr,   # input cols (tile of D for up-GEMM)
):
    # Program IDs
    pid_m = tl.program_id(0)  # which row block
    pid_n = tl.program_id(1)  # which output col block

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator for output tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Tile over intermediate dimension (MLP_D)
    for k_start in range(0, MLP_D, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # ---- Up-projection tile: X @ UpW^T → [BLOCK_M, BLOCK_K] ----
        # UpW is [MLP_D, D], so UpW^T is [D, MLP_D]
        # We compute: up_tile[m,k] = sum_d X[m,d] * UpW[k,d]
        up_tile = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]

            # Load X tile [BLOCK_M, BLOCK_D]
            x_tile = tl.load(X_ptr + rm[:, None] * stride_xm + rd[None, :] * stride_xd,
                             mask=(rm[:, None] < M) & (rd[None, :] < D), other=0.0)

            # Load UpW tile [BLOCK_K, BLOCK_D]
            uw_tile = tl.load(UpW_ptr + rk[:, None] * stride_um + rd[None, :] * stride_ud,
                              mask=(rk[:, None] < MLP_D) & (rd[None, :] < D), other=0.0)

            # Accumulate: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_K]
            up_tile += tl.dot(x_tile, tl.trans(uw_tile))

        # ---- LeakyReLU²: in registers ----
        act_tile = tl.where(up_tile >= 0, up_tile, up_tile * neg_slope)
        act_tile = (act_tile * act_tile).to(tl.bfloat16)  # [BLOCK_M, BLOCK_K]

        # ---- Down-projection: accumulate into output ----
        # DownW is [D, MLP_D], we need DownW[rn, rk] → [BLOCK_N, BLOCK_K]
        dw_tile = tl.load(DownW_ptr + rn[:, None] * stride_dm + rk[None, :] * stride_dd,
                          mask=(rn[:, None] < D) & (rk[None, :] < MLP_D), other=0.0)

        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc += tl.dot(act_tile, tl.trans(dw_tile))

    # Store output tile
    tl.store(Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_od,
             acc.to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < D))


def megakernel_mlp_fwd(x: Tensor, up_w: Tensor, down_w: Tensor,
                        ln_scale: float = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Fused RMSNorm → Up GEMM → LeakyReLU² → Down GEMM.
    x: [B, T, D]  up_w: [MLP_D, D]  down_w: [D, MLP_D]
    Returns: [B, T, D]
    """
    orig_shape = x.shape
    D = x.shape[-1]
    MLP_D = up_w.shape[0]

    # Step 1: Fused RMSNorm * scale (lightweight)
    x_2d = x.reshape(-1, D)
    x_norm = F.rms_norm(x_2d, (D,), eps=eps) * ln_scale  # this is fast, dim=512
    x_norm = x_norm.to(torch.bfloat16).contiguous()
    M = x_norm.shape[0]

    # Step 2: Fused double-GEMM megakernel
    out = torch.empty(M, D, device=x.device, dtype=torch.bfloat16)

    # Tile sizes — fit within SRAM
    BLOCK_M = 32
    BLOCK_N = 32    # tile output D
    BLOCK_K = 64    # tile MLP_D
    BLOCK_D = 64    # tile input D for up-GEMM

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))

    _fused_double_gemm_kernel[grid](
        x_norm, up_w, down_w, out,
        M, D=D, MLP_D=MLP_D,
        stride_xm=x_norm.stride(0), stride_xd=x_norm.stride(1),
        stride_um=up_w.stride(0), stride_ud=up_w.stride(1),
        stride_dm=down_w.stride(0), stride_dd=down_w.stride(1),
        stride_om=out.stride(0), stride_od=out.stride(1),
        neg_slope=0.5,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )
    return out.reshape(orig_shape)


# ============================================================
# Autograd wrapper
# ============================================================

class MegakernelMLP(torch.autograd.Function):
    """
    Forward: megakernel (fused double-GEMM, no intermediate in HBM)
    Backward: recompute via standard ops (torch.compile handles it)
    """
    @staticmethod
    def forward(ctx, x, up_w, down_w, ln_scale):
        ctx.save_for_backward(x, up_w, down_w)
        ctx.ln_scale = ln_scale
        return megakernel_mlp_fwd(x, up_w, down_w, ln_scale)

    @staticmethod
    def backward(ctx, grad_output):
        x, up_w, down_w = ctx.saved_tensors
        ln_scale = ctx.ln_scale

        with torch.enable_grad():
            x_det = x.detach().requires_grad_(True)
            up_w_det = up_w.detach().requires_grad_(True)
            down_w_det = down_w.detach().requires_grad_(True)

            x_norm = F.rms_norm(x_det, (x_det.size(-1),)) * ln_scale
            h = F.leaky_relu(F.linear(x_norm, up_w_det.to(x_norm.dtype)), negative_slope=0.5)
            h = h.square()
            out = F.linear(h, down_w_det.to(h.dtype))
            out.backward(grad_output)

        return x_det.grad, up_w_det.grad, down_w_det.grad, None


def fused_mlp(x: Tensor, up_w: Tensor, down_w: Tensor, ln_scale: float = 1.0) -> Tensor:
    """Drop-in replacement for the MLP block's forward pass."""
    return MegakernelMLP.apply(x, up_w, down_w, ln_scale)


# ============================================================
# Tests
# ============================================================

def test_correctness():
    torch.manual_seed(42)
    device = 'cuda'
    B, T, D, MLP_D = 4, 128, 512, 2048
    ln_scale = 0.333

    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    up_w = torch.randn(MLP_D, D, device=device, dtype=torch.bfloat16) * 0.02
    down_w = torch.randn(D, MLP_D, device=device, dtype=torch.bfloat16) * 0.02

    # Reference
    x_norm = F.rms_norm(x.reshape(-1, D), (D,)) * ln_scale
    x_norm = x_norm.reshape(B, T, D)
    h = F.leaky_relu(F.linear(x_norm, up_w), negative_slope=0.5).square()
    ref = F.linear(h, down_w)

    # Megakernel
    out = megakernel_mlp_fwd(x, up_w, down_w, ln_scale=ln_scale)

    abs_err = (ref.float() - out.float()).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    ref_norm = ref.float().abs().mean().item()
    rel_err = mean_err / (ref_norm + 1e-8)

    print(f"Megakernel MLP correctness:")
    print(f"  Max abs error:  {max_err:.6e}")
    print(f"  Mean abs error: {mean_err:.6e}")
    print(f"  Relative error: {rel_err:.6e}")
    print(f"  Ref mean abs:   {ref_norm:.6e}")

    passed = rel_err < 0.1  # generous for bf16 double-GEMM
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_backward():
    torch.manual_seed(42)
    device = 'cuda'
    B, T, D, MLP_D = 2, 64, 512, 2048
    ln_scale = 0.333

    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    up_w = torch.randn(MLP_D, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    down_w = torch.randn(D, MLP_D, device=device, dtype=torch.bfloat16, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    up_ref = up_w.detach().clone().requires_grad_(True)
    down_ref = down_w.detach().clone().requires_grad_(True)

    x_norm = F.rms_norm(x_ref.reshape(-1, D), (D,)) * ln_scale
    x_norm = x_norm.reshape(B, T, D)
    h = F.leaky_relu(F.linear(x_norm, up_ref), negative_slope=0.5).square()
    ref_out = F.linear(h, down_ref)
    ref_out.sum().backward()

    fused_out = fused_mlp(x, up_w, down_w, ln_scale)
    fused_out.sum().backward()

    dx_err = (x.grad.float() - x_ref.grad.float()).abs().mean() / (x_ref.grad.float().abs().mean() + 1e-8)
    duw_err = (up_w.grad.float() - up_ref.grad.float()).abs().mean() / (up_ref.grad.float().abs().mean() + 1e-8)
    ddw_err = (down_w.grad.float() - down_ref.grad.float()).abs().mean() / (down_ref.grad.float().abs().mean() + 1e-8)

    print(f"\nMegakernel MLP backward:")
    print(f"  dx relative error:      {dx_err:.6e}")
    print(f"  d_up_w relative error:  {duw_err:.6e}")
    print(f"  d_down_w relative error:{ddw_err:.6e}")

    passed = dx_err < 0.1 and duw_err < 0.1 and ddw_err < 0.1
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def benchmark():
    torch.manual_seed(42)
    device = 'cuda'
    B, T, D, MLP_D = 4, 128, 512, 2048
    ln_scale = 0.333
    WARMUP, ITERS = 100, 500

    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    up_w = torch.randn(MLP_D, D, device=device, dtype=torch.bfloat16) * 0.02
    down_w = torch.randn(D, MLP_D, device=device, dtype=torch.bfloat16) * 0.02

    def unfused():
        xn = F.rms_norm(x.reshape(-1, D), (D,)).reshape(B, T, D) * ln_scale
        h = F.leaky_relu(F.linear(xn, up_w), negative_slope=0.5).square()
        return F.linear(h, down_w)

    for _ in range(WARMUP):
        unfused()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        unfused()
    torch.cuda.synchronize()
    unfused_us = (time.perf_counter() - t0) / ITERS * 1e6

    for _ in range(WARMUP):
        megakernel_mlp_fwd(x, up_w, down_w, ln_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        megakernel_mlp_fwd(x, up_w, down_w, ln_scale)
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - t0) / ITERS * 1e6

    print(f"\nMegakernel MLP benchmark (B={B}, T={T}, D={D}, MLP_D={MLP_D}):")
    print(f"  Unfused (PyTorch): {unfused_us:.1f} µs")
    print(f"  Megakernel:        {fused_us:.1f} µs")
    print(f"  Speedup:           {unfused_us/fused_us:.2f}x")

    # Also benchmark with torch.compile for fair comparison
    unfused_compiled = torch.compile(unfused, dynamic=False, fullgraph=True)
    for _ in range(WARMUP):
        unfused_compiled()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        unfused_compiled()
    torch.cuda.synchronize()
    compiled_us = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"  torch.compile:     {compiled_us:.1f} µs")
    print(f"  vs compile:        {compiled_us/fused_us:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("MEGAKERNEL MLP: Fused Up→LeakyReLU²→Down (no HBM intermediate)")
    print("=" * 60)

    ok1 = test_correctness()
    ok2 = test_backward()
    benchmark()

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if ok1 and ok2 else 'SOME FAILURES'}")
