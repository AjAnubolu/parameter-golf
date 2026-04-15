"""
Fused Triton kernels for Parameter Golf.
Target: reduce kernel launch overhead and memory bandwidth.

Kernel 1: Fused RMSNorm + Linear (forward only — backward via autograd)
Kernel 2: Fused LeakyReLU² (leaky_relu(x, 0.5).square())
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
import torch.nn.functional as F


# ============================================================
# Kernel 1: Fused RMSNorm + Scale (elementwise, not the matmul)
# RMSNorm output feeds into F.linear, but fusing norm+matmul is
# very hard to beat cuBLAS. Instead we fuse:
#   rms_norm(x) * ln_scale_factor
# into a single kernel to avoid materializing the normalized tensor
# when it's immediately scaled. The matmul still uses cuBLAS.
# ============================================================

@triton.jit
def _rmsnorm_scale_fwd_kernel(
    X_ptr, Out_ptr,
    scale_factor,  # scalar: ln_scale_factor
    N: tl.constexpr,  # hidden dim
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS norm
    ms = tl.sum(x * x, axis=0) / N
    rms = tl.rsqrt(ms + eps)
    out = x * rms * scale_factor

    tl.store(Out_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


def fused_rmsnorm_scale(x: Tensor, scale_factor: float = 1.0, eps: float = 1e-6) -> Tensor:
    """Fused RMSNorm * scale_factor. Avoids materializing intermediate."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _rmsnorm_scale_fwd_kernel[(M,)](
        x_2d, out,
        scale_factor,
        N=N,
        eps=eps,
        BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)


# ============================================================
# Kernel 2: Fused LeakyReLU²
# Replaces: F.leaky_relu(x, 0.5).square()
# Saves one full tensor write+read (the intermediate after leaky_relu)
# ============================================================

@triton.jit
def _leaky_relu_sq_fwd_kernel(
    X_ptr, Out_ptr,
    numel,
    neg_slope: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # leaky_relu then square
    y = tl.where(x >= 0, x, x * neg_slope)
    y = y * y
    tl.store(Out_ptr + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _leaky_relu_sq_bwd_kernel(
    X_ptr, DOut_ptr, DIn_ptr,
    numel,
    neg_slope: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(DOut_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # y = leaky_relu(x)^2
    # dy/dx = 2 * leaky_relu(x) * d_leaky_relu(x)/dx
    #       = 2 * leaky_relu(x) * (1 if x >= 0 else neg_slope)
    lrelu = tl.where(x >= 0, x, x * neg_slope)
    dlrelu = tl.where(x >= 0, 1.0, neg_slope)
    dx = dout * 2.0 * lrelu * dlrelu
    tl.store(DIn_ptr + offs, dx.to(tl.bfloat16), mask=mask)


class FusedLeakyReLUSq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, neg_slope=0.5):
        ctx.save_for_backward(x)
        ctx.neg_slope = neg_slope
        out = torch.empty_like(x)
        numel = x.numel()
        BLOCK = 1024
        grid = ((numel + BLOCK - 1) // BLOCK,)
        _leaky_relu_sq_fwd_kernel[grid](x, out, numel, neg_slope=neg_slope, BLOCK=BLOCK)
        return out

    @staticmethod
    def backward(ctx, dout):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        numel = x.numel()
        BLOCK = 1024
        grid = ((numel + BLOCK - 1) // BLOCK,)
        _leaky_relu_sq_bwd_kernel[grid](x, dout, dx, numel, neg_slope=ctx.neg_slope, BLOCK=BLOCK)
        return dx, None


def fused_leaky_relu_sq(x: Tensor, neg_slope: float = 0.5) -> Tensor:
    """Fused leaky_relu(x, neg_slope).square() with custom backward."""
    return FusedLeakyReLUSq.apply(x, neg_slope)


# ============================================================
# Kernel 3: Fused Residual Mix + Scale-Add
# Replaces: mix[0]*x + mix[1]*x0  (then later: x_in + scale * attn_out)
# Two common patterns fused into single kernels
# ============================================================

@triton.jit
def _residual_mix_kernel(
    X_ptr, X0_ptr, Mix0_ptr, Mix1_ptr, Out_ptr,
    M, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused: out = mix0[None,None,:] * x + mix1[None,None,:] * x0"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x0 = tl.load(X0_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    m0 = tl.load(Mix0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    m1 = tl.load(Mix1_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    out = m0 * x + m1 * x0
    tl.store(Out_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


def fused_residual_mix(x: Tensor, x0: Tensor, mix: Tensor) -> Tensor:
    """Fused mix[0]*x + mix[1]*x0. mix shape: (2, dim)."""
    orig_shape = x.shape
    N = x.shape[-1]
    M = x.numel() // N
    x_2d = x.reshape(M, N)
    x0_2d = x0.reshape(M, N)
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    mix_bf16 = mix.to(x.dtype)
    _residual_mix_kernel[(M,)](
        x_2d, x0_2d, mix_bf16[0], mix_bf16[1], out,
        M, N=N, BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)


@triton.jit
def _scale_add_kernel(
    X_ptr, Y_ptr, Scale_ptr, Out_ptr,
    M, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused: out = x + scale[None,None,:] * y"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(Scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    out = x + s * y
    tl.store(Out_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


def fused_scale_add(x: Tensor, y: Tensor, scale: Tensor) -> Tensor:
    """Fused x + scale[None,None,:] * y."""
    orig_shape = x.shape
    N = x.shape[-1]
    M = x.numel() // N
    x_2d = x.reshape(M, N)
    y_2d = y.reshape(M, N)
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _scale_add_kernel[(M,)](
        x_2d, y_2d, scale.to(x.dtype), out,
        M, N=N, BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)


# ============================================================
# Test suite
# ============================================================

def test_rmsnorm_scale():
    torch.manual_seed(42)
    x = torch.randn(4, 128, 512, device='cuda', dtype=torch.bfloat16)
    scale = 0.333

    # Reference
    ref = F.rms_norm(x, (512,)) * scale

    # Fused
    out = fused_rmsnorm_scale(x, scale_factor=scale)

    err = (ref.float() - out.float()).abs().max().item()
    print(f"RMSNorm+Scale max error: {err:.6e}  ({'PASS' if err < 1e-2 else 'FAIL'})")
    return err < 1e-2


def test_leaky_relu_sq():
    torch.manual_seed(42)
    x = torch.randn(4, 128, 2048, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    # Reference forward
    ref = F.leaky_relu(x_ref, negative_slope=0.5).square()
    # Fused forward
    out = fused_leaky_relu_sq(x)

    fwd_err = (ref.float() - out.float()).abs().max().item()
    print(f"LeakyReLU² fwd max error: {fwd_err:.6e}  ({'PASS' if fwd_err < 1e-2 else 'FAIL'})")

    # Backward
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)
    out.backward(grad_out)

    bwd_err = (x_ref.grad.float() - x.grad.float()).abs().max().item()
    print(f"LeakyReLU² bwd max error: {bwd_err:.6e}  ({'PASS' if bwd_err < 1e-2 else 'FAIL'})")
    return fwd_err < 1e-2 and bwd_err < 1e-2


def test_residual_mix():
    torch.manual_seed(42)
    x = torch.randn(4, 128, 512, device='cuda', dtype=torch.bfloat16)
    x0 = torch.randn(4, 128, 512, device='cuda', dtype=torch.bfloat16)
    mix = torch.randn(2, 512, device='cuda', dtype=torch.float32)

    # Reference
    mix_cast = mix.to(x.dtype)
    ref = mix_cast[0][None, None, :] * x + mix_cast[1][None, None, :] * x0

    # Fused
    out = fused_residual_mix(x, x0, mix)

    err = (ref.float() - out.float()).abs().max().item()
    print(f"ResidualMix max error: {err:.6e}  ({'PASS' if err < 1e-2 else 'FAIL'})")
    return err < 1e-2


def test_scale_add():
    torch.manual_seed(42)
    x = torch.randn(4, 128, 512, device='cuda', dtype=torch.bfloat16)
    y = torch.randn(4, 128, 512, device='cuda', dtype=torch.bfloat16)
    scale = torch.randn(512, device='cuda', dtype=torch.float32)

    # Reference
    ref = x + scale.to(x.dtype)[None, None, :] * y

    # Fused
    out = fused_scale_add(x, y, scale)

    err = (ref.float() - out.float()).abs().max().item()
    print(f"ScaleAdd max error: {err:.6e}  ({'PASS' if err < 1e-2 else 'FAIL'})")
    return err < 1e-2


def benchmark_all():
    """Benchmark fused vs unfused on realistic sizes."""
    import time
    torch.manual_seed(42)
    device = 'cuda'
    # Model dims: batch=4, seq=128 (per GPU), dim=512, mlp_dim=2048
    B, T, D = 4, 128, 512
    MLP_D = 2048
    WARMUP, ITERS = 50, 200

    # --- RMSNorm+Scale ---
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    scale = 0.333
    for _ in range(WARMUP):
        F.rms_norm(x, (D,)) * scale
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        F.rms_norm(x, (D,)) * scale
    torch.cuda.synchronize()
    unfused_us = (time.perf_counter() - t0) / ITERS * 1e6

    for _ in range(WARMUP):
        fused_rmsnorm_scale(x, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused_rmsnorm_scale(x, scale)
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"RMSNorm+Scale:  unfused={unfused_us:.1f}µs  fused={fused_us:.1f}µs  speedup={unfused_us/fused_us:.2f}x")

    # --- LeakyReLU² ---
    x = torch.randn(B, T, MLP_D, device=device, dtype=torch.bfloat16)
    for _ in range(WARMUP):
        F.leaky_relu(x, 0.5).square()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        F.leaky_relu(x, 0.5).square()
    torch.cuda.synchronize()
    unfused_us = (time.perf_counter() - t0) / ITERS * 1e6

    for _ in range(WARMUP):
        fused_leaky_relu_sq(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused_leaky_relu_sq(x)
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"LeakyReLU²:     unfused={unfused_us:.1f}µs  fused={fused_us:.1f}µs  speedup={unfused_us/fused_us:.2f}x")

    # --- ResidualMix ---
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    x0 = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    mix = torch.randn(2, D, device=device, dtype=torch.float32)
    mix_bf = mix.to(torch.bfloat16)
    for _ in range(WARMUP):
        mix_bf[0][None,None,:]*x + mix_bf[1][None,None,:]*x0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mix_bf[0][None,None,:]*x + mix_bf[1][None,None,:]*x0
    torch.cuda.synchronize()
    unfused_us = (time.perf_counter() - t0) / ITERS * 1e6

    for _ in range(WARMUP):
        fused_residual_mix(x, x0, mix)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused_residual_mix(x, x0, mix)
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"ResidualMix:    unfused={unfused_us:.1f}µs  fused={fused_us:.1f}µs  speedup={unfused_us/fused_us:.2f}x")

    # --- ScaleAdd ---
    y = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    s = torch.randn(D, device=device, dtype=torch.float32).to(torch.bfloat16)
    for _ in range(WARMUP):
        x + s[None,None,:]*y
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        x + s[None,None,:]*y
    torch.cuda.synchronize()
    unfused_us = (time.perf_counter() - t0) / ITERS * 1e6

    for _ in range(WARMUP):
        fused_scale_add(x, y, s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused_scale_add(x, y, s)
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"ScaleAdd:       unfused={unfused_us:.1f}µs  fused={fused_us:.1f}µs  speedup={unfused_us/fused_us:.2f}x")


if __name__ == "__main__":
    print("=== Correctness Tests ===")
    all_pass = all([
        test_rmsnorm_scale(),
        test_leaky_relu_sq(),
        test_residual_mix(),
        test_scale_add(),
    ])
    print(f"\nAll tests: {'PASS' if all_pass else 'FAIL'}")

    print("\n=== Benchmarks ===")
    benchmark_all()
