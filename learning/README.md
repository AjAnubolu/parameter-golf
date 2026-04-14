# ML Systems & Training Lessons

Hard-won lessons from the Parameter Golf competition. Each document covers a
specific issue we hit, why it happened, and how to fix it.

## Index

| File | Topic | TL;DR |
|---|---|---|
| [nccl_process_group.md](nccl_process_group.md) | NCCL destroy/reinit crash | Never destroy+reinit NCCL in the same process. Use Gloo, subprocesses, or parallelize. |
| [gptq_debugging.md](gptq_debugging.md) | GPTQ quantization bugs | 4 bugs took BPB from 4.71 to 0.005 gap. Hessian precision and scale search matter. |
| [torch_compile_vs_triton.md](torch_compile_vs_triton.md) | Custom kernels vs torch.compile | torch.compile already fuses elementwise ops. Custom Triton is slower at dim=512. |
| [flash_attention_versions.md](flash_attention_versions.md) | FA2 vs FA3 | FA3 is SM90-only (H100), 15% faster. 900 extra steps in 600s = real BPB gain. |
| [distributed_training_pitfalls.md](distributed_training_pitfalls.md) | 8-GPU training gotchas | Watchdog timeouts, collective ordering, rank-0 OOM, gradient clipping order. |
| [slot_and_ttt.md](slot_and_ttt.md) | Test-time adaptation | Embedding SLOT fails, logit L-BFGS SLOT works. Pre-quant TTT >> post-quant TTT. |
| [quantization_compression.md](quantization_compression.md) | Fitting in 16 MB | int6 GPTQ + SDClip + Brotli + byte-shuffle. Compression-quantization interaction. |
