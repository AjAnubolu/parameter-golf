# exp_074_prequant_ttt — Pre-quant AdamW TTT (READY, untested)

**Hypothesis**: Running AdamW TTT on the **full-precision EMA model before GPTQ** should give a much larger BPB improvement than post-quant SGD TTT.

**Source**: [PR #1364](https://github.com/openai/parameter-golf/pull/1364) reports −0.027 BPB from this technique alone (1.1025 BPB 3-seed mean).

## Why this works

Post-quant SGD TTT on int6 weights is unstable — we observed +0.030 BPB
penalty with naive SGD TTT on a GPTQ-quantized model (see PR #756's
25 failed attempts). Running TTT **before** quantization:

1. Avoids optimizer instability on the quantized weight manifold
2. Lets GPTQ see the TTT-adapted Hessians during calibration
3. Uses AdamW (not SGD) for better adaptation dynamics

## Flow

```
Train 600s → EMA model (bf16)
           → AdamW TTT on full-precision model (3 epochs)
           → GPTQ quantize the adapted model
           → Sliding window eval (no further TTT)
```

## NCCL Timeout Fix

Pre-quant TTT runs for ~13 minutes on rank 0 only, exceeding NCCL's
default watchdog timeout (600s). Fix:

```python
if distributed:
    dist.barrier()
    dist.destroy_process_group()
# ... rank 0 runs TTT ...
if distributed:
    dist.init_process_group(backend="nccl", device_id=device)
    for p in base_model.parameters():
        dist.broadcast(p.data, src=0)
```

## Running

```bash
PREQUANT_TTT_ENABLED=1 PREQUANT_TTT_EPOCHS=3 PREQUANT_TTT_LR=3e-4 \
  PREQUANT_TTT_FREEZE_BLOCKS=2 GPTQ_ENABLED=1 GPTQ_N_BATCHES=64 \
  TTT_ENABLED=0 EVAL_STRIDE=64 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Result

Targeting ~1.10-1.12 BPB (a −0.01 to −0.027 BPB gain from the post-EMA 1.142).
Full PR #1364 reports 1.1025 at 6 epochs; we use 3 epochs to halve the TTT time.
