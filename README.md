# Parameter Golf — Experiments

Experiment log for OpenAI's [Parameter Golf](https://github.com/openai/parameter-golf) competition.

**Goal**: minimize `val_bpb` on FineWeb within a 16 MB artifact + 10 min training on 8×H100 SXM.

## Current Best

**1.1191 BPB** (3-seed mean) — `exp_031_submission/` — Full Hessian GPTQ int6 + SGD TTT.

## Repository Layout

```
experiments/
├── exp_031_submission/       # Our validated SOTA (1.1191 BPB)
├── exp_072_slot_qkgain/      # SLOT score-first (failed, 1.1493)
├── exp_073_kernels/          # Custom Triton kernel exploration
├── exp_074_prequant_ttt/     # Pre-quant AdamW TTT (untested)
├── exp_075_lbfgs_slot/       # L-BFGS causal SLOT (untested)
└── exp_076_depth_recurrence/ # MLP-only weight sharing (untested)
```

Each experiment directory contains a self-contained `train_gpt.py` (~88-105KB) and a short README describing the hypothesis, technique, and result.

## Running

All experiments use the same interface:

```bash
# 8×H100 SXM, competition configuration
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Experiment-specific env vars are documented in each experiment's README.

## Requirements

- 8×H100 SXM (Hopper) — Flash Attention 3 required for full speed (~86 ms/step)
- PyTorch 2.x + CUDA 12.x
- FineWeb SP1024 training data
