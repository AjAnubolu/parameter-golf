#!/bin/bash
# Parameter Golf - Run all 3 experiments sequentially on 8xH100 SXM
# Total time: ~12 min setup + 3x ~25 min = ~90 min
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/results"

echo "============================================"
echo "  Parameter Golf - 3 Experiment Queue"
echo "============================================"

# === SETUP (one-time) ===
echo ""
echo "=== Setup: Install dependencies ==="
pip install tiktoken blobfile tqdm lm_eval sentencepiece 2>/dev/null

echo "=== Setup: Build Flash Attention 3 (Hopper kernels, ~12 min) ==="
pip install flash-attn --no-build-isolation 2>&1 | tail -5
python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" 2>/dev/null \
  || python -c "from flash_attn import flash_attn_func; print('FA2 fallback (slower)')"

echo "=== Setup: Download training data ==="
python -c "
import subprocess, os
os.makedirs('data', exist_ok=True)
if not os.path.exists('data/cached_challenge_fineweb.py'):
    subprocess.run(['wget', '-q', '-O', 'data/cached_challenge_fineweb.py',
        'https://raw.githubusercontent.com/openai/parameter-golf/main/data/cached_challenge_fineweb.py'])
" 2>/dev/null
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 2>&1 | tail -3
echo "=== Setup complete ==="

# ============================================================
# EXPERIMENT 1: Pre-quant AdamW TTT
# Train → EMA → AdamW TTT on full-precision → GPTQ → eval
# Expected: ~0.01-0.027 BPB improvement over baseline
# ============================================================
echo ""
echo "============================================"
echo "  EXP 074: Pre-quant AdamW TTT (seed 1337)"
echo "============================================"
cp "$SCRIPT_DIR/exp_074_prequant_ttt/train_gpt.py" ./train_gpt.py

PREQUANT_TTT_ENABLED=1 \
PREQUANT_TTT_EPOCHS=3 \
PREQUANT_TTT_LR=3e-4 \
PREQUANT_TTT_FREEZE_BLOCKS=2 \
GPTQ_ENABLED=1 \
GPTQ_N_BATCHES=64 \
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$SCRIPT_DIR/results/exp074.log"

echo ""
echo "--- EXP 074 RESULTS ---"
grep -E "val_bpb|DIAGNOSTIC|prequant_ttt:(done|finished)|sliding" "$SCRIPT_DIR/results/exp074.log" | tail -10
echo ""

# ============================================================
# EXPERIMENT 2: L-BFGS Causal SLOT
# Train → GPTQ → L-BFGS SLOT eval (logit-space, 25 iters)
# Expected: significant if SLOT is legal
# ============================================================
echo "============================================"
echo "  EXP 075: L-BFGS Causal SLOT (seed 1337)"
echo "============================================"
cp "$SCRIPT_DIR/exp_075_lbfgs_slot/train_gpt.py" ./train_gpt.py

LBFGS_SLOT_ENABLED=1 \
LBFGS_SLOT_ITERS=25 \
LBFGS_SLOT_HISTORY=20 \
LBFGS_SLOT_FOCAL=128 \
LBFGS_SLOT_CLAMP=5.0 \
LBFGS_SLOT_LR=1.0 \
GPTQ_ENABLED=1 \
GPTQ_N_BATCHES=64 \
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$SCRIPT_DIR/results/exp075.log"

echo ""
echo "--- EXP 075 RESULTS ---"
grep -E "val_bpb|lbfgs_slot|sliding" "$SCRIPT_DIR/results/exp075.log" | tail -10
echo ""

# ============================================================
# EXPERIMENT 3: Depth Recurrence (MLP-only sharing)
# Layers 3,4,5 share MLP weights, WD=0.09
# Expected: ~0.02 BPB improvement from more effective depth
# ============================================================
echo "============================================"
echo "  EXP 076: Depth Recurrence (seed 1337)"
echo "============================================"
cp "$SCRIPT_DIR/exp_076_depth_recurrence/train_gpt.py" ./train_gpt.py

DEPTH_RECURRENCE_ENABLED=1 \
DEPTH_RECURRENCE_LAYERS=3,4,5 \
MUON_WD=0.09 \
ADAM_WD=0.09 \
GPTQ_ENABLED=1 \
GPTQ_N_BATCHES=64 \
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$SCRIPT_DIR/results/exp076.log"

echo ""
echo "--- EXP 076 RESULTS ---"
grep -E "val_bpb|sliding|depth_recurrence" "$SCRIPT_DIR/results/exp076.log" | tail -10
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE - SUMMARY"
echo "============================================"
echo ""
echo "EXP 074 (Pre-quant AdamW TTT):"
grep "val_bpb" "$SCRIPT_DIR/results/exp074.log" | tail -1
echo ""
echo "EXP 075 (L-BFGS SLOT):"
grep "val_bpb" "$SCRIPT_DIR/results/exp075.log" | tail -1
echo ""
echo "EXP 076 (Depth Recurrence):"
grep "val_bpb" "$SCRIPT_DIR/results/exp076.log" | tail -1
echo ""
echo "Done! Results saved in $SCRIPT_DIR/results/"
