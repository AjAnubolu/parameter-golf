#!/bin/bash
# Parameter Golf - exp_074: Pre-quant AdamW TTT
# Requirements: 8xH100 SXM, PyTorch 2.x, CUDA 12.x
# Expected time: ~35 min total (12 min FA3 build + 10 min train + 13 min TTT + 5 min GPTQ/eval)
set -e

echo "=== Step 1: Install dependencies ==="
pip install tiktoken blobfile tqdm lm_eval sentencepiece 2>/dev/null

echo "=== Step 2: Build Flash Attention 3 (Hopper kernels) ==="
echo "This takes ~12 minutes. DO NOT SKIP - FA3 gives ~86ms/step vs ~100ms with FA2."
pip install flash-attn --no-build-isolation 2>&1 | tail -5
# Verify FA3
python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" 2>/dev/null \
  || python -c "from flash_attn import flash_attn_func; print('FA2 fallback (slower)')"

echo "=== Step 3: Download training data ==="
# The script auto-downloads data, but we can pre-fetch for speed
python -c "
import subprocess, os
os.makedirs('data', exist_ok=True)
if not os.path.exists('data/cached_challenge_fineweb.py'):
    subprocess.run(['wget', '-q', '-O', 'data/cached_challenge_fineweb.py',
        'https://raw.githubusercontent.com/openai/parameter-golf/main/data/cached_challenge_fineweb.py'])
" 2>/dev/null
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 2>&1 | tail -3

echo "=== Step 4: Run experiment ==="
echo "Training 600s → EMA → Pre-quant AdamW TTT (3 epochs) → GPTQ → Sliding eval"
PREQUANT_TTT_ENABLED=1 \
PREQUANT_TTT_EPOCHS=3 \
PREQUANT_TTT_LR=3e-4 \
PREQUANT_TTT_FREEZE_BLOCKS=2 \
GPTQ_ENABLED=1 \
GPTQ_N_BATCHES=64 \
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee exp074_results.log

echo "=== Done! Check exp074_results.log for val_bpb ==="
grep -E "val_bpb|DIAGNOSTIC|prequant_ttt|sliding" exp074_results.log | tail -20
