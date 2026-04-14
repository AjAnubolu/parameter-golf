# Distributed Training Pitfalls

## 1. NCCL Watchdog Timeout

**Symptom**: Process dies with SIGABRT after ~600 seconds of inactivity.

**Cause**: NCCL has a background heartbeat thread. If a rank doesn't participate
in any collective operation for `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (default 600s),
the watchdog kills the process.

**When it bites**: Any time one rank does long solo work (TTT, data processing,
model surgery) while other ranks wait.

**Fix**: Either involve all ranks in the work, or increase the timeout:
```python
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"
# Must be set BEFORE dist.init_process_group()
```

## 2. All Ranks Must Call Collectives Together

**Symptom**: Hang or crash when only some ranks call a collective.

```python
# BUG: Only rank 0 calls eval, which internally does all_reduce
if rank == 0:
    eval_val(model)  # Has dist.all_reduce inside — ranks 1-7 never call it

# FIX: All ranks must call it
val_loss = eval_val(model)  # All ranks participate
if rank == 0:
    print(val_loss)  # Only rank 0 prints
```

Every `dist.all_reduce`, `dist.broadcast`, `dist.barrier`, `dist.reduce_scatter`
is a **collective** — all ranks must call it or the program hangs.

## 3. torchrun Kills All Ranks on Any Failure

**Symptom**: You see `FAILED` with `Signal 6 (SIGABRT)` or `Signal 11 (SIGSEGV)`
on one rank, and all other ranks get `SIGTERM`.

**Cause**: `torchrun` monitors all child processes. If any one dies, it sends
SIGTERM to all others and reports the first failure. The error trace might say
"rank 7 failed" but the root cause could be rank 0 running out of memory —
the other ranks fail because NCCL can't reach rank 0.

**Debug tip**: Check the earliest error, not the last. The first `exitcode` in
the error report is the root cause.

## 4. OOM on One Rank Kills Everything

**Symptom**: CUDA OOM on rank 0 (which has extra work like logging, checkpointing)
while ranks 1-7 are fine.

**Cause**: Rank 0 often holds extra tensors:
- Validation data
- EMA state dict
- SWA accumulator
- Checkpoint buffers

**Fix**: Offload rank-0-only state to CPU:
```python
if rank == 0:
    ema_state = {k: v.cpu() for k, v in model.state_dict().items()}
```

## 5. Non-Determinism Across Ranks

**Symptom**: Gradients diverge across ranks, training becomes unstable.

**Cause**: Each rank processes different data, but they must agree on:
- Model state (via all-reduce of gradients)
- Random seed for dropout, etc.
- Learning rate schedule

**Common bug**: Using `torch.manual_seed(seed)` without accounting for rank:
```python
# BUG: All ranks get different random sequences
torch.manual_seed(seed + rank)  # For data sampling — intentionally different

# But model init must be the same:
torch.manual_seed(seed)  # Same seed for weight init
model = build_model()
# Then set per-rank seed for data:
torch.manual_seed(seed + rank)
```

## 6. SWA/EMA GPU→CPU Transfers Per Step

**Symptom**: Training is slower than expected, profiler shows lots of D2H transfers.

**Cause**: Code like this runs every N steps:
```python
swa_state[name] += t.detach().cpu()  # GPU → CPU transfer per parameter
```

Each `.cpu()` call is a synchronous GPU→CPU transfer. For 27M params with 70+
tensors, that's 70 kernel launches + 70 synchronizations.

**Fix**: Keep SWA state on GPU (it fits), or batch the transfer:
```python
# Keep on GPU:
swa_state[name] += t.detach()  # No .cpu(), stays on GPU

# Or batch transfer at the end:
state_dict = model.state_dict()
cpu_state = {k: v.cpu() for k, v in state_dict.items()}  # One batch
```

## 7. dist.barrier() Doesn't Guarantee Order

**Symptom**: File written by rank 0 isn't visible to rank 1 after barrier.

```python
if rank == 0:
    torch.save(model.state_dict(), "checkpoint.pt")
dist.barrier()
if rank == 1:
    model.load_state_dict(torch.load("checkpoint.pt"))  # File might not exist!
```

**Cause**: `dist.barrier()` synchronizes NCCL operations, not filesystem
operations. On networked filesystems, the file might not be visible yet.

**Fix**: Add a small sleep or use filesystem sync:
```python
dist.barrier()
time.sleep(1)  # Give NFS time to propagate
# Or use dist.broadcast to send the data directly
```

## 8. gradient_clip_norm With Distributed

**Symptom**: Training diverges or is unstable with gradient clipping.

```python
# BUG: Clip before all-reduce — each rank clips its own partial gradient
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
dist.all_reduce(grad)  # Reduced gradient may exceed max_norm

# FIX: All-reduce first, then clip
dist.all_reduce(grad)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
```
