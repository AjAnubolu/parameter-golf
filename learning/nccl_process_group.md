# NCCL Process Group Destroy/Reinit Bug

## The Problem

When doing distributed training on 8xH100, we needed rank 0 to run a long
solo task (pre-quant TTT, ~13 minutes) while ranks 1-7 wait. The naive approach:

```python
# Destroy so ranks 1-7 don't NCCL-timeout while rank 0 works
dist.destroy_process_group()

# Rank 0 does work alone...
if rank == 0:
    run_ttt(model)  # 13 minutes

# Reinitialize and broadcast results
dist.init_process_group(backend="nccl", device_id=device)
for p in model.parameters():
    dist.broadcast(p.data, src=0)  # CRASHES HERE
```

## What Happens

After `destroy_process_group()` + `init_process_group()`:
- NCCL internally creates TCP sockets for rank-to-rank communication
- The old sockets from the destroyed group aren't fully cleaned up
- The new group tries to connect on the same ports
- Result: `socketStartConnect: Connect failed: Software caused connection abort`
- Manifests as either NCCL error or SIGSEGV (signal 11)

## Why NCCL Timeout Forced This

NCCL has a watchdog thread that monitors collective operations. Default timeout
is 600 seconds. If any rank doesn't participate in a collective for >600s, the
watchdog kills the process with SIGABRT.

Our TTT takes ~13 minutes (780s) > 600s timeout. So we can't just have idle
ranks sit there — the watchdog kills them.

## Failed Fix Attempts

### Attempt 1: `dist.barrier()` before broadcast
```python
dist.barrier()  # Doesn't help — NCCL watchdog is independent of barriers
dist.broadcast(p.data, src=0)
```
Barrier itself times out because the watchdog thread runs independently.

### Attempt 2: Set `NCCL_TIMEOUT=3600` env var
```python
# At launch: NCCL_TIMEOUT=3600 torchrun ...
```
Wrong env var name. The actual var is `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, and
even that doesn't reliably prevent the watchdog on all PyTorch versions.

### Attempt 3: Destroy + reinit (current)
```python
dist.destroy_process_group()
# ... rank 0 works ...
dist.init_process_group(backend="nccl")
```
Crashes with socket connection abort as described above.

## Correct Solutions

### Option A: Use Gloo backend for the reinit
```python
dist.destroy_process_group()
# ... rank 0 works ...
dist.init_process_group(backend="gloo")  # Gloo uses different sockets
for p in model.parameters():
    dist.broadcast(p.data, src=0)  # Works over Gloo
# Optionally switch back to NCCL for GPU ops
dist.destroy_process_group()
dist.init_process_group(backend="nccl")
```

### Option B: Run single-GPU work in a subprocess
```python
if rank == 0:
    # Fork a subprocess that imports torch fresh, no dist
    subprocess.run(["python", "run_ttt.py", "--model", "checkpoint.pt"])
    # Load the TTT'd model back
    model.load_state_dict(torch.load("ttt_model.pt"))
dist.broadcast(model.state_dict(), src=0)
```

### Option C: Set heartbeat timeout high enough
```python
# Before init_process_group:
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"  # 30 min
# Then just barrier:
if rank == 0:
    run_ttt(model)
dist.barrier()  # Ranks 1-7 wait up to 30 min
dist.broadcast(...)
```

### Option D: Parallelize TTT across all ranks
```python
# Split val data chunks across 8 GPUs
my_chunks = all_chunks[rank::world_size]
for chunk in my_chunks:
    # Each rank TTT's on its own chunks
    train_on_chunk(model, chunk)
# All-reduce the weight updates
for p in model.parameters():
    dist.all_reduce(p.data, op=dist.ReduceOp.AVG)
```
Best option if TTT can be parallelized — 8x faster, no timeout issues.

## Key Lesson

Never assume you can safely destroy and reinitialize an NCCL process group
within the same process. NCCL manages internal state (sockets, shared memory,
GPU memory) that doesn't fully clean up on destroy. If you need to break out
of distributed mode temporarily, use Gloo, subprocesses, or parallelize the
work across all ranks.
