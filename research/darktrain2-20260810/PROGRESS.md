# darktrain phase 2 — real training consumer (2026-08-10)

Branch: `lane/cx-darktrain2`  
Base: `be0b0240c222f78ec7c099078e7786778083021f`  
Rig: box1, 2x RTX PRO 6000 Server Edition, all GPU work under one bounded
`flock /tmp/memra-gpu.lock` hold per experiment block.

## Goal

Compose memra's v1 background-job runner with a real PyTorch optimizer workload while the
promoted PP-2 Step-3.7 serving configuration is resident. Measure the VRAM launch contract,
serve QoS in absent/running/parked cells (N=3 interleaved), SIGSTOP/SIGCONT behavior,
checkpoint durability, and a genuine `refused_vram` decision.

Serving configuration is pinned to:

```text
/home/ubuntu/step37/models/step-3.7-flash IQ4_XS+MTP
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
MEMRA_CTX=262144
MEMRA_MOE_GROUPED=1
MEMRA_PREFILL_TICK=2048
```

## Initial contract questions

- The runner checks free VRAM only before launch. Does importing PyTorch allocate CUDA memory,
  and does the real job remain within its declared peak after CUDA initialization and optimizer
  state creation?
- Does `PYTORCH_CUDA_ALLOC_CONF`/`PYTORCH_ALLOC_CONF` change the measured reservation or only
  fragmentation behavior for this stable-shape workload?
- Does process-group SIGSTOP stop optimizer progress quickly, and does it retain the CUDA context
  and allocations as shown by `nvidia-smi`?
- Can an atomic model+optimizer+step checkpoint survive checkpoint-mode preemption and resume at
  the next valley?
- Does an intentionally impossible budget reach `refused_vram` without launching the trainer?

## Status

- The required first commit landed as `7f26a9a4`; the real PyTorch trainer and box1 harness
  landed in `409493bd`, with its measured 19 GiB budget in `694bff65`.
- Read the predecessor lane, the train-loop pilot recipe, and
  `~/projects/darklanes/sft-pipeline/`. The private corpus stayed read-only and the seam job used
  synthetic tensors; no training data left box1.
- Installed PyTorch 2.11.0+cu128 in an isolated box1 venv. Import and
  `torch.cuda.is_available()` allocated no GPU memory; first CUDA context creation appeared as
  676 MiB in `nvidia-smi` while PyTorch reported only 2 MiB reserved.
- Ran two allocator arms under one GPU-lock hold. The 16 GiB frozen bank plus real rank-16 LoRA
  forward/backward/AdamW loop peaked at 18,230 MiB by NVML with the native allocator and
  18,250 MiB with `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`; the option did not help.
- Started the exact promoted PP-2 Step-3.7 IQ4_XS+MTP serving shape for the QoS block. Rep 1
  absent was 8/8 byte-identical. Rep 1 running returned 8/8 HTTP successes but one of eight
  deterministic completions differed from the absent golden. The harness emitted exit 86,
  stopped both processes, verified both GPUs empty, and released the lock.
- Per the explicit P0 stop rule, the remaining N=3 cells, normal valley resume, real
  checkpoint/relaunch, and fresh `refused_vram` block were not run. A direct five-step trainer
  smoke did produce an atomic 797,357-byte checkpoint, but that is not a runner resume proof.
- Raw receipts and their manifest landed in `a1dfc679`. The final bounded verdict is in
  `RESULTS.md`.
