# Darktrain phase 2: real-consumer results (2026-08-10)

## Verdict

**NO — the darklane is not usable for real training on the serving pair today.** Keep
`MEMRA_BG_JOB` opt-in and off for production serving.

The first co-located QoS cell violated the byte-exactness invariant. With the promoted PP-2
Step-3.7 configuration, all eight train-absent requests returned the same 326-byte completion.
With the real PyTorch optimizer running at admission, seven requests returned that completion
and one returned a different 310-byte completion. All requests were HTTP successes and both
outputs reached the 64-token length limit; the failure is content, not transport. The harness
returned 86, shut down the trainer and server, verified both GPUs at 0 MiB, released the lock,
and did not run another GPU cell.

This establishes an operational P0 under co-location, not its physical cause. There was no
captured CUDA error, OOM, panic, or worker death. The stopped CPU process may still have had
previously queued GPU work, but the receipt does not distinguish that from a batching,
prefix-cache, or fresh-boot determinism defect. That isolation belongs in a new lane; the strict
serving gate fails regardless.

## Scope and provenance

- Rig: box1, two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, 97,887 MiB each. Each
  experiment block held `/tmp/memra-gpu.lock` for its full lifetime.
- Serving artifact:
  `/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf`
  plus `/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf`.
- Serving shape: `MEMRA_PP_STAGES=2`, `MEMRA_PP_DEVICES=0,1`, `MEMRA_CTX=262144`,
  `MEMRA_MOE_GROUPED=1`, `MEMRA_PREFILL_TICK=2048`; speculative decode variables were unset.
  PP-2 placement selected K=0 as designed.
- Server source commit: `188154299064a42b67fc8eb1f41757cf6237300d`; binary SHA-256:
  `e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3`.
  The current branch has no runtime-source diff from that commit in `Cargo.toml`, `Cargo.lock`,
  `build.rs`, or `crates/`.
- Trainer: PyTorch 2.11.0+cu128, CUDA 12.8, isolated box1 venv. The real seam workload held a
  16 GiB BF16 frozen bank, ran four frozen 4096x4096 projections with rank-16 adapters, and
  executed BF16 forward, backward, and AdamW steps on GPU 0. It is a synthetic LoRA-class
  systems workload, not a scored model-quality arm.
- The private `~/projects/darklanes/sft-pipeline/` corpus was read only. Synthetic tensors and
  checkpoints remained on box1; no darklanes-repo files were modified.

## P0 exactness receipt

Requests used the same prompt, `temperature=0`, `seed=3407`, streaming, a 64-token limit, and a
barrier release at concurrency 8.

| Cell | HTTP success | Completion hashes | Byte result |
|---|---:|---|---|
| rep1 absent | 8/8 | 8 x `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` | PASS; 326-byte golden |
| rep1 running | 8/8 | 7 x golden; 1 x `7a5032f2d723e3cf9ef788fdc9d4067fe2eb909157189b666430b7997a56961f` | **P0 FAIL**; divergent result is 310 bytes |

The differing suffixes make the failure visible rather than hash-only:

```text
golden:    ... prevents interactive requests from missing their latency targets, since the operator is measuring that.
divergent: ... prevents queuing delays that would break user-facing response time targets. Wait, make that
```

The divergent request's first token arrived 158.224 ms after barrier release. The trainer was
first observed in `/proc` state `T` at 144.609 ms, 13.615 ms earlier. Process stop therefore did
not make this admitted request byte-identical; `/proc T` is not evidence that another CUDA
context has no queued work.

Raw receipts: [`qos/driver.log`](raw/qos/driver.log),
[`rep1-absent/qos-rows.jsonl`](raw/qos/rep1-absent/qos-rows.jsonl),
[`rep1-running/qos-rows.jsonl`](raw/qos/rep1-running/qos-rows.jsonl), and both cell server logs.

## VRAM-budget contract edges

The runner's contract is a launch admission check, not runtime containment. The child must report
and obey an honest peak. NVML/process memory is the relevant measurement: PyTorch allocator
statistics omit CUDA-context and other non-allocator memory.

| Edge | Observation | Outcome |
|---|---|---|
| Import | `import torch` took 0.810 s, left `torch.cuda.is_initialized()==false`, and created no NVML GPU process. | Held: import did not grab the declared budget. |
| Availability probe | `torch.cuda.is_available()` returned true while CUDA remained uninitialized and no process memory appeared. | Held. |
| First CUDA context | PyTorch reported 0 MiB allocated / 2 MiB reserved; NVML reported 676 MiB for the process. | Important edge: budget checks based only on `memory_reserved()` undercount by about 674 MiB immediately. |
| Launch admission, 18 GiB arm | With serve resident, minimum free memory was 40,914 MiB; the 18,432 MiB declaration plus 2,048 MiB serving headroom fit. The runner launched once. | Held at launch. |
| Runtime, native allocator | Peak NVML was 18,230 MiB; peak PyTorch reserved was 17,538 MiB; peak allocated was 16,826.251 MiB. No child audit violation across 515 steps. | Held, but the 18 GiB declaration had only 202 MiB NVML margin. |
| Runtime, 19 GiB QoS declaration | The trainer reached 18,230 MiB under a 19,456 MiB declaration before the P0 cell. | Held with 1,226 MiB observed margin. |
| Deliberately low 2 GiB smoke declaration | At step 3, NVML reached 2,054 MiB while PyTorch reserved only 1,362 MiB. The child emitted `budget_violation` and exited 72; the runner itself would not have stopped the overshoot. | Contract edge exposed: honest child audit works; runner containment does not exist. |
| `refused_vram` | The planned impossible-budget block was after QoS and was not run once the P0 stop fired. | **Not re-proven in this lane.** The 2 GiB child violation is not a `refused_vram` receipt. |

### Allocator arm

Both allocator arms were fresh server boots in one lock hold. Values below are single arms, not
cross-run medians. Optimizer p50 is over the logged step samples (every fifth step plus step 1).

| Arm | Logged steps | NVML peak | PyTorch reserved peak | Inactive-split peak | Optimizer step p50 |
|---|---:|---:|---:|---:|---:|
| native default | 104 through step 515 | 18,230 MiB | 17,538 MiB | 39.749 MiB | 1.1865 ms |
| `max_split_size_mb:128` | 105 through step 520 | 18,250 MiB | 17,558 MiB | 39.749 MiB | 1.1860 ms |

`max_split_size_mb:128` did not help this stable-shape job: NVML and reserved peaks were each
20 MiB higher, inactive-split peak was unchanged, and optimizer step time was flat. Current
[PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html) describe
`PYTORCH_ALLOC_CONF` as the canonical name (`PYTORCH_CUDA_ALLOC_CONF` is the compatibility
alias) and `max_split_size_mb` as a native-allocator fragmentation control for borderline OOMs,
not a process-memory limit. It cannot implement the darklane budget contract.

The practical contract must therefore use all three of these:

1. A child readiness handshake after CUDA initialization and at least one representative full
   optimizer step, carrying an NVML-observed peak and a conservative margin.
2. Runtime monitoring of actual per-process GPU memory, with a loud policy action on overshoot.
   A cooperative PyTorch `per_process_memory_fraction` limit can be defense in depth, but the
   runner must still account for context/non-allocator memory.
3. A workload-specific, frozen budget established from repeat peaks. Allocator tuning is not a
   substitute for containment.

Raw receipts: [`03-training-smoke.log`](raw/03-training-smoke.log),
[`04-training-smoke-pass.log`](raw/04-training-smoke-pass.log), and
[`allocator/`](raw/allocator/).

## Serve QoS impact

The requested design was N=3 cells per arm in a balanced interleave. The P0 stop ended the block
after 2 of 9 cells, so these are partial single-cell observations and **not campaign medians**.
Each observed cell used eight concurrent requests; p99 is nearest-rank and therefore the maximum
of eight. Both cells were fresh boots in the same lock hold. GPU temperatures during sampling
were 31-38 C absent and 32-39 C running.

| Condition | Cells / requests | TTFT p50 | TTFT p99 | End-to-end p50 | End-to-end p99 | Server step p50 | Exactness |
|---|---:|---:|---:|---:|---:|---:|---|
| train absent | 1 / 8 | 160.272 ms | 385.143 ms | 3.346391 s | 3.393323 s | 49.289 ms | 8/8 match |
| train running at admission | 1 / 8 | 190.252 ms (+18.706%) | 416.029 ms (+8.019%) | 3.382954 s (+1.093%) | 3.429976 s (+1.080%) | not scraped | **1/8 differs** |
| train parked before admission | 0 / 0 | not run | not run | not run | not run | not run | not run |

The absent server-step statistic was scraped after the warmup plus measured burst. The running
probe returned the P0 exit before the post-cell metrics scrape, so inventing or comparing its
step p50 would be invalid. No QoS acceptance claim can be made from this incomplete N.

## Park, resume, and memory release

| Check | Receipt | Result |
|---|---|---|
| Busy edge to stopped process | Request barrier release to `/proc` state `T`: 144.609 ms, N=1. | The process stopped within the predecessor's 500 ms bound, but this is end-to-end polling latency, not raw `kill(SIGSTOP)` syscall latency. |
| Optimizer progress while stopped | Last pre-stop log: step 65 at 01:51:55.313 UTC. Next log: step 70 at 01:51:58.694, only after shutdown cleanup sent CONT+TERM; then `term_exit`. | Park halted user-space stepping. Normal valley resume was not observed. |
| VRAM while stopped | Before admission: trainer 18,230 MiB, server 46,154 MiB on GPU 0. After state `T`, total GPU-0 use remained 64,620 MiB in every 500 ms sample from 01:51:55.679 through 01:51:58.179. | **SIGSTOP releases no framework VRAM.** |
| Shutdown cleanup | Final `nvidia-smi`: 0 MiB used and no compute process on either GPU. | No stopped orphan remained. |

The 16 GiB bank is live tensor memory, so allocator cache flushing cannot reclaim it. Even for
unused cache, current [`torch.cuda.memory.empty_cache()` documentation](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html)
says only unoccupied cached memory is released for other applications; it does not enlarge
PyTorch's own usable pool. A SIGSTOPped process cannot cooperatively call it anyway. If a valley
must return VRAM rather than only stop compute submission, the usable mechanism is
checkpoint-and-exit (or process termination), not stop-mode parking.

### Checkpoint durability

The direct five-step real trainer smoke atomically wrote a 797,357-byte adapter + AdamW + step/RNG
checkpoint, exited, and the file re-hashed as
`5f6aa7e3dc9d9f00f0d761938705a7e6fec121c11429a2b4f16d2df090311709`.
That proves the real consumer's writer survives process exit. It does **not** prove runner
preemption, reload, or resume-past-checkpoint: the planned checkpoint-mode cell was not run after
the P0. The fresh `refused_vram` arm was likewise not run.

## Required changes before another real-training trial

1. Preserve the production default: no background trainer unless explicitly configured.
2. Isolate the exactness failure with absent A/A, already-parked, and running controls at c=1 and
   c=8, holding prompt, seed, cache state, and fresh-boot order constant. Capture output bytes on
   every request as this harness does.
3. Do not dispatch an interactive request merely because the trainer's CPU process reached `T`.
   Add a cooperative trainer safe point that synchronizes its CUDA work and acknowledges
   quiescence before serving, or use checkpoint-and-exit and wait for process/VRAM disappearance.
   Measure the acknowledgement path; do not infer quiescence from `/proc` state.
4. Add the post-initialization/full-step budget handshake and runtime NVML guard described above.
5. Only after exactness is green, complete the balanced N=3 absent/running/parked QoS block, the
   real checkpoint/relaunch receipt, and the impossible-budget `refused_vram` receipt.

## Evidence index

- Machine, source, package, and model inventory: [`00-box1-inventory.log`](raw/00-box1-inventory.log),
  [`01-box1-python-repos.log`](raw/01-box1-python-repos.log), and
  [`02-box1-venv-setup.log`](raw/02-box1-venv-setup.log).
- Machine-readable reduction: [`analysis-summary.json`](raw/analysis-summary.json), which records
  `campaign_complete=false`, 2 observed of 9 expected cells, `all_exact=false`, and null
  checkpoint/refusal results.
- QoS/P0 driver: [`06-qos-driver.log`](raw/06-qos-driver.log).
- All receipt hashes: [`SHA256SUMS`](raw/SHA256SUMS); `sha256sum -c` passes 56/56 files.

No public performance board moves from this incomplete, failed campaign.
