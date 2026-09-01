# lane/cx-concurrent-prefill — different-prefix saturation

Branch base: `d43f9e27` (serve-ready receipt train).
Receipt rig: box1 `<rented-box-ip>`, 2x RTX PRO 6000 Blackwell Server Edition,
PP-2 devices 0/1 under `/tmp/memra-gpu.lock`.

The requested `~/.lanectl/inbox/cx-concprefill.md` did not exist at lane start.
The adjacent live coordination brief assigns new GPU verification to box1 and
reserves box2 for serving.

## Increment 1 — anatomy harness, no serving-policy change

### Read verdict

The interactive worker has two mutually exclusive long-prompt shapes:

- cross-request fresh-prime batching requires each whole prompt to be no longer
  than `MEMRA_PRIME_BATCH_MAX_T` (default 2048) and to fit inside the current
  interactive tick budget;
- multi-session interactive prefill uses a 1024-token per-request tick budget.

A different-prefix 4096-token burst therefore cannot enter `prime_cache_batch`.
Phase (b) walks every active request's 1024-token `prefill_tick` call serially,
then phase (c) advances the background decode rows. Four requests consume four
serial prime calls before one decode tick is reached.

### Measurement

`concurrent_prefill.py` drives barrier-synchronized, exact 4096-token,
different-prefix requests while background decode streams remain live. Each cell
records:

- aggregate prefill tokens divided by wall time to the last first-visible token;
- per-request TTFT and prompt/cache usage;
- background visible-token rate and per-stream inter-token p95 during the burst.

`run-box1.sh` runs the Step trial config, one warmup, then:

- mixed load: 1/2/4 simultaneous primes, N=3, with four background decode streams;
- prime-only control: four simultaneous primes, N=3.

The server runs with `MEMRA_TTFT_TRACE=1` and `MEMRA_TICK_TRACE=1`.
The latter now reports interactive prefill wall time, serial call/token counts,
batched call/token counts, and the following decode phase time. The trace change is
debug-only and does not alter admission, batching, chunking, kernels, or arithmetic.

Raw logs will live under `raw/box1/`. The anatomy verdict must be committed before
any scheduler or engine behavior changes.

## Increment 2 — box1 anatomy verdict

**VERDICT: CAPACITY-SATURATED.** Different-prefix concurrency does not unlock
additional pair throughput. The pair stays in one approximately 0.58-0.67K tok/s
compute class; it does not approach the earning model's 3K tok/s floor.

The measured multi-session path has a bounded segmentation tax: 1024-token outer
calls deliver 580 tok/s at c=4 versus 674 tok/s for the one-call solo warmup. That
14% gap is fill/drain and per-call geometry overhead, not latent parallel capacity.
Even perfect recovery to the solo class would remain 4.5x below 3K. A larger tick
could trade decode QoS for some of that bounded gap, but cannot change the capacity
verdict.

### Aggregate concurrent prefill, 4096 tokens/request

Box1, Step trial config, N=3 barrier bursts per cell. Mixed cells kept four
background decode streams live. TTFT p95 is across all requests in the three bursts.

| load | simultaneous primes | aggregate prefill tok/s median (range) | TTFT p95 | background visible tok/s | background inter-token p95 |
|---|---:|---:|---:|---:|---:|
| mixed | 1 | **568.2** (565.4-568.4) | 7.245 s | 2.6 | 3.582 s |
| mixed | 2 | **575.3** (574.7-576.0) | 14.255 s | 1.3 | 3.576 s |
| mixed | 4 | **577.6** (576.8-577.9) | 28.404 s | 0.7 | 7.102 s |
| prime-only control | 4 | **580.5** (579.7-580.5) | 28.265 s | — | — |

The c=4 mixed result is only 0.5% below prime-only. Decode competition therefore
does not explain the capacity ceiling; prefill itself owns the wall.

### Serialization location

- Target 4096-token prompts produced zero `[prime-batch]` admissions. The mixed
  server's two batch calls were the four 128-token background prompts coalescing at
  startup, totaling only 512 tokens.
- A typical c=4 prefill tick logged four serial calls, 4096 total prompt tokens,
  7.0-7.1 seconds of prefill wall, then a 27-34 ms decode phase.
- The mixed run recorded 85 serial calls / 90,112 tokens. The prime-only run recorded
  49 serial calls / 53,248 tokens. No target-prompt batch call occurred.
- This follows the code predicate exactly: whole-prompt batch admission requires
  `ql <= 2048` and `ql <= budgets[0]`, while a multi-session interactive tick has a
  1024-token budget.

### Pair utilization and thermal regime

Across the three c=4 mixed bursts, one-second `nvidia-smi` samples (82 per GPU)
showed median GPU utilization of 78.5% on device 0 and 86% on device 1; c=4
prime-only measured 80.5% / 86%. Median power was 280 W / 330 W mixed and
285 W / 335 W prime-only. The stage-1 card is already near continuous work while
stage-0 exposes the bounded pipeline fill/drain bubbles.

Cards were 26 C / 0 MiB at lock acquisition, reached at most 54 C, and returned to
0 MiB before releasing the lock. No OOM, CUDA error, panic, server death, or request
error occurred.

### Direction

Do not build a multi-request prime scheduler on this evidence. Its theoretical
ceiling is the 674 tok/s solo class, not 3K. Closing the earning-tier gap requires
more pairs or a new per-prime compute mechanism; concurrent scheduling can improve
fairness and recover at most the bounded segmentation loss, but cannot supply the
missing 4-5x capacity.

Raw receipts:

- `raw/box1/anatomy-20260808T170819Z.log`
- `raw/box1/mixed-client-20260808T170819Z.jsonl`
- `raw/box1/prime-only-client-20260808T170819Z.jsonl`
- `raw/box1/mixed-server-20260808T170819Z.log`
- `raw/box1/prime-only-server-20260808T170819Z.log`
- `raw/box1/mixed-gpu-20260808T170819Z.csv`
- `raw/box1/prime-only-gpu-20260808T170819Z.csv`
- `raw/box1/client-table-20260808T170819Z.tsv`
- `raw/box1/tick-table-20260808T170819Z.tsv`

## Increment 3 — implementation disposition

No scheduler or engine fix was built. The mission made implementation conditional
on scheduler headroom, and the anatomy measured a capacity-saturated pair:
different-prefix c=4, mixed c=4, and prime-only c=4 all occupy the same aggregate
throughput class. Interleaving or broadening batch admission could recover only the
bounded 1024-token segmentation loss and would not supply the missing 4-5x.

The lane therefore keeps the instrumentation and reproducible harness, records the
negative implementation decision, and proceeds directly to the unchanged-runtime
target-rig battery.

## Increment 4 — final target-rig battery

Box1 ran the required gates against the same branch and Step trial model under
exclusive flock windows:

| surface | production result | canary |
|---|---|---|
| CUDA kernels | `kernel-check` ALL GREEN | n/a |
| PP split | unsplit/serial/pipeline bit-identical and pipeline live | overlap disable broke liveness |
| segmentation chunks | bit-identical at 4096/513/512/256/64 | assertion broke |
| per-tick segmentation | bit-identical across budgets 0/1024/513/512/256/64 and splits 64/256/512 | assertion broke |
| generation | prefill/decode and batched-prime/tokenwise argmax all 6776, MATCH | n/a |
| speculative decode | SELF-CONSISTENCY PASS for K=1..8 | n/a |
| Step B>1 geometry | c=2/c=4 byte-identical to c=1; live B=2 evidence | B=1 re-pin broke batched evidence |

The B=1 geometry canary deliberately emits engine errors and a failing internal
verdict; its wrapper then records `CANARY OK`, proving the gate rejects a disabled
batched path. The combined battery exits 0 with
`=== concurrent-prefill gates PASS`.

Both cards were 0 MiB at the final snapshot and the box1 flock was free after the
run. Raw receipts are under `raw/box1/gates/`, led by:

- `raw/box1/gates/gates-summary-20260808T172104Z.log`
- `raw/box1/gates/kernel-check-20260808T172104Z.log`
- `raw/box1/gates/ppsplit-20260808T172104Z.log`
- `raw/box1/gates/chunkinv35-20260808T172104Z.log`
- `raw/box1/gates/tickinv35-20260808T172104Z.log`
- `raw/box1/gates/run-gen-20260808T172104Z.log`
- `raw/box1/gates/run-spec-20260808T172104Z.log`
- `raw/box1/gates/b2geo35-20260808T172104Z.log`
