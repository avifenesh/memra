# Q35 carried-prime exactness hotfix and Q27 hit-tail fence

## Verdict

Both shipped regressions are fixed at source `065a705d63461b2e3a7d945f824024fe0c482886`:

- Q35 completed all 80 scored cells and all 2,300 requests at exactly 60 tokens. The frozen
  c=4 cold+mixed slice is 200/200 exact across N=5, all 40 required base cells are clean, and
  routed-MoE emitted zero carried prime batches.
- Q27 completed all 70 scored cells and all 1,400 requests at exactly 60 tokens. Frozen mixed
  c=4 cache-hit TTFT is p50 18.497 ms / p95 19.820 ms, restoring the sold 18.573/21.565 ms
  envelope from the broken 269 ms tail. Its clean throughput knee is c=16, up from c=12.
- Both models pass the standard target-rig correctness gates, serial-cache exactness, accounting,
  and every frozen sellability criterion. No admission defer, VRAM defer, step-OOM park, server
  fatal marker, or post-run GPU process was captured.

The frozen reducer nevertheless returns **`P0_REGRESSION`** because it treats any old/new
envelope decrease as a failure: Q27 mixed-c4 output is 144.245 versus 144.462 tok/s
(`-0.217 tok/s`, `-0.150%`). This is the only reported regression; Q27 and Q35 are individually
`SELLABLE`. The new median lies inside the old run's 143.626--144.637 tok/s raw range, and the
comparison is not a same-window interleaved A/B, so this evidence does not support a causal
throughput-regression claim. It also does not waive the frozen reducer's red verdict.

The release class is **v0.81.1 patch**, not v0.82.0: this is a narrow repair for defects shipped
in v0.81.0, with no mechanism/board promotion. Do not merge or tag yet. The orchestrator must
resolve the strict `-0.150%` comparison with a same-window decision and still run the required
Vast 2x RTX PRO 6000 pre-release battery. This lane did not merge, tag, or push.

## Frozen local reproduction

- Model: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, SHA-256
  `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`.
- Workload: frozen sellgate mixed90 c=4 cell: 20 requests, 18 full-cache hits, two cold misses,
  4,860 prompt tokens/request, and exactly 60 requested completion tokens.
- Harness: `tools/q35-cold-mixed-gate.py`, which imports the frozen sellgate replay and refuses
  workload-shape drift before running the cell.
- Rig: local RTX 5090 under both GPU coordination locks and the lane's 210--1200 MHz thermal cap.
  This is correctness-only evidence; no local throughput claim is made.

| Arm | Observed prime path | Cold-miss result | Cell |
|---|---|---|---|
| v0.81.0 naked default | one fresh B=2 batch, then four B=2 calls with carried rows | both 26/60, `finish_reason=stop` | FAIL |
| full rollback, `MEMRA_PRIME_BATCH=1` | serialized primes | both 60/60, `length` | PASS |
| forced complete-fresh control | one B=2, 9,720-token batch, `carried=0 partial=0` | both 60/60, `length` | PASS |
| fixed naked default | complete-fresh remains eligible; routed-MoE carried chunks serialize | both 60/60, `length`; zero carried batch lines | PASS |

The baseline returned 1,132 output tokens: 18 hits at 60 plus two misses at 26. The fixed cell
returned all 1,200. Client usage and engine counters agree in both runs; neither records an
admission defer, VRAM defer, or step-OOM park.

## Fault boundary and mechanism

The coldhol scheduler change broadened batch candidates from complete fresh prompts to one
bounded cold continuation chunk per eligible session, then repeatedly called
`prime_cache_batch`. In the failing run, the two cold sessions traversed:

```text
B=2 tokens=2048 carried=0 partial=2
B=2 tokens=2048 carried=2 partial=2  # three calls
B=2 tokens=1528 carried=2 partial=0
```

Those calls cover both 4,860-token prompts exactly. The same requests pass when all batches are
disabled and when a real complete-fresh B=2 batch is retained but continuation calls are removed.
The demonstrated failure boundary is therefore the repeated carried concat-prime program on Q35
routed MoE, not prime batching in general.

The requests contain no stop strings. `advance_sample_emit` appends the sampled token, checks it
against the model's EOS set, and finishes `StopReason::Eos`; budget exhaustion would instead
produce 60 tokens and `length`. `tok-check` proves id 248046 is `<|im_end|>` and the server
declares the same id as EOS. The constant 26th generated token was therefore a real, falsely
selected EOS--not a 26-token scheduler boundary or HTTP truncation. The failed requests were
distinct newly admitted `n_cached=0` sessions, so shared stop-state or continuation-slot reuse is
not required. Exact prompt sums and reconciled counters provide no stale-position symptom.

This evidence identifies the faulty scheduler/engine composition. It does not claim a deeper
floating-point primitive defect. The immediate safety boundary is to withhold carried routed-MoE
batches until a dedicated gate covers repeated real-size chunks followed by the serving
batched-decode trunk.

The Q27 tail has a separate scheduler boundary. In the broken run, five >100 ms full-cache hits
arrived while the worker was synchronously priming an unrelated cold 4,860-token request; the
replacement request had not crossed the HTTP/channel boundary before the worker entered prime.
That made one cold-prime call part of cache-hit TTFT despite the hit needing no prefill.

## Fix

- `carried_prime_batch_eligible()` keeps dense architectures eligible and rejects routed-MoE
  through `Arch::is_moe()`. The predicate applies only to bounded carried `cold_chunk`
  candidates; complete-fresh batching is unchanged.
- A fully restored cache hit that has emitted no token fences unrelated interactive cold prefill
  until its first decode. When a completed request opens a client-concurrency slot, the existing
  4 ms batch-formation window also acts as bounded refill grace so the replacement can cross the
  HTTP/channel boundary before another synchronous cold prime starts.
- Cold-only traffic and all later hit tokens keep the existing ordering. Dense Q27 continuation
  batching remains active; the box1 campaign counted 825 Q27 prime-batch calls and zero for Q35.
- Unit tests enumerate current routed-MoE versus dense eligibility and pin the fully-cached,
  unemitted first-token predicate.
- `serve-smoke` runs the frozen Q35 mixed-c4 cell, requires 20/20 exact 60-token completions, and
  rejects any routed-MoE `carried>0` prime-batch log.

## Box1 N=5 requalification

The scored run used one physical RTX PRO 6000 Blackwell Server Edition GPU on provisioned Sbox
box1, one model resident at a time, with odd boots ordered Q27/Q35 and even boots Q35/Q27. The
single-GPU lock ran from 18:21:54Z to 19:15:28Z. The observed thermal regime peaked at 68 C,
2,422 MHz, 525.13 W, and 77,845 MiB used. Every median below is N=5.

| Metric | Q27 | Q35 |
|---|---:|---:|
| clean scored cells | 70/70 | 80/80 |
| exact 60-token requests | 1,400/1,400 | 2,300/2,300 |
| required base cells | 40/40 clean | 40/40 clean |
| mixed-c4 hit TTFT p50 / p95 | 18.497 / 19.820 ms | 7.418 / 10.260 ms |
| mixed-c4 output | 144.245 tok/s | 404.810 tok/s |
| clean mixed-throughput knee | c=16 | c=40 |
| capacity headroom over sold c=4 | 300% | 900% |
| prime-batch calls | 825 | 0 |

Q27's median mixed path is 144.245 at c4, 179.143 at c8, 185.568 at c12, 187.799 at c16,
then 185.960 at c20. Q35 rises from 404.810 at c4 through 491.542 at c16, 511.059 at c32,
and 515.732 at c40 before declining to 502.520 at c48.

The strict Q27 c4 comparison is fully visible rather than summarized away:

| Repetition | Hit TTFT p95 | Output tok/s |
|---:|---:|---:|
| 1 | 18.973 ms | 147.642 |
| 2 | 20.081 ms | 143.609 |
| 3 | 19.172 ms | 144.245 |
| 4 | 22.640 ms | 143.598 |
| 5 | 19.829 ms | 144.328 |

Pooling the 90 c4 hits yields p50 18.497 / p95 19.820 ms. The frozen reducer compares the
five-cell output median, 144.245, to the older campaign median, 144.462, and records the sole
`P0_REGRESSION` entry (`-0.150%`).

The remote checkout was detached at the exact source with an empty status. Its stale `target/`
directory was removed before this work and rebuilt from source; it is recoverable by rebuilding
and no source or research evidence was removed. Fresh binary hashes include server
`3b8ec4ed328dc4f1ddaae3b4f170510435c98a1d0c6ce84b0021fa60bda5976e`, kernel-check
`efcca6fce71027e3ba8588c89c5e067b291db8390c60f365da0c21424ad4cae1`, run-gen
`1102f6fe307c20c64a043ea16241228f8f2a16ed39aa25afffd3ec51821d296b`, and run-spec
`0a4593ac760ab9b39dd863a0877e9c1903fdd570899457c5395aa21eb197e8d2`.

## Gates

| Gate | Result | Receipt |
|---|---|---|
| Final-tree fresh local release build | PASS, CUDA 13.1 / sm_120a | [`build.log`](raw/battery-r3/build.log) |
| Full local `cargo test` | PASS, aggregate 441 passed / 0 failed / 2 GPU-explicit ignores | [`cargo-test.log`](raw/battery-r3/cargo-test.log) |
| Local `kernel-check`, both manifests | PASS, `ALL GREEN (106 cells, 1 skipped)` | [`kernel-check.log`](raw/battery-r3/kernel-check.log) |
| Local Q27/Q35 `run-gen` | PASS, prefill/decode and batched-prime/tokenwise argmax MATCH | [`Q27`](raw/battery-r3/run-gen-q27.log), [`Q35`](raw/battery-r3/run-gen-q35.log) |
| Local Q27/Q35 `run-spec` K=1..8 | PASS, 8/8 each | [`Q27`](raw/battery-r3/run-spec-q27.log), [`Q35`](raw/battery-r3/run-spec-q35.log) |
| Local `serve-smoke` | PASS, 0 failed; Q35 20/20 exact and carried path absent | [`serve-smoke.log`](raw/battery-r3/serve-smoke.log) |
| Local c=64 serve stress | PASS, 64/64 well formed, worker alive, log clean | [`serve-stress.log`](raw/battery-r3/serve-stress.log) |
| Box1 standard correctness | PASS, kernel + both-model run-gen + both-model K=1..8 | [`gates/`](raw/box1-accept/gates/) |
| Box1 serial-cache exactness | PASS for both models | [`exactness/`](raw/box1-accept/exactness/) |
| Box1 frozen N=5 | all 150 cells clean; 3,700/3,700 exact; 0 short | [`analysis.json`](analysis.json) |
| Box1 shutdown/seal | `REQUAL2_COMPLETE`, `DRIVER_EXIT rc=0`, GPU/ports/lock clear | [`orchestrator.log`](raw/box1-accept/orchestrator.log) |

The first local battery attempt is retained under `raw/battery/`: build/tests passed, but an
incorrect authoritative model-directory override caused required kernel cells to fail closed.
The corrected final-tree run is `raw/battery-r3/`.

The first box1 attempt is retained under `raw/box1-attempt1/` and excluded from scoring. It
passed all standard gates plus Q27 rep 1 and Q35 rep 1, then stopped because the upstream driver
required every model to report a positive prime-batch count. That assertion was stale once Q35
carried batches were intentionally disabled: Q27 counted 168, Q35 counted 0, and both verdicts
were PASS. A sealed one-hunk driver copy preserves the Q27 positive-count requirement and instead
requires Q35 to have zero `carried>0` lines. Original driver hash is `e3a65b...cedd`, patched
driver `de4442...6492`, and patch `59a0e5...9526`.

## Evidence seals and release boundary

- Final local battery manifest: `d3df2fe39dc9ced8aa3b17c642347544942ad73a3265dba98448c1744c05517c`.
- Complete box1 campaign manifest: `5804c57af75bd5b738a77a0ba175eb92f4e2684da1fe3b93f058b18ab2d8b727`.
- Frozen analysis SHA-256: `092c8e42489009984c363afe992482f4ef59799797ada00ed4d11a0eea41c2e3`.
- Lane-wide raw manifest: `263d3db1f02b0efce19532936b6104a47841165c4f43083f36e847070d3e3153`.

The Sbox/PRO 6000 evidence is research qualification, not the repository's pre-release shipping
gate. Before merge or tag, run `kernel-check` ALL GREEN, affected-model `run-gen` argmax MATCH,
and `run-spec` K=1..8 PASS on the designated Vast 2x RTX PRO 6000 verification box. The strict
Q27 output comparison also remains an explicit orchestrator decision; no tag is authorized by
this lane.
