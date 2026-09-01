# cx-requal2 progress — 2026-08-12

## Scope

Re-sweep the frozen Q27 and Q35 sold envelopes on the shipping naked-default
scheduler from `main` at `584ed0af05e5f8a29b318a410a8e76ed8a08292f` or
newer. Re-run the unchanged 40-cell qualification battery for both models and
an interleaved N>=5 knee sweep at Q27 c=4,8,12,16,20 and Q35 c=4,16,32,40,48
under one `/tmp/memra-gpu.lock` hold on box1.

## Gates

- [x] Read `CLAUDE.md`, the lane inbox, the old sealed requalification, and the
      frozen harness machinery.
- [x] Confirm a clean dedicated `lane/cx-requal2` worktree at `584ed0af0`.
- [x] Re-check the lane inbox before remote setup and each major phase.
- [x] Verify box1, its single-card GPU state, source revision, artifacts, and
      frozen workload inputs without touching the GPU outside the flock.
- [x] Fresh-build the shipping runtime from clean `main @584ed0af0+`.
- [x] Run the unchanged Q27 and Q35 40-cell qualification batteries.
- [x] Run the extended model-specific knee grids interleaved N>=5 under the
      same uninterrupted GPU-lock hold.
- [x] Retrieve and verify raw JSONL/log receipts and manifests locally.
- [x] Reduce hit TTFT p50/p95, mixed throughput, knee, and c=4 headroom for each
      model; diff every sold-envelope value against the old sealed result.
- [x] Record any regression as a P0; otherwise enumerate stale submission
      surfaces and exact follow-up deltas without editing those surfaces.
- [x] Commit `RESULTS.md`, raw JSONL, and complete lane evidence locally. Do not
      merge, tag, push, update boards, format, or bypass hooks.

## Log

- 2026-08-12T15:57:06Z — Started from a clean dedicated worktree on
  `lane/cx-requal2` at `584ed0af05e5f8a29b318a410a8e76ed8a08292f`.
- 2026-08-12T15:57:06Z — Read the owner brief and repository laws. The prior
  envelope remains immutable measurement history; this lane will publish a new
  result and raw evidence only. No remote GPU work has started.
- 2026-08-12T16:00:16Z — Box1 is reachable. A five-second flocked preflight
  found physical GPU0 idle at 0 MiB / 0% / 27 C; GPU1 was also idle but is out
  of this lane's run shape. The four pinned Q27/Q35 target and draft artifacts
  remain staged on local NVMe at their frozen byte sizes.
- 2026-08-12T16:11:41Z — The first SSH attempt offered agent keys before the
  named key and was rejected without login; all subsequent calls pin
  `IdentitiesOnly=yes`. No GPU command ran outside the flock.
- 2026-08-12T16:11:41Z — `origin/main` advanced during setup from `584ed0af0`
  to `e78054f5f` via the OpenRouter-submission documentation merge. The staged
  build source will be that current `584ed0af0+` head unless a newer head is
  observed before transfer.
- 2026-08-12T16:11:41Z — Added a hash-pinned adapter around the unchanged frozen
  replay. The fixed interpretation is GPU0 only, one model resident at a time,
  odd/even repetition boot-order reversal, exactly 40 base cells per model at
  c=1,2,4,8, and both frozen cold/mixed90 arms at every requested extension:
  Q27 c=12,16,20 and Q35 c=16,32,40,48. Local shell syntax, Python compilation,
  frozen hashes, workload parsing, and rotation-order checks pass.
- 2026-08-12T16:17:23Z — Fresh release build from detached, clean shipping
  `main @e78054f5fec808703d050a5d9545f2ac2cc162cb` passed on box1. The checkout
  began without `target/`; installed Rust 1.97.1 and CUDA 13.2 produced
  `kernel-check` `93715901...3f7`, `run-gen` `eb1eda21...800`, `run-spec`
  `24042b52...d59`, and `memra-server` `2ab01ba5...61a`.
- 2026-08-12T16:18:40Z — The detached all-phase driver acquired
  `/tmp/memra-gpu.lock` immediately and began the GPU0 correctness battery. It
  will retain this one hold through all ten model boots; GPU1 remains unused.
- 2026-08-12T16:21:42Z — The single-card exactness battery passed: kernel-check
  `ALL GREEN (95 cells, 13 skipped)`; Q27 and Q35 each passed prefill/decode and
  batched-prime/tokenwise argmax MATCH plus all eight K=1..8 self-consistency
  rows. Both serial cache gates subsequently reconciled exactly 27,702 cached
  tokens across client usage and both engine counters.
- 2026-08-12T16:28:35Z — Q27 repetition 1 completed 14/14 clean cells. Its
  single-boot mixed path was 186.161 -> 186.816 -> 186.297 output tok/s at
  c=12/16/20, a first-decline knee of c=16, with 169 continuation prime-batch
  calls captured. N=1 is not the campaign verdict.
- 2026-08-12T16:31:36Z — **P0: Q35 failed the new naked default.** Repetition 1
  produced only 6/16 clean cells. The first required failure was mixed c=4:
  both cold misses returned HTTP 200 / `finish_reason=stop` at 26/60 tokens,
  leaving 18/20 requests and 1,132/1,200 output tokens. Short completions also
  appeared at c=8,16,32,40,48. Cached/prompt counters still reconciled and no
  OOM/admission event explains the truncation.
- 2026-08-12T16:33:05Z — The fail-fast attempt released the flock cleanly with
  GPU0 empty; its complete round-1 raw evidence is sealed under manifest
  `bd6b819e3e784af1b59b960515866f52346e067dfc7478206889738e76e55bf2`.
  The recorder is being changed only to retain failed verdicts while continuing
  later repetitions; no workload, prompt, arm, width, or runtime setting moves.
- 2026-08-12T16:35:00Z — The complete evidence campaign acquired one outer
  `/tmp/memra-gpu.lock` hold after repeating the entire exactness battery. The
  inbox later acknowledged the P0 and assigned this matrix as the separate
  coldfix lane's acceptance evidence; this lane remained evidence-only.
- 2026-08-12T17:26:08Z — The campaign completed all ten model boots under the
  same lock. Q27 passed 70/70 cells. Q35 failed every repetition, with 34/40
  base and 41/80 total cells clean; 714/2,300 responses stopped at 26/60 tokens.
  GPU0 was empty and the lock was free after exit.
- 2026-08-12T17:29:00Z — The remote full manifest verified all 111 entries and
  hashed to `bbae0cf4d0861e4534411254db17958717bc9089ac78d84ed99266e41a9ea76b`.
  The local rsync copy verified byte-for-byte and contains all ten replay JSONL
  files plus correctness, exactness, server, system, and thermal logs.
- 2026-08-12T17:31:00Z — Frozen reduction emitted `P0_REGRESSION`: Q27 remained
  clean and confirmed knee c=16, but c=4 hit TTFT p95 moved 21.565 -> 269.139 ms;
  Q35 is not qualified at c=4. All 714 Q35 short responses were HTTP 200,
  `finish_reason=stop`, exactly 26 tokens; no admission defer, OOM park, or
  accounting drift was observed.
- 2026-08-12T17:34:00Z — Refreshed `origin/main` and live BitRouter PR #814.
  OpenRouter is already submitted with the old 40/40 narrative; Surplus/Onlist
  retained earlier capacity claims; BitRouter remains open at `6e4729e23756`
  and its manifest has no numeric capacity fields. These are reported as
  follow-ups only. No public or runtime surface was mutated.
- 2026-08-12T17:34:11Z — Report-to-analysis assertions, all 111 manifest
  checks, Python compilation, shell syntax, shellcheck, perf-board drift, and
  diff whitespace checks passed. The intended evidence tree contains ten
  scored replay JSONLs and no nsys artifact.
