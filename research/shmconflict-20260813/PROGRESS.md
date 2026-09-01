# Prefill FA shared-memory conflict progress

Date: 2026-08-13
Branch: `lane/cx-shmconflict`
Base: `v0.81.3` (`7cf5fd842ebc76f6e8a82910a8e6d4b864b6b42d`)
Rig: local RTX 5090 Laptop GPU, owner-imposed 210--1200 MHz thermal cap

## Objective

Localize the replay-heavy shared-memory instructions in the frozen cached-prefill
`fa_prefill_qw_db` serving path, then test the smallest bit-identical layout fix. A useful
negative result still requires a committed per-instruction attribution table.

## Required evidence read before implementation

- Predecessor profile: `main@055adf47d:research/fa3softmax-20260813/PROFILE.md`, especially the
  frozen 4,860-token request and the four 6.42--6.45x shared-wavefront measurements.
- Predecessor verdict: `main@055adf47d:research/fa3softmax-20260813/RESULTS.md`; its scheduling
  candidate lost on both models and was removed, so this lane does not reopen scheduling.
- Kernel primitives and base loop: `crates/memra-engine/cu/flash_attn.cu:103-145` and
  `:1105-1255` at the lane base.
- Cached/chunk implementation: `crates/memra-engine/cu/flash_attn.cu:4732-4908` at the lane base.
- Dispatch and dynamic-shared-memory calculation:
  `crates/memra-engine/src/lib.rs:9575-9583` and `:9627-9658` at the lane base.

## Starting source hypothesis, not attribution

The target body currently stages Q, K, V, and P row-major and calls the unswizzled `ld_A` /
`ld_A_trans` helpers. Existing XOR-swizzled helpers elsewhere in the same translation unit prove
that a layout-only mechanism is available, but source shape alone does not identify which target
instruction dominates the measured replay. Per-PC NCU/SASS attribution must decide among the
one-time Q loads, per-tile K loads, per-tile P loads, and per-tile V-transpose loads before any
kernel edit.

## Non-negotiable boundaries

- Reuse the predecessor's exact serving request and shapes; do not invent a microbenchmark shape.
- Hold `/tmp/memra-5090.lock` for every timed or profiler run and require an idle
  `nvidia-smi --query-compute-apps` preflight.
- Local measurements are relative-only under the existing clock cap; do not change the cap.
- Do not commit NCU/Nsys report files. Commit only tee-first logs and extracted tables.
- A layout-only candidate must be bit-identical. Any differing bit stops the lane verbatim.
- Do not touch decode, the live serve box, generated boards, tags, merges, pushes, or formatting.
- If a tile-shape change becomes necessary, declare the new numeric class before running it.

## Planned evidence sequence

1. Import the predecessor profiling/request harness verbatim with recorded source hashes.
2. Rebuild an isolated `v0.81.3` baseline and capture per-PC shared wavefront metrics plus SASS.
3. Commit the array/instruction attribution table before changing the kernel.
4. Test XOR swizzle or padding only on the attributed arrays; build in a second isolated target.
5. Stop immediately on any exactness difference; otherwise profile the same four shapes.
6. If replay falls, run the complete required exactness battery and N>=5 interleaved timing.
7. Record `GO`, `PARTIAL`, or `NO-GO`, commit the lane, and stop.

## Status

`COMPLETE; LOCAL GO; PRO CONFIRMATION REQUIRED`

- Worktree verified clean and isolated on the requested branch.
- `HEAD` exactly matched `v0.81.3` before this progress file.
- Required predecessor evidence, target kernel, and dispatch were read.
- Current NVIDIA documentation was checked for the metric semantics: shared wavefronts are
  serialized service passes, while excessive wavefronts quantify work beyond the ideal request.
- The predecessor request harness was copied verbatim and hash-checked.
- Fresh `v0.81.3` Q27/Q35 captures reproduce all four aggregate ratios exactly.
- Per-PC source counters localize replay to every row-major operand: Q/K/V `ldmatrix` loads are
  8.00x ideal; P stores and reloads are 4.00x. Recurring K/V loads are 97.03--97.22% of excess.
- The committed attribution and raw tables are in [`ATTRIBUTION.md`](ATTRIBUTION.md) and `raw/`.
- Candidate v1 applies only the selected XOR permutation to Q/K/V/P staging and matching
  `ldmatrix` addresses. It does not change tile geometry, arithmetic, accumulation order, or
  shared-memory capacity.
- The orchestrator preserved that candidate at `c6921f5de` after the harness sweep. Its first
  three build attempts stopped while compiling the unchanged `cu/hybrid.cu`: first
  `Segmentation fault (core dumped)`, then twice `LLVM ERROR: out of memory`. Steering identified
  exhausted RAM-backed `/tmp` as the environmental cause and reclaimed it; these failures are
  not candidate evidence.
- The unchanged candidate then built successfully with `TMPDIR=/home/avifenesh/tmp-lanes` on the
  disk-backed root filesystem. Engine binaries completed in 4m07s and `memra-server` in 19.96s.
  The full resumed log is `raw/candidate-v1/build-retry-resume.log`; all candidate binary hashes
  are recorded in `raw/candidate-v1/binary-sha256.txt`.
- Static resource extraction keeps the actual-shape `fa_prefill_qw_db` at 255 registers and the
  same shared allocation, but adds an 8-byte stack frame. The hd128 twins rise from 230/233 to
  254/255 registers. `raw/candidate-v1/flash-resource-usage.log` retains both fatbin dumps; NCU
  must check whether the actual-shape stack produces local traffic.
- The predecessor interleaved harness was imported byte-verbatim at SHA-256
  `625602ab55e1d4b42c1e7dd73d9cc3219dd4f215e6aac34cb7af206a3faca590`, then changed only to
  select this lane's isolated baseline and candidate target directories.
- A four-arm frozen-shape gate ran under one lock hold with empty per-arm compute-app checks.
  Baseline and candidate match exactly on Q27 (`200ec271...88e7`) and Q35
  (`b723be26...be1`); every scored request reports 4,860 prompt tokens and `cached_tokens=0`.
  The single-run timings in this identity gate are unscored and support no performance verdict.
- Candidate NCU reduces all four aggregate ratios from 6.424672--6.452196x to
  1.046995--1.047378x, a 99.13% reduction in excess wavefronts. Q/K/V are exactly 1.00x; P retains
  a 2.00x residual. See [`PROFILE-CANDIDATE.md`](PROFILE-CANDIDATE.md) and the combined raw tables.
- Dynamic shared memory and 8.33% occupancy are unchanged. Candidate v1's profiled duration is
  32--34% shorter, but its new 8-byte stack frame produces two local stores and two local loads.
  These are single profiler captures, not scored timing evidence.
- A source-liveness-only v2 moved `kv_head`, `causal_i`, and `q_pos0w` below Q staging. It built in
  an isolated target but left `fa_prefill_qw_db` at `REG:255 STACK:8`; SASS moved the stack slot
  into the PV/output accumulator path instead of removing it. V2 failed its static objective and
  was rejected without a GPU run. Its build, hashes, resource dump, and SASS are under
  `raw/candidate-v2/`.
- The source was restored byte-for-byte to the profiled v1 before proceeding to the complete
  exactness battery.
- The full candidate battery then ran from `2026-08-13T02:18:52Z` through
  `2026-08-13T02:34:59Z` under one uninterrupted `/tmp/memra-5090.lock` hold. Its fail-closed
  preflight hash-matched the v1 source, four candidate binaries, both models, both drafts, and
  every prompt before the first GPU command.
- Required manifests finished `ALL GREEN (106 cells, 1 skipped)`; model-backed Q27 finished
  `ALL GREEN (107 cells, 3 skipped)`; model-backed Q35 finished
  `ALL GREEN (113 cells, 1 skipped)`.
- Q27 and Q35 `run-gen` both report prefill/decode argmax `MATCH` and batched-prime/tokenwise
  argmax `MATCH`. Both `run-spec` runs report eight per-K self-consistency passes for K=1..8 and
  the terminal `SELF-CONSISTENCY PASS` verdict.
- Both Q27 and Q35 are chunk-invariant on the pinned 97- and 149-token prompts: chunk sizes
  64 and 32 are bit-exact to 2048 for logits and produce identical 48-step streams.
- The raw gate logs, input hashes, provenance, postflight, and verified manifest are sealed under
  `raw/candidate-v1/gates/`.
- The frozen predecessor measurement harness then completed all 20 arms under a second single
  lock hold. Each arm used a fresh server and namespace, the exact 4,860-token request, 60 output
  tokens, and no cache hit. Q27 alternated baseline/candidate; Q35 reversed the leading arm.
- Candidate v1 won every paired repetition on both models. Median candidate deltas are Q27
  prime `-1.652%`, prefill `+1.680%`, cold TTFT `-1.617%`; Q35 prime `-2.788%`, prefill
  `+2.868%`, cold TTFT `-2.747%`.
- Q27's median prime gap is 91.921 ms versus baseline/candidate ranges of 2.056/9.596 ms. Q35's
  median prime gap is 40.622 ms versus ranges of 0.902/0.631 ms. The wins are outside per-arm
  spread, not flat.
- All 20 scored outputs retain one text hash per model. The raw requests, server traces,
  telemetry, compute-app checks, results, provenance, and verified manifest are under
  `raw/measurement/`; `raw/measurement-driver.log` is the tee-first orchestration log.
- Local verdict is `GO`: direct replay fell by 99.13%, exactness stayed green, and both served
  models won outside spread. Final disposition must name the required Vast 2x RTX PRO 6000
  confirmation; this lane does not run it.
- [`RESULTS.md`](RESULTS.md) records the final local GO, resource caveat, complete exactness and
  timing evidence, and the explicit Vast 2x RTX PRO 6000 pre-release follow-up. Per lane scope,
  stop here without merge, tag, push, board edit, or live-server action.
