# Eager-parity B=1 progress

## 2026-08-10 — lane opened

- Branch/worktree: `lane/cx-eagerpar` in `wt-cx-eagerpar`, base `dc77de73`.
- Target rig: box1 cloud pair, 2x RTX PRO 6000 Server Edition; every bounded GPU block holds `/tmp/memra-gpu.lock`.
- Fixed serve shape: PP-2 on devices 0,1; context 262144; grouped MoE on; prefill tick 2048; Step-3.7 Flash IQ4_XS + MTP.
- Starting receipt: the b1fix default sends Step3.5/3.7 B=1 through `step35_decode_batch_layers`, producing golden completion SHA-256 `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` across 35 fresh boots / 150 requests. N=5 c=1 sustained decode fell from eager 85.423 tok/s to batched 81.399 tok/s (-4.710%).
- Objective: recover most of that loss with a B=1-specialized entry into the same batched arithmetic, without creating another numeric class.
- Kill gate: specialized and current batched trunk must be byte-identical over the required fresh-boot matrix and preserve the B=1 -> B>1 transition class. Any arithmetic-order or kernel-class change is out.
- Required closeout: anatomy timings; c=1 interleaved N=5 A/B; c=2/c=4 sanity; kernel-check; run-gen argmax; run-spec K=1..8; b2geo35 transition; `RESULTS.md` promote/hold verdict and raw receipts.
- Constraints acknowledged: no push, no `cargo fmt`, no `rustup`, no `nsys`; small commits; preserve unrelated work.
- Inbox check: `~/.lanectl/inbox/cx-eagerpar.md` was absent at lane start.

Next: inspect the current and pre-b1fix Step35 dispatch/anatomy, establish remote build provenance, then run a bounded baseline anatomy block.

## 2026-08-10 — static anatomy and rig preflight

- Current B=1 Step35 PP-N enters `step35_decode_batch_layers` on each stage; pre-b1fix source `188154299064a42b67fc8eb1f41757cf6237300d` entered `decode_layers_eager` instead.
- The batched Step35 arithmetic itself selects the same `m=1` projection, norm, rope, attention, gate, and MoE classes when `B=1`. Its visibly B-general state loop additionally launches two device-to-device copies per layer: `q[row] -> q_row` before `fa_decode_kvmod`, then `a_row -> attn[row]` afterward. With `B=1`, both source ranges are the whole allocation and neither copy performs arithmetic.
- Candidate hypothesis: a B=1 entry can pass the whole `q` allocation directly to the unchanged FA call and write FA directly into the whole `attn` allocation. This removes 2 x layer-count copy launches/allocations while keeping every arithmetic kernel, its arguments/shape, and its order intact. Measure before editing.
- Existing diagnostic `MEMRA_BATCH_PHASE=1` provides sync-bounded phase ranks and isolates those copies in `attn per-seq: q/a dtod copies`; its own documentation says shares/rank only, not absolute wall time.
- Box1 preflight at 2026-08-10T16:40Z: idle compute-app list; CUDA 13.2 / nvcc 13.2.51; Nsight Compute 2026.1 is installed, but this lane will prefer the existing bounded phase hook and will not use `nsys`.
- Proven baseline binaries remain present byte-for-byte: fixed `memra-server` SHA-256 `6a7c2046eb3197773def91baf012abd629e0b0ced239ec2d38016c93be5ca7e5`; eager base SHA-256 `e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3`.
- Remote `memra-cx-b1fix` contains unrelated untracked b2geo logs; they will not be touched. A separate `memra-cx-eagerpar` directory will carry this lane.

Next: build the missing eager baseline bench without changing source, then run one bounded locked baseline anatomy block for batched and eager trunks.

## 2026-08-10 — baseline anatomy block complete

- One bounded box1 lock hold ran the current batched trunk and pre-b1fix eager source on the same PP-2/model geometry. Full driver, cell, thermal, GPU, artifact, and binary receipts are in `raw/anatomy-baseline/`.
- Uninstrumented engine bench (`64` steps, one discarded warmup plus N=3 timed reps per arm): batched `81.8 tok/s` vs eager `86.0 tok/s`, `-4.884%` / `+0.597 ms/token` for batched. This independently reproduces the live b1fix N=5 `-4.710%` class-level loss.
- Thermal caveat: the anatomy wall cells were one fixed-then-eager order (batched start 26/26 C, eager start 32/33 C). Treat this as mechanism confirmation, not the final A/B verdict; the candidate performance gate will be N=5 interleaved with per-arm idle handling.
- Sync-bounded batched phase run covered 16 total tokens (warmup + measured): `q/a dtod copies` consumed `15.0 ms`, 6.0% of the deliberately inflated 250.6 ms diagnostic total. The exact structural count is 45 layers x 2 copies = 90 copy launches/token. The eager chain has no corresponding copies.
- Anatomy verdict: B-generality copy/issue overhead is large enough to explain the observed `0.597 ms/token` gap; removing only these whole-row B=1 copies is the first candidate. No arithmetic fusion is justified at this point.

Next: implement the minimal B=1 state-path specialization inside `step35_decode_batch_layers`, compile, and run the direct bit-identity gate before any broad battery or performance claim.

## 2026-08-10 — candidate increment and first kill gate

- Code increment `711fbcaa`: inside `step35_decode_batch_layers`, only the `B=1` attention state path bypasses the general row materialization. It passes the whole existing `q` allocation into the same `fa_decode_kvmod` call and writes that call directly into the whole existing `attn` allocation. All arithmetic calls before/after it and the full B>1 loop are unchanged; no fusion, environment flag, or dispatch class was added.
- Local `cargo check -p memra-engine --lib` passed with CUDA 13.1 / sm_120a. No formatting command was run.
- Isolated remote release build passed with CUDA 13.2 / sm_120a. Candidate `memra-server` SHA-256 is `43ad098d46bb26d644ba0b742d92f3f014d9287ac72e8a0edb8ebf9dac3ba608`; candidate `decode-batch-bench` SHA-256 is `91544a0d32977dac792bc548b609ce26dafda1c4ce92bf7fb071534660bb7e3f`.
- First fresh-boot live c=1 kill gate **PASS** under the exact serve shape: 1/1 request, 0 errors, 326 bytes, exact frozen batched-trunk golden SHA-256 `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`. Full boot/server/GPU/request receipt is under `raw/smoke-c1/`.

Next: candidate anatomy block. Require the q/a-copy bucket to disappear and a wall recovery before expanding to c=1 x10 and the transition/correctness batteries.

## 2026-08-10 — candidate anatomy passes

- One bounded candidate/eager box1 block is archived under `raw/anatomy-candidate/`.
- Uninstrumented engine bench (`64` steps, one discarded warmup plus N=3 timed reps per arm): candidate `85.4 tok/s`, eager `85.9 tok/s`, candidate within `-0.582%` of eager. Against the separately receipted current batched `81.8 tok/s`, candidate is `+4.401%` and recovers 85.7% of the `81.8 -> 86.0` engine gap.
- Same-order thermal caveat remains (candidate start 27/27 C, eager start 33/34 C); the final live-server verdict will be interleaved N=5 with idle handling.
- Mechanism check is exact: the sync-bounded `q/a dtod copies` bucket fell from `15.0 ms` / 6.0% to `0.0 ms` / 0.0%, and the code removes the structural 90 copy launches/token. No other candidate mechanism is needed.

Next: run the required c=1 x10 fresh-boot one-hash matrix against the frozen batched golden. Stop on any divergence; if green, proceed to the transition cell and standard battery.

## 2026-08-10 — c=1 x10 frozen-hash matrix passes

- Required fresh-boot c=1 matrix **PASS** in one bounded lock window: 10/10 boots, 10/10 requests, 0 errors, 0 divergences, exactly one completion class.
- All ten 326-byte completions equal the frozen current-batched SHA-256 `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
- Full per-boot server logs, scheduler metrics, GPU samples, request rows, response hashes, and thermal snapshots are under `raw/matrix-c1x10/`; reduced counts are in its `summary.json`.

Next: run the live-default b2geo35 static/transition gate. Its stateful cell must prove `ready=1 -> ready>=2` while every early/late completion remains in the same golden class.

## 2026-08-10 — live B=1 -> B>1 transition passes

- Naked `step35-b2-geometry-gate.sh` **PASS** on the candidate under live defaults.
- Static c=1 reference, both c=2 rows, and all four c=4 rows were byte-identical.
- Stateful cell proved the early row emitted while alone, then the tick trace crossed `ready=1 -> ready>=2`; early and both late completions matched the reference.
- Non-vacuity evidence: decode chunk cap `8`; server logged `[step35-batch] first B>1 ... B=2 layers=[0,22)`.
- Raw gate, server, and sliced transition tick logs plus reduced summary are under `raw/transition/`.

Next: run the engine exactness battery (`decode-batch-gate` B=1/2/4/8), then kernel-check, run-gen, and run-spec K=1..8. Any byte or argmax failure stops promotion.

## 2026-08-10 — core battery passes

- `kernel-check`: **ALL GREEN** against the CPU reference.
- PP-2 `decode-batch-gate --batch 1,2,4,8 --steps 24 --reps 2 --plen 520`: **ALL GREEN**; every split repeat and unsplit comparison had 0 differing logits bits, 0 failing arms; B=8 epilogue passed.
- One bounded core lock window, with raw stderr/stdout and thermal receipts under `raw/gates/core/`.

Next: separate bounded generation block for Step3.7 `run-gen` argmax and MTP `run-spec` K=1..8 self-consistency.

## 2026-08-10 — generation/spec battery passes

- Step3.7 `run-gen`: prefill/decode argmax **MATCH**; batched-prime/tokenwise argmax **MATCH**.
- Step3.7 + external MTP `run-spec`: K=1..8 each **self-consistency PASS** against the live B=1 batched target; aggregate verdict **SELF-CONSISTENCY PASS**.
- One bounded generation lock window, with raw stderr/stdout and thermal receipts under `raw/gates/generation/`.

Next: rigorous live-server performance A/B — N=5 interleaved candidate vs current batched, c=1 sustained decode; c=2/c=4 sanity. Promotion still depends on those results and final receipt integrity.

## 2026-08-10 — interleaved live A/B passes

- One bounded box1 lock window ran five paired fresh-process rounds, alternating order as `current/candidate`, `candidate/current`, `current/candidate`, `candidate/current`, `current/candidate`. All 40 measurement points and 80 request rows passed; 0 errors and 0 short completions. Raw server logs, request/point JSONL, and per-arm thermal snapshots are under `raw/perf/`.
- Required c=1 sustained decode metric, `(completion_tokens - 1) / (latency - TTFT)`, N=5 median: current batched `81.472 tok/s`; specialized `85.041 tok/s`; **+4.381%**. This recovers 90.5% of the prior `81.399 -> 85.423` gap and ends only 0.447% below the separately receipted old eager median.
- c=2 aggregate N=5 median: `113.694 -> 119.943 tok/s`, **+5.496%**. c=4: `139.479 -> 144.665 tok/s`, **+3.718%**. Both sanity cells improve.
- Short eight-token TTFT N=5 median: `70.297 -> 67.844 ms`, **-3.491%**.
- Thermal regime: fresh server per arm, alternating order, no artificial cooldown; all captured temperature snapshots were 26–36 C. Candidate source/binary and current b1fix-identical binary hashes are recorded in `raw/perf/summary.json` and `driver.log`.

Next: freeze checksums, write `RESULTS.md`, audit the complete diff and receipt links, then commit the final promote verdict without pushing.

## 2026-08-10 — lane complete

- Final verdict: **PROMOTE** the Step35 B=1 specialized entry as the naked default. It preserves the current batched numeric class, removes only arithmetic-free row copies, recovers 90.5% of the prior gap, and passes every required identity/correctness/transition gate.
- `RESULTS.md` records the anatomy, design contract, complete gate table, N=5 interleaved A/B, provenance, and the scoped promotion decision.
- `raw/SHA256SUMS` freezes all 204 raw evidence files; manifest SHA-256 is `20636c29fc90bb241d2dfc6e701cda8938386f4421957627bbcda96dfc5b9896`.
- Final audit: all manifest entries verify; all JSON parses; all three lane drivers pass `bash -n`; report links resolve; `git diff --check` passes; box1 has no compute applications after cleanup.
- No perf-board update was warranted, and no push, tag, merge, or release was performed.
