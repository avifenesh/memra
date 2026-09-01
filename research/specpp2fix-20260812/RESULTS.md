# PP-2 speculative-decode resume verdict

Date: 2026-08-12

Branch: `lane/cx-specpp2fix`

Runtime source tested: `0abdf957ee15b64271e9cb49b1830408ba7f4f3d`

Rig: box1, 2x RTX PRO 6000 Blackwell Server Edition

## Verdict

**No new runtime fix is warranted.** The requested PP-2 + speculative-decode illegal address does
not reproduce on the current tree, because the tree already contains the complete #87
reverse-publication fix and containment traps. Forced speculative serving is exact and stable in
both PP placement orders, naked dual mode, and the serial rollback. It should nevertheless remain
policy-disabled on sharded PP-2: the final c=8 timing receipt measures K=1 speculative serving at
only **52.09%** of plain throughput.

This lane therefore changes only research harnesses and receipts. It does not change runtime code,
admission defaults, generated perf boards, or release state.

## Freshness boundary

The measurements are authoritative for runtime source `0abdf957e`. A final fetch at
2026-08-12T03:35:48+03:00 found `origin/main` had advanced to `3143c4674` after the box1 battery.
The late-main audit found:

- `spec.rs`, `decode.rs`, and `decode_batch.rs` have identical Git object ids between this lane and
  `origin/main`; the #87 reverse fences and speculative math did not move.
- The `pp.rs` / worker / server-main delta is the separately gated `cx-sec8` peer-integrity
  hardening: startup policy, an every-8,192-boundary-copy re-probe serviced between scheduler
  ticks, and operator metrics. It does not change the spec ownership path measured here.
- The later `cx-prefixmoney` merge adds research reports and deferred harnesses only. This lane
  explicitly ran with `MEMRA_PREFIX_CACHE_MB=0`.

Per the brief, those late changes were not merged into this freshness-gated lane. The orchestrator
must rebase/promote it and rerun the required current-main battery; this receipt must not be
misrepresented as a test of `3143c4674`.

## Task-premise correction

The inbox says `MEMRA_SERVE_SPEC=0` still stands because #87 is open. That is not the current
repository contract:

- `MEMRA_SERVE_SPEC` is globally on and the #87 PP-2 quarantine is documented as lifted.
- PP-2 engagement is default-off through the placement-aware `MEMRA_SPEC_GATE` policy
  (`LOW=0/HIGH=1`) because plain batching wins, not because the path is unsafe.
- The original fixes are all ancestors of this lane: `7450928b4` fences spec verify,
  `4c72d637d` extends the reverse fence to eager and batched PP bodies, `80b2ddf45` fences stage-KV
  admission, `c41dc3452` removes the quarantine, and merge `61f513f34` closes #87.

The evidence-backed historical root cause remains the one recorded in
[`../pp2spec-crash-20260807/PROGRESS.md`](../pp2spec-crash-20260807/PROGRESS.md): stage-stream
allocations could be freed and reused before the primary stream consumed its queued reads. The
victim seed became partially NaN, the NaNs spread across the draft-logit row, device argmax left
its `0x7fffffff` initializer intact, and the next embed gather dereferenced roughly 4.6 TiB beyond
the table. `PpNRt::fence_stages_behind` supplies the missing reverse order; out-of-vocabulary
argmax traps contain any recurrence before dereference.

No current fault was captured, so there is no second root cause to name. Launch blocking and
Compute Sanitizer were deliberately not invoked after the bare runs stayed clean: without a bare
faulting operation they cannot produce the brief's required quoted attribution, and the original
investigation already records debugger/synchronization perturbation masking this race.

## Frozen artifacts

| Artifact | SHA-256 |
| --- | --- |
| Step 3.7 Flash IQ4_XS trunk | `b940497a9cec2f801f07e3a9783f2115fd8bf79cbd453225b4f73d86bcd11259` |
| External MTP Q8_0 draft | `469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57` |
| Golden response | `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` |
| `memra-server` | `25f2114b548400736b5ba03023f7c6f11dcec078f3d64b53e28f87d04cbbcc60` |

The release build used CUDA 13.2 with auto-detected sm_120a. Research-only commits after the build
caused Cargo no-op rebuilds; no compiled source changed.

## Reproduction attempts

Every run held `/tmp/memra-gpu.lock`, found both cards idle before launch, forced speculative
admission with `MEMRA_SERVE_SPEC=1` plus `MEMRA_SPEC_GATE=0`, and attached the external drafter.

| PP placement / K | Shape | Result | Aggregate throughput |
| --- | --- | --- | ---: |
| dev0 -> dev1, K=1 | c=8, 16 x 64 | 16/16 OK, 0 errors, 1,024 tokens | 53.727 tok/s |
| dev0 -> dev1, K=3 | c=8, 32 x 128 | 32/32 OK, 0 errors, 4,096 tokens | 46.417 tok/s |
| dev1 -> dev0, K=1 | c=8, 32 x 128 | 32/32 OK, 0 errors, 4,096 tokens | 68.767 tok/s |

All three failure scans are empty. The K=1 runs contain 40 and 152 `spec-acc` burst lines,
respectively; the reverse placement also passed the 64 MiB production-slot peer probe in both
directions. The servers stopped normally and left no compute process. Raw receipts:
[`raw/box1/repro-baseline/`](raw/box1/repro-baseline/),
[`raw/box1/repro-k3-c8/`](raw/box1/repro-k3-c8/), and
[`raw/box1/repro-k1-c8-dev10/`](raw/box1/repro-k1-c8-dev10/).

## Exactness gates

The final run is [`raw/box1/gates-v2/`](raw/box1/gates-v2/). Every command exit receipt is zero,
the targeted failure scan is empty, and the driver ends in `SPEC_PP2FIX_GATES_PASS`.

| Gate | Result |
| --- | --- |
| `kernel-check` with Step manifest | ALL GREEN, 87 cells; 21 unrelated optional-model cells skipped |
| spec-verify PP split, dev0 -> dev1 | T=2,5,9; 16 rounds x 3 reps; zero differing bits, including `h_seed` |
| spec-verify PP split, dev1 -> dev0 | T=2,5,9; 16 rounds x 3 reps; zero differing bits, including `h_seed` |
| batched PP split | B=1,4,8; 16 steps x 2 reps; zero differing bits |
| `run-gen` | prefill/decode and batched-prime/tokenwise argmax MATCH |
| `run-spec` dev0 -> dev1 | K=1..8 self-consistency PASS |
| `run-spec` dev1 -> dev0 | K=1..8 self-consistency PASS |

The first gate invocation in [`raw/box1/gates/`](raw/box1/gates/) completed every substantive
gate successfully but its final case-insensitive reducer matched benign `mismatch=0` fields. Commit
`0abdf957e` narrowed the scan to real uppercase failure markers; the entire battery was rerun into
`gates-v2` rather than treating the harness false red as a pass.

## Golden matrix and sticky-crash soak

The generated summary is
[`raw/box1/serve-validation/summary.json`](raw/box1/serve-validation/summary.json).

- Forced K=1 speculative serving matched the pinned golden response **62/62** times across
  c=1,2,4,8,16 in both naked dual and `MEMRA_DUAL_PP=0` serial arms.
- Three fresh naked-dual server boots each completed 64 requests at c=8: **192/192** requests,
  **12,288** generated tokens, zero shed, zero errors.
- Per-boot aggregate throughput was 54.267, 54.327, and 54.335 tok/s.
- Server and kernel scans contain no CUDA error, illegal address, sentinel, Xid, panic, worker
  death, or nonzero peer mismatch.
- Thermal receipt: N=3 fresh boots, no artificial cooldown; 3,938 samples at 250 ms, 27-51 C.

## N=5 interleaved timing

The final timing summary is
[`raw/box1/timing/summary.json`](raw/box1/timing/summary.json). Both arms used naked dual PP-2,
the same binary and model artifacts, c=8, one 16-token warmup, then 32 measured requests x 128
tokens. Pair order alternated inside one uninterrupted GPU-lock hold; no artificial cooldown was
used.

| Arm | N | Aggregate tok/s by rep | Median |
| --- | ---: | --- | ---: |
| spec off | 5 | 132.369, 132.568, 132.577, 132.689, 132.734 | **132.577** |
| forced spec K=1 | 5 | 69.041, 69.035, 69.054, 69.080, 69.058 | **69.054** |

- Spec-on / spec-off: **0.52086x**, or **-47.914%**.
- Cell-wide acceptance from server burst counters: 9,000 / 11,400 = **78.947%**. This includes
  one 8/8 accepted warmup burst per rep; excluding those five warmups, the measured-window rate is
  8,960 / 11,360 = **78.873%**.
- All 320 measured requests completed, producing 40,960 tokens with zero errors or shed.
- Thermal receipt: N=5/arm, 6,728 samples at 250 ms, 28-51 C.

The result confirms the existing policy. Acceptance is healthy, but current PP-2 speculative
verify serializes the two balanced stages per request; it does not engage the plain dual-active
batch overlap that supplies the faster denominator. There is no board-moving win and no release
or default change to make.
