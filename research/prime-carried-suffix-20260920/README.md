# Carried sub-floor suffix prime: one numeric program again (#379 regression)

Verdict: `tools/spec-on-cache-hit-gate.sh qwen` is ALL GREEN again on the local RTX 5090 Laptop
(24 GB, CUDA 13.1) with the fix in `crates/memra-engine/src/spec/prime.rs`. Main since
cc9d593be (#379, 2026-09-09) failed its spec==plain identity law on the two suffix-fed restore
cells (`r3`, `g2`) deterministically. The cause is a two-programs crossing, the class memra's
CLAUDE.md calls "one numeric program per request".

## Symptom

Cells `r3` ("extended repeat") and `g2` ("turn 2") both restore `106 of 119 prompt tokens + draft
plane from cache [suffix queued]` and queue a 13-token suffix. Spec-on and spec-off boots agree
for about 200 characters and then flip one near-tie token (`Duration: 45 min` against
`Duration: 15 min`). Cold cells and full-cover hits (`r1`, `r2`, `g1`) are byte-identical.
Three runs, same bytes, same spec stats (rounds 21, drafted 63, accepted 27): external drafter,
then twice with the built-in MTP head. `raw/hitgate-rerun2/` is the tip run
(main `fdb781362`).

## Attribution

| Binary | Configuration | Result | Raw |
| --- | --- | --- | --- |
| main `afb8f7f4d` (2026-09-05) | built-in MTP, defaults | ALL GREEN | `raw/hitgate-afb8/` |
| main `fdb781362` (2026-09-20) | built-in MTP, defaults | 2 FAIL (`r3`, `g2`) | `raw/hitgate-rerun2/` |
| main `fdb781362` | `MEMRA_PRIME_YIELD=1` | 2 FAIL (`r3`, `g2`), same bytes | `raw/hitgate-yield-on/` |
| main `fdb781362` | `MEMRA_PRIME_TOKENWISE=1` | 5 FAIL (every cell) | `raw/hitgate-tokenwise/` |
| `git bisect run` over `afb8f7f4d..fdb781362` | tip gate script held constant, 10 steps, all conclusive | first bad `cc9d593be` (#379); parent `4896327d9` green | `raw/bisect/` |

The yield door is irrelevant: #379 routes the MTP prime through the shared walker in both arms
(research/prefill-fairness-20260908/MTP.md), and only the walker's program matters here.

## Mechanism

Before #379 a prefix-cache restore with a suffix fed it inside
`HybridModel::spec_session_from_restored` ("[suffix fed]"), whose program law says: mirror the
plain prefill tick arm for arm, eager `decode_step_h` below `PRIME_MIN_T` (16) and
`prime_cache` at or above it. The comment in that function already records why: on the
batched-serving numeric class (qwen35, qwen38) the generate path's tokenwise arm is
`spec_target_step_h`, the batched T=1 program, and it produced "ULP-different suffix rows"
and the near-tie flip at generated token about 8 on qwen r3 on 2026-08-18.

#379 defers the suffix to the worker's prime walker ("[suffix queued]"). `trunk_schedule`
kept the cold prime's legacy call for an unsegmented sub-floor segment,
`target_step: !segmented`, which is `spec_target_step_h`. A 13-token queued suffix with no
stable-boundary stop is exactly that shape. The plain boot still primes the same 13 tokens
through eager `decode_step`. Two programs, one request.

`MEMRA_PRIME_TOKENWISE=1` failing every cell is the same mechanism from the other side: under
the override the walker runs the whole cold prompt through `spec_target_step_h`, while the plain
boot's tokenwise arm is `decode_step`.

## Fix

`trunk_schedule` takes `carried` (`base > 0`, the restored or continuation shape). A carried
segment batches at or above `PRIME_MIN_T` unless the override is set; its sub-floor and
override segments use `decode_step_h`. The cold prime is unchanged: `spec_target_step_h` for an
unsegmented sub-floor prompt, batched non-final segments under the override. Unit tests cover
the gate's 13-token queued shape, a split carried suffix, the carried override, and the two
legacy cold schedules (`cargo test -p memra-engine --lib prime::tests`, 8 passed,
`raw/hitgate-fix/unit-tests.log`).

## Gate on the fix

`raw/hitgate-fix/`: memra-server `da3aa017a6a4977f...` built from this branch at
`1bc716343` (main `dbf88d467` plus the fix), qwen35-9b NVFP4 with the built-in MTP head,
`NVIDIA_TF32_OVERRIDE=0`, rig lock held per boot. 62 checks, ALL GREEN; `r3` and `g2` are byte
identical to the spec-off twin. The spec-on server log still says `[suffix queued]` on those
cells: the suffix still goes through the walker, only its program changed. The first attempt
(`attempt1-lock-wait.log`) never booted because another battery held `/tmp/memra-5090.lock`; it
was stopped and re-queued.

The perf-ci rows appended by `tools/local-ci.sh --perf` on this branch are the freshness
receipt the pre-push perf gate reads (`research/tune-data/perf-ci.jsonl`).

## `tools/local-ci.sh --perf` on this branch (2026-09-21, local RTX 5090 Laptop)

`raw/local-ci-perf-stages.log`. Every stage through the serving battery passed: decode-dc,
graph-decode, graph-session, serve-stress (c=64), the accept gate, the lock self-tests and the
spec-on-cache-hit gate. The run stopped at the "memra-engine lib suite (GPU-only #[ignore]
tests)" stage with 4 failures, none in this diff and none reachable from `spec/prime.rs`:

| Test | Failure | Origin |
| --- | --- | --- |
| `model_memory::native_tests::glm_indexed_mla_kda_state_materialization` | "two native CUDA devices are required; never skip" | cc348cce3 (GLM TP device owners) |
| `model_memory::native_tests::glm_peer_admission_materialization_trim_and_refill` | same, single-GPU rig | cc348cce3 |
| `dsv4_grouped::half2_chain_identity_tests::cuda_half2_chain_identity` | "GU half2 launcher did not engage" | 8a52da7c7 (#318) |
| `dsv4_graph::tests::replay_rust_partial_submission_and_capture_cleanup` | see log | a383ffc8c (#358) |

Two of them require two native CUDA devices by their own message; this rig has one. So the perf
stage did not run and no perf-ci rows could be appended honestly on this rig. The push carries
the logged `MEMRA_SKIP_PERF_CI=1` with this section as its record (`.git/memra-gate-skips.log`
row on the pushing clone). The correctness receipt for this change is the gate above.
