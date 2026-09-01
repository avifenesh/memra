# SWA ring flag-ON validation progress

Lane `lane/cx-ringval`, fixed runtime base `019428e217e297cb5981d201a4a520aee69222a6`,
started 2026-08-10.

## Contract

- Measurement and validation lane for `MEMRA_SWA_RING=1`; the flag remains default OFF.
- Box1 rented cloud pair: two RTX PRO 6000 Blackwell Server Edition GPUs, PP-2 on devices 0,1.
- Pinned Step-3.7-flash IQ4_XS trunk plus Q8_0 MTP artifact under
  `/home/ubuntu/step37/models/step-3.7-flash`.
- Serve shape: `MEMRA_CTX=262144`, `MEMRA_MOE_GROUPED=1`, and
  `MEMRA_PREFILL_TICK=2048`.
- Every bounded GPU block holds `/tmp/memra-gpu.lock`; raw stdout and stderr are retained before
  reduction.
- Required receipts: wrap-crossing teacher-forced OFF/ON exactness, b1fix golden hash with the
  ring flag ON, serving/correctness gates, honest 262k capacity OFF/ON, and lapped-affinity decline
  followed by a clean cold re-prime.
- No origin push, tag, release, `rustup`, or `nsys`. Stop after committed `RESULTS.md`.

## Status

- Read `CLAUDE.md` and `research/kv256-20260809/RESULTS.md`, including the complete Box1
  follow-up, before remote work.
- Read the b1fix one-hash receipt and capbase capacity harness/methodology.
- Worktree started clean on the dedicated branch at the fixed runtime base above.
- `~/.lanectl/inbox/cx-ringval.md` was absent at intake and at the preflight before the build
  block; the exact path will be checked again before every later block.
- Box1 preflight at 2026-08-10T06:28:53Z found both GPUs idle at 0 MiB and 0% utilization,
  26/27 C, with no compute applications. CUDA reports 13.2.
- The initial exact-tip build block ended at 2026-08-10T06:34:39Z with the worktree clean at
  `019428e217e297cb5981d201a4a520aee69222a6` and CUDA auto-selecting `sm_120a`. It rebuilt the
  engine binaries, but a later serving preflight proved that `memra-server` itself was stale: its
  mtime remained 2026-08-10T04:40:40Z, its SHA-256 was the pre-ring b1fix-parent value
  `6a7c2046...e5ca7e5`, and the binary did not contain the ring server markers. The original
  receipt remains under `raw/build-20260810T063055Z/`; it is superseded for the server binary by
  a clean-target rebuild.
- The first wrap-harness attempt stopped before measurement with the quoted loader error
  `ERROR: model has no MTP/NextN head`. The standalone draft loader is keyed by
  `MEMRA_MTP_DRAFT`; `MEMRA_DRAFT` alone only marks spec-serving intent. The failed raw receipt is
  retained under `raw/wrap-20260810T063818Z/`; the harness now supplies both variables.
- The corrected replay arms under `raw/wrap-20260810T064018Z/` both completed: each consumed
  9,216 tokens with 4,096-row chunks, so the second append crossed the 4,639-row physical tail in
  both the trunk SWA cache and persistent MTP scratch. The 35 sampled teacher-forced rows are
  byte-identical (`20e1a2a7...e7de`) and both arms report NLL/token 0.78327 over 9,215 targets.
- That block's subsequent `run-gen` diagnostic unintentionally tokenized the complete nominal
  `p4-16k.txt` into 23,770 tokens and stopped with the captured
  `DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")`; no cause beyond that reported CUDA
  error is inferred. A separate fixed-binary probe uses the exact first 9,216 ids from the successful
  replay to compare the complete step-15 teacher-forced logit row across OFF and ON.
- The fixed-binary full-row comparison completed under one lock at 2026-08-10T07:40:04Z. With
  N=1 per arm, OFF and ON produced byte-identical 128,896-float step-15 logit vectors
  (`0fdc84a9...c912`), identical forced output tokens, and identical teacher-forced summaries.
  The preflight and final snapshots show no other GPU compute application; the complete receipt is
  retained under `raw/logits-20260810T065008Z/`.
- The first ring-ON serving attempt under `raw/serve-exactness-20260810T074152Z/` sent no
  requests. Its startup assertion caught the stale server binary because the ready server logged
  no `capped at 4639 rows` admission shape. The block exited nonzero after cleanup, with the
  exact server log and stale process identity retained. A new target directory is required before
  any serving or capacity result is accepted.
- A clean-target build under `raw/clean-build-20260810T074946Z/` completed at
  2026-08-10T07:53:33Z from the same clean source commit with CUDA 13.2 / `sm_120a`. It rebuilt
  `memra-server`, every engine validation binary, and `tok-check` without consuming the old target.
  The corrected server SHA-256 is `7f04f767...d9b9cef`, its mtime is after the build start, and
  the executable contains the ring admission and lapped-checkpoint markers. The remote repo's
  ignored `target` symlink now resolves to `/home/ubuntu/memra-cx-ringval-target-ringval`.
- The corrected ring-ON serving exactness block under
  `raw/serve-exactness-20260810T075504Z/` passed on two fresh boots. The c=1 probe matched the
  pinned 326-byte golden hash, and the c=4 barrier produced 4/4 matches with zero errors or
  divergences. Both server logs report `61248 capped at 4639 rows` for plain SWA KV and
  `63104 capped` for spec KV.
- The matched full-context capacity block under `raw/capacity-20260810T075756Z/` completed with
  N=1 per arm and c=24 offered. Ring OFF first deferred at 2 active 262,144-token sessions; ring
  ON first deferred at 12, a measured 6.0x session-capacity ratio. The admission cost was
  21,894 MB/session OFF versus 6,123 MB/session ON (3.576x), while all 24 requests per arm
  eventually completed and both arms captured zero failure lines and zero step-OOM parks.
- The lapped-checkpoint block under `raw/lap-20260810T080122Z/` passed with N=1. A 9,216-token
  session lapped its position-1,024 checkpoint; the 2,048-token affinity request logged the exact
  `SWA ring lapped checkpoint 1024` decline, computed all 2,048 tokens cold with zero cache hits or
  affinity rewinds, and produced the same 17-byte output hash (`67f6e242...fc02`) as a fresh-server
  cold reference. The ring-specific prefix-cache refusal also logged on both requests.
- The clean-binary ring-ON core battery under `raw/core-gates-20260810T080351Z/` is green:
  `kernel-check` matched its CPU references; the PP-2 decode matrix was bit-identical for
  B=1,2,4,8; `run-gen` passed prefill/decode and batched-prime/tokenwise argmax gates; and
  `run-spec` passed self-consistency for every K=1 through K=8 with the external Q8_0 draft.
- The Step35 segmentation block under `raw/invariance-20260810T081034Z/` passed all four
  assertions with the ring enabled. Naked chunk sizes 4096/513/512/256/64 and tick budgets
  0/1024/513/512/256/64 plus split points 64/256/512 were bit-identical. Both canaries restored
  their legacy arithmetic, produced the pinned divergences, and were caught by the gates.
- The unchanged stock `serve-smoke` under `raw/serve-smoke-20260810T082015Z/` completed with
  exit 1 and exactly one top-level failed cell: its flat prefix-cache accounting gate. All 14
  sub-failures are the ring contract's expected zero-hit/zero-insert/zero-LCP result; the server
  explicitly logged prefix-cache refusal. Plain APIs, streaming, completions, determinism,
  concurrency, long generation, spec/plain identity, all sampled truncation arms, and the two
  affinity runs passed. The final server also quoted the same shutdown-only
  `CUDA_ERROR_DEINITIALIZED` pending-flush line present in the accepted flag-OFF b1fix receipt,
  after `0 in flight` and `drain complete`; no request failed.

## Pending blocks

- [x] Clean-target exact-tip server rebuild and corrected provenance receipt.
- [x] Wrap-crossing teacher-forced OFF/ON comparison.
- [x] Ring-ON b1fix one-hash golden probe and c=4 barrier burst.
- [x] Stock serve-smoke completed with the expected prefix-cache-only scoped red; all applicable
  API/exactness cells passed.
- [x] Honest full-262k session capacity OFF/ON (2 -> 12, 6.0x; N=1 per arm).
- [x] Lapped affinity decline and clean cold re-prime.
- [x] Raw manifest and `RESULTS.md` complete. Terminal documentation commit is the final lane action.
