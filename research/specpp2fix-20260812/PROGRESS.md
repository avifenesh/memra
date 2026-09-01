# PP-2 speculative-decode root cause and fix

Date: 2026-08-12  
Branch: `lane/cx-specpp2fix`  
Box1 clone: `~/memra-cx-specpp2fix` (bundle transport only)

## Objective

Reproduce and fix the PP-2 + MTP speculative-decode illegal-address failure at concurrency 4 and
above without changing exact outputs. Validate both naked dual-active PP and
`MEMRA_DUAL_PP=0` serial PP, then record exactness, soak, and interleaved timing evidence.

## Frozen constraints

- Keep PP-2 admission unchanged while diagnosing; task #87 already established a spec-path defect.
- Capture quoted CUDA failure evidence before drawing a root-cause conclusion.
- Preserve the Step 3.7 Flash model, Q8_0 MTP drafter, prompt/template, and golden-response hash.
- Required exactness gates: `run-spec` K=1..8, `run-gen` argmax, decode-batch 0-diff, and one-hash
  golden identity at c=1,2,4,8,16 in dual and serial arms.
- Required soak: three fresh boots times 64 c=8 requests, spec on, zero errors.
- Required timing: N=5 interleaved c=8 spec on/off under one GPU-lock hold, including accept rate
  and aggregate token throughput.
- Do not merge, tag, push, update generated perf boards, or run repository-wide formatting.

## Starting point

- Local head: `8b2ba8c883152fdbb9f9bbd800a055ad03fe80c4`.
- Worktree was clean at lane start.
- Steering read from `~/.lanectl/inbox/cx-specpp2fix.md`; it matches the owner task verbatim.
- Task-specific prior memory search returned no entry; repository evidence is authoritative.

## Work log

### 2026-08-12

- Created this progress ledger as the first repository write.
- Resume: re-read the frozen inbox brief and both prior PP-2/spec investigations. The existing
  box1 clone initially had the older `lane/dualpp-default-flip` branch checked out at
  `0edc57b9c`; updated that clone in place through an incremental bundle (no re-clone, no origin
  push), switched it to this lane, and verified local and box1 HEAD are both
  `ac2437a323711c1e364ee6d0a25c9151206ecbd8` with clean worktrees.
- Committed the resume ledger at `903a493d5e56d18ac15c14310deeb41cb3f729d9`, transferred that
  exact tip to the existing box1 clone by incremental bundle, and built the release server and
  correctness binaries there with CUDA 13.2 / sm_120a. Build log:
  `raw/box1/build-baseline/build.log`.
- The repository history contradicts the task's stale quarantine premise: the original #87
  investigation records the all-NaN draft-logit / argmax-sentinel failure and reverse-publication
  race as fixed, the current `docs/FLAGS.md` says the quarantine is lifted, and the later
  `specpp2-20260810` study keeps PP-2 speculative serving policy-off for measured throughput rather
  than correctness. No post-fix runtime change was found in the current-main delta that would
  reopen that ownership path.
- Tried to reproduce on the unchanged release binary under the box1 GPU lock. All requests used
  Step 3.7 Flash IQ4_XS plus its external Q8_0 MTP drafter, explicit PP-2, forced speculative
  admission (`MEMRA_SERVE_SPEC=1`, `MEMRA_SPEC_GATE=0`), and eight concurrent sessions:

  | Placement / K | Requests x max tokens | Result | Spec evidence |
  | --- | ---: | --- | ---: |
  | dev0 -> dev1, K=1 | 16 x 64 | 16/16 OK, 0 errors, 1,024 tokens | 40 `spec-acc` bursts |
  | dev0 -> dev1, K=3 | 32 x 128 | 32/32 OK, 0 errors, 4,096 tokens | active (server log) |
  | dev1 -> dev0, K=1 | 32 x 128 | 32/32 OK, 0 errors, 4,096 tokens | 152 `spec-acc` bursts |

  Each server stopped normally, left both cards idle, and produced an empty scan for CUDA errors,
  illegal accesses, sentinel ids, worker death, or panic. The reverse-placement run also passed
  both 64 MiB production-slot peer probes in both directions. Raw roots:
  `raw/box1/repro-baseline/`, `raw/box1/repro-k3-c8/`, and
  `raw/box1/repro-k1-c8-dev10/`.
- Because the bare failure has not reproduced in either placement or at K=1/K=3, there is no
  operation to attribute with launch blocking or Compute Sanitizer and no evidence basis for a
  second runtime fix. Continue with the frozen exactness, golden, soak, and timing receipt on this
  unchanged baseline; only reopen source diagnosis if one of those stages captures a quoted fault.
- Added and committed the box1 gate, serving-validation, and timing harnesses. The first correctness
  invocation ran every substantive gate green, then the final failure reducer false-reded because
  case-insensitive `MISMATCH` also matched benign `mismatch=0` counters. Narrowed that reducer in
  `0abdf957e` and reran the entire battery into `raw/box1/gates-v2/`; the second invocation ended
  cleanly with `SPEC_PP2FIX_GATES_PASS`.
- Final exactness: kernel-check ALL GREEN (87 cells, 21 optional-model skips); spec-verify
  T=2,5,9 zero differing bits in both device orders; batched PP B=1,4,8 zero differing bits;
  `run-gen` argmax MATCH; `run-spec` K=1..8 self-consistency PASS in both device orders.
- Final golden/soak: the c=1,2,4,8,16 matrix matched the frozen hash 62/62 across naked dual and
  serial rollback; three fresh dual boots completed 192/192 c=8 requests and 12,288 generated
  tokens with zero errors. Summary: `raw/box1/serve-validation/summary.json`.
- Final N=5 interleaved c=8 timing, one lock hold: spec off median 132.577 tok/s; forced K=1 spec
  median 69.054 tok/s; spec/on = 0.52086x (-47.914%). Cell-wide acceptance was 9,000/11,400
  (78.947%; 78.873% after excluding the five 8/8 warmup bursts). All 320 measured requests
  completed. Summary: `raw/box1/timing/summary.json`.
- Copied all 241 raw box1 receipt files into this lane. Both cards were idle after every stage.
  No merge, tag, push, board update, source formatting, or runtime/default change was performed.
- Post-run freshness check: fetched `origin/main` at 2026-08-12T03:35:48+03:00 and found it had
  advanced to `3143c4674`. The spec/eager/batched decode source objects are identical; the relevant
  late runtime delta is peer-integrity re-probing between scheduler ticks, and the prefix-cache
  merge is research-only while this battery pinned cache to zero. Did not merge it into this lane;
  orchestrator promotion must rebase and rerun the current-main gates.

## Current status

Complete. The current tree already carries the #87 fix; the reported failure did not reproduce,
all frozen correctness and soak gates passed, and forced K=1 speculative serving is 47.914% slower
than plain at c=8. No runtime code changed. See `RESULTS.md` for the final receipt and policy
verdict.
