# Current-main composition and Gemma hidden exports

The correction at `33144df6d12c58a7821b351e18490cf8eff38f3e` completed its
source-bound native campaign: selected callers 27/27, original T16 and the additional
15/16/17 direct/batch/carried matrix passed. Publication
`18684d53b9dbfec597eddba1b2f0335c08217200` preserves its raw logs. The original
`65510819` campaign remains FAILED, 26/27: seed error 4.4796142578125 and the five
separate logit errors remain in its original records. No new result replaces those bytes.

This composition merges `871693e08725b47bea114e6f37bf7303c3ed8d0d` after a fresh
fetch. It includes #547's v3 build and release gates, the carried-prefix scheduling and
grid-aligned republication corrections, and the newer worker/tier recovery changes.
The prime fallback still uses the actual qualified Eager walk. Production diagnostic
redaction, generation-bound request guards, prime exclusion and the Qwen source pin,
original T16 oracle, `.005` limit and 27-case census are unchanged.

## Overlaps and source boundaries

- `worker.rs`: preserve upstream admission/recovery/prefix handling with the lane's
  request snapshot and entry guards, including combined request walks. The private test
  diagnostic recorder remains under `cfg(test)`; the client error stays sanitized.
- `spec/prime.rs` and `spec.rs`: retain the upstream carried sub-floor Eager schedule,
  tokenwise override and grid-aligned publication rules. An unqualified ordinary prime
  still delegates to the receipt-backed Eager program when these callers reach it.
- `hybrid_forward.rs`: retain upstream lockstep shared-expert per-row arithmetic and
  CPU row policy with the existing fallback and hidden-export adapter.
- Retain the relocated expert-bank gate API and upstream release tooling. The flags
  conflict combines release and rewrite-provenance meanings. Remove only three exact
  numerical-blob allowlist entries that no longer match the upstream raw-byte scanner;
  preserve the remaining matching, previously authorized gzip exception.
- No unmerged #541 root-adoption source is selected. The public prime signatures are
  unchanged. A later selected `PreparedModelSource`/bound-source composition must keep
  one prepared input through loading and identity capture and requalify affected paths.

The fetched main tree contains three distinct `LOCK.json` / `lock.json` pairs under
`research/spill-c-20260919/{pro-single-day15-review,pro-single-day16-review,pro-single-day16}/gputests`.
They cannot both materialize on a case-insensitive Darwin filesystem. Preserve both Git
blobs unchanged; the build package must use the committed Git objects and materialize a
clean Linux checkout, then validate every actual input there. A Darwin status difference
for these three historical files is not a source edit or a receipt rewrite.

## Additional non-PLE Gemma proof

`qualify-gemma-hidden.py` requires a selected real GGUF with its complete independently
approved SHA256. It refuses a reused result directory and runs inside the existing
physical-card wrapper. For HPOST=0 and HPOST=1 separately, it inspects that artifact,
uses the existing independent verify-prefill versus T1 capture to obtain an Eager-only
receipt, then invokes `rewrite_identity_gate gemma-hpost` in a fresh admitted process.
An admission failure stops the phase; no fallback format, manufactured receipt or
synthetic fixture substitutes for the missing prerequisite.

The native mode requires the canonical Gemma program, no PLE operation, one device,
Eager-only admission, and explicit FAST=0/HPOST at exec. Per mode it runs 12 streams:
direct widths 15/16/17 at fresh and 11-token carried positions, then B3 widths
[15,16,17] and [17,16,15] with the middle stream carried. Each stream keeps four own
greedy continuations and checks their logits against its independent T1 cache.

Before prime, it persists raw T1 rows, independently computed normalized rows, expected
exports, expected seed and seed logits. Scalar f64 RMS evaluation over the loaded,
identity-bound output-norm weights and epsilon supplies the expected normalized values;
it never reads candidate prime exports to construct them. Candidate seed, full hidden
stack, logits and the real `hidden_postnorm_row` pooling-consumer output are retained.
Every row and the last-row seed are checked at the fixed absolute `.005` limit. Raw
HPOST-off exports and all logits must also retain T1 bits. Raw versus normalized rows
must differ beyond that same limit, preventing a vacuous normalization control.

Across the two process settings, raw T1, independently normalized values, actual pooling
outputs and all five logit rows must be byte-identical (156 comparisons). The changed
HPOST export convention is the intended difference. CPU controls reject missing/double
normalization, the first value outside the fixed boundary, nonfinite inputs, missing
cross-mode outputs and failed evidence quarantine. This proves the affected export and
pooling surface only; it is not whole-Gemma, HTTP, quality or performance qualification.

## Fresh build and native scope

Both builds must use the final reviewed commit, fresh owned output directories, verified
toolchain/CUDA inputs and CPU-only staging. Existing ELFs and receipts are preserved.

1. Run the unchanged #547 `tools/qualify-release.py build` v3 recipe with Linux
   bubblewrap. Its fingerprinted compiler view builds exactly kernel-check, run-gen,
   run-spec, argmax-margin-probe, memra-server and tok-parity. Generic capture uses
   those exact six bytes on physical NVML GPU0 under its fresh UUID lease; seal only
   after the wrapper closes. Each shipping OS profile needs its own qualified bytes.
2. The #542 owned v2 builder remains the separate producer for its specialized gate
   and test executables. It does not claim v3 release qualification. Run the composed
   engine/server CPU suites on Linux and retain the actual build and executable hashes.
   Its input closure now also names the Gemma controller, and its test executable
   inventory additionally includes the engine lib suite for this composition check.
3. Requalify the 27 selected callers including all twelve real worker diagnostics,
   original T16 and its matrix, plus baseline admission/MTP and changed transfer and
   restored-prefix/prime regressions. The fresh v3 generic release campaign replaces
   the duplicate legacy battery invocation in the new plan; it does not relabel the
   earlier battery receipt. Exact source, binary and request identities changed, so
   earlier campaign results cannot authorize the composed binaries.
4. Run the separate Gemma phase only once its actual artifact is selected and staged.
   The current-main/Qwen phases need not wait for that artifact or unmerged #541 work.

Inspect existing job namespaces and processes before dispatch. Acquire every physical
card through the authoritative wrapper, recheck actual UUID availability, preserve all
failures, and stop rather than retry a failed phase unchanged. No capacity is acquired
by these scripts. Freeze and review source/controllers before any new native campaign.
