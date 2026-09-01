# IQ4_XS qmatvec bandwidth feasibility study — progress

Status: complete (read-only analysis lane)

- [x] Read the lane inbox and repository instructions.
- [x] Confirm dedicated branch `lane/cx-qmatvec-study` at base `4fa5a266` with a clean worktree.
- [x] Inventory the committed ncuspike, solgap, prior IQ4_XS, and q27 evidence.
- [x] Derive IQ4_XS + q8_1 bytes, operations, arithmetic intensity, and roofline placement.
- [x] Trace the byte-permutation/codebook path and compare it with upstream llama.cpp IQ4_XS MMVQ.
- [x] Analyze launch geometry, occupancy, coalescing, and latency-hiding constraints at `n_embd=4096`.
- [x] Rank exactness-preserving and changes-bytes rewrite candidates.
- [x] Write and evidence-check `REPORT.md`; commit only this study directory after final checks.

Verdict: **GO** to an exactness-preserving dense wide-load + `byte_perm` implementation A/B. The
current constant-cache bug is already fixed; the remaining opportunity is fewer/wider load and
lookup instructions. The 50.81% NCU replay rate is not the unperturbed runtime aggregate: exact
logical-byte accounting gives 63.56% of card bandwidth and a 1.80 ms/token absolute qmatvec floor.

Constraints held: no GPU runs, kernel/runtime edits, dispatch arms, flags, formatting, perf-board changes,
merge, tag, or push.
