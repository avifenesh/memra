# optipipe increment 1 progress

Started: 2026-08-10T23:23:39Z
Lane: `lane/opti1`
Base: `107201f0` (opti0 TX-release seam merged)
Rig: box1, 2x RTX PRO 6000 Server Edition

## Contract

- Add a forced-diagnostic state fork for one session's consecutive PP-2 K=1 rounds.
- Keep two alternating snapshot/seed generations alive until generation-tagged reconcile.
- On a forced hit, retain the optimistic state without restoring it.
- On a forced miss, drain optimistic stage 0, restore the pre-fork generation, and continue serially.
- Keep ring sessions and round-stream bursts out of the fork path.
- Do not add controller/admission policy, change defaults, publish board numbers, push, tag, or release.

## Success gates

- [x] Forced-hit next-round bytes match serial.
- [x] Forced-miss restore then serial continuation matches serial.
- [x] Abort with a fork in flight drains and tears down cleanly.
- [x] Long alternating-generation hit/miss stress stays exact.
- [x] `MEMRA_SWA_RING=1` refuses the fork path.
- [x] Default serial path is one hash across 10 fresh boots.
- [x] `run-spec` K=1..8 passes.
- [x] `kernel-check` is ALL GREEN.
- [x] Reconcile miss latency and per-device fork peak-memory delta are retained with raw logs.

## Progress

- [x] Read `CLAUDE.md`, DESIGN section 2 and section 6, and opti0 `RESULTS.md`.
- [x] Confirm `lane/opti1` is clean at the merged seam.
- [x] Revalidate CUDA stream/event ordering against current NVIDIA documentation.
- [x] Confirm increment 1 remains forced diagnostics only; increment 2 owns the controller.
- [x] Map current cache, recurrent snapshot, seed, and ticket ownership.
- [x] Implement the forced-hit fork and exact state-identity harness.
- [x] Implement miss reconcile and generation-tagged abort teardown.
- [x] Allocate and refresh both snapshot generations through each PP stage's owning engine.
- [x] Run local checks and the complete requested box1 gate battery.
- [x] Write `RESULTS.md` with increment-2 GO/NO-GO.

## Notes

- The requested `~/.lanectl/inbox/opti1.md` was absent at lane open and appeared during the final
  audit. Its active additions are snapshot/HBM accounting and retention of the merged-seam floor
  triple; `RESULTS.md` records both. The pinned artifact loads as Step35 and has no GDN layers, so
  its measured recurrent snapshot bill and verify-GDN HBM traffic are both zero; no bandwidth
  classification is fabricated for a kernel the target does not execute.
- The design requires generation ownership, not round parity inference: every in-flight ticket
  must carry the generation it keeps alive until hit, miss, or abort retires it.
- If the full miss path cannot be proved in this pass, stop after a committed forced-hit fork and
  receipt the remaining miss/reconcile work, per the lane's L-effort escape hatch.
- Never run `cargo fmt`, `rustup`, or `nsys` in this lane.

## Final gate snapshot

- Final source under test: `4ac646c3`; `0c2a7e2a` adds only the fresh-boot receipt harness.
- Forced hit: 15/15 exact; forced miss: 15/15 exact; alternate: 132 attempts, 66 hit / 66 miss.
- Abort: one generation drained, followed by a healthy 15-round exact session.
- Ring and round-stream: zero attempts, one explicit refusal each.
- Miss reconcile: N=15, median 0.071 ms, range 0.063-0.083 ms; bounded block 26 C to 33 C.
- Full-CTX peak: off/hit both 68,497 MiB on dev0 and 76,881 MiB on dev1; sampled delta 0 MiB.
  Exact incremental payload is 65,812 bytes on dev0 and zero on dev1 for this zero-GDN target.
- Standing battery: serial fresh boots 10/10 one hash, run-spec 8/8, run-gen argmax MATCH,
  kernel-check ALL GREEN (376 OK lines).
