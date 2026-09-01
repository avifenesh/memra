# dualpp2 PRO-pair RE-GATE — RESULTS (box1, 2x RTX PRO 6000 Blackwell Server Edition)

Source commit (frozen): `64a869252029610b833e9b469f4708044b558b3e` (branch lane/cx-dualpp2)
Rig: box1, 2x RTX PRO 6000 Blackwell Server Edition. One inherited GPU lock hold across all stages.
Golden hash: `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`

## Verdict: ALL PASS — gates 1-3 GREEN => gate-4 (default flip) promotion eligible (OWNER-GATED)

The dual-active PP path (MEMRA_DUAL_PP=1 + MEMRA_PP_OVERLAP=1) is bit-identical to serial, adds zero
cross-device slot collisions, and adds no admission thrash over serial on this pair. It remains
DEFAULT OFF; flipping it to default is the separate owner-gated orchestrator promotion (step 4:
default flip + serve-shape N=5 A/B), NOT done here.

## Gate 1 — correctness (STAGE_CORRECTNESS_PASS 19:12Z)
PP split bit-identity B=1..5 BIT-IDENTICAL; dual liveness +8 overlaps/rep. Raw: raw/box1-regate/correctness/.

## Gate 2 — serve-stress admission-thrash (RESUME_SERVESTRESS_PASS 19:29:28Z)
verdict=PASS, dual_adds_no_thrash=True. serial/dual/teeth each 64/64 completed, worker alive, logs
clean. Admission-counter delta serial==dual (admission_session_defers / admission_vram_defers /
step_oom_parks = 0/0/0 both arms) => blocker #6 live PASS. Teeth non-binding on the 192GB pair at
c=64 (defers==0; admission math not exercised at this headroom) — recorded, not a lane failure.
NOTE: this summary was RE-REDUCED with the canary-fixed bad-log grep (see below); the original
false-FAIL summary is preserved as servestress/summary.json.false-fail. Raw: raw/box1-regate/servestress/.

## Gate 3 — 10-boot alternating soak + x100 cross-device collision soak (SLOT_SOAK_PASS + RESUME_SOAK_PASS 19:59:33Z)
Protocol: 10 alternating fresh boots (serial/dual), c=1..17 one-hash matrix then rotated mixed widths.
- one_hash_matrix: PASS, 34 points, widths 1-17, every request == golden hash.
- serial arm: 929/929 golden_matches (101 points x 5 boots). dual arm: 929/929 golden_matches.
- slot totals: collisions=0 across pairs=9123
  (slot_0_uses == slot_1_uses == pairs; perfectly balanced dual-slot use, ZERO collisions).
- per-boot dual overlaps 1473..1916, slot_collisions 0 every boot.
- thermal regime: sm_clock 180..2422 MHz, temp 26..52C, 14078 samples @250ms, no artificial cooldown.
Raw: raw/box1-regate/soak/ (driver.log, slot-soak.json, per-boot server logs, gpu.csv).

## Canary fix (why this was a RE-gate, not the first gate)
The prior servestress reducer + soak assert_clean matched benign peer-probe telemetry `mismatches=0`
via a case-insensitive `MISMATCH` token, forcing verdict=FAIL on GREEN data (all arms completed,
worker alive, no thrash). Fixed at source: MISMATCH made case-sensitive + `mismatches=[1-9]` added
for real nonzero counts + `illegal` narrowed to `illegal memory access`/`ILLEGAL_ADDRESS`.
Canary-verified: 0 on the clean captured log, 7/7 on injected faults, 0 on benign `mismatches=0`.
This is the vacuous-canary class the directive warns about; fixed, not worked around.

## Follow-ups keyed off this re-gate
- cx-peerprobe RESULTS.md native two-card cells: box1 is that native PP-2 window — flip PENDING to
  these receipts + hash (kimi b8a2fe62). See QUEUE-peerprobe-runtime.md item A.
- Runtime peer re-probe (kimi b336ba03): this soak bounds the run-time corruption class on THIS pair
  (0 divergences under sustained load); the general-case idle re-probe is a queued sec follow-up.
