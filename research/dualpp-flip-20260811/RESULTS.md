# Dual-PP default flip — box1 PRO-pair validation (source e94699eba)

Owner call 2026-08-11: "make it the default, create regression tests to validate safety."
Battery: /tmp/dualpp-flip-battery.sh (one flock hold, canary-fixed assert_clean from the
dualpp2 re-gate, same model/drafter/golden/qos_probe as the re-gate). Raw: raw/box1/.

## Verdict: ALL PASS — flip validated on the PRO pair

| Stage | Result |
|---|---|
| 0. kernel-check (incl. new dual-pp-default-flip manifest cell) | ALL GREEN |
| 1. NAKED default (zero dual/overlap env) | dual ENGAGED by default: overlaps=265, slot_pairs=265, slot_collisions=0; golden 21b8293f matched at c=1,2,8,16 (qos exactness=match, 0 divergences); log clean |
| 2. Rollback seam MEMRA_DUAL_PP=0 | serial path; same golden matched c=1,2,8,16; log clean — one flag restores the pre-flip naked path |
| 3. N=5/arm interleaved c=8 perf (one window, warmup discarded) | naked(dual) median 158.065 vs rollback(serial) 133.553 agg tok/s = +18.354% |

Perf rows: naked [157.662, 157.99, 158.065, 158.204, 158.241] (spread 0.37%),
rollback [133.413, 133.54, 133.553, 133.564, 133.681] (spread 0.20%), temps 40-44C both arms,
clocks 2317-2400 MHz (thermal-start/end.csv). The +18.4% naked-default gain at c=8 is consistent
with the dualpp1 flagged-arm receipt (+20.753% at c8) measured pre-flip.

## Safety regression surfaces (owner-ordered)
- pp.rs `flip_*` pure-resolution tests (mode resolve / overlap-follows-mode / route matrix) — 11/11 engine lib.
- worker.rs `flip_naked_default_schedules_dual_on_pp2_and_one_flag_restores_serial` — 191/191 server suite.
- kernel-check `dual-pp-default-flip` manifest cell — battery-wired (H100 law 3), GREEN above.
- decode-batch-gate negative cells now pin MEMRA_DUAL_PP=1 + MEMRA_PP_OVERLAP=0 explicitly and
  restore both env vars (the flip makes unset no longer mean OFF).

## Semantics shipped
MEMRA_DUAL_PP: unset=Auto (dual exactly in the re-gated regime: PP-2 fence, double-slot, peer
transport, B>=2; serial degrade elsewhere — naked PP-3 and host-bounce keep decoding),
0=Off (rollback seam; unset overlap follows OFF), 1=Forced (pre-flip explicit request; binding
refusals stay reachable). MEMRA_PP_OVERLAP unset follows the mode (ON under Auto).
