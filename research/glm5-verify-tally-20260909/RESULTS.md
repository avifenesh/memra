# GLM verify tally status, 2026-09-09

Verdict: **TALLY BANKED; CANDIDATE UNQUALIFIED**. rev: 2026-09-23

TALLY.md contains the measured ranked tables at t2/4/7, launch and gap census,
method, identity and raw archive hash. At t4: 2933 launches/round, 65.067 per
trunk layer. The selected E4M3 six-projection target costs 3811.757 us/round;
the 70%-HBM screening ideal is 619.128 us. Candidate launch elimination
estimates 651.270 us at t2 and 712.470 us at t4 before matvec savings.

The fusion is not yet qualified. No byte-exact oracle PASS, ABBA saving,
KEEP verdict, serving win or default change is claimed by this bank commit.
The next cell runs the byte-exact real-input oracle at all three widths
before any warmed ABBA x5 timing. KEEP still requires >=0.5 ms/round under
the existing conditional width mix. Negative results remove the candidate
and its door in this lane. The three prior settled component verdicts stand.

No local cargo or GPU work ran. Push uses MEMRA_SKIP_PERF_CI=1.
publicity: skipped - maintenance research record.
