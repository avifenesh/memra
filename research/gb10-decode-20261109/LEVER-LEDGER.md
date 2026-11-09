# GB10 single-user decode: lever ledger (round 5)

One row per lever or measurement, in the order they ran. Status: shipped, negative, design, measurement.

| # | lever | exactness | result | status | receipts |
|---|---|---|---|---|---|
| r5-0 | scout: c1 decode at 1k / 131k / 260k, nsys of a steady decode window at 1k and 131k | measurement | 22.1 / 10.3 / 6.9 tok/s after the first token (29.8 / 13.5 / 9.4 inside bursts); round 89.5 / 195.8 / 294 ms against a 70.7 / 91.0 / ~111 ms floor at 226 GB/s; the three gaps in README.md | measurement | `scout/` |
| r5-1 | the burst ends at the round its public stream ends (EOS, stop id, spent budget) | visible stream identical (burst size is content-neutral); every cell served the same text in both arms, greedy, sampled and the resumed second turn | GB10: g1k-t2 -0.80 s (-47.9%), s1k -0.35 s (-7.5%), g131k -2.37 s (decode -19.1%), g1k flat (EOS in the burst's last round); RTX 5090: g1k -0.46 s (-18.5%), g1k-t2 -0.29 s (-37.9%), s1k -0.43 s (-17.2%); spreads at most 0.14 s; round cost inside bursts unchanged | shipped | `eos-unit2/`, `eos-5090/` |
| r5-2 | multi-row verify attention | bit-identical per row by construction | design: MULTIROW-ATTN-DESIGN.md | design | |
| r5-3a | verify column states stay in the scan's slabs | bit-identical (same bytes) | pending | | |
| r5-3b | state slot ring through the state pointer table | bit-identical | not started | | |
| r5-4 | K re-priced on GB10 (cold-long rung K=2 vs K=3, for memra#1093) | greedy byte-identical; sampled distribution-exact | pending | measurement | |
| r5-5 | matvec schedules toward the ceiling | bit-identical schedules only | not started | | |
| r5-6 | one graph for the eager single-session round | bit-identical | not started | | |
