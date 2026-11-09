# GB10 single-user decode: lever ledger (round 5)

One row per lever or measurement, in the order they ran. Status: shipped, negative, design, measurement.

| # | lever | exactness | result | status | receipts |
|---|---|---|---|---|---|
| r5-0 | scout: c1 decode at 1k / 131k / 260k, nsys of a steady decode window at 1k and 131k | measurement | 22.1 / 10.3 / 6.9 tok/s after the first token (29.8 / 13.5 / 9.4 inside bursts); round 89.5 / 195.8 / 294 ms against a 70.7 / 91.0 / ~111 ms floor at 226 GB/s; the three gaps in README.md | measurement | `scout/` |
| r5-1 | the burst ends at the round its public stream ends (EOS or a stop id; a gone client too) | visible stream identical (burst size is content-neutral); every cell served the same text in both arms, greedy, sampled and the resumed second turn | GB10: g1k-t2 -0.80 s (-47.9%), s1k -0.35 s (-7.5%), g131k -2.37 s (decode -19.1%), g1k flat (EOS in the burst's last round); RTX 5090: g1k -0.46 s (-18.5%), g1k-t2 -0.29 s (-37.9%), s1k -0.43 s (-17.2%); spreads at most 0.14 s; round cost inside bursts unchanged. At the rebased head: 22 of 22 warm == cold pair cells on the RTX 5090 and 6 of 6 on GB10 (7 cases moved from the affinity tier to an exact hit that the grid law rewinds), 14 of 14 pooled-cohort streams identical to main | shipped | `eos-unit2/`, `eos-5090/`, `eos-pairs-5090/`, `eos-pairs-unit1/`, `eos-pool-5090/` |
| r5-2 | multi-row verify attention | bit-identical per row by construction | design: MULTIROW-ATTN-DESIGN.md | design | |
| r5-3a | verify column states stay in the scan's slabs | bit-identical (same bytes) | pending | | |
| r5-3b | state slot ring through the state pointer table | bit-identical | not started | | |
| r5-4 | K re-priced on GB10: the cold-long rung at K=2 against K=3 (for memra#1093), 12 boots k3 k2 k2 k3 x3 on unit 1 | greedy text identical across arms; sampled stays distribution-exact | K=2 does not win: T=1 1.3k / 3.6k / 5.8k at 0.987 / 0.969 / 0.968 of K=3 (K=2 faster in 4 / 0 / 0 of 36 boot pairs), greedy 3.6k 0.998 (0 of 36), TTFT unchanged; four of six sampled prompts 2.5-6.5% slower, two flat. GB10 keeps K=3 on the rung. Lead: the 34-token prompts ran 1.9% and 6.9% faster at K=2 (the door moved them too on GB10's cost-model K) | measurement | `kcl-unit1/` |
| r5-5 | matvec schedules toward the ceiling | bit-identical schedules only | not started | | |
| r5-6 | one graph for the eager single-session round | bit-identical | not started | | |
