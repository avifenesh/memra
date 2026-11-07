# prefix-budget-20261107: the discrete-card prefix cache (budget, insert clamp, victim order, reclaim)

Lane `fix/discrete-prefix-budget`, base main `6f5e03e74a`. Decision record:
[`docs/decisions/PREFIX-CACHE-DISCRETE.md`](../../docs/decisions/PREFIX-CACHE-DISCRETE.md).
Model in every cell: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`),
`MEMRA_CTX=32768` unless stated.

## The finding (one RTX PRO 6000 Max-Q, 96 GB): `pro6000-finding/`

The spec-cache-mixed gate (`tools/spec-cache-mixed-gate.sh`: one greedy warm request with a 520-token
shared prompt, then c8 unique-prompt batch load with six greedy repeats of the shared prompt 3 s apart,
`MEMRA_MAX_SESSIONS=12`):

- boot line: `budget 2260MB (2259943424 B, derived: 2 x 1129971712 B max entry ... boot driver free
  86614081536 B, post-reserve clamp 85003468800 B)`;
- a 64-token seed is 158.8 MB, the shared 480-token spec-boundary entry 172.1 MB: every entry carries
  156.9 MB of GDN recurrent state, against 29.7 KB per cached token;
- 14 seeds fill the budget; the shared entry was evicted before the first repeat:
  `cached=[0, 0, 0, 0, 0, 0]`, 105 eviction lines.

## Arms

| arm | binary | what it runs |
|---|---|---|
| base | main `6f5e03e74a` | two-entry budget, no insert clamp, plain LRU, reclaim flush |
| lru | candidate, `MEMRA_PREFIX_EVICT=lru` | session-peak budget, live-free insert clamp, plain LRU, partial reclaim |
| gds | candidate (default) | the same with the GreedyDual-Size order |
| off | `MEMRA_PREFIX_CACHE_MB=0` | no prefix cache (the exactness reference) |

The RTX 5090 Laptop is memory-bound for this model: boot free 10,053,877,760 B, and the session peak
of every measured config exceeds it, so the budget is the two-entry floor (2,259,943,424 B) in every 5090
cell; the budget arm's effect is measured on the PRO 6000 (owed).

## Policy simulator: `sim/policy-sim.txt`, `harness/policy_sim.py`

Entry bytes 156.9e6 + 29,700 x tokens, the 32-token grid, whole-entry restores, no session pressure.
Arms: lru, gds (always admit, shipped), gds-refuse (admits only by displacing lower priority), door
(second sight of the 64-token family). At 2,260 MB:

| shape | lru | gds | gds-refuse | door |
|---|---|---|---|---|
| mixed (14 seeds per repeat), repeats hit | 0/6 | 6/6 | 6/6 | 5/6 |
| incident replay (5090 day-20 geometry, 1 GiB), computed tokens | 28,896 | 28,896 | 30,528 | 45,376 |
| agentic prime then measure (29 turns), computed tokens | 261,534 | 261,534 | 261,534 | 286,558 |
| the same at 8 GB | 261,534 | 163,806 | 163,806 | 274,366 |

## The incident replay, RTX 5090 (scored): `rtx5090/phase1d/`

`tools/prefix-policy-ab.py` (deleted with the SLRU door, recovered from `9466b891` as
`harness/prefix-evict-ab.py`, arms renamed lru / gds), the day-20 geometry of
`docs/decisions/PREFIX-CACHE-POLICY.md`: budget 1024 MiB, ctx 16384, cohort 1248/1344/1440/1536 ids
seeded twice, one loop 10,912 + 160 ids per turn over 12 turns, returns every third turn,
`MEMRA_REUSE_POOL=0`, AB/BA x 5 pairs, one lock hold, a cache-off boot for the cold digests. Binary
`7b385110e4` (its victim-order code is the shipped code; run with `MEMRA_ENV_AUDIT=warn` because its door
registry predates the `MEMRA_PREFIX_EVICT` row; the first attempt refused on exactly that and is kept in
`attempt1-env-audit-refused/`). Verdict, verbatim:

```text
PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=792920064 turns=12 start_tokens=10912 grow=160 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens lru_median=29600 gds_median=29600 (N=10 each) pairs_lru_better=0/10 pairs_gds_better=0/10 ties=10/10 return_cached lru_median=1696 gds_median=1696 loop_cold_after_1 lru_max=0 gds_max=0 refusals lru=0 gds=0 temp_c=78.0..80.0 power_limit_w=None -> INCONCLUSIVE
```

Ten ties of ten: both orders computed 29,600 tokens per run (the day-20 LRU median, to the token), returned
1,696 cached tokens to the cohort, and matched the cold digests 28/28. The only difference is one more
eviction per run under gds (20 vs 19). Loop TTFT p50 188-194 ms in both arms.

## Preview, RTX 5090 (single runs, not scored): `rtx5090/preview1a/`

The mixed shape once per arm on the first candidate `7b385110e4`: the partial reclaim and the order, but
no live-free insert clamp and no owed-bytes term.

| arm | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims | evicted in reclaims |
|---|---|---|---|---|---|---|
| base (phase0, `6f5e03e74a`) | 2/6 | 610.7 | 111.6 | 1294 | 6 | 11,949 MB |
| gds, partial reclaim, no clamp | 6/6 | 329.0 | 103.0 | 1307 | 21 | 8,575 MB |
| lru, partial reclaim, no clamp | 2/6 | 763.6 | 98.5 | 1347 | 24 | 9,382 MB |
| off (phase0) | 0/6 | 573.9 | 122.4 | 621 | 0 | 0 |

Value per byte kept the shared entry; the partial reclaim alone fired almost every admission because the
next seed refilled what it freed, and the batch tier paid for it. That is the reason for the live-free
insert clamp in the scored candidate.

## Scored arms, RTX 5090: FILL
