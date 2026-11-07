# prefix-budget-20261107: the discrete-card prefix cache (budget, victim order, admission reclaim)

Lane `fix/discrete-prefix-budget` (memra#1090), base main `6f5e03e74a`, rebased on `05a8a63503`.
Decision record: [`docs/decisions/PREFIX-CACHE-DISCRETE.md`](../../docs/decisions/PREFIX-CACHE-DISCRETE.md).
Model in every cell: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`),
`MEMRA_CTX=32768` unless stated. Card for every scored cell: one RTX 5090 Laptop GPU (24 GB),
balanced platform profile, 74-80 C under load. Local absolute paths in the receipts are replaced by
placeholders (`<lane>`, `<models>`, ...).

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

The 24 GB RTX 5090 Laptop is memory-bound for this model: boot free 10,053,877,760 B at
`MEMRA_MAX_SESSIONS=12` (8,342,601,728 B at the default caps), and the session peak of every measured
configuration exceeds it, so the derived budget is the two-entry floor (2,259,943,424 B) in every 5090
cell. The budget arm changes nothing on this card; its effect is the PRO 6000's (owed).

## Harness (`harness/`)

- `mixed.py`: the spec-cache-mixed workload with every request streamed (TTFT per row, greedy text per
  row, batch rows tagged `batch-<worker>-<item>` so texts compare across arms).
- `agentic.py`: the `ab_agent.py run` shape (prime every turn with `max_tokens=1`, then the measured pass)
  with per-turn TTFT and cached tokens. The workload replays 29 turns of 18 private coding-agent sessions;
  the committed rows keep the numbers and a sha256 of each output (`sanitize_agentic.py`), not the texts.
- `arm.sh`: one boot, one workload, 250 ms `nvidia-smi` telemetry (device free floor) and 1 s
  `/proc/meminfo` MemAvailable; `phase*.sh` interleave the arms, each phase under its own gpulock lease.
- `policy_sim.py`: the trace-free policy simulator below; `prefix-evict-ab.py`: the incident replay.

## Policy simulator: `sim/policy-sim.txt`

Entry bytes 156.9e6 + 29,700 x tokens, the 32-token grid, whole-entry restores, no session pressure.
Arms: lru, gds (always admit, shipped), gds-refuse (admits only by displacing lower priority), door
(second sight of the 64-token family). At 2,260 MB unless stated:

| shape | lru | gds | gds-refuse | door |
|---|---|---|---|---|
| mixed (14 seeds per repeat), repeats hit | 0/6 | 6/6 | 6/6 | 5/6 |
| incident replay (5090 day-20 geometry, 1 GiB), computed tokens | 28,896 | 28,896 | 30,528 | 45,376 |
| agentic prime then measure (29 turns), computed tokens | 261,534 | 261,534 | 261,534 | 286,558 |
| the same at 8 GB | 261,534 | 163,806 | 163,806 | 274,366 |
| mixed and agentic at the PRO 6000's 54,277 MB | 6/6, 29/29 | 6/6, 29/29 | - | 5/6, 11/29 |

## The incident replay (scored): `rtx5090/phase1d/`

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

Ten ties of ten: both orders computed 29,600 tokens per run (the day-20 LRU median, to the token),
returned 1,696 cached tokens to the cohort and matched the cold digests 28/28. The only difference is one
more eviction per run under gds (20 vs 19). Loop TTFT p50 188-194 ms in both arms. Value per byte does not
lose the shape the 2026-09-21 decision was made on.

## The live-free insert clamp, measured and rejected: `rtx5090/phase1a-clamp/`, `rtx5090/preview1a/`

The first scored candidate (`39ec446dd5`) also applied GB10's live-free insert clamp on discrete cards. The
mixed shape, three runs per arm interleaved ABCDE EDCBA ABCDE, `MEMRA_MAX_SESSIONS=12`:

| arm | repeats hit (each run) | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims | device free min MiB |
|---|---|---|---|---|---|---|
| base (`6f5e03e74a`) | 2, 3, 3 | 489 | 94.9 | 1471 | 6 | 1267 |
| clamp + partial reclaim, LRU | 0, 0, 0 | 763 | 93.1 | 1456 | 0 | 2630 |
| clamp + partial reclaim, GDS | 1, 1, 1 | 758 | 96.2 | 1489 | 0 | 2611 |
| clamp + reclaim flush, GDS | 1, 1, 1 | 797 | 94.4 | 1521 | 0 | 2611 |
| no clamp + partial reclaim, GDS (`7b385110e4`) | 6, 6, 6 | 350 | 95.5 | 1438 | 18 | 1235 |
| off (one run) | 0 | 664 | 110.2 | 688 | 0 | 2931 |

Under load the clamp held the cache to about one entry, and since an insert never evicts itself, every
new seed displaced the shared entry. Without the clamp the value-per-byte order kept it through every
repeat. Greedy texts matched the cache-off run on every compared row in every arm. The clamp is not in
the shipped change; the preview (`preview1a/`, single runs of `7b385110e4`) had shown the same 6/6.

The agentic c8 preview (`preview1b/`) answered 408 on both base and `7b385110e4`: 29 prompts of up to
15k tokens at c8 do not reach a first token inside the 90 s streaming deadline on 24 GB, so the scored c8
cell runs with `MEMRA_STREAM_TTFT_MS_MAX=900000` as a reclaim stress cell, compared only across arms.

## Scored arms, final candidate: FILL
