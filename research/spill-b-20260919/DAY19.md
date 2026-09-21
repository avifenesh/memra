# WP-B day 19: both capture sites on the GDN prime grid (memra#602), the #379 gate green, both cards

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`. Day 18 aligned the plain prompt-end seed
and left the spec session's `capture_at` at the prompt end; `tools/spec-on-cache-hit-gate.sh qwen` then
read red on r3 and g2 because its identity law compares spec-on with spec-off and only the spec-off side
had moved onto the grid. Lead ruling: the grid law applies to every prefix-cache capture the server makes,
the spec session's `capture_at` included; one capture law, one restore program. This day completes the
fix, restates the #379 gate's accounting under the law without touching its identity assertion, fixes the
evict-reclaim gate's V3 arithmetic in the gate, and reruns everything on both cards on the completed fix.
Every GPU cell is N=1, `executed-not-qualified` (or `failed` where the gate's exit says so); no timing is
compared with any other card.

## The fix (commit `49d79b946`: worker.rs and spec.rs)

- Cold spec session: `capture_at` is `seed_capture_boundary(prompt.len(), 0)` (a `snapshot_at` boundary,
  already on the grid, keeps precedence); an aligned length under the entry floor is the same typed,
  counted `seed REFUSED (grid)` line as the plain seed's. The capture boundary is the prime split in its
  own right and the affinity boundary rides through `ckpt_at`: `trunk_schedule` takes both stops and the
  engine captures where a chunk ends on `capture_at`; the old `min()` dropped whichever stop came later.
- Restored spec session: `republish_at` is the render-stable boundary ahead of the restored length (as
  before, grid-aligned) or, when none lies ahead, the seed's grid boundary above the restored length
  (`Covered` publishes nothing). The engine's prompt-end republish (`spec.rs`) fires only when the prompt
  end is itself a grid multiple, and a restored session's `capture_at` is `None` off the grid when no
  republish boundary was armed.
- No prime program moves: the splits the fix adds are the measured bit-identical ones (the cold digests of
  the twin gate equal day 18's 12/12 on the local card).

## The gates

- `tools/spec-on-cache-hit-gate.sh`: the identity law (spec-on text == spec-off text) is untouched. The
  accounting clauses state the law's numbers with the reason beside them: the s and sp identical repeats
  restore the whole published entry, `capture_len(P)` (64 of 106 on this fixture), and cold-prime the rest
  from the grid; g2 restores `capture_len(P1)`; g3 and g4 restore the republished render-stable boundary
  (96 of 119, the raw-completion guard window on the grid), never g2's whole prompt. A new `fc` pair keeps
  the whole-prompt full-cover shape and its `restore-full-cover` boundary-draw site exercised: PROMPT
  padded with " ok" words until `/v1/tokenize` counts a multiple of 32 (128 tokens, 22 filler words),
  sent twice sampled at one seed. The first day-19 run also asserted a `restore-suffix-feed` site; the
  deferred restore draws its first token in the prime path, that site belongs to the legacy non-deferred
  shape and is unreachable from these cells, and the assertion (this lane's own addition, never shipped)
  was removed the same day at `fa5d7a014`; the first run's log is kept.
- `tools/prefix-evict-reclaim-gate.py`: V3 adds `pool_retained_bytes` to the driver delta and the trim
  release. Its docstring already called those bytes counted headroom; the code read the driver delta alone
  and passed on day 13 and on base only while the shape evicted a second (busy-peer) seed beside E1. A
  gate bug fixed in the gate; the fix under test was not touched for it.
- `tools/kv-host-tenant-reclaim-gate.sh`: the r7 and r8 promote equalities read `capture_len` of the
  leader's prompt_tokens, because the spec publications those cells restore are on the grid now (the
  target card's first run today read `insert (spec-boundary): 64 tokens` for the 83-token leader and
  `hit: 64 of 96`, and failed the two whole-prompt equalities and nothing else; the rerun on the corrected
  gate is green).

## Cards, binaries, regimes

Local: NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, driver 595.84, clocks not pinned, first telemetry
sample 15 MiB in every collector cell, cells between 62 and 88 C, peak draw 179.2 to 179.9 W; the collector
on `/tmp/memra-5090.lock` (inherited FD, `tier-lock-proof.py`), the #379 gate and local-ci on their own
flock of the same file. Target: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W,
driver 580.178.04, lane C running beside this lane at the start (its collector held the lock; every cell
of this lane got the lock on its first attempt, `retries.log` absent), first sample 0 MiB, 32 to 61 C,
peak draw 496 to 504 W; the collector on `/tmp/memra-gpu.lock`. Artifact on both:
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`; the #379 gate uses `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` without the
external drafter (absent on this rig, the assertion no-ops as local-ci's WARNING branch says).

Binaries: fix `49d79b946`, local SHA-256
`20fa4dc7028793a3c203d1c0c277bbba20406e93af632cc13f19b0d7c27db65a` (`rtx5090-day19/build-fix/`, 45 s
incremental), target SHA-256 `35c077002b9f586b1583ca9a5134803532628e677e79dc82da8fb370a0a63d54`
(`pro-single-day19/build-fix/`, 35 s). Base: day 18's `origin/main` `1b354be59` builds on each card
(local `79e549b3...`, target `449a7430...`). The commits after the cells (`fa5d7a014`: the #379 site
list, lane A's equalities, docs) touch no crate; the cells that read those gates were rerun on them.

| card | cell | binary | collector status | samples | temp C | peak `memory.used` MiB | peak draw W |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| 5090 | `gate-fix-day16-shape` | fix | `executed-not-qualified` (exit 0) | 743 | 62..88 | 21,785 | 179.2 |
| 5090 | `restore-fix-t10` | fix | `executed-not-qualified` (exit 0) | 298 | 65..87 | 22,105 | 179.9 |
| 5090 | `hitgate-base`, `hitgate-fix` (first gate source) | base, fix | exit 1, exit 1 (below) | own logs | | | |
| 5090 | `hitgate-base-2`, `hitgate-fix-2` (corrected gate) | base, fix | see below | own logs | | | |
| 5090 | `local-ci` (correctness stage), `local-ci-perf` (`--perf`) | fix | see below | own logs | | | |
| PRO 6000 | `gate-fix-day16-shape` | fix | `executed-not-qualified` (exit 0) | 403 | 32..60 | 22,067 | 504.2 |
| PRO 6000 | `restore-fix-t10` | fix | `executed-not-qualified` (exit 0) | 164 | 39..58 | 26,259 | 500.9 |
| PRO 6000 | `serve-smoke` | fix | `executed-not-qualified` (exit 0) | 86 | 42..55 | 20,277 | 496.4 |
| PRO 6000 | `cache-meter` | fix | `executed-not-qualified` (exit 0) | 29 | 40..46 | 17,845 | 180.7 |
| PRO 6000 | `kv-host-fix` (whole-prompt equalities) | fix | `failed` (exit 1, the two equalities) | 56 | 39..52 | 20,277 | 486.5 |
| PRO 6000 | `kv-host-fix-2` (gate at `fa5d7a014`) | fix | `executed-not-qualified` (exit 0) | | | | |
| PRO 6000 | `evict-reclaim-fix` | fix | `executed-not-qualified` (exit 0) | 385 | 39..61 | 94,484 | 499.3 |
| PRO 6000 | `evict-reclaim-base` | base | `executed-not-qualified` (exit 0) | | | | |

`tools/tier-battery.py --validate` exit 0 on every collector cell of the target card.

## The twin gate and the restore gate on the completed fix

Local RTX 5090, day-16 shape (cohort 1,250/1,350/1,450/1,550; 12 turns 11,000 + 150), verbatim:

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=789118976 turns=12 cold_turns_after_1=0 cached_ok=11/11 lines_ok=12/12 evictions=10 cohort_evictions=4 self_evictions=0 refused_or_skipped=0 effective_free_ok=12/12 identity_ok=12/12 grid_ok=31/31 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
```

The twelve measured digests and the twelve cold digests equal day 18's (`verify-day19.py`): the spec-side
change moved nothing on the plain path. Restore gate, turn 10, verbatim:

```text
PREFIX-RESTORE-IDENTITY: target=12350 points=5 identical=5/5 grid_ok=5/5 grid=32 12288(seed12288,restored12288,off0,suffix62):yes 12320(seed12320,restored12320,off0,suffix30):yes 12200(seed12160,restored12160,off0,suffix190):yes 12250(seed12224,restored12224,off0,suffix126):yes 12300(seed12256,restored12256,off0,suffix94):yes V1=ok V2=ok -> PASS
```

Target card, the same two gates on the fix, verbatim:

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=789118976 turns=12 cold_turns_after_1=0 cached_ok=11/11 lines_ok=12/12 evictions=14 cohort_evictions=4 self_evictions=0 refused_or_skipped=0 effective_free_ok=12/12 identity_ok=12/12 grid_ok=31/31 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
PREFIX-RESTORE-IDENTITY: target=12350 points=5 identical=5/5 grid_ok=5/5 grid=32 12288(seed12288,restored12288,off0,suffix62):yes 12320(seed12320,restored12320,off0,suffix30):yes 12200(seed12160,restored12160,off0,suffix190):yes 12250(seed12224,restored12224,off0,suffix126):yes 12300(seed12256,restored12256,off0,suffix94):yes V1=ok V2=ok -> PASS
```

## The #379 gate (`tools/spec-on-cache-hit-gate.sh qwen`), local RTX 5090

First run, the gate source at `49d79b946` (the day-19 accounting plus the `restore-suffix-feed` site this
lane had added), verbatim summaries:

```text
base: SPEC-ON-CACHE-HIT GATE: 3 FAILURE(S) (qwen)
      FAIL: s7 sampled hit restores the whole published entry (cached == cap(prompt) == 64)   (and s1234, s99991, sp: the identical repeat restores 106 of 106 on base)
      FAIL: g2 restores g1's published entry (cached == cap(P1) == 64)   FAIL: g3's restored prefix == turn 2's republished boundary (96 of 119)   FAIL: g4 (repeat of turn 2) restores the whole REPUBLISHED entry (cached == 96)
      FAIL: boundary site restore-suffix-feed never fired (untested code path)
fix:  SPEC-ON-CACHE-HIT GATE: 1 FAILURE(S) (qwen)
      FAIL: boundary site restore-suffix-feed never fired (untested code path)
```

On the fix every identity clause was green (`r1`..`g2 spec==plain byte identity`, the s and sp hits'
bytes equal to their cold leaders at the same seed, the fc full-cover hit's bytes equal to its leader's,
g4's continuation equal to g2's), every accounting clause read the law's number (64 of 106, 64 of 119,
96 of 119, the fc pair 128 of 128), and the one red was the site this lane had wrongly required: the
deferred restore primes the carried suffix through the walker and draws its first token in the prime
path, so `restore-suffix-feed` is the legacy non-deferred shape's site. The assertion was removed at
`fa5d7a014` (the day's own addition, one run old, never a shipped check) and both arms were rerun on the
corrected gate. Base's three shell-level failures are the accounting blocks (its entries sit at the
prompt end) plus the same site line; its identity clauses were green because both of its boots ran the
same off-grid program (the day-18 finding).

Reruns on the corrected gate (`fa5d7a014`), verbatim summaries:

```text
fix:  SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)          (61 ok, 0 FAIL)
base: SPEC-ON-CACHE-HIT GATE: 2 FAILURE(S) (qwen)       (54 ok, 7 FAIL, every one an accounting clause)
      FAIL: s7 sampled hit restores the whole published entry (cached == cap(prompt) == 64)   (s1234, s99991, sp likewise)
      FAIL: g2 restores g1's published entry (cached == cap(P1) == 64)
      FAIL: g3's restored prefix == turn 2's republished boundary (96 of 119)
      FAIL: g4 (repeat of turn 2) restores the whole REPUBLISHED entry (cached == 96)
```

On the fix every cell is green: the greedy r1..r3 and the growth turns g1 and g2 byte-identical between
the spec-on and spec-off boots (`r1`, `r2`, `r3`, `g1`, `g2 spec==plain byte identity`), the three seeded
sampled repeats and the penalized one restoring 64 of 106 with bytes and acceptance equal to their cold
leaders, the sampled suffix hits reproducing at one seed, the plane-less refusal real, the on-grid fc
pair a true full-cover hit (128 of 128) with bytes and acceptance equal to its leader, the growth turns
restoring 64 then 96 with g4 reproducing g2's continuation and acceptance, the boundary sites
`cold-prime`, `burst-tail-commit` and `restore-full-cover` all fired, and the sampled first token drawn.
On base the identity clauses are all green as well (both of its boots restore the same off-grid entry);
its seven reds are exactly the accounting clauses the law moved, which is what "red on base" should
read: the restore point, not the bytes, is what the pre-fix binary gets wrong on this fixture.

`tools/local-ci.sh` (the correctness stage, once, on the fix, the rig lock held for the whole run):
exit 0; inside it the hit gate ran in both postures, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` twice
(default and `MEMRA_HITGATE_TEETH=1`), the gemma arm SKIP for its absent artifact, three pair-only GPU
tests SKIP on a one-card rig, everything else PASS (`rtx5090-day19/local-ci/local-ci.log`).

## The evict-reclaim gate under the corrected V3 (target card), verbatim

```text
fix:  PREFIX-EVICT-RECLAIM: entry_bytes=1590853632 reclaim_credit_bytes=1591000000 driver_free_delta_bytes=1442840576 trim_released_bytes=1442840576 pool_retained_bytes=148013056 p2=admit-same-tick busy_overlap_s=21.658 identity=aa6cc3291b981646 V1=ok V2=ok V3=ok V4=ok -> PASS
base: PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=1751000000 driver_free_delta_bytes=1610612736 trim_released_bytes=1610612736 pool_retained_bytes=140549120 p2=admit-same-tick busy_overlap_s=21.659 identity=aa6cc3291b981646 V1=ok V2=ok V3=ok V4=ok -> PASS
```

The numbers are day 18's to the byte (same card, same shape): the fix's reclaim evicts exactly E1 because
the 71-token busy peer's seed is refused (`e0_busy_peer_entry_bytes` 0), base's evicts E1 plus that seed;
under V3 as it is now stated (`driver_free_delta + pool_retained >= E1 - one granule`, likewise for the
trim release) both arms pass, and the ~140 to 148 MB the pool retains on this card reads as the headroom it
is. The tokens are byte-identical across arms and days (`identity=aa6cc3291b981646`).

## Lane A's gate, serve-smoke, cache-meter (target card, the fix)

`tools/kv-host-tenant-reclaim-gate.sh fix`, gate at `49d79b946` (whole-prompt equalities): `GATE:
kv-host-tenant-reclaim (fix arm) FAIL (2 assertions)`, exactly `beta's entry promotes at r7 ...` and
`fix: r8 promotes the reclaimed admission (cached_tokens == r6 prompt_tokens)`; the server log:
`insert (spec-boundary): 64 tokens` for the 83-, 89-, 86-, 93- and 95-token leaders (96 for the 96-token
one, on the grid), `hit: 64 of 96`, `hit: 64 of 108`; r7 `cached_tokens` 64 = `capture_len(83)`, r8 64 =
`capture_len(95)`. Rerun on the gate at `fa5d7a014`: `GATE: kv-host-tenant-reclaim (fix arm) PASS`, 28
`ok` lines, no `FAIL`. `serve-smoke.sh`: `serve-smoke: 0 failed` (spec, gemma4 and Q35 arms SKIP for absent
artifacts, as on days 14, 15 and 18). `cache-meter-gate.py --n 5 --k 256`: `cache-meter-gate: 0 failed`.

## Local checks (the tree at `fa5d7a014`)

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy -p memra-server -p memra-engine --offline --all-targets -- -D warnings` | PASS (`rtx5090-day19/clippy.log`) |
| `cargo test -p memra-server --offline` | `758 passed; 0 failed; 8 ignored` (`test-server.log`) |
| `bash tools/check-flags.sh` | `no uncovered runtime names` (865 reads; no new `MEMRA_*` read) |
| `bash tools/docs-registry-census.sh`, `git diff --check` | PASS, clean |
| `tools/local-ci.sh` (correctness stage) | exit 0, hit gate `ALL GREEN` in both postures (above) |
| `tools/local-ci.sh --perf` (the pre-push hook's perf-ci freshness gate names it for engine-file changes; `spec.rs` moved today) | see the closing note below |
| `verify-day19.py` | `DAY19 REPLAY OK` |

## Closing note: the perf battery and the push

`tools/local-ci.sh --perf` on the local card (the pre-push hook's perf-ci freshness gate named it: engine
files touched after the last battery, `spec.rs` today): exit 0, the correctness stage inside it green
again (hit gate `ALL GREEN` in both postures), then `perf stage: 0 fail, 0 warn` with the cells this rig
has (`26b-plain-short: 208.24 tok/s [OK]`, `qwen9b-plain-short: 138.54 tok/s [OK]`; the 31B, e4b and
26b-spec cells SKIP for absent artifacts), two rows appended to `research/tune-data/perf-ci.jsonl` (committed
with this record). The first push attempt at `49d79b946` was refused by that gate (`push-1.log`); the
attempt after the battery was allowed (`push-2.log`: `3e261c1c2..fa5d7a014`). No skip variable, no
`--no-verify`, at either attempt.

## Boundaries and record

- Forbidden list honored: the identity law `spec-on text == spec-off text` untouched; no prime program
  moved; no skip variable, no `--no-verify` (the first push attempt at `49d79b946` was refused by the
  hook's perf-ci freshness gate: `engine files touched after the last perf-ci battery. Run:
  tools/local-ci.sh --perf`; the commit was shipped to the target card as a git bundle and the battery is
  queued on the local card, see the closing table); no third lock name; no bare GPU run; `/root/artifacts`
  and `/root/memra-spill` untouched; no host, id, location or cost in a tracked file.
- Receipts: `rtx5090-day19/` (build, collector cells, the `hitgate-*` directories with the gate's evidence
  and logs, `local-ci/`, `local-ci-perf/`, chain and driver logs, `card-before-*.csv`, CPU check logs,
  push logs) and `pro-single-day19/` (the target card's `b-day19` mirror). Drivers:
  `run-day19-*.sh`, `chain-day19-*.sh`, `pro-single-day19/{build-arm,run-all,run-rest,cache-meter-cell}.sh`.
  Replay: `verify-day19.py`.
