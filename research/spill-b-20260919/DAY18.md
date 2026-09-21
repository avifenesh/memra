# WP-B day 18: the prompt-end seed lands on the GDN prime grid (memra#602), both cards

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `9a1a717ce` at
`7578d6bf4` and again with `1b354be59` (the lane C integration, #599) at `03beaa0b9`, the commit that
carries the fix. Day 17 named the two numeric programs behind the day-16 digest divergence: the cold prime
(every `prime_cache` call starting on the 32-token GDN WY-chunk grid) and the restored render of a growing
`prompt_ids` conversation (a restore at the prompt-end seed boundary, an arbitrary position, then one
off-grid suffix prime call), and the engine's own law (`align_prime_ranges_to_gdn`) says the second is not
bit-identical to the first under the chunked GDN scan. Owner law: one numeric program per request; a
restored suffix produces the cold prime's bytes. This day fixes it at the capture, turns the twin gate's
identity column into a verdict, adds the restore-point gate, and runs both on both cards, base and fix.
Every GPU cell is N=1, `executed-not-qualified` in the collector's vocabulary or `failed` where the gate's
exit code says so; no timing is compared with any other card.

## The fix (worker.rs, commit `269178070`, merged as `03beaa0b9`)

`seed_capture_boundary(prompt_len, hit_len)` decides where the prompt-end seed publishes:

- `AtPromptEnd` when the prompt end is a multiple of `Engine::gdn_chunk_size()`: the whole-prompt seed
  publishes at prefill-done, as before.
- `AtBoundary(b)` otherwise: `b` is the largest grid-aligned fed-length under the prompt end that leaves at
  least `PRIME_MIN_T` (16) prompt tokens behind it (`grid_align_boundary_within`, the same function the
  LCP, first-message and stable-boundary captures already use; a 1..15 remainder steps one grid unit
  further down, the W1 floor). The boundary is armed as `Session::seed_at`, honored by `prefill_tick`'s
  `bound_rem` beside `snapshot_at` and `ckpt_at`, and published by `maybe_prefix_seed` the instant
  `fed == seed_at`; the remainder then primes as one more `prime_cache` call from a grid start, the split
  the measured law calls bit-identical to the monolithic prime.
- `Covered` on a hit whose aligned boundary does not lie `PRIME_MIN_T` past the restored entry (an exact
  re-send, or a re-send extended by less than a grid step): nothing to publish, silently, the same outcome
  `prefix_seed_deepens` already gave.
- `Refused { aligned }` when the aligned length is under `PREFIX_CACHE_MIN_TOKENS` (64): a typed
  `[prefix-cache] seed REFUSED (grid): ...` line and the new `prefix_cache_seed_grid_refusals` counter
  (`/metrics`, operator-only). A prime path that does not stop on the armed boundary refuses the
  off-grid prompt-end state with the same line instead of publishing it.

No prime program changed, no new numeric program, no new `MEMRA_*` read (the grid is
`Engine::gdn_chunk_size()`). Sessions whose seed boundary stops inside the prompt prime alone (the concat
prime cannot stop; the rule `snapshot_at` already had); in-batch fanout participants clear the seed as
before; the eager-only arm clears it with `snapshot_at`; the solo prefill widening ignores it like
`ckpt_at`. `cached_tokens` reports the restored (aligned) length because that is what was restored. The
spec session's `capture_at` publication (prompt end, the engine's post-prime capture) is **not** moved; see
"The #379 gate" below for what that costs today. Unit tests:
`seed_capture_boundary_lands_on_the_grid_or_refuses` (the day-17 lengths 12,350 -> 12,320 and 12,200 ->
12,160; lengths just above and below a grid line; the exact prompt-end line; the sub-floor refusal band
65..79 -> `Refused { aligned: 32 }`; covered re-sends; the day-16 chain step),
`seed_boundary_inside_prompt_is_a_stop_only_below_the_prompt_end`, the `/metrics` key.

## The gates

- `tools/prefix-newest-turn-fits-gate.py`: V5 identity (every turn's text equals the calibration boot's
  cold completion of the same prompt) and V6 grid (every `insert (seed)` of the measured boot, cohort and
  turns, has exactly `capture_len(prompt_tokens)` tokens; every restored length is the published one and
  a grid multiple) are verdict clauses. V1 reads `cached_tokens == the previous turn's published length`
  (equality; the old `>= prompt_tokens` only an off-grid prompt-end entry satisfies); V2 counts exactly
  one seed publication per turn and a hit of the published length; `seed REFUSED (grid)` counts as a
  refusal; the `[primeseg]` receipts (`MEMRA_DEBUG_PRIMESEG=1`, a documented diagnostic) and the count of
  off-grid call starts are recorded.
- `tools/prefix-restore-identity-gate.py` (new; Python beside the twin gate so it inherits the lock,
  collector and default-policy discipline, imports the twin gate's id generators and `capture_len`): the
  day-17 target prompt (turn 10 of the day-16 chain, 12,350 ids) restored from five entries, one
  `cache_salt` each; V1 identity per point, V2 grid per point (published == `capture_len(p)`, restored ==
  published, a grid multiple, every hit prime call `grid_off=0`). The first cells of this day ran the
  gate's mistaken default of turn 12 (12,650 ids); the default is turn 10 since `66c0aea89` and those cells
  are kept as extra receipts on a prompt that is not the near-tie.
- `tools/kv-host-tenant-reclaim-gate.sh`: unchanged in substance. Its promote equalities were moved to the
  aligned capture before the run and moved back after the target card showed why: that gate boots with
  spec at its default, every leader publishes `insert (spec-boundary)` at the prompt end (83 tokens for an
  83-token prompt), and the plain seed never runs there. The reason now sits beside the equalities.

## The cards and their regimes

Local: NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, driver 595.84, clocks not pinned; every cell's
first telemetry sample read 15 MiB (the card was alone); cells ran between 54 and 89 C, peak draw 177.5
to 182.2 W; every GPU command went through `tools/tier-battery.py --rig rtx5090 --external-lock`, the
canonical `/tmp/memra-5090.lock` inherited as an FD and proven by `tier-lock-proof.py` (`lock.json`,
`cell/LOCK.json` in every cell). Builds and CPU checks under `systemd-run --user --scope -p
CPUQuota=1200% -p MemoryMax=28G` (the two release builds at 800% each, side by side). Target: one NVIDIA
RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W, driver 580.178.04, idle at 0 MiB and 32 C before
the chain, no other lane running (`tmux ls`: no server); every GPU command through `tools/tier-battery.py
--rig pro-single` on `/tmp/memra-gpu.lock`, inherited where the gate takes a lock, held by the collector
for serve-smoke and cache-meter (the day-14 shape). Artifact on both: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`.

Binaries. Local base: `origin/main` `1b354be59` built fresh in its own worktree
(`rtx5090-day18/build-base/`, exit 0, 11:42:47 to 11:45:44 UTC, SHA-256
`79e549b339d22f8940267e130a6a743ea1df621c47a3017ddf6aba6f032c4b43`). Local fix: the lane tip `03beaa0b9`
(`build-fix/`, exit 0, 14 s incremental, SHA-256
`81056589fb25e8ea47218228db3d98ea3d5badc7a490cf3b41e94e7bb2c04375`). Target fix: `03beaa0b9`
(`pro-single-day18/build-fix/`, exit 0, 38 s incremental, SHA-256
`3ea188e7be820dcffe723730777b55659e19761e7dc684f636150246df8e3ef9`). The commit after the cells
(`66c0aea89`) rewrites `% c == 0` as clippy's `is_multiple_of` (identical arithmetic), fixes the restore
gate's default turn, restores lane A's equalities and adds the docs; no cell was re-run for it.

| card | cell | binary | collector status | telemetry samples | temp C | peak `memory.used` MiB | peak draw W |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| 5090 | `gate-fix-day16-shape` | fix | `executed-not-qualified` (exit 0) | 710 | 54..88 | 21,785 | 182.2 |
| 5090 | `restore-fix` (turn 12, 12,650) | fix | `executed-not-qualified` (exit 0) | 312 | 64..88 | 22,201 | 179.6 |
| 5090 | `gate-base-day16-shape` | base | `failed` (exit 1, the gate's FAIL) | 712 | 68..89 | 21,785 | 179.4 |
| 5090 | `restore-base` (turn 12, 12,650) | base | `failed` (exit 1) | 302 | 65..88 | 22,329 | 177.5 |
| 5090 | `restore-base-t10` (turn 10, 12,350) | base | `failed` (exit 1) | 295 | 65..87 | 22,169 | 182.1 |
| 5090 | `restore-fix-t10` (turn 10, 12,350) | fix | `executed-not-qualified` (exit 0) | 301 | 69..88 | 22,105 | 180.9 |
| 5090 | `hitgate-fix` (#379 gate, own flock) | fix | exit 1 (2 failures, below) | own log | | | |
| 5090 | `hitgate-base` (#379 gate, own flock) | base | exit 0, `ALL GREEN` | own log | | | |
| PRO 6000 | `gate-fix-day16-shape` | fix | `executed-not-qualified` (exit 0) | 403 | 32..60 | 22,067 | 504.5 |
| PRO 6000 | `restore-fix` (turn 12) | fix | `executed-not-qualified` (exit 0) | 168 | 40..58 | 26,323 | 500.7 |
| PRO 6000 | `restore-fix-t10` (turn 10) | fix | `executed-not-qualified` (exit 0) | 164 | 42..58 | 26,259 | 502.6 |
| PRO 6000 | `serve-smoke` | fix | `executed-not-qualified` (exit 0) | 86 | 42..56 | 20,277 | 497.1 |
| PRO 6000 | `cache-meter` | fix | `executed-not-qualified` (exit 0) | 29 | 40..46 | 17,845 | 183.8 |
| PRO 6000 | `evict-reclaim` | fix | `failed` (exit 1, V3) | 385 | 39..61 | 94,484 | 498.0 |
| PRO 6000 | `kv-host-fix` | fix | `failed` (exit 1, the two moved equalities) | 55 | 44..57 | 20,405 | 485.2 |
| PRO 6000 | `evict-reclaim-base` | base (`1b354be59`, built there, SHA-256 `449a7430e99791c288c9abca614a63024f488e610b0ae2834ce8c54543b1bc13`) | `executed-not-qualified` (exit 0) | | | | |
| PRO 6000 | `kv-host-fix-2` (gate at `66c0aea89`) | fix | `executed-not-qualified` (exit 0) | | | | |

`tools/tier-battery.py --validate` exit 0 on every collector cell of the target card.

## Local RTX 5090: the twin gate on the day-16 lru shape (cohort 1,250/1,350/1,450/1,550; 12 turns 11,000 + 150)

Base (`origin/main` `1b354be59`), verbatim:

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=793870336 turns=12 cold_turns_after_1=0 cached_ok=11/11 lines_ok=12/12 evictions=10 cohort_evictions=4 self_evictions=0 refused_or_skipped=0 effective_free_ok=12/12 identity_ok=11/12 grid_ok=0/31 grid=32 off_grid_calls=11 V1=ok V2=ok V3=ok V4=ok V5=FAIL V6=FAIL -> FAIL
```

The mechanics (V1 to V4) pass; the identity clause fails on turn 10 exactly as day 17 recorded (restored
`22f023976ebc22fd`, cold `4415b7e361fc6f6b`, `finish_reason` `length` versus `stop`), every one of the
twelve measured digests equals day 17's cell B digest (12/12, `verify-day18.py`), every seed is off the
grid (`grid_ok=0/31`: 11,000, 11,150, ... published whole; expected 10,976, 11,104, ...), and eleven of
the twelve turns' restored suffix calls start off the grid. Turn 10's receipts, verbatim:

```text
[prefix-cache] hit: 12200 of 12350 prompt tokens from cache (model gate)
[primeseg] call start=12200 take=150 grid_off=8 bound_rem=None ckpt_at=None snapshot_at=None q=150 budget=1024
[prefix-cache] insert (seed): 12350 tokens, 523.6MB (resident 1042.8MB / 1074MB, model gate, ns "grow")
```

Fix (`03beaa0b9`), verbatim:

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=789118976 turns=12 cold_turns_after_1=0 cached_ok=11/11 lines_ok=12/12 evictions=10 cohort_evictions=4 self_evictions=0 refused_or_skipped=0 effective_free_ok=12/12 identity_ok=12/12 grid_ok=31/31 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
```

All twelve turns identical to the calibration boot's cold completion, every published entry on the grid
(cohort 1,250 -> 1,216, 1,350 -> 1,312, 1,450 -> 1,408, 1,550 -> 1,504; turns 11,000 -> 10,976,
11,150 -> 11,104, ..., 12,350 -> 12,320, 12,650 -> 12,608), every prime call of the measured boot on the
grid. The fix's twelve cold digests (its calibration boot) equal day 17's cold digests 12/12: the cold
program did not move. Turn 1 (cold, cache on) now stops on its seed boundary and the split is the law's
bit-identical one (turn 1 identical to the cache-off render); turn 10 restores 12,160 and primes 160 from
a grid start, publishes 12,320, primes the last 30, and produces the cold bytes. Verbatim:

```text
turn 1:  [primeseg] call start=10240 take=736 grid_off=0 bound_rem=Some(736) ckpt_at=None snapshot_at=None q=760 budget=1024 seed_at=Some(10976)
turn 1:  [prefix-cache] insert (seed): 10976 tokens, 482.8MB (resident 883.1MB / 1074MB, model gate, ns "grow")
turn 1:  [primeseg] call start=10976 take=24 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=24 budget=1024 seed_at=None
turn 10: [prefix-cache] hit: 12160 of 12350 prompt tokens from cache (model gate)
turn 10: [primeseg] call start=12160 take=160 grid_off=0 bound_rem=Some(160) ckpt_at=None snapshot_at=None q=190 budget=1024 seed_at=Some(12320)
turn 10: [prefix-cache] insert (seed): 12320 tokens, 522.7MB (resident 1040.7MB / 1074MB, model gate, ns "grow")
turn 10: [primeseg] call start=12320 take=30 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=30 budget=1024 seed_at=None
```

| turn | prompt | base cached | base published | fix cached | fix published | expected capture | base == cold | fix == cold | digest (fix = cold) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 11,000 | 0 | 11,000 | 0 | 10,976 | 10,976 | yes | yes | `9745f43fe07640e1` |
| 2 | 11,150 | 11,000 | 11,150 | 10,976 | 11,104 | 11,104 | yes | yes | `66d394ced7203380` |
| 3 | 11,300 | 11,150 | 11,300 | 11,104 | 11,264 | 11,264 | yes | yes | `24a97a2867768b2d` |
| 4 | 11,450 | 11,300 | 11,450 | 11,264 | 11,424 | 11,424 | yes | yes | `876c0bf0f375834e` |
| 5 | 11,600 | 11,450 | 11,600 | 11,424 | 11,584 | 11,584 | yes | yes | `7abc2552932a6a15` |
| 6 | 11,750 | 11,600 | 11,750 | 11,584 | 11,712 | 11,712 | yes | yes | `24a97a2867768b2d` |
| 7 | 11,900 | 11,750 | 11,900 | 11,712 | 11,872 | 11,872 | yes | yes | `22f023976ebc22fd` |
| 8 | 12,050 | 11,900 | 12,050 | 11,872 | 12,032 | 12,032 | yes | yes | `01062972f3feb195` |
| 9 | 12,200 | 12,050 | 12,200 | 12,032 | 12,160 | 12,160 | yes | yes | `22f023976ebc22fd` |
| 10 | 12,350 | 12,200 | 12,350 | 12,160 | 12,320 | 12,320 | **NO** (`22f02397`) | yes | `4415b7e361fc6f6b` |
| 11 | 12,500 | 12,350 | 12,500 | 12,320 | 12,480 | 12,480 | yes | yes | `b6b5b760960efe97` |
| 12 | 12,650 | 12,500 | 12,650 | 12,480 | 12,608 | 12,608 | yes | yes | `a60e5ad8da5b0484` |

## Local RTX 5090: the restore-identity gate (the 12,350 prompt from five entries)

Base, verbatim (the day-17 probe E result to the point, on a fresh `main` build):

```text
PREFIX-RESTORE-IDENTITY: target=12350 points=5 identical=3/5 grid_ok=2/5 grid=32 12288(seed12288,restored12288,off0,suffix62):yes 12320(seed12320,restored12320,off0,suffix30):yes 12200(seed12200,restored12200,off8,suffix150):yes 12250(seed12250,restored12250,off26,suffix100):NO 12300(seed12300,restored12300,off12,suffix50):NO V1=FAIL V2=FAIL -> FAIL
```

Both off-grid flips are the day-17 stream (`"_\t\t\"\t\t\"\t"`, 8 tokens, `length`, `22f023976ebc22fd`)
against the cold `"_\n"` (3, `stop`, `4415b7e361fc6f6b`), first differing character 1; the 12,200 point
holds by a near-tie that did not flip and fails V2 (`off8`).

Fix, verbatim:

```text
PREFIX-RESTORE-IDENTITY: target=12350 points=5 identical=5/5 grid_ok=5/5 grid=32 12288(seed12288,restored12288,off0,suffix62):yes 12320(seed12320,restored12320,off0,suffix30):yes 12200(seed12160,restored12160,off0,suffix190):yes 12250(seed12224,restored12224,off0,suffix126):yes 12300(seed12256,restored12256,off0,suffix94):yes V1=ok V2=ok -> PASS
```

The seeds of 12,200, 12,250 and 12,300 publish 12,160, 12,224 and 12,256; every hit restores the
published length and primes its suffix from a grid start (`off_grid_calls` 0 on every point); all five
produce the cold bytes.

The turn-12 cells (the gate's first default; 12,650 ids, a prompt that is not a near-tie): base
`identical=5/5 grid_ok=2/5 ... V1=ok V2=FAIL -> FAIL` (three off-grid restore points, no flip on this
prompt; V2 red on the mechanism, which is what V2 is for), fix `identical=5/5 grid_ok=5/5 ... -> PASS`.

## Target card (one RTX PRO 6000 Blackwell): the fix

Twin gate, day-16 shape, verbatim:

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=789118976 turns=12 cold_turns_after_1=0 cached_ok=11/11 lines_ok=12/12 evictions=14 cohort_evictions=4 self_evictions=0 refused_or_skipped=0 effective_free_ok=12/12 identity_ok=12/12 grid_ok=31/31 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
```

Restore-identity gate, turn 10, verbatim:

```text
PREFIX-RESTORE-IDENTITY: target=12350 points=5 identical=5/5 grid_ok=5/5 grid=32 12288(seed12288,restored12288,off0,suffix62):yes 12320(seed12320,restored12320,off0,suffix30):yes 12200(seed12160,restored12160,off0,suffix190):yes 12250(seed12224,restored12224,off0,suffix126):yes 12300(seed12256,restored12256,off0,suffix94):yes V1=ok V2=ok -> PASS
```

(and the turn-12 cell `target=12650 ... identical=5/5 grid_ok=5/5 ... -> PASS`). `serve-smoke.sh`:
`serve-smoke: 0 failed` (spec, gemma4 and Q35 arms SKIP for absent artifacts, as on days 14 and 15).
Cache-metering arm (`cache-meter-gate.py --n 5 --k 256`): `cache-meter-gate: 0 failed`, every accounting
line `ok` (K = 256 is on the grid, so the shared-prefix entry and its `cached_tokens == K` are unchanged
by the law).

`tools/prefix-evict-reclaim-gate.py`, verbatim:

```text
PREFIX-EVICT-RECLAIM: entry_bytes=1590853632 reclaim_credit_bytes=1591000000 driver_free_delta_bytes=1442840576 trim_released_bytes=1442840576 pool_retained_bytes=148013056 p2=admit-same-tick busy_overlap_s=21.658 identity=aa6cc3291b981646 V1=ok V2=ok V3=FAIL V4=ok -> FAIL
```

What moved against day 13's fix run on the same card (`entry_bytes=1592160256 ...
driver_free_delta_bytes=1610612736 trim_released_bytes=1610612736 ... -> PASS` after its V3 fix): the
entry is 1,306,624 B smaller (the P1 seed now publishes at its aligned boundary, at most 47 tokens
short of the prompt end), the reclaim credit matches it (V1 ok), the tokens are byte-identical (V4 ok,
the same `aa6cc3291b981646` as day 13), and the settle returned 1,442,840,576 B to the driver while the
pool retained 148,013,056 B, which is V3's shortfall. Whether the retained bytes come from the fix (the
cold prime now ends in a short remainder call from the seed boundary, a new small prime-workspace shape
the pool keeps) or from the card's state today was decided by the base arm on the same card
(`run-rest3.sh`: base built from `origin/main` `1b354be59` there), verbatim:

```text
PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=1751000000 driver_free_delta_bytes=1610612736 trim_released_bytes=1610612736 pool_retained_bytes=140549120 p2=admit-same-tick busy_overlap_s=21.658 identity=aa6cc3291b981646 V1=ok V2=ok V3=ok V4=ok -> PASS
```

The attribution is in the two arms' own lines. Base evicted 1,751,161,856 B (`evicted_prefix_bytes` on
its settle line): E1 plus the busy peer's own 159,001,600 B seed, `insert (seed): 71 tokens, 159.0MB`, and
the trim returned 1,610,612,736 B, 18 MB above E1, while the pool retained 140,549,120 B. The fix evicted
1,590,853,632 B, exactly E1, because the 71-token busy peer published nothing:
`[prefix-cache] seed REFUSED (grid): prompt 71 tokens is off the 32-token prime grid and its aligned
boundary 32 is under the 64 token entry floor` (twice, both boots; `e0_busy_peer_entry_bytes` 0 in the
fix's calibration, 159,001,600 in base's), so the same ~140 to 148 MB pool retention is no longer covered
by a second entry's bytes and V3's `driver free rising by >= E1 - one granule` is exposed. The pool
retention is present in both arms and is the card's, not the fix's; what the fix moved is the busy peer's
seed, from an off-grid 71-token entry that could only ever serve an exact re-send to a typed refusal. The
finding is about this lane's own day-13 gate: V3's arithmetic does not include `pool_retained_bytes` and
passed on day 13 and on base today only because the shape evicted a sub-floor seed beside E1. It is not
changed here (the result was seen first); the follow-up is V3 stated on `driver_free_delta +
pool_retained >= E1 - granule`, or a busy peer whose prompt has a grid-aligned entry, decided by the lead.

`tools/kv-host-tenant-reclaim-gate.sh fix`: `GATE: kv-host-tenant-reclaim (fix arm) FAIL (2
assertions)`, the two being exactly the equalities this day had moved to the aligned capture before the
run. The server log says why they were wrong to move: every entry there is `insert (spec-boundary): 83
tokens` (89, 86, 93, 95, 96, 108 for the others), the spec path's prompt-end publication, and the
promotes restore the whole leader prompt (`hit: 83 of 96`, `hit: 95 of 108`; r7 `cached_tokens` 83 == r1
`prompt_tokens` 83, r8 95 == r6 95). Every other clause held: the tenant-share evictions name acme, each
is followed by acme's own demote landing, `[prefix-host] promote:` names beta at r7 and acme at r8, no
evaporation line, no wasted reclaim. The equalities are the original ones again at `66c0aea89` with the
reason beside them, and the fix arm re-ran on that gate source after the base evict-reclaim arm:
`GATE: kv-host-tenant-reclaim (fix arm) PASS`, 28 `ok` lines, no `FAIL`, the same `insert
(spec-boundary): 83 tokens` entries and `hit: 83 of 96` / `hit: 95 of 108` promotes. Nothing the fix
touches runs in that gate; the day's only change to it is the comment saying so.

## The #379 gate (`tools/spec-on-cache-hit-gate.sh qwen`), local RTX 5090: red on the fix, green on base

Run as `local-ci` runs it (the gate takes `flock -w 300 /tmp/memra-5090.lock` around every boot itself,
so not through the collector; the 9B trunk `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`; the external drafter
`draft-9b-owntrim-nvfp4head-q4blk.gguf` is absent on this rig, so the drafter-attach assertion no-ops
exactly as `local-ci`'s WARNING branch says). Verbatim:

```text
fix:  ok=52 FAIL=2   FAIL: r3 spec-on text != spec-off text (identity law)   FAIL: g2 spec-on text != spec-off text (identity law)   SPEC-ON-CACHE-HIT GATE: 2 FAILURE(S) (qwen)
base: ok=54 FAIL=0   SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)
```

The two reds are the suffix-fed hits, and the cached counts say what they are: on the spec-on boot r3 and
g2 report `cached_tokens` 106 of 119 (the spec session published its entry at the 106-token prompt end,
`insert (spec-boundary)`, 106 % 32 = 10, and the 13-token suffix is under `PRIME_MIN_T`, so the spec
side's carried suffix rides `decode_step_h` tokenwise, its own documented program); on the spec-off boot
they report 64 of 119 (the plain seed published `capture_len(106) = 64`, the restore is on the grid and
the 55-token suffix primes from a grid start). The texts agree through character 35 and diverge at 36
(`"...systematic catalog of monthly inspection tasks"` versus `"...systematic catalog of the observatory
maintena..."`). On base both boots restored the same off-grid 106-token entry and walked the same 13
tokens tokenwise, so the identity law held because both sides ran the same second program; the twin gate
shows which side is the cold render now (the plain one, V5 12/12 on both cards). The gate is red for a
true reason: the spec path's prompt-end capture is still off the grid, and its identity law compares
spec-on with spec-off, not with cold.

This is the conflict of the day, stated rather than resolved by weakening anything: the fix cannot keep
that gate green without moving the spec session's `capture_at` onto the same grid, and that in turn moves
the gate's own fixture, whose sampled cells (`s`, `sp`, `g4`) assert whole-prompt "FULL-COVER" hits
(`cached == prompt_tokens`) and require the `restore-full-cover` boundary site to fire, which only an
on-grid prompt can do once seeds are aligned. That is the spec-side half of memra#602 and a fixture
redesign of a gate this lane does not own; both are named in the issue comment for the lead's decision,
and neither is started here. No clause of any gate was relaxed after a result.

## Local checks (the merged tree at `66c0aea89`)

| Check | Result |
| --- | --- |
| `git merge --no-ff origin/main` twice (`9a1a717ce`, then `1b354be59` after #599 landed mid-day); pushes through the pre-push hook (perf board, flags census, releasability and docs-registry censuses, workflow keys, public boundary) | allowed: `c4894627a..7578d6bf4`, `7578d6bf4..03beaa0b9`, `03beaa0b9..66c0aea89` |
| `cargo fmt --all -- --check` | PASS (`fmt-check.log`) |
| `cargo test -p memra-server --offline` | `758 passed; 0 failed; 8 ignored` (`test-server-2.log`) |
| `cargo clippy -p memra-server --offline --all-targets -- -D warnings` | PASS after the two `is_multiple_of` rewrites (`clippy.log` red, `clippy-2.log` green) |
| `bash tools/check-flags.sh` | `no uncovered runtime names` (865 reads; no new `MEMRA_*` read) |
| `bash tools/docs-registry-census.sh` | PASS |
| `git diff --check` | clean |
| Full GPU exactness battery | NOT RUN (the change is the prefix-cache capture position; the serving-shape gates above are its gates) |

## Boundaries and record

- Forbidden list honored: no V4.1 code, no external dependency, no new numeric program and no change to a
  prime program (the split the fix adds is the law's bit-identical one; the cold digests equal day 17's),
  no `unsafe`, no `--no-verify`, no skip variable, no third lock name, no bare GPU run (the #379 gate holds
  the canonical lock itself), `/root/artifacts` and `/root/memra-spill` untouched, no other lane's
  worktree touched, `main` untouched, no host, id, location or cost in a tracked file.
- Receipts: `rtx5090-day18/` (builds, one collector directory per cell with `CELL.jsonl`, `lock.json`,
  `command.log`, `command.gpu.csv`, `command.capture.json`, `gate-source.txt`, `binary.sha256` and the
  gate's `cell/`; the two `hitgate-*` directories with the gate's evidence and log; `chain*.log`, the
  driver logs, `card-before-*.csv`, the CPU check logs, `push-*.log`) and `pro-single-day18/` (the target
  card's `b-day18` mirror: builds, cells, validate logs, drivers). Drivers: `build-day18.sh`,
  `run-day18-gate.sh`, `run-day18-restore.sh`, `run-day18-hitgate.sh`, `run-day18-hitgate-base.sh`,
  `chain-day18-*.sh`; target card `pro-single-day18/{build-arm,run-all,run-rest2,run-rest3,cache-meter-cell}.sh`.
  Replay: `verify-day18.py`.
- The base worktree (`wt-spill-b-base`) was removed after its binary was copied; the release binaries live
  under the untracked `target/bins/`.
