# WP-B day 14: the newest turn fits the prefix cache (#523 items 1 and 3)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `be07f2d36` at
`bd6a990ed`. Commits this day: `f42d921db` (the fix), `8c813fc3b` (the gate and the target-card drivers),
`d520f745f`, `d27badac1` and `a178620c8` (gate corrections after the first two target-card rounds, the
TESTING.md row), plus this record and the receipts. **Not merged, released, deployed, or a default
promotion.** Every target-card cell is N=1, `executed-not-qualified` in the collector's vocabulary; the card
ran at its 600 W cap; no timing is compared between binaries or boots. The default policy stays SLRU (#523
item 2, the interleaved A/B, is not this day's work).

## The defect, read from the code and the issue

- Under `MEMRA_PREFIX_CACHE_POLICY=slru` (the default) `capacity_victim_with` names the probation LRU. When
  a growing conversation's newest entry is the only probation member beside a promoted cohort, the insert
  loop evicts the entry it just inserted (`insert probation ... resident 9604 / 8590 MB` then
  `evict (Probation LRU)` of the same entry in the incident's slot log), and the snapshot preflight
  (`prepare_snapshot`) counted only probation bytes as reclaimable, so it refused the publication outright
  when probation was empty. Either way the turn ran cold (`cached_tokens=0`, a full re-prefill) while the
  cohort sat protected, and the only trace was one once-announced
  `[prefix-cache] snapshot skipped: cannot fit beside leased/protected entries; request continues` line.
- `cached_tokens` decides the bill and whether the cache engaged at all; the production workaround was a
  launcher's `MEMRA_PREFIX_CACHE_POLICY=lru`, a deployment choice, not an engine invariant.

## The fix (`f42d921db`, `crates/memra-server/src/worker.rs`)

- `room_victim_with(slru, keep)`: the probation LRU first, exactly as before; when probation is exhausted
  or its only remaining member is the entry being inserted, the protected LRU goes next, oldest first. An
  insert therefore never selects itself. Both the insert loop and `prepare_snapshot` use it;
  `prepare_snapshot` counts every unleased byte as reclaimable (`evictable_bytes`).
- Refusals are typed and in bytes, one formatter per shape, on both publication paths:
  `[prefix-cache] insert refused: entry N exceeds budget M (<why>, model <m>[, ns "<salt>"])` and
  `[prefix-cache] insert refused: entry N cannot fit beside L leased bytes (budget M, <why>, ...)`. The
  once-announced `snapshot skipped` line is gone; the unpinned insert path gets the leased preflight too
  (before, such an entry evicted every unleased neighbour and then itself). The preflight evict line names
  the segment: `evict (snapshot preflight, Protected LRU)`.
- Victim selection and accounting only: `prefix_snapshot` and the restore path are untouched, no numeric
  program moved, no new `MEMRA_*` read (flags census clean), the VMM door untouched.
- CPU arms, all in `worker::tests` and all green locally (`memra-server` 748 passed, 0 failed, 6 ignored):
  `prefix_cache_slru_newest_turn_fits_beside_a_protected_cohort` (the incident's bytes: 8192 MiB budget,
  80 % protected, six promoted 30k conversations, one tenant 105k to 168k by 300 per turn, entries 2.7 to
  4.3 GB; all 211 turns insert, the cohort goes first oldest first, an insert is never its own victim, no
  refusal); `prefix_cache_oversized_insert_refuses_with_the_typed_line_and_evicts_nothing` (both paths
  refuse, nothing evicted, `skips_budget` 2, the exact line
  `[prefix-cache] insert refused: entry 11 exceeds budget 10 (seed, model m, ns "t")`);
  `prefix_cache_slru_fitting_inserts_keep_the_same_victims_in_the_same_order` (for entries that fit without
  touching protected bytes the raw capacity victim and the room victim are the same entry, 19 victims in the
  same order, and the demotion shape is unchanged);
  `prefix_cache_newest_turn_beside_a_leased_predecessor_fits_or_refuses_loudly` (the spec-session shape
  where the predecessor is still leased: fits while `entry + leased <= budget`, otherwise the typed leased
  line, nothing evicted). `prefix_cache_lru_policy_evicts_the_global_oldest_not_only_probation` keeps the
  `lru` rollback arm honest.

## The gate (`tools/prefix-newest-turn-fits-gate.py`, final source `a178620c8`)

Serving shape on the real `memra-server`, one card, plain path (`MEMRA_SERVE_SPEC=0`), `MEMRA_CTX=16384`,
`MEMRA_PREFIX_CACHE_MB=1024`, the DEFAULT policy (the boot line must report SLRU; the gate never sets the
policy), greedy, `prompt_ids` requests, under the collector's inherited canonical lock (`--external-lock
@COLLECTOR_LOCK_FD@`, `tier-lock-proof.py` verified before any port binds).

- Calibration boot, prefix cache OFF (`MEMRA_PREFIX_CACHE_MB=0`): the identical request sequence, so each
  turn's own retained footprint and effective free after each turn are known for this binary on this card
  with no cache activity at all.
- Measured boot: a second tenant (`cache_salt=cohort`) sends prompts of 2,800, 3,000 and 3,200 ids twice
  each; the second send is a whole-entry hit whose lease promotes the entry, so the cohort ends PROTECTED
  (737,943,552 B inside the 858,993,459 B protected share, no demotion, 0 evictions: asserted). The growing
  tenant (`cache_salt=grow`) then replays 8 turns: 9,200 ids, then +300 per turn to 11,300. The pressure
  arithmetic is read from the server's own `insert probation` lines (29,750 B per token plus 156,716,667 B
  fixed for this artifact) and the gate REFUSES unless cohort <= protected share, cohort + turn-1 entry >
  budget (737,943,552 + 430,416,667 > 1,073,741,824) and every turn's entry <= budget: the incident's
  shape at a small budget.
- Verdicts, bytes from the server's `[prefix-cache]` lines and `/metrics`: V1 every turn k >= 2 reports
  `usage.prompt_tokens_details.cached_tokens >= ` turn k-1's `prompt_tokens`; V2 turn 1 publishes and every
  later turn hits exactly once and publishes, no `insert refused` or `snapshot skipped` line for the growing
  tenant; V3 after every turn the calibration boot's effective free (`cuda_driver_free_bytes +
  cuda_pool_cached_bytes`) equals the measured boot's effective free plus the cache's resident bytes within
  64 MiB (the cache costs exactly what it holds, so every evicted byte came back); V4 at least one
  `evict (... Protected LRU)` line. Per turn the gate records the completion digest, the same prompt's cold
  digest from the calibration boot, and the server's per-request receipt (this artifact prints
  `[spec-k] ... prompt=N cached=M lcp=L`, not the `[glm5-spec] route= ... cold= restored=` line, which is a
  draft-capable-model receipt). Exit 0 PASS, 1 FAIL, 2 `REFUSED: ...`.

## Target-card cells (one RTX PRO 6000 Blackwell, 96 GB, 600/600 W, `--rig pro-single`, `/tmp/memra-gpu.lock`)

Binaries built natively at their exact refs (receipts `build-main/`, `build-fix/`, `dirty.txt` empty):

| Arm | Source | `memra-server` SHA-256 |
| --- | --- | --- |
| base | `be07f2d36` (`origin/main`) | `9c38a4c20e413299e937b882b8cbae148c568e9d7a3e8b572161f0f69147a539` |
| fix | `d520f745f` (the lane tip when built; every later commit touches only `tools/`, `docs/`, `research/`) | `a3ea4bc927a13f03b3946b24f1404d595ef55873e2def9d29d5e8ae7bf5c8d40` |

Artifact `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (the same one every earlier B cell used). The same two binaries,
the same prompts and the same cohort ran in all three rounds; only the gate's V3 arithmetic changed.

### The first sitting's chain (kept under `refused-sitting1/`)

Every cell of the first sitting's chain REFUSED before any verdict, on the driver, not on anything measured:
`REFUSED: [Errno 17] File exists: '<receipts>/gate-main'` (the driver ran `mkdir -p` before the collector,
which creates `--out` itself and refuses one that exists) and
`REFUSED: --external-lock requires exactly one @COLLECTOR_LOCK_FD@ argument` (serve-smoke and cache-meter
carry no FD placeholder; the collector holds the lock for them, the day-13 shape). Its two binaries were
built at `0e7741b76` and `f42d921db`, not the required arms (`memra-tier` moved between `0e7741b76` and
`be07f2d36`, and `memra-engine` and `memra-kv` depend on it), so both arms were rebuilt.

### Round 1, gate `d520f745f`: V3 as `consumed == grew` (`gate-main`, `gate-fix`, both collector `failed`)

```text
base be07f2d36: PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=7 cached_ok=0/7 lines_ok=0/8 evictions=0 protected_evictions=0 refused_or_skipped=1 effective_free_ok=0/0 V1=FAIL V2=FAIL V3=FAIL V4=FAIL -> FAIL
fix  d520f745f: PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 protected_evictions=2 refused_or_skipped=0 effective_free_ok=6/8 V1=ok V2=ok V3=FAIL V4=ok -> FAIL
```

The fix's V3 error equalled, to the byte, the base's effective-free consumption on the same turn with zero
cache activity (base: no insert, no eviction, no hit): 1,111,666,004 B on turn 1, 610,359,380 B on turn 2,
20,160,000 B on turns 3 to 7, 28,878,336 B on turn 8. That is the request sequence's own retained footprint
(pool `used` that stays after the request retires; 67,200 B per additional prompt token on turns 3 to 8),
identical across the two binaries and present with the cache doing nothing. Round 1 had no control for it.

### Round 2, gate `d27badac1`: V3 as `consumed - calibration consumed == grew` (`gate-main-final`, `gate-fix-final`, both collector `failed`)

```text
base be07f2d36: PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=7 cached_ok=0/7 lines_ok=0/8 evictions=0 protected_evictions=0 refused_or_skipped=1 effective_free_ok=0/0 V1=FAIL V2=FAIL V3=FAIL V4=FAIL -> FAIL
fix  d520f745f: PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 protected_evictions=2 refused_or_skipped=0 effective_free_ok=7/8 V1=ok V2=ok V3=FAIL V4=ok -> FAIL
```

The calibration boot added the control: the fix's error was exactly 0 on turns 2 to 8 and 149,094,400 B on
turn 1. The recorded states say why: with the cache off, the cohort's second sends are cold prefills, and
that cohort phase leaves 149,094,400 B more retained footprint before turn 1 (calibration pool `used`
17,758,320,040 B) than the cache-on cohort (three cold sends, three restores; non-prefix pool `used`
17,609,225,640 B). Turn 1 converges both boots to the same non-prefix `used` (18,720,891,644 B in both), so
turn 1's per-turn delta differed while the states agreed. In round 2's own rows the state identity
`calibration effective free after k == measured effective free after k + resident prefix bytes` held
exactly on every turn (turn 1: 82,663,431,428 B on both sides; turn 2: 82,053,072,048 B). Per-turn deltas
need aligned starting states and the cohort phase does not give them; states do not.

### Round 3, gate `a178620c8`: V3 as the state identity after every turn (`gate-main-r3`, `gate-fix-r3`)

Same artifact, same prompts (9,200 to 11,300 ids), same cohort (737,943,552 B), same two binaries, back to
back on the same card. Verdict lines, verbatim:

```text
base be07f2d36: PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=7 cached_ok=0/7 lines_ok=0/8 evictions=0 protected_evictions=0 refused_or_skipped=1 effective_free_ok=8/8 V1=FAIL V2=FAIL V3=ok V4=FAIL -> FAIL
fix  d520f745f: PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 protected_evictions=2 refused_or_skipped=0 effective_free_ok=8/8 V1=ok V2=ok V3=ok V4=ok -> PASS
```

The gate is **red on `main` and green on the fix** on the target card. Both cells ran through the collector
(`failed` for exit 1, `executed-not-qualified` for exit 0, `"qualification": false` in both captures).

#### Base `be07f2d36`, per turn (`gate-main-r3/cell/TURNS.md`)

Every turn after the first is cold: `cached_tokens=0` in the response, `cached=0 lcp=0` in the server's
per-request receipt, no hit, no insert, no eviction, a full re-prefill (2.6 to 3.3 s per turn, recorded, not
compared). The only cache line the growing tenant produced, once, at turn 1:

```text
[prefix-cache] snapshot skipped: cannot fit beside leased/protected entries; request continues
```

| turn | prompt_tokens | cached_tokens | prev prompt_tokens | server receipt | hit | insert | evict | skipped | effective free after | resident prefix bytes | cache-off effective free after | V3 error | elapsed s | text sha256[:16] | == cold |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 9200 | 0 | - | cached=0 lcp=0 | none | none | none | 1 | 81925487876 | 737943552 | 82663431428 | 0 | 2.639 | 254a65a01730e58b | yes |
| 2 | 9500 | 0 | 9200 | cached=0 lcp=0 | none | none | none | 0 | 81315128496 | 737943552 | 82053072048 | 0 | 2.77 | 24a97a2867768b2d | yes |
| 3 | 9800 | 0 | 9500 | cached=0 lcp=0 | none | none | none | 0 | 81294968496 | 737943552 | 82032912048 | 0 | 2.854 | 65eeb1ef8f716af1 | yes |
| 4 | 10100 | 0 | 9800 | cached=0 lcp=0 | none | none | none | 0 | 81274808496 | 737943552 | 82012752048 | 0 | 2.909 | 64a99158cb668b0c | yes |
| 5 | 10400 | 0 | 10100 | cached=0 lcp=0 | none | none | none | 0 | 81254648496 | 737943552 | 81992592048 | 0 | 3.046 | c6b9d167a76a942e | yes |
| 6 | 10700 | 0 | 10400 | cached=0 lcp=0 | none | none | none | 0 | 81234488496 | 737943552 | 81972432048 | 0 | 3.098 | 8e4798b352770d9a | yes |
| 7 | 11000 | 0 | 10700 | cached=0 lcp=0 | none | none | none | 0 | 81214328496 | 737943552 | 81952272048 | 0 | 3.188 | f34ee12b3db5cb9c | yes |
| 8 | 11300 | 0 | 11000 | cached=0 lcp=0 | none | none | none | 0 | 81185450160 | 737943552 | 81923393712 | 0 | 3.33 | ee53848835a29d90 | yes |

The base's V3 holds (nothing was evicted, and the cache costs exactly its 737,943,552 resident bytes); the
red is V1, V2 and V4: seven cold turns, no publication, no protected eviction. The `[glm5-spec] route= ...
cold=1 restored=0` line the issue quotes is a draft-capable-model receipt; this artifact prints the
`[spec-k]` admission receipt instead, and it reads `cached=0 lcp=0` on all eight turns.

#### Fix `d520f745f`, per turn (`gate-fix-r3/cell/TURNS.md`)

| turn | prompt_tokens | cached_tokens | prev prompt_tokens | server receipt | hit | insert | evict (segment) | refused/skipped | effective free after | resident prefix bytes | cache-off effective free after | V3 error | elapsed s | text sha256[:16] | == cold |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 9200 | 0 | - | cached=0 lcp=0 | none | 9200 tok 430.1MB | 2800 tok 240.0MB (Protected) | 0 | 81735433476 | 927997952 | 82663431428 | 0 | 2.641 | 254a65a01730e58b | yes |
| 2 | 9500 | 9200 | 9200 | cached=9200 lcp=9200 | 9200 of 9500 | 9500 tok 439.0MB | 3000 tok 246.0MB (Probation), 3200 tok 251.9MB (Protected) | 0 | 81183970480 | 869101568 | 82053072048 | 0 | 0.249 | 24a97a2867768b2d | yes |
| 3 | 9800 | 9500 | 9500 | cached=9500 lcp=9500 | 9500 of 9800 | 9800 tok 447.9MB | 9200 tok 430.1MB (Probation) | 0 | 81145992880 | 886919168 | 82032912048 | 0 | 0.251 | 65eeb1ef8f716af1 | yes |
| 4 | 10100 | 9800 | 9800 | cached=9800 lcp=9800 | 9800 of 10100 | 10100 tok 456.8MB | 9500 tok 439.0MB (Probation) | 0 | 81108015280 | 904736768 | 82012752048 | 0 | 0.252 | 64a99158cb668b0c | yes |
| 5 | 10400 | 10100 | 10100 | cached=10100 lcp=10100 | 10100 of 10400 | 10400 tok 465.7MB | 9800 tok 447.9MB (Probation) | 0 | 81070037680 | 922554368 | 81992592048 | 0 | 0.253 | c6b9d167a76a942e | yes |
| 6 | 10700 | 10400 | 10400 | cached=10400 lcp=10400 | 10400 of 10700 | 10700 tok 474.6MB | 10100 tok 456.8MB (Probation) | 0 | 81032060080 | 940371968 | 81972432048 | 0 | 0.254 | 8e4798b352770d9a | yes |
| 7 | 11000 | 10700 | 10700 | cached=10700 lcp=10700 | 10700 of 11000 | 11000 tok 483.5MB | 10400 tok 465.7MB (Probation) | 0 | 80994082480 | 958189568 | 81952272048 | 0 | 0.256 | f34ee12b3db5cb9c | yes |
| 8 | 11300 | 11000 | 11000 | cached=11000 lcp=11000 | 11000 of 11300 | 11300 tok 492.5MB | 10700 tok 474.6MB (Probation) | 0 | 80947386544 | 976007168 | 81923393712 | 0 | 0.256 | ee53848835a29d90 | yes |

The growing tenant's server lines, verbatim, turns 1 and 2 (the room came from the protected cohort, oldest
first; the entry never evicted itself):

```text
[prefix-cache] evict (snapshot preflight, Protected LRU): 2800 tokens, 240.0MB (model gate, ns "cohort")
[prefix-cache] insert probation (seed): 9200 tokens, 430.1MB (resident 928.0MB / 1074MB, model gate, ns "grow")
[prefix-cache] demote (protected bytes): 246.0MB to probation (model gate, ns "cohort")
[prefix-cache] hit: 9200 of 9500 prompt tokens from cache (model gate)
[prefix-cache] evict (snapshot preflight, Probation LRU): 3000 tokens, 246.0MB (model gate, ns "cohort")
[prefix-cache] evict (snapshot preflight, Protected LRU): 3200 tokens, 251.9MB (model gate, ns "cohort")
[prefix-cache] insert probation (seed): 9500 tokens, 439.0MB (resident 869.1MB / 1074MB, model gate, ns "grow")
```

From turn 3 on, each turn's hit promotes the previous entry, the protected share (858,993,459 B) demotes the
turn before it to probation, the preflight evicts that probation entry (`evict (snapshot preflight,
Probation LRU)`), and the new turn publishes: `cached_tokens` equals the previous turn's `prompt_tokens` on
every turn, and `prefix_cache_bytes` equals exactly the resident entries the lines describe.

- **The refusal line.** No turn of the growing tenant was refused on the card (`refused_or_skipped=0`): the
  shape requires every turn to fit, and every turn did. The typed refusal exists for the entry that cannot
  fit and is pinned by the CPU arm, verbatim from the test:
  `[prefix-cache] insert refused: entry 11 exceeds budget 10 (seed, model m, ns "t")`, and for the leased
  boundary
  `[prefix-cache] insert refused: entry 6 cannot fit beside 5 leased bytes (budget 10, snapshot preflight, model m)`.
  The once-announced `snapshot skipped` line the base prints does not exist in the fix.
- **What changed and what did not.** Changed: victim selection when probation is empty or holds only the
  newcomer (protected LRU, oldest first, no floor at `protected_target_bytes`), the preflight's reclaimable
  set (every unleased byte), the refusal lines (typed, in bytes, on both paths), the preflight evict line
  (names the segment). Not changed: captured and restored bytes.
- **The trade-off, named (integ14 review finding 1, lead decision: keep the behaviour, document it).** The
  operator-visible SLRU contract moved. The real rule is now: every unleased byte, protected or not, is
  reclaimable for a publication that fits the budget; the protected share bounds only demotion and pinned
  admission; only leases are untouchable. Scan resistance survives only for entries that fit the free or
  probation share: a one-hit entry larger than the free share beside a promoted cohort evicts that cohort
  oldest-first instead of running cold, so scan resistance for such entries is gone under SLRU. This gate
  shows the extreme of it: one growing tenant with one hit per turn removed the other tenant's entire
  737,943,552 B protected cohort by turn 2. That is what #523 item 1 asked for (the newest turn fits,
  protected oldest first); whether SLRU stays the default is #523 item 2, the interleaved A/B on the
  incident's shape, still open. `docs/SERVING.md` and the `MEMRA_PREFIX_CACHE_PROTECTED_PCT` row in
  `docs/FLAGS.md` now state this rule instead of "probation LRU before protected LRU".
- **The refusal lines are throttled (integ14 review finding 2).** Removing the one-shot `ANNOUNCED` guard
  had made both typed refusals per-request stderr, unbounded on a saturated cache.
  `PrefixRefusalAnnouncer` (worker.rs) prints the first refusal, suppresses and counts identical repeats
  (same kind, bytes, budget, leased bytes, model and salt key), prints again on a changed shape carrying the
  previous shape's count, and prints every 64th identical repeat with its count;
  `prefix_cache_skips_budget` and `prefix_cache_skips_pinned` count every refusal regardless. Unit tests
  `prefix_refusal_announcer_prints_first_changed_and_every_nth_identical_refusal` and
  `prefix_cache_repeated_refusals_count_every_time_and_print_once`. A log throttle and documentation need no
  GPU rerun; the target-card cells above stand (no refusal was printed in any of them, so the throttle did
  not take part). Every completion digest is identical across the two binaries on all
  8 turns and all 6 cohort sends (`254a65a01730e58b`, `24a97a2867768b2d`, `65eeb1ef8f716af1`,
  `64a99158cb668b0c`, `c6b9d167a76a942e`, `8e4798b352770d9a`, `f34ee12b3db5cb9c`, `ee53848835a29d90`),
  and identical to the cache-off calibration boot's cold completion of the same prompt on all 8 turns in
  both arms: the fix's seven restored turns produce the cold turns' bytes. Not changed either: the
  SLRU default, the protected share, the lease rules, the `lru` rollback arm.
- **V3 in bytes.** After every turn, in both arms, `calibration effective free == measured effective free +
  resident prefix bytes` with error 0 (turn 1 of the fix: 82,663,431,428 = 81,735,433,476 + 927,997,952;
  turn 2: 82,053,072,048 = 81,183,970,480 + 869,101,568). Nine evictions totalling 3,771,900,000 B of
  entry bytes over the eight turns cost effective free nothing beyond what the cache holds.
- **A recorded observation, cause not inferred.** The request path retains device memory that grows with
  the longest prompt seen: 962,571,604 B at the first 9,200-token request after a cache-off cohort,
  610,359,380 B more at 9,500 tokens, then 20,160,000 B per additional 300 tokens (67,200 B per token), and
  28,878,336 B on the last turn; identical to the byte across the two binaries and across all three rounds,
  and present with the cache doing nothing. It is not prefix-cache memory and this gate does not judge it;
  the numbers are in every cell's `calibration.json`.

## Checks actually run

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` (merged tree) | PASS |
| `cargo test -p memra-server --offline --no-fail-fast` (dev, local, `CPUQuota=1200% MemoryMax=28G`; `pro-single-day14/local-checks/test-server.log`) | 748 passed, 0 failed, 6 ignored; the five prefix-cache tests above ran and passed |
| `cargo clippy -p memra-server --offline --all-targets -- -D warnings` (dev, local, CPU quota) | PASS (`local-checks/clippy-server.log`) |
| `bash tools/check-flags.sh`; `python3 tools/check-public-boundary.py check`; `git diff --check` | PASS (no uncovered runtime `MEMRA_*` name; boundary 0 new matches) |
| Post-review battery (throttle + docs, `local-checks/post-review/`): `cargo fmt --all -- --check`, `cargo test -p memra-server`, `cargo clippy -p memra-server --all-targets -- -D warnings`, `tools/check-flags.sh`, `tools/docs-registry-census.sh`, `git diff --check`, all under the CPU quota | fmt PASS; 750 passed, 0 failed, 6 ignored (the two throttle tests included); clippy PASS after one lint fix (`is_multiple_of`); flags PASS; docs-registry census PASS (58 tables, 903 rows); diff check PASS. No GPU rerun: a log throttle and documentation change no numeric program and no victim selection; no refusal line was printed in any target-card cell |
| Native release builds, one RTX PRO 6000 Blackwell (`build-main/`, `build-fix/`) | exit 0 / exit 0, `dirty.txt` empty |
| `tools/prefix-newest-turn-fits-gate.py` on `main` `be07f2d36`, rounds 1, 2, 3 | `-> FAIL` in every round (red, as the defect requires) |
| `tools/prefix-newest-turn-fits-gate.py` on the fix, rounds 1, 2, 3 | round 1 `V1=ok V2=ok V3=FAIL V4=ok -> FAIL`; round 2 `V1=ok V2=ok V3=FAIL V4=ok -> FAIL` (turn 1 only); round 3 `V1=ok V2=ok V3=ok V4=ok -> PASS` |
| `tools/serve-smoke.sh <Qwen3.8 artifact> /nonexistent-draft` through the collector on the target card (`serve-smoke/`; the battery rebuilt `memra-server` from the checkout `d520f745f`, binary `a3ea4bc9...`, byte-identical to `bins/fix`) | `serve-smoke: 0 failed`; spec, gemma4 and Q35 arms `SKIP` (no draft or model on this box) |
| `tools/cache-meter-gate.py --n 5 --k 256` against the fix binary, native path, `MEMRA_SERVE_SPEC=0`, through the collector (`cache-meter/`) | `cache-meter-gate: 0 failed` |
| `tools/tier-battery.py --validate` on every cell (`validate-*.log`) | `capture-integrity`, 0 refused, `"qualification": false` everywhere; the gate cells carry the collector status their exit code earned |
| `research/spill-b-20260919/verify-day14.py` (offline replay of the mirrored receipts) | `DAY14 REPLAY OK` (`pro-single-day14/verify-day14.log`): both verdicts re-derived from the per-turn rows, binaries bound to their build receipts and sources, 600 W rig line, shape gates, 7 cold server receipts on the base, cross-binary and cold-boot identity, V3 state error 0 on all 16 turn rows, rounds 1 and 2 checked as failed cells with the stated reasons |
| Full GPU exactness battery (`kernel-check`, `run-gen`, `run-spec`), local 5090 cells | NOT RUN (no kernel or numeric change; the local card carried the lead's perf battery) |

Receipt directories under `pro-single-day14/` (mirror of the target card's `b-day14`, binaries and bundles
excluded): `refused-sitting1/` (the first sitting's refused chain and its build receipts), `build-main/`,
`build-fix/`, `gate-main`, `gate-fix` (round 1), `gate-main-final`, `gate-fix-final` (round 2),
`gate-main-r3`, `gate-fix-r3` (round 3), `serve-smoke/`, `cache-meter/`, the driver scripts, driver logs,
exit files and `validate-*.log`. Each gate cell holds `LOCK.json`, `rig.json`, `shape.json`,
`calibration.json` (rounds 2 and 3), `summary.json`, `TURNS.md`, `VERDICT.txt`, and per boot `server.log`;
the collector adds `CELL.jsonl`, `command.log`, `command.gpu.csv` (250 ms telemetry) and
`command.capture.json`.

## Boundaries and record

- Every GPU command on the target card went through `tools/tier-battery.py --rig pro-single` with the
  canonical lock (the gate cells with `--external-lock`, inherited FD, proof in each cell's `LOCK.json`;
  serve-smoke and cache-meter under the collector's own hold). No third lock name; no bare GPU run; no
  `--no-verify`; no skip variable; no touch of `/root/artifacts`, `/root/memra-spill`, other lanes'
  worktrees or `main`. Local builds and tests under `systemd-run --user --scope -p CPUQuota=1200%
  -p MemoryMax=28G`. No local GPU cell.
- Native checkout `/root/wt-b` synced by git bundle (detached at the gate tip; the binaries record their
  own source in `build-*/source.txt`). Receipts mirrored from the target card's `b-day14` to
  `pro-single-day14/`.
- Push: the pre-push hook allowed the lane push at `d27badac1` (perf board, flags census, releasability and
  docs-registry censuses, public boundary all OK); later commits pushed the same way where the hook allowed.
