# WP-B day 16: the day-15 policy A/B replayed on the second card class (local RTX 5090, #523 item 2)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `34ed99dfc` at
`0416a2e72`. This is the confirmation cell lead ruling 19 owes (`INTEGRATION-DAY12.md`, verbatim): "the local
5090 replay of the same harness (from history) is owed as a confirmation cell, not as a blocker." The decision
(`docs/decisions/PREFIX-CACHE-POLICY.md`) is already landed on the target card's verdict; this day either
confirms the mechanism reading on the second card class or reports a disagreement for the lead to decide. It
changes no code, no default, no gate, and no byte. Written in two parts: the **pre-registration** below was
committed before any GPU run of the day; the **result** sections follow it and quote the runs verbatim.

## Pre-registration (committed before the runs)

### The harness, from history, off the lane

The harness `tools/prefix-policy-ab.py` and the two-arm binary exist only at `9466b8912` (the day-15
pre-registration commit; the landing `87d9e5963` deleted the SLRU arm and the harness with the door). Neither
comes back onto the lane. A DETACHED worktree of that tree (`git worktree add --detach <path> 9466b8912`,
removed with its target dir when the day closes) builds `memra-server` release with its own
`CARGO_TARGET_DIR` under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`
(`build-day16.sh`, receipt `rtx5090-day16/build-tip/`: `source.txt` = `9466b8912`, `dirty.txt` empty,
`exit` 0, `binary.sha256`). The collector and the lock-proof tool at that ref already know
`--rig rtx5090` and `/tmp/memra-5090.lock`; the harness is run from that worktree through the collector with
the inherited canonical lock (`run-day16-ab.sh`; `--gpu-lock /tmp/memra-5090.lock` so the proof binds the
FD to the rig's lock). If another job holds the lock the driver waits 90 s and retries, six attempts at
most; it never kills a holder. No bare GPU run.

### The card and the scaling

Local NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, `power.limit` reported `[N/A]` by nvidia-smi on this
laptop (`power.max_limit` 175 W); the second card class after the target card (one RTX PRO 6000 Blackwell,
97,887 MiB, 600 W). The day-15 cell does not fit this card unscaled: the target's 250 ms telemetry peaked at
28,947 MiB `memory.used` and the settled `/metrics` live footprint (driver used minus pool-cached) reached
27.9 GB, both above 24,463 MiB. So the shape is scaled to the budget this card allows, **1024 MiB**, half of
day 15's 2048 MiB, and the byte SHARES are what is preserved, not the token counts:

- An entry costs `156,893,184 B + 29,696 B/token` on this artifact (exact, from the day-15 `/metrics`
  deltas: 388,521,984 B at 7,800 tokens, 394,461,184 B at 8,000; the day-15 least-squares fit over the
  MB-rounded insert lines gave 157,260,000 + 29,650). The fixed part is the model's recurrent state; it
  does not scale with the budget, so halving the budget and halving the tokens would NOT halve the entries.
  The token counts below are chosen so every entry's share of the budget matches the target's.
- **Cohort**: four tenants of 1,250, 1,350, 1,450 and 1,550 ids (target 7,800 to 8,400), each seeded twice.
  Cohort bytes 793,870,336 B = 73.9 % of the 1,073,741,824 B budget (target 74.0 %), inside the 80 %
  protected share (858,993,459 B).
- **Loop**: the agent tenant replays 12 turns from 11,000 ids, **+150 ids per turn** (target 27,300 and
  +300), to 12,650. Entry 1 = 483,549,184 B = 45.0 % (target 45.0 %); entry 12 = 532,547,584 B = 49.6 %
  (target 49.6 %); entry 11 + entry 12 = 1,060,640,768 B <= budget, margin 1.2 % (target 1.2 %);
  cohort + entry 1 = 1,277,419,520 B > budget. The growth is halved with the budget so that its share of
  the budget over the loop stays the target's 4.6 %; the incident's "about 300 tokens per turn" is a token
  count, and what the policy sees is bytes.
- **Returns**: unchanged in structure, after loop turns 3, 6, 9 and 12 one cohort tenant continues (its
  prompt plus 150 new ids), round robin; every cohort tenant once more after the loop. 28 requests per run,
  the same roles in the same order as day 15, so the per-request rows line up with day 15's table.
- **Server**: `MEMRA_CTX=16384` (day 14's local-friendly envelope; the longest prompt is 12,650 and a
  request with `max_tokens` given is sized to its prompt, so the envelope only bounds parked-session
  capacity, which is what this card cannot spare at 32768), `MEMRA_PREFIX_CACHE_MB=1024`,
  `MEMRA_SERVE_SPEC=0`, `MEMRA_COMPAT=openai`, `MEMRA_TTFT_TRACE=1`, greedy, `max_tokens=8`, `prompt_ids`
  requests, one boot per run, the boot line must report `byte-SLRU` for A and `plain-LRU` for B. Every
  other harness parameter is day 15's.
- **Schedule**: `AB-0, BA-0, AB-1, BA-1, ..., AB-4, BA-4`, five pairs per order, 20 runs, one lock hold,
  one thermal window, 250 ms telemetry from the collector, plus the cache-off boot for the cold digests. A
  one-pair smoke (`ab-smoke`, no verdict by design) runs first, as on day 15, to catch a memory or driver
  fault before the long cell and to read this card's own `insert` lines against the arithmetic above.
- **Shape gates** (the harness refuses, exit 2, unless): cohort <= protected share; cohort + entry 1 >
  budget; every loop entry <= budget; entry 11 + entry 12 <= budget. Read from this card's own first run.

### The rule, unchanged

The day-15 pre-registered rule, verbatim from DAY15.md: precondition, every completion digest identical
across the twenty runs and the cache-off boot on all 28 requests (a mismatch is a FAIL of the day, exit 1);
primary, total computed tokens across the replay, lower is better; secondary, the cohort tenants'
`cached_tokens` on their returns, stated whichever way it points; win, better on the primary at every one of
the ten pairs in both orders, a tie at any pair is not "better", otherwise INCONCLUSIVE. No timing is
compared with the target card or any other box; the metric is computed tokens and the digest precondition.
This day lands nothing new: `WINNER=lru` confirms the decision on the second card class; anything else is
reported to the lead as a disagreement, plainly, and the decision text is not touched.

### Predicted outcome (from the day-15 measured mechanism, written before the runs)

Day 15 read three mechanisms from the server's own lines (loop served identically by both arms; after a
cohort return SLRU evicts the loop's newest entry and the next turn computes two growths instead of one;
after the loop SLRU protects the dead last loop entry over the fresh cohort entry so the last tenant's
final is cold under SLRU and a hit under LRU). Replaying those three mechanisms over the scaled bytes above,
every eviction decision falls the same way (the arithmetic is in this file's git history and in
`verify-day16.py`'s expected table): predicted primary **slru 31,700 vs lru 29,550** computed tokens per
run, lru better by **2,150** (3 x 150 on the post-return turns plus the 1,700-token final hit) at every
pair, deterministically; predicted secondary, return cached slru 0, lru 1,700; loop cold turns after turn 1,
0 in both arms; refusals 0; digests 28/28. Day 15 is the reference for what a deviation would mean: a
different token count with the same three mechanisms is a fit difference to record; a pair where lru is not
better is a disagreement to stop on and report.

## Result: the smoke cell on the local RTX 5090, verbatim, and why the day stopped there

Binary `4a55ad8e656b15b5a5c3c47977551052b56d1f3fec86681d47ff607871536721` built from `9466b8912` in the
detached worktree (`build-tip/`, `dirty.txt` empty, 2 min 57 s with the rig's sccache), artifact
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, NVIDIA GeForce RTX 5090 Laptop GPU (24,463 MiB, driver 595.84,
`power.limit` `[N/A]`, `power.max_limit` 175 W), through `tools/tier-battery.py --rig rtx5090` with the canonical
`/tmp/memra-5090.lock` inherited by the harness (proof in `ab-smoke/lock.json` and `ab-smoke/cell/LOCK.json`, same
device and inode), the lock free on the first attempt. One pair per order, four runs plus the cache-off boot, 28
requests each, 386.7 s, 1,525 telemetry samples at 250 ms (temperature 54 to 88 C over the cell, peak draw
192.3 W, peak `memory.used` 23,033 MiB of 24,463; the first and last samples read 15 MiB, so the card was alone for the
whole cell and every byte above that is this cell's own), collector status `failed`, exit 1:

```text
PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=793870336 turns=12 start_tokens=11000 grow=150 return_every=3 pairs_per_order=1 runs=4 requests_per_run=28 digests_identical=26/28 computed_tokens slru_median=31700 lru_median=29550 (N=2 each) pairs_slru_better=0/2 pairs_lru_better=2/2 ties=0/2 return_cached slru_median=0 lru_median=1700 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=69.0..75.0 power_limit_w=None -> DIGEST-FAIL
```

The shape gates held from this card's own `insert` lines (fit 29,700 B/token + 156,895,000 B; cohort 793,870,336 B
= 73.9 %, turn 1 45.0 %, turn 12 49.6 % of the budget, every gate true), exactly the pre-registered shares.

Per pair (computed tokens, lower is better; N=2 runs per arm, a smoke, no verdict by design):

| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 31700 | 29550 | 2150 | lru | 0 | 1700 | 0 | 0 | 191.16/280.406 | 184.097/187.013 | 74.0->71.0 | 71.0->69.0 |
| BA-0 | BA | 31700 | 29550 | 2150 | lru | 0 | 1700 | 0 | 0 | 188.215/285.607 | 193.092/197.588 | 75.0->73.0 | 69.0->75.0 |

### What agreed with the target card

The byte arithmetic. Every run's primary and secondary landed on the pre-registered prediction to the token:
slru 31,700 and lru 29,550 computed tokens (lru better by 2,150 at both pairs), cohort returns cached 0 under slru
and 1,700 under lru, loop cold turns after turn 1 = 0 in both arms, no refusal line. The per-request rows
(`ab-smoke/cell/REQUESTS.md`) line up with day 15's table row for row: the return publications evict the loop's
newest entry under SLRU (`evict 11300 (Probation)`, `11750`, `12200`, `12650`) and the hit entry under LRU
(`evict 11150 (Protected)`, `11600`, `12050`, `12500`), so the post-return turns compute 300 under SLRU and 150
under LRU (rows 12, 16, 20); after the loop SLRU evicts the fresh cohort-4 entry (row 25, `evict 1700
(Probation)`) and cohort-4's final is cold (row 27, 1,850 computed) while LRU evicts the dead `12650` and serves the
final from cache (`hit 1700`, 150 computed). `day16-predict.py`'s 28-row table matches every run's `cached_tokens`
on all 28 requests. Those are the day-15 mechanisms 2 and 3, made by the policy under test, on the second card.

### What did not: the precondition, and a confound

1. **Digest identity failed, 26/28.** The two slru runs differ from the cache-off boot at request 14 (loop
   turn 6, prompt 11,750 = 11,600 restored + 150 prefilled): run digest `70efa8d085a5531a`, cold `24a97a2867768b2d`,
   8 tokens and `finish_reason: length` on both sides. The two lru runs differ at request 20 (loop turn 10, prompt
   12,350 = 12,200 restored + 150): run `22f023976ebc22fd` (8 tokens, `length`), cold `4415b7e361fc6f6b` (3
   tokens, `stop`). Deterministic per arm: runs 1 and 4 agree with each other, runs 2 and 3 agree with each other,
   so identity ACROSS runs held and identity against the cold boot did not, at one restored-suffix request per
   lineage (the two arms restore different chains after turn 4, so the near-tie that flips sits at a different
   turn in each). The target card held 28/28 across 20 runs and the cache-off boot on day 15, and 8/8 on day
   14's twin gate. By the pre-registered rule a mismatch is a FAIL of the day, not a data point, so this card
   produced `-> DIGEST-FAIL` and no verdict. Not diagnosed here; two candidates, neither established: the
   restored-prefix-plus-suffix prefill and the cold monolithic prefill are different programs (the
   `splitiso-20260813` class) and are not bit-identical on this card's kernel arms, so a KV lineage of restores
   drifts in its low bits until a greedy near-tie flips; or the VRAM-pressure path below changes a workspace
   geometry on this card. Day 14's twin gate (V3, restored == cold on every turn) has never run on the 5090 and
   is the direct probe.
2. **The admission reclaim ladder evicted prefix entries in every run, in both arms.** The card cannot hold the
   model (about 19 GB with the CUDA context and workspaces, read from the target card's own settled `/metrics`),
   the 1024 MiB budget, the 2.1 GB prefill workspace an 11k-token prompt is charged, the parked plain sessions and
   the admission floor (`[admit-trim] ... floor_bytes=2147483648`) at once. So at every loop turn from 2 to 12 the
   server's `[admit-oom] reclaim-on-defer: evicted N prefix entries + M plain ... parked sessions (global LRU)`
   ran BEFORE the prefill, 11 events and 12 prefix entries per run (2 at turn 2, the cohort's `1450` and
   `1550`; then one per loop turn), the same in both arms, and `reclaim spared the prompt's prefix entry` each
   time. The policy's own `[prefix-cache] evict` lines fell to 9 (slru) and 7 (lru) per run against the target
   card's 21 and 19; the loop-turn evictions on this card were the reclaim's, not the policy's. The target card
   logged zero `[admit-oom]` and zero `[admit-trim]` lines in the whole 20-run cell. The reclaim takes the global
   oldest unleased entry, which at every loop turn coincided with the entry each policy would have taken next,
   which is why the primary still matched the prediction to the token; but the receipts cannot prove a choice the
   policy never had to make. The return and final publications, where the two arms differ and the day-15
   reading lives, were the policy's own decisions (their evict lines are in the rows above).

The shape cannot be replayed on this card without the confound: four cohort entries cost 4 x 157 MB = 628 MB
fixed before a single token, so the 80 % protected share needs a budget of at least 785 MB and the loop's two
consecutive entries beside it need the 1024 MiB used here, while the card's free VRAM after the model, the
prefill workspace and the 2 GiB admission floor is under 1 GB. The harness pins `MEMRA_MAX_SESSIONS=4` and the
parked-session count; changing it means changing the harness, which this day does not do.

### Stopped here, by the brief

The brief: "if the 5090 disagrees, stop and report, the lead decides." The disagreement is on the precondition,
and the 20-run cell (`ab-full`) would end in the same `DIGEST-FAIL` by the same rule with the same confound, so it
was NOT run; no GPU time went into a cell whose verdict rule was already failed. Prediction, mechanism rows and
the failing precondition are all in the smoke receipts, replayed offline by `verify-day16.py`.

## Checks actually run

| Check | Result |
| --- | --- |
| Native release build, detached worktree at `9466b8912`, own target dir, CPU quota (`build-tip/`) | exit 0, `dirty.txt` empty |
| `tools/prefix-policy-ab.py --pairs 1` through the collector, `--rig rtx5090`, inherited canonical lock (`ab-smoke/`) | `-> DIGEST-FAIL`, exit 1, collector `failed`; 4/4 runs on the predicted primary; digests 26/28 |
| `research/spill-b-20260919/verify-day16.py` (offline replay of the smoke receipts) | `DAY16 REPLAY OK` (receipts consistent; precondition FAILED re-derived; prediction matched 28/28 rows in every run; 12 reclaim evictions per run counted from the server logs) |
| `day16-predict.py --day15` (the corrected model against day 15's measured runs) | slru 132,300, lru 122,700, evictions 21/19, return cached 0/8,700: exact |
| `cargo fmt --all -- --check`, `bash tools/docs-registry-census.sh`, `git diff --check` (lane tree, CPU quota) | see the commit log line and STATE.md |
| `tools/tier-battery.py --validate` on the smoke | NOT RUN: the collector already labels the capture `failed`; `--validate` asserts `executed-not-qualified` captures, and there is nothing to qualify here |
| The 20-run cell (`ab-full`) | NOT RUN (precondition failed on the smoke; the brief says stop and report) |
| Full GPU exactness battery | NOT RUN (no code change; the digest finding above is the one to chase, on its own lane) |

## Boundaries and record

- Every GPU command on this card went through `tools/tier-battery.py --rig rtx5090` with the canonical
  `/tmp/memra-5090.lock` (inherited FD, proof in `lock.json` and `cell/LOCK.json`); no third lock name; no bare
  GPU run; no `--no-verify`; no skip variable; no other lane's worktree touched, `main` untouched; nothing of
  V4.1; no external dependency; no captured, restored or served byte changed; the harness and the SLRU arm were
  used from the detached worktree only and never returned to the lane. Build and cell under
  `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`.
- The detached worktree and its target dir were removed when the day closed; the binary's SHA-256 and source ref
  are in `build-tip/`.
- No timing is compared with the target card. The TTFT columns are this card's own record of its own runs.
