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
