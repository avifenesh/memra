# WP-A day 54: OWED item 10, the publishes still on the tick (a census and a price per publisher first)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `33a7ecd63` (item 21 closed). The lead's order: items 10 to 14,
item 10 per publisher. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell `executed-not-qualified`.

## 1. Pre-registration (committed before any code)

**The item** (`OWNER-THREAD-OFFLOAD.md` days 24 to 26, item 2): the publishes that still run on the tick under the door:
the fanout leader's snapshot and the pause sweep's boundary snapshot (`prefix_snapshot` direct); the
`dspark-boundary` publish (the drafter's `export_tail`); the `glm5-boundary` publish (latent tails); and every `OnTick`
answer of the two capture routes (`prefix_capture_off_tick` for the seed and LCP-split publishers,
`prefix_spec_capture_off_tick` for the spec-boundary publishers). None has a price on record. Per publisher, the price
comes first, then a design where the price is one the tenant sees.

**What each publisher needs to run, read from the code** (no support claim for any family is made or implied):

| publisher | where | what exercises it |
|---|---|---|
| fanout leader snapshot, then the siblings' restores | `dedup_interactive_prefixes` | cold interactive requests admitted together with the same first 64 tokens (`MEMRA_PREFIX_DEDUP` is the served default), plain sessions |
| pause park snapshot (V's shape 1) | the pause sweep | `MEMRA_KV_PAUSE_DEMOTE=1`, a tool-calling turn (V's cell) |
| seed and LCP-split publishes answered `OnTick` | `prefix_insert_from_session` | a cache the capture route refuses: TP shards, latent planes, a layer not at the boundary, no logits, no model; or the route latched |
| spec-boundary publishes answered `OnTick` | `prefix_insert_from_spec_boundary` | the same refusals on the spec route, plus a DFlash tail (`dspark-boundary`) and latent tails (`glm5-boundary`) |

The 27B on one RTX PRO 6000 exercises the first two and whatever route refusals its caches hit. The `dspark-boundary`
publish needs a DFlash drafter with `MEMRA_DSPARK_BOUNDARY_CAPTURE=1` (default OFF by design), the `glm5-boundary`
publish a GLM-5 tensor-parallel deployment, the latent refusals an MLA model: none is on the target box, and each is its
own family's program. They are recorded as not measured here, and their designs wait for an artifact and a rig (the
lead's call).

**Step 1, the lines (log only; no behavior or numeric program changes).**

1. Each route records why it answered `OnTick` (one `&'static str` per return site, kept on the host cache), and each
   caller that then runs the tick program prints, under the door only, `[prefix-cache] on-tick publish: publisher=<why>
   (the <capture|spec> route answered on-tick: <reason>); snapshot X ms, insert Y ms, B MB`. Door OFF prints nothing
   new.
2. The fanout: `[prefix-dedup] on-tick publish: snapshot X ms (B MB), N sibling restore(s) Y ms, insert Z ms` beside
   the existing `[prefix-dedup] B=..` line.
3. The pause park: `[prefix-host] pause park snapshot on the tick: X ms (B MB)` before its off-tick demote.

The times are the owner thread's wall time around each part. The copies they enqueue run on the owner stream; their GPU
time reaches the tenant through the next decode kernel, which the stall cells read.

**Step 2, the cells** (one RTX PRO 6000 Blackwell, the 27B NVFP4 MTP artifact, the door ON, the PRO environment of the
S sittings; `pro-single-day54/`, one collector hold per cell):

- `fanout` (a new `stall_cell.py` mode): the tenant streams; at the fire, four identical fresh prompts are posted at
  once (so the dedup groups them), at 72 and at 4096 tokens; against `prime` at the same lengths (one fresh prompt):
  the fanout's own on-tick share is its stall minus the single prime's (the prime runs on the tick in both). N=5 per
  order, both orders (`fanout prime x5`, `prime fanout x5`), five boots each at each length.
- `pause` (V's cell, door ON, `MEMRA_KV_PAUSE_DEMOTE=1`): the park snapshot line read; the tenant stall read as V
  read it.
- The census: the identity gate (default and plain boots), the hit gate ON, and the pause gate on the tip binary, with
  every `on-tick publish` line counted by publisher and reason.

**The rule, stated before any cell.** Per publisher on this card: its price is the median of its owner time and, for the
fanout, the stall difference (fanout minus prime) per order. A publisher whose tenant-visible share is above 1.0 ms in
both orders (the resolution of the S sittings' e2e clauses) gets a design pre-registered next, in price order; one at or
below 1.0 ms in both orders is closed as priced, with its receipt; one not exercised on this card stays open, owed to
its family's artifact and rig. A route refusal the census shows on the 27B is its own publisher row.

**What each card decides.** The target card only (the tenant's tick is the target's). The 5090 reads the same cells
after its reset, a compatibility reading.

**Budget.** 0.4 agent-day: the lines and their census 0.1, the stall mode and the sitting 0.1, the card 0.2 (about an
hour).

## 2. As built (`17d43a6ac`), the CPU cells, and the sitting prepared

- Step 1: `HostPrefixCache::on_tick_reason` (a `&'static str`) is set before every `OnTick` answer of
  `prefix_capture_off_tick` (its reasons: the capture path latched, no transfer engine, an SWA ring cache,
  tensor-parallel shards, no model, latent planes, the cache position off the token boundary, no boundary logits, a KV
  layer not at the boundary, the path latched while settling), of `host_capture_submit` (no host tier, no transfer
  engine) and of `prefix_spec_capture_off_tick` (the same kind, plus a DFlash drafter tail, latent boundary tails, the
  capture snapshot off its boundary, the boundary outside the committed tokens, a KV layer shorter than the boundary,
  no KV rows). Each route's combined refusal is now one `else if` chain over the same conditions; day 24's census pins
  the chain's `dspark_tail` arm by its new text. The lines print as section 1 states; the fanout's line prints under
  the door too (section 1 left its condition unstated; door OFF prints nothing new anywhere).
- Census `day54_the_on_tick_lines_are_log_only` (every `OnTick` answer preceded by its reason; the refusal names; no
  decision reads the reason; the door guard on the lines; the fanout's order), its teeth checked (one reason removed:
  `an OnTick answer without its reason`).
- CPU cells, green: server lib `925 passed; 0 failed; 25 ignored`; clippy `-D warnings`; fmt; `git diff --check`.
- `stall_cell.py` gains `fanout`, `fanout-long` and `prime-short` (four identical fresh prompts posted at once, the
  slowest wall recorded with each member's wall and cached tokens; the single-prime control), smoke-checked against a
  fake local server (four members, `STALL REPLAY: PASS`); the earlier modes' programs and receipts unchanged.
- The reader `day54-reading.py`, checked on a synthetic fixture (every price line, the census row, the rule's three
  answers).
- The sitting `pro-single-day54/`: `build.sh <tip>`, `driver.sh` (the short and long paired cells, the pause cell,
  the census gates, the hit gate, the reader). One change from section 1's environment, stated here before any cell:
  the paired cells boot with `MEMRA_MAX_SESSIONS=8` (the fanout needs the tenant plus four intruders; the S sittings'
  4 would queue the fourth), both modes alike; the pause cell keeps 4. About 1.5 hours of card time on one RTX PRO 6000
  Blackwell with the 27B artifact; any host class, recorded (the arms are compared within the hold).

## 3. The sitting on the target card, as it ran (the lead's run; `pro-single-day54/box/`)

- Run by the lead on BOX28: one RTX PRO 6000 Blackwell Workstation Edition; the host reads `AMD Ryzen 9 5900XT 16-Core
  Processor` (32 CPUs, 121 GB), a Zen 3 class, recorded as read (the arms are compared within each hold). A fresh
  clone at `b05e79f3e`, `build.sh b05e79f3e` `rc=0` (the tip `072659f1639f6db9..`; markers `on-tick wording: 2 dedup: 1
  park: 1`), `driver.sh` with `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock` exported. Cells: `ab-short-cell rc=0` 19:17:03Z,
  `ab-long-cell rc=0` 19:47:26Z, `pause-cell rc=0` 19:50:28Z, `census-cell rc=0` 19:54:15Z, `hitgate-on rc=0`. Replays 20,
  20 and 3 of 3 `STALL REPLAY: PASS`. The model sha256 `1facf36c2db359dc..`.
- Mirror: 328 files, 327 of 327 against the box's `box-mirror-manifest.sha256`, 0 mismatched (the lead's `lead-a54.out`
  and `lead-box-after.txt` beside them); the ELF by hash (`box-binaries.sha256`).
- Thermal regime: boot starts 29 C (the first) and 51 to 66 C after; telemetry 29 to 77 C, SM up to 2850 to 2857 MHz.
- **The box's reader stopped** after the two fanout prices: `NotADirectoryError: .. '/root/spill-receipts/a-d54/pause/
  binary.sha256/server.log'`: the pause glob `b*` matched the cell's `binary.sha256`. A reader defect, corrected (the
  boot directories `b[0-9][0-9]` only) before its pause and census parts were read; the corrected reader re-run here on
  the mirror prints the box's lines unchanged and the rest (`reading-day54-corrected.log`).

**The reading, verbatim** (the corrected reader over the mirror):

```
DAY54 PRICE fanout (short) stall fanout-minus-prime o1=+6.50 o2=+6.50 ms rule >1.0 both -> DESIGN NEXT
DAY54 PRICE fanout (long) stall fanout-minus-prime o1=+896.82 o2=+895.60 ms rule >1.0 both -> DESIGN NEXT
DAY54 PRICE pause park snapshot owner N=30 median=0.73 ms (pause stall median 64.00 ms, N=30) rule >1.0 -> CLOSED AS PRICED
DAY54 CENSUS no on-tick publish line from either capture route on this card
DAY54 PRICE dspark-boundary (a DFlash drafter tail) -> NOT MEASURED HERE (not exercised by the 27B on one card)
DAY54 PRICE glm5-boundary (latent tails, tensor parallel) -> NOT MEASURED HERE (not exercised by the 27B on one card)
DAY54 PRICE latent planes (MLA models) -> NOT MEASURED HERE (not exercised by the 27B on one card)
DAY54 VERDICTS fanout (short): DESIGN NEXT; fanout (long): DESIGN NEXT; pause park: CLOSED AS PRICED
```

- The census gates are green: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` default and plain, `KV-HOST-PAUSE-DEMOTE
  GATE: ALL GREEN`, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, each exit 0.

**Verdicts, as registered.** The fanout publisher: **design next** (both lengths). The pause park snapshot: **closed as
priced** (0.73 ms of owner time; the pause cell's 64 ms stall is the tool turn's own prime on the tick, which the stall
reads whole). The seed, split and spec-boundary route refusals: **none on this card** (no on-tick publish line in any
gate or cell: the 27B's caches never hit a refusal). The DFlash, GLM-5 and latent publishers: **not measured here**,
owed to their families' artifacts and rigs.

**What the fanout's price is made of** (readings, no verdict; they shape the design's pre-registration):

1. At 72 words the four identical prompts share a 95 to 98-token prefix, and the fanout adds 6.50 ms to the tenant's
   stall over one prime of the same prompt. Its on-tick parts: the leader's snapshot 0.72 ms (160 MB), the three
   sibling restores 1.19 ms, and the insert 2.26 ms, which is the evicted entry's demote pre-submit (steady 2.09 ms: the
   pinned lease allocation, item 19), not the fanout's.
2. At 4096 tokens the shared prefix is capped at 1024 tokens (`prefix=1024` on every group), so each of the four
   members primes its own 3072-token suffix on the tick: the +896 ms is three extra suffix primes, the four requests'
   own work, not the publisher. The single-prime control does not match four requests there; the price is recorded as
   read, and the design's cells use the short shape and a matched control. The groups also split when the four
   arrivals straddle a tick (`{4: 33, 3: 9, 2: 8}` and `{4: 27, 3: 15, 2: 8}` of 50 in the two orders).
3. The long fanout's insert holds the owner 18.25 ms, the evicted long entry's demote pre-submit (18.11 ms steady, 32
   pinned leases of 30.4 MB): item 19's price again. The first demote of each boot holds it 78 ms in `spans` (the staging
   set's first allocation) on this host class, against about 20 ms on the 9950X class (DAY49): item 19's other half.
