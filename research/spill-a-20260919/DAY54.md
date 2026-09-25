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
