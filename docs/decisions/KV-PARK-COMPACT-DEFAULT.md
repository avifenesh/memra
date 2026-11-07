# Continuation-pool park compaction is the default on its two receipted cards (2026-11-04)

**Chosen:** a parked plain-pool session's `Cache` compacts from its ladder cap to its committed
length at park time, and the resume grows it back to the request's charged cap through the
checkpoint-restore extension before any suffix primes. The default holds on the RTX 5090 class
and on the full-power RTX PRO 6000 Blackwell Workstation Edition, keyed by device name
(`park_compact_for_device` in `crates/memra-server/src/worker.rs`). `MEMRA_KV_PARK_COMPACT` is
deleted.

**Rejected:**

- Keeping the door default-off past its decide-by (2026-10-06). The owner's 2026-11-04 ruling:
  a winner with a positive, correct receipt becomes a default keyed to the card it came from,
  and its switch goes.
- One default for every card. The receipts cover two card classes. The RTX PRO 6000 Blackwell
  Max-Q and Server Edition variants, GB10, B200 and H100 have no cell, so they park whole, the
  same scope `tier_transfer::kv_vmm_placement_for_device` gives its Workstation receipt.
- An environment switch beside the key. The doctrine for a hardware-shaped default is
  detection, and no launcher or gate in the repo set the variable.

**Measurement that settled it:** `research/spill-b-20260919/DAY38.md`, the door's deciding cell,
plain path (`MEMRA_SERVE_SPEC=0`), default against `=1`, both orders, one binary per card. The
pre-registered rule (DAY38 1.6) reads PROMOTE-ELIGIBLE when P1 and P2 pass. The registered P1
read FAIL (no reading) on both cards: the compacting arm resumed turns the other arm primed
cold, and the near-tie residual made some of them differ. Addendum D (1.11) re-registered P1
and P2 before its runs, and both cards then passed:

| card | P1' (resumed turns equal across arms) | P2' (step-OOM adjacency) | rule |
|---|---|---|---|
| RTX PRO 6000 Blackwell Workstation Edition, 27B (2.3) | 30 of 30 resumed on both arms, no digest difference, both orders | PASS in both modes, red arm red | PROMOTE-ELIGIBLE |
| RTX 5090 Laptop GPU, 9B (2.4) | 20 of 20, no digest difference, both orders | PASS in both modes, red arm red | PROMOTE-ELIGIBLE |

Readings that rode with the rule (no bound): the park-time copy cost 1.3, 2.6 and 8.2 ms p50 at
the three fed lengths on the target card and 1.1 and 2.8 ms p50 at 6,144 and 30,720 on the 5090
(N=100 each); resumed-turn E2E p50 equal within a few percent on both cards (531 against 533 ms
on the target card, N=45 per boot; 481 to 483 against 466 to 469 ms on the 5090, N=30). The
freed ladder-cap allocations stay in the stream-ordered pool: at idle the compacting arm reads
1.24 GB less driver-free and 1.37 GB more pool-cached memory on the target card, 0.54 GB less
driver-free on the 5090. Resumed turns flip against their cold prime at the same rate on both arms (3 of 30 and
2 of 20), the continuation pool's documented near-tie residual, not a property of compaction.
The VMM interaction pair (`MEMRA_KV_ALLOCATOR=vmm`) showed no digest difference.

**Scope:** the receipts are one model per card on the plain path, both of the qwen35 hybrid
class (full-attention KV plus recurrent state). SPEC and DSpark pools park live engine sessions and
were never in scope. A card class joins the default only with its own DAY38-shape cell. The device
key covers the whole RTX 5090 class (the desktop card with the Laptop GPU the cell ran on), the
class scope every per-device default in the engine uses.

**What compaction refuses** (`park_compact_rows`, each refusal named on the `[kv-reuse]
park-compact declined` line; the cache parks whole):

- latent (MLA/DSA) KV planes: `CacheSnapshot` carries none and `restore_cache_checkpoint` never
  writes one, so a compacted glm5_next or glm_dsa cache would resume over empty latent rows;
- the eager-only class (`HybridModel::eager_only_class`: Gemma-4, GLM-5.3-Flash, glm_dsa,
  MiniMax-M3), whose contract has no prefix snapshot;
- an SWA ring, a distributed TP KV mirror and `MEMRA_PP_HOST_BOUNCE=1`, the refusals the door
  carried;
- a cache whose position is not its fed length, or one already at its committed length.

Only an entry the park compacted resumes below the request's charged cap (`ReuseEntry::compacted`),
so the admit-time grow never runs on a cache the compactor declined.

**Owed:** no cell has run compaction on a family outside the qwen35 class. On these two cards it
still reaches every plan the refusals above let through, among them step35 (Step-3.7, whose
blocking gate on the RTX PRO 6000 is `step-pro`), hy3 and the dense and MoE qwen3 packs. Each
joins the evidence with a plain-path identity cell on its card: every resumed turn read against
the same turn primed cold, with the near-tie residual counted as DAY38 counts it.
