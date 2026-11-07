# Memory-shaped admission is the default on the RTX PRO 6000 Workstation Edition, full power and Max-Q (2026-11-04, Max-Q 2026-11-07)

**Chosen:** on the full-power RTX PRO 6000 Blackwell Workstation Edition and, since 2026-11-07, the
Max-Q Workstation Edition (each from its own cell), a batched worker that serves that one card
admits by memory, in four parts. (a) A request that bounds neither `max_tokens` nor `max_ctx` is
charged `prompt + 32768 + 8` (clamped to the trained context) instead of the `MEMRA_CTX`
envelope, and 32768 is its output budget. The charge never exceeds the envelope the unarmed
program would charge, so a `MEMRA_CTX` below the open output still bounds a naked request (its
budget is then the envelope less the prompt and the 8 slack rows). (b) The admission reclaim flush demotes device prefix
entries into the pinned host tier up to the arrival's shortfall before dropping the rest. (c) A
memory-deferred request requeues until both tiers are exhausted and `MEMRA_ADMIT_DEFER_BUDGET_MS`
(8 s) is spent, then answers 429 with `Retry-After`. (d) Every headroom reading subtracts the
prefill workspace and the capped prefix seeds that still-priming sessions owe, and a prefill CUDA
OOM on a session that emitted nothing parks and requeues. `MEMRA_ADMIT_BY_MEMORY` is deleted;
`MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` and `MEMRA_ADMIT_DEFER_BUDGET_MS` stay as runtime parameters. The
boot line `[admit-mem] door=ON` or `door=OFF` states the arm and its reason. Code:
`MemoryAdmitConfig::for_deployment` in `crates/memra-server/src/admit_memory.rs` (the card keys:
`HardwareTarget::is_pro6000_workstation` and `is_pro6000_maxq`).

**Rejected:**

- The unarmed program on that card, the previous default. It charges a naked request the whole
  context envelope and its memory defer never ends in a refusal the client can retry.
- An open-output value of 2048 or 8192. DAY32's pre-registered selection rules read on that card:
  R1 (zero truncation) selects none (2048, 8192 and 32768 truncate 10, 10 and 8 open rows), R2
  (natural cover) selects none (the largest natural completion is 193,178 tokens), R3 (registry
  agreement) reads 32768 and R4 (the surveyed convention) is the remaining context. 32768 is the
  one value a rule selects and the measured value that truncates the fewest open requests. 8192
  was a placeholder the code itself called unmeasured.
- The RTX 5090 class for now. It read V-DOOR PASS on day 34 and day 36, but on the uncapped seed
  booking; its half of the final-booking cell (DAY40, `rtx5090-day40/chain.sh`) has not read.
- A multi-card deployment of that card (PP stages, a TP pair, EP; `worker::is_multi_device_deployment`):
  the deciding cell ran one card, and no cell measured the open-output cap, the bounded defer or
  the pending-prime booking across devices. It keeps the unarmed program until one does (Step-3.7
  PP on a pair is the first such cell).
- The Server Edition, the B200 route the mechanism was first built for and every other card: no
  cell in this repo. A launcher that set the retired name is refused at boot; that
  route needs its own cell before it arms.
- Keeping the env door as a rollback seam. Owner, 2026-11-04: winners with positive receipts
  become defaults keyed to the card each receipt came from, and their switches go.

**Measurement that settled it:** `research/spill-b-20260919/DAY40.md` 2.1, one RTX PRO 6000
Blackwell Workstation Edition at 600 W, Qwen3.8-27B at the checkpoint's context, the final
booking (`S40` = `b46ae200e`: the capped seed booking and day 39's revised prime term), arms `off`,
`on2048`, `on8192` and `on32768` in both orders, burst 64. Every DAY34 1.6 term (V-CRASH, V-ID,
V-TRUNC band and conservation, V-ALLOC, V-RETRY, V-OOM, G-BOOK, PARK) passes on all eight boots,
`V-DOOR ... -> PASS`. `on32768` admits 46 of the 64-request burst and refuses 18 with a typed 429
in both orders, in an 80 to 92 s window against `off`'s 150 to 152 s; `on2048` and `on8192` admit
64 of 64. The seed cap bound on every armed boot. No panic, respawn, fatal or engine error on any
boot.

The Max-Q row is DAY52 section 2 (`research/spill-b-20260919/pro-maxq-day52/`), pre-registered
before any Max-Q boot: the same S40 program, model, arms, orders and readers on one RTX PRO 6000
Blackwell Max-Q Workstation Edition beside training on the box's other cards, its two orders in two
windows. `V-DOOR ... -> PASS` with no boot excluded; every term passes on all eight boots, and
`on32768` again admits 46 of the burst and refuses 18 with a typed 429 in both orders. The card held
300 W for about three minutes of each order before the box's cap timer returned it to 280 W; no term
reads power. Greedy outputs equal the Workstation receipt on 414 of 416 sequential rows (a reading
with no bound). The open-output value stays 32768: R3 reads no boot.

**Not measured:** the envelope cap. Every cell ran at a `MEMRA_CTX` wider than its open output
(65,536 on the 5090, the checkpoint's 262,144 on the target card), where the cap changes nothing.
It exists because the 32768 default would otherwise let a naked request on a narrow `MEMRA_CTX`
(the 8192 several serving gates boot with) book four times the KV its operator set.

Also not measured: whole-session resume of naked multi-turn conversations. Armed, a naked
request's session cap is its own charge (`prompt + 32768 + 8`, at most the envelope) instead of the
constant `MEMRA_CTX` envelope, the shape a bounded request already has on every card. A later turn's
prompt is longer, so its cap is larger than the parked session's, and the exact and text tiers of
the spec pool and the plain continuation probe (each requires the parked cap to cover the new one)
miss it. The spec and plain affinity tiers (default on) still resume it, growing the parked
session and rewinding to its last prompt-end checkpoint, so the previous reply is primed again;
without them the prefix cache serves the turn. DAY40's continuation turns read the prefix cache on both arms (1440 of about
1570 tokens cached), so no receipt shows the cost. The cell that would: naked back-to-back
multi-turn requests on that card, unarmed against armed, reading resume hits and TTFT per turn.
If it regresses, the pool tiers would admit a smaller parked cap and grow it as park compaction
does, or the owner takes the trade.

**What would change it:** the 5090 half of DAY40 reading V-DOOR PASS on its registered tree adds
the RTX 5090 class (its own SELECT read R1 and R2 8192 on day 36, so its open-output value is its
own question). A deployment whose registry advertises another `max_output` sets
`MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` to it.
