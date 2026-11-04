# Memory-shaped admission is the default on the RTX PRO 6000 Workstation Edition (2026-11-04)

**Chosen:** on the full-power RTX PRO 6000 Blackwell Workstation Edition, the batched worker
admits by memory, in four parts. (a) A request that bounds neither `max_tokens` nor `max_ctx` is
charged `prompt + 32768 + 8` (clamped to the trained context) instead of the `MEMRA_CTX`
envelope, and 32768 is its output budget. (b) The admission reclaim flush demotes device prefix
entries into the pinned host tier up to the arrival's shortfall before dropping the rest. (c) A
memory-deferred request requeues until both tiers are exhausted and `MEMRA_ADMIT_DEFER_BUDGET_MS`
(8 s) is spent, then answers 429 with `Retry-After`. (d) Every headroom reading subtracts the
prefill workspace and the capped prefix seeds that still-priming sessions owe, and a prefill CUDA
OOM on a session that emitted nothing parks and requeues. `MEMRA_ADMIT_BY_MEMORY` is deleted;
`MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` and `MEMRA_ADMIT_DEFER_BUDGET_MS` stay as runtime parameters. The
boot line `[admit-mem] door=ON` or `door=OFF` states the card's arm. Code:
`MemoryAdmitConfig::for_device` in `crates/memra-server/src/admit_memory.rs`.

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
- The Max-Q and Server editions, the B200 route the mechanism was first built for and every other
  card: no cell in this repo. A launcher that set the retired name is refused at boot; that
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

**What would change it:** the 5090 half of DAY40 reading V-DOOR PASS on its registered tree adds
the RTX 5090 class (its own SELECT read R1 and R2 8192 on day 36, so its open-output value is its
own question). A deployment whose registry advertises another `max_output` sets
`MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` to it.
