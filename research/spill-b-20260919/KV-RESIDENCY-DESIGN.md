# KV residency: the four options, their gates and one recommended order (WP-B day 26, 2026-09-22)

Design note for memra#539 (contiguous per-session KV at `request_ctx_cap`, deep-copy prefix hits, no preemption of an
emitting session) read against memra#552 (the tier program's G0 to G7 qualification) and
`docs/decisions/KV-PHYSICAL-RECLAIM.md` (the G1 reclaim criterion and the `--kv-allocator vmm` door, decide-by
2026-10-04). The census behind every number here is `DAY26.md` section 1 (file and line anchors on tree `1c66ff10e`);
the before receipt is `DAY26.md` section 2. No engine change accompanies this note. The owner decides the order.

## What the census fixes before any option

- A session's device KV is allocated once, at admission, at `max_ctx = ctx_cap` rows for every trunk full-attention
  layer (`Cache::new_inner`, `crates/memra-kv/src/lib.rs:2892-2905`), and it is never grown, shrunk, trimmed or shared
  for the session's lifetime. `ctx_cap` is `prompt + max_tokens + 8` when the client bounds output and the whole server
  context when it does not (`worker.rs:21705-21745`), unless `MEMRA_ADMIT_BY_MEMORY=1` re-charges the open case at
  `prompt + 8192 + 8`.
- Bytes per token per class (q8_0 K, q5_1 V, one `n_head_kv x head_dim = 1024` row per full-attention layer, 1,856 B
  per layer per token): Qwen3.5-9B 8 full-attention layers, 14,848 B/token plain, 16,704 spec; Qwen3.8-27B (the mint
  and the class Qwen3.8 belongs to) 16 layers, 29,696 plain, 31,552 spec. Recurrent layers cost a fixed state, not a
  per-token row.
- At the served context the open arm therefore books and allocates `262144 x 31,552 = 8,270,192,640 B` per request
  on the 27B and `262144 x 16,704 = 4,378,853,376 B` on the 9B, whatever the prompt; the bounded arm allocates
  `(P + max_tokens + 8) x bpt`. Used KV is `(P + G) x bpt`. The two ratios have different causes: the open arm's is
  the cap rule (policy); the bounded arm's is the client's unused `max_tokens` (layout); the prefix hit's extra copy is
  the sharing rule (layout plus cache design).
- What the gate ladder means here. `KV-PHYSICAL-RECLAIM.md` defines G1 (driver-visible reclaim: (a) to (e), the
  series lift under ruling 6); lane D's `G2-PROTOCOL.md` defines G2 (the host/device transfer envelope, N>=5 both
  orders); #552 criterion 4 names the rest as "model/byte/serving/performance cells" with source, binary and artifact
  identities. I found no tracked text spelling out G0 and G3 to G7 individually (searched `research/spill-*`,
  `docs/decisions`, the lead's FREEZE and HANDOVER notes, and the darklanes spec and research trees); the mapping
  below uses G1 and G2 by their definitions and names the byte, serving and performance cells each option owes in
  the vocabulary #552 uses. The one-program law (`CLAUDE.md`, "One numeric program per request") applies to every
  option: a request may never produce tokens under two numerical programs unless the transition is forbidden or
  proven bit-identical in a serving-shape gate.

## (a) Admission by memory with a bounded default when `max_tokens` is omitted

What it is. Policy, no numeric change. The door exists: `MEMRA_ADMIT_BY_MEMORY=1` (`admit_memory.rs:84`, default
OFF, `docs/FLAGS.md` row with decide-by 2026-09-23) makes `request_ctx_cap` return
`charged_ctx_tokens(prompt, None, MEMRA_ADMIT_OPEN_OUTPUT_TOKENS = 8192, model_ctx)` on the open arm
(`worker.rs:21717-21723`, `admit_memory.rs:139-147`). Because `ctx_cap` is both the charge and the allocation
(`worker.rs:25336`, `24718`), the door already bounds the allocation, not only the book.

What the API contract says today. `docs/SERVING.md:912-914`: `max_tokens` omitted means a context-bounded budget
(session context minus prompt, capped at the trained context), "the OpenAI default-when-omitted semantics, not a
silent 128-token truncation". `docs/SERVING.md:373-386` already recommends sending `max_tokens` and quotes the
stranding it measured at `MEMRA_CTX=32768`. Deployments with a registry output limit never reach the open arm:
`lib.rs:8646-8656` resolves an omitted `max_tokens` to `default_output_length` (or `max_output_length`) before the
worker sees it, so the fleet shapes that pin those are already bounded and unchanged by this option.

What would change for a client. On a naked boot (no registry limit) a request that omits `max_tokens` would stop at
about 8,200 generated tokens with `finish_reason: "length"` where today it runs to the context. `usage` reports it.
Nothing else moves: same prompt, same tokens up to the stop, same program.

Gate owed. A decision cell, not a qualification: the day-26 mix with the door OFF and ON on the same binary, N>=5
per arm per order, both orders, both cards; per request the allocated and used bytes, `finish_reason`, and the
`[admit-mem]` receipt line; the concurrency the card admits at the served context under OFF and ON. Byte identity is
by construction (only the cap moves) and is still recorded: completion digests OFF versus ON equal on every request
whose generation ends before the bound. Maps to: a serving cell under #552 criterion 4; touches no G1 or G2 surface.

Cost. 0.5 agent-day (the cell and the FLAGS decision; the code exists). The decide-by is tomorrow.

What it moves on the target card at the served context. From the census: the open arm's allocation goes from
8.27 GB to `(P + 8,200) x 31,552`, about 0.35 GB at a 3k prompt; the number of open sessions the free device memory
admits goes from single digits to the `MEMRA_MAX_SESSIONS` ceiling (64). The bounded arm and the prefix copy are
untouched.

## (b) Grow-on-demand contiguous planes with a reserved virtual range (the VMM path)

What it is. `KvAllocator::Vmm` / `Cache::new_with_allocator` / `KvPlane` (`crates/memra-kv`), the door
`kv-tier-gate --kv-allocator vmm` from days 8 to 12. Today the door reserves a fixed VA range per plane and maps the
whole range at construction; demote unmaps and releases whole 2,097,152 B granules inside the demoted range and
restore maps them back at the same VA. Grow-on-demand is the same mechanism with a different trigger: reserve the VA
for `max_ctx` rows at construction, map physical granules only as `pos` crosses a granule boundary (the append path
that publishes `len_d`), and never move a byte.

What the receipts say (`KV-PHYSICAL-RECLAIM.md`). `ACTIVE-8K G1 PASS` on both card classes: released equals
reacquired equals 201,326,592 B, driver free moved by exactly that, all 32 planes at their original VA, tokens,
logits and state byte-identical. 32k: `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)`
on both cards, residual 2,097,152 B per cycle, drift 0. The mechanism that makes device memory come back is this
one; the pooled trim was rejected by measurement (`RECLAIM-DIAG: freed but not observable`).

What a grow step costs on the tick. Not measured yet. The receipts record whole-range map and unmap of
905,969,664 B at 32k, not a per-granule `cuMemCreate + cuMemMap + cuMemSetAccess` inside a decode tick. On the 27B a
2 MiB granule holds 1,927 K rows or 2,730 V rows per layer, so a session maps one granule per plane about every
1,900 tokens, 32 planes per boundary. The cost sits on the tick that crosses the boundary (a stall class like lane
A's day-16 cell, `stall_median`), not on every token; pre-mapping the next granule off the boundary is the obvious
follow-up if the stall is visible. The pool arithmetic also changes: VMM planes are outside the CUDA pool, so
`cuda_pool_cached_bytes` stops describing them and the effective-free reading admission uses
(`effective_free_bytes`) must count mapped VMM bytes explicitly.

The one-program law under a reallocation. There is no reallocation: the plane's address never changes, the kernels
read the same pointers, and the mapped prefix holds the same bytes. The law is satisfied by construction for the
tokens; the gate still has to prove it (run-gen argmax and run-spec K=1..8 pooled versus grown, the hit gate and the
twin gate on both allocators, byte tape equal). The failure mode is a map fault or a stall, never a different token.

Gate owed. G1 (exists for demote and restore; a grow series cell is new: N>=5 boundary crossings in one process,
driver free falling by exactly the mapped granules, restore drift 0), G2 untouched (no host transfer), byte cells
(kernel-check, run-gen, run-spec, `prefix-newest-turn-fits`, `qwen-hit-gate`, `serve-smoke`, `cache-meter`) on both
allocators with equal lines, a serving stall cell at the granule boundary (tick latency, N>=5 both orders), and the
admission accounting cell (booked versus mapped versus used). Maps to #552 criterion 2 (active-KV capture, demote,
load, resume with fault arms; the `device-short` arm gets a grow twin) and criterion 4.

Cost. 3 to 4 agent-days: the append-path trigger and the plane bookkeeping (1), admission and metrics accounting of
mapped bytes (1), the grow series cell, stall cell and the serving-shape battery on both cards (1 to 2). The door's
decide-by (2026-10-04) is the natural decision point; promotion without a grow cell would only promote the
demote and restore arms.

What it moves. Allocated becomes mapped: the physical footprint follows `pos` to within one granule per plane
(32 x 2 MiB = 64 MiB on the 27B, 16 MiB on the 9B), so the open arm's 8.27 GB becomes about `(P + G) x bpt + 64 MiB`
and the bounded arm's unused `max_tokens` slack stops costing physical memory. The book, however, still charges
`ctx_cap` unless admission moves to mapped bytes, which is the accounting half of (a): (b) without a memory-shaped
book frees bytes the gate will not hand out.

## (c) Paged KV (block table, kernel changes on every attention path)

What it is. Fixed-size token blocks from a per-device pool, a block table per session, every KV reader and writer
addressing rows through the table. This is the "rejected alternative" of `KV-PHYSICAL-RECLAIM.md` for the reclaim
question (reclaim becomes pool-available bytes, not driver bytes), and it is #539's item 1.

Which kernels. Every path that takes a KV plane pointer plus a row index. From `docs/KERNELS.md` on this tree: the
decode family `fa_decode_f32`, `fa_decode_vec_q*` (about 30 variants) and `fa_decode_combine*`; the prefill and prime
family `fa_prefill_q*`, `fa_prefill_qw*` (`_hd128`, `_db*`, `_t3`, `fa2`, the `*_prime_table` twins that read the live
causal depth); the append and quantize writers `append_quantize_kv_q8_0_q5_1*` (rows, dc, seqs, inc, prime_table);
the batched decode pointer table `decode_v2_rope_fa_rows` with its `{k, v, len, base, ctr, back}` per-rank slab
(`MEMRA_ROWS_TAB_RESTAGE`); the spec verify repair `copy_batch_uniform_kv_u8_set_len`; the TP KV sidecar
(`TpKvRankAllocationShape`); and, outside the Qwen class, the MLA latent planes (`memra_mla_*_live_kernel`) and the
Gemma windowed planes. The graph decode path bakes plane pointers at capture, so a block table means either a
per-session indirection buffer the graph reads or a re-capture on every block boundary. Arch guards: the naked
sm_120a build and the `memra_hopper_mma` twins of any kernel that has one.

The bit-identity gate a paged layout owes. The layout is addressing, not arithmetic, only if every kernel visits the
same rows in the same order with the same accumulation grouping. The FA kernels tile the KV sequence; a block size
that is not a multiple of the tile (or a tile that straddles a block boundary) changes the online-softmax grouping and
the result is not bit-identical. The gate is therefore not argmax alone: byte tape (logits and state) paged versus
contiguous on every kernel in the list above, at every tile and block boundary shape, plus run-gen argmax and
run-spec K=1..8 per family, plus the serving-shape identity gates on both cards. That census is why this is the
largest item: it is a rewrite of the KV addressing contract under every optimized path, with a proof obligation per
kernel, on two arch guards, and it also moves the prefix cache (entries become block lists), the tier (demote and
restore per block, the `KvPlane` contract), PP placement and TP sharding.

Gate owed. G1 restated (a block pool must itself be VMM-backed to satisfy the driver-free criterion; a pool of blocks
inside the CUDA pool is exactly the `not-applicable-pooled` case), G2 for block-granular transfers, the full byte
battery above, the #539 acceptance serving cell (a 2k prompt on a 1M server holds 2k tokens of blocks; 16 sessions
sharing an 84k prefix hold one copy once (d) lands; a session demoted mid-decode resumes byte-identical), and the
#552 fault arms per block. Performance cells: decode and prefill tok/s paged versus contiguous, N>=5 both orders,
both cards, since indirection costs a load per tile.

Cost. 10 to 15 agent-days on the owner's scale: kernel twins and their per-kernel byte proofs (5 to 7), the cache,
prefix, tier, PP and TP surfaces (3 to 4), the batteries on both cards (2 to 4).

What it moves. The same physical result as (b) for a single session (footprint follows use to a block) plus the
ability to place a session's rows anywhere, which is the precondition for (d) and for preempting an emitting
session by block. Without (d) it does not remove the N private copies of a shared prompt.

## (d) Prefix-hit sharing without copy (copy-on-write pages)

What it is. Depends on (c). A hit maps the entry's blocks into the new session's table with a refcount instead of
`copy_u8_into` per plane (`worker.rs:13547-13553`); the first write into a shared block copies that block only; the
prefix cache becomes a radix over block hashes; eviction frees blocks whose refcount reaches zero. The hybrid
mid-entry refusal (`worker.rs:13470-13476`, recurrent state exists only at the captured endpoint) still holds: a
shared block prefix ends at a block boundary at or before the entry's endpoint, and the recurrent state is copied,
never shared.

Gate owed. The hit gate's identity (restored bytes equal the entry bytes, completion digests equal the cold
digests, `qwen-hit-gate ALL GREEN` and the twin gate `PASS`) on the shared path; the #539 c=16 cell (one copy of an
84k prefix, all admitted, all byte-identical to solo); refcount and eviction fault arms in the #552 criterion 2
shape (a shared block evicted under a live lease must refuse, never free); the writer proof that no kernel writes
into a shared block (append targets only the session's own tail block; a shared partial tail block is copied first).

Cost. 3 to 5 agent-days after (c) lands.

What it moves. The prefix copy: today a hit costs `P x bpt` of fresh device memory per session on top of the entry
(the day-14 rows: a 9,200-token hit copied 430 MB into a fresh cache and kept the 430 MB entry); with (d) sixteen
sessions on one 84k prefix hold one copy.

## Recommended order, with the reason

1. **(a) first**, this week, as the decision cell for the door whose decide-by is 2026-09-23. The census says the
   open arm's allocation is the cap rule, not the layout: `262144 x bpt` regardless of prompt. That is the largest
   ratio in the mix and it moves with zero numeric risk (only the cap changes, the program is the same). The client
   change is one documented `finish_reason: "length"` at about 8,200 tokens on naked boots, which the fleet's
   registry deployments never see. If the owner keeps the door OFF, record why in FLAGS and the door is deleted per
   the door-hygiene rule; either way the question closes tomorrow.
2. **(b) second**, on the VMM door's own decide-by (2026-10-04). Its reclaim receipts exist on both card classes,
   the address never moves so the one-program law holds by construction, and it makes admission by memory honest:
   mapped equals used to a granule, so a memory-shaped book can hand out what the card actually has. It also fixes
   the bounded arm's slack (unused `max_tokens`) without touching a kernel. The grow series cell and the boundary
   stall cell are the two new gates; the rest of the battery exists.
3. **(c) only if the measured residual justifies a kernel rewrite** after (a) and (b): what remains is the granule
   rounding (64 MiB per 27B session), the impossibility of moving a live session's rows, and the N private copies of
   a shared prompt. Those are real on the shared-prefix serving shape (#539's c=16 cell), but they are a
   week-class rewrite with a per-kernel byte proof on two arch guards, and (d) is the part that actually pays for
   it. Price it in agent-days (10 to 15 plus 3 to 5), decide on the shared-prefix workload the fleet carries, and do
   not start it to fix a ratio that (a) and (b) already remove.
4. **(d) rides on (c)**; it is not schedulable alone.

The owner decides. The before receipt for whichever order is chosen is `DAY26.md` section 2 (both cards, N=5 per arm
per order, both orders, `executed-not-qualified`).

## Day 27 addendum: two findings from the census, one fixed, one for the owner (2026-09-22)

Both receipts are in `DAY27.md`; every cell is `executed-not-qualified`.

**Finding 1, fixed: the prefix-cache budget's context term.** The derived budget `min(2 x entry(ctx), boot_free - 1.5
GiB)` read `ctx` as `MEMRA_CTX` when set and a literal 8192 when unset, while the cap rule (`request_ctx_cap` through
`resolve_env_ctx`) reads the checkpoint's context when unset. On the target card (27B, `MEMRA_CTX` unset) that was
`2 x 400,162,816 = 800,325,632 B` of budget beside `262,144 x 31,552 = 8,271,167,488 B` sessions, and the day-26
deferred shape evicted 43 of 45 spec-boundary entries before their continuation (`cached=0` on 15 of 15). The fix
(`crates/memra-server/src/worker.rs`, `prefix_budget_ctx` = `resolve_ctx` per loaded model inside
`init_prefix_cache_budget`, tree `de2c781e6`) moves only the context term: the entry is `262,144 x 29,696 +
156,893,184 = 7,941,521,408 B`, the budget `15,883,042,816 B` under the unchanged clamp `84,909,096,960 B`. After cell
on the target card: 15 of 15 continuations hit (`cached_tokens` 1440 / 3104 / 5760), P and G equal to day 26 on 45 of
45, 45 entries resident at 11,986,255,872 B, 0 evictions; idle retention rose from 30.1 GB to 39.1 GB because the
entries now stay (that is the budget's purpose; they yield to sessions through `alloc_with_single_reclaim_retry`).
Consequence for the options above: (a) and (b) are unchanged; the shared-prefix argument for (c)/(d) now has a
prefix cache that actually retains on the target card at the served context, so the `c=16` cell of #539 can be run
against a working cache rather than a starved one.

**Finding 2, for the owner: what the 30 GB at idle is, and what the park door can reach.** Day 26 attributed the
retention to four parked whole-session entries across two pools. The gauges say `continuation_pool_entries=0` in every
target-card cell and `spec_pool_entries=2`: the mix is spec-path and a spec session parks in the spec pool only
(`worker.rs:20880-20925`). The 30 GB is pool reserved minus the post-warmup used (about 30.6 GB): two parked spec
sessions at `ctx_cap` (up to 2 x 8.27 GB when the last requests were open, 2 x 183 MB when they were bounded), the
prefix entries, and the CUDA pool's cached free blocks (12 GB or 29 GB, the complement of the parked bytes). What the
parked sessions bought on days 20 and 26: `continuation_pool_hits=0`, `spec_pool_hits=0`, every `prompt_ids` replay
`plain-affinity: declined (no checkpoint retained ...)`, every chat continuation `spec-affinity: declined (history
diverged at 48 of checkpoint ...)`. `MEMRA_KV_PARK_COMPACT` (`0 = OFF by design`, no `decide-by:` in its row) compacts
plain-pool parks only; by design it cannot touch the spec pool that holds the bytes here. The pre-registered park
cell (`DAY27.md` 2.4, 2.5) measures the door against the default on both cards; the pool's cached blocks are the
`--kv-allocator vmm` door's subject (decide-by 2026-10-04), not the park door's. Owner's decision, recorded on #539:
the park policy (TTL, per-pool caps at the served context, the spec pool's scope) is a product switch, not a lane's.
