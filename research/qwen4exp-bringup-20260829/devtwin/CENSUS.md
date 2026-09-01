# devtwin: host-twin crossing census (work item 1) — what still crosses after the yarn lane's device scorer

Lane: qwen4exp device host-twins (owner-sequenced from PROFILE-7 §2 verify decomposition,
mtp11 AUDIT §4, and the round-3 graphs doctrine note). Code refs at branch tip 1da5d6136,
`crates/memra-engine/src/qwen4exp_gpu.rs` unless said otherwise. Geometry: 48 layers, all
MoE (512 experts, top-10); 12 QSA layers (1 per 4); 36 GDN layers, one of which (layer 1)
carries PLE; MTP draft block = 1 QSA layer + MoE on card 1.

## 1. Remaining host decisions + crossings, per FORWARD (single-card route)

"BLOCKING dtoh" = `dtoh_view`/`clone_dtoh` — a stream sync that drains everything queued
before it; these are the per-layer bubbles PROFILE-7 attributed ~12 ms/round to. h2d into
parked slots is pageable staging (quasi-blocking), not a stream drain.

| # | site | code | host decision | crosses | count/forward | class |
|---|------|------|---------------|---------|---------------|-------|
| R1 | MoE router dtoh | `moe_route_slots` L6777 (graph driver t==1); `moe_forward` L7724 (eager t==1 + verify/draft chunks); TP2 L12902 | none yet (the GEMV is device) | [t,512] f32 (2 KB × t) | **48** | BLOCKING dtoh, one per MoE layer — routing is layer-sequential, so each is a full pipeline drain |
| R2 | router softmax-top-10 | `host_route_softmax_topk` L1349 | softmax over 512, top-10 by (weight desc, index asc — `total_cmp`), renorm with the 6.1035156e-5 denominator floor | — | 48 × t rows | HOST TWIN (the decision this lane moves) |
| R3 | selection h2d | L6783-6784 (slots); L7804-7806 (grouped: sel/w/tok arrays) | — | t==1: sel 40 B + w 40 B; grouped: 3 × t×10×4 B | 48 | h2d into parked slots (the count-gated pack twin `tp2_pack_bytes` L13346 is the TP2-graph form of the same product) |
| I1 | QSA idx_proj dtoh | `qsa_forward` L7197 | none yet | [t,640] f32 (2.5 KB × t) | **12** | BLOCKING dtoh — feeds the HOST raw-key cache (`MixerState::Qsa.raw_keys`) + query prep |
| I2 | indexer selection | `indexer_select_rows` L1599 | full-row structural fast path (complete <= 512 → EVERY position < 2051 selects nothing); past 2051: HOST pooled-key extend (mean+rmsnorm+rope, L1447), HOST query prep, device scoring (`qsa_index_score_f32`, yarn lane), **score-slab dtoh L1737 + HOST top-k** (`top_blocks_ascending`, pinned tie rule score-desc/index-asc) | below horizon: nothing; past: [rows × n_blocks] f32 slab back | 12 | HOST TWIN. The yarn lane moved SCORING only; the pooled-cache math, query prep, slab readback, and top-k SELECTION still cross |
| I3 | mask/selpos h2d | L7296-7299 (dense mask) / L7266-7269 (blocklist) | host renders `rowsel_to_mask`/`rowsel_positions` | t×t_kv u8, or ≤2052 i32/row | 12 | h2d (host-rendered, but derivable from base_pos/t alone below the horizon) |
| P1 | PLE n-gram gather | `ple_block` L8267-8297 | host hashing + gather from the 102 GB host table | h2d [t,2560] f32 | 1 | HOST BY DESIGN — stays (table is host-resident; bounds the census, not this lane) |
| P2 | PLE gate dots dtoh | L8332 | — | t f32 per stream | **4** | BLOCKING dtoh |
| P3 | PLE gate scalars | L8337-8348 | signed-sqrt + sigmoid on host | h2d t f32 per stream (L8352) | 1 host loop + 4 h2d | HOST TWIN (movable; small — 1 layer) |
| H1 | head readback | L6235/6250/6258 | argmax sink / last-row | 4t B or 1 row | 1 | mtp11 territory, already minimal |

Blocking dtoh per forward: **48 (router) + 12 (idx_proj) + 4 (PLE) = 64**, plus the head.
The t==1 decode step pays the same class (PROFILE-7: "the t=1 step pays the same class
per forward") — and under an ARMED verify, decode graphs are structurally OFF
(`state.verify.is_none()` in graphs_mode, the mtp11 fix), so every zero-draft round is a
full 64-drain eager step too.

## 2. Per spec ROUND at ship admission (dev1, K=5, adapt k_lo=1 + pmin 0.3)

| phase | router dtoh | idx dtoh | PLE dtoh | other |
|---|---|---|---|---|
| verify chunk (t=j+1) | 48 | 12 | 4 | S3 argmax drain (mtp11) |
| draft chain (j steps, card 1) | j × 1 | j × 1 | 0 (no PLE in MTP) | S1/S2 2j argmax/prob dtoh (host arm) |
| replay (t=a+1, card 1) | 1 | 1 | 0 | S7 P2P (instrumented) |
| zero-draft round (t==1 trunk) | 48 | 12 | 4 | commits via device argmax (mtp11) |

The draft-step router+indexer twins are exactly what PROFILE-8 §4 named: they serialize
the host INSIDE each chain step, which is why the mtp11 deferred readback measured
~0.01-0.02 ms/round instead of the 0.67 ms class — its ceiling rises only when these
move device-side (this lane), and its banked ladder is the re-measure baseline (item 4).

## 3. What does NOT cross (checked, so nobody re-audits)

- MoE dispatch after routing: grouped path (gufuse/sel_v3/axpy combine) and shared-expert
  tail are all-device (L7742-8143, `moe_shared_tail` L8144). The per-expert PREFILL
  executor consumes host routes by construction (host gathers per expert) — prefill
  keeps host routing, amortized over the whole prefill, out of scope.
- GDN layers: no host twin at all (the capturable class — 35 interior graphs exist).
- Verify accept walk / rewind / admission arithmetic: host by design over the S3 drain
  (4-byte argmaxes), already minimal (mtp11).
- planes_to_wide dtoh (L5880) = goldens capture only; trace-mode dtoh (L10157) = trace.

## 4. Consequences (why the router is first)

- 48 of 64 blocking drains per forward are the router boundary; it is also THE
  structural blocker the round-3 doctrine named: "MoE routing is a HOST twin by lane
  doctrine, so a whole-step graph is structurally impossible" (PROFILE-2 §graphs). A
  device router makes interior+route+tail one capturable span for the 35 PLE-free GDN
  layers and unblocks multi-layer capture sets (item 4).
- The indexer's hot-path crossing at the measured shapes (all < 2051 fill) is I1+I3
  only — the SELECTION never engages below the horizon (structural full rows). Killing
  I1 below the horizon = device-resident raw-key cache append + lazy host materialization
  at the first scored row; past the horizon the score-slab dtoh + host top-k (I2) is the
  yarn-lane residue this lane's item 3 moves.
- The TP2 route keeps its host boundary (L12882) this lane unless re-gated (tp2-gate
  24/24 required if touched).
