# TP/EP position-split C4 store (memra #710, the "head-split KV" lane, 2026-09-26)

Owner direction 2026-09-25: "flip and continue to head split". TP/EP replicated every layer's
KV cache on both ranks, so a session held about 800k tokens plain and 300k with DSpark, against
PP-2's 1M.

Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`. Hardware:
2x RTX PRO 6000 Blackwell Server Edition (SE pair). One scored campaign at a time under
`/tmp/memra-gpu.lock`.

## Why a position split, not a head split

DSv4-Flash has `num_key_value_heads = 1`. Each position keeps one 512-wide latent, and all 64
query heads read it. Attention TP2 splits the query heads, so there is no KV head to split. The
bytes live in the C4 layers' compressed stores. There are 21 C4 layers, and each keeps one
2 KB row per 4 positions: 10.75 KB per token per rank of the 13.8 KB total at 1M.

## What changed

- **Store.** In each C4 layer, compressed block `c` lives on rank `c & 1`, at row
  `win + (c >> 1)`. Both ranks still run the compressor, since its arithmetic is replicated.
  Only the owner stores the emitted block.
- **Recent ring.** The non-owner keeps its own computed copy of each block in a tagged ring
  (slot `(c >> 1) % R`, tag `c`). `R` is `transient_rows / 8 + 4`, rounded up to a power of
  two. Rows emitted in the current transaction, and recent rows in general, therefore never
  cross the link.
- **Attention.** A gather kernel builds each query's selected rows in slot order. Each row comes
  from the local store, the recent ring, or the peer's store (a direct P2P read). The kernel
  rewrites the indices the way the host-C4 `C4Gather` does, so the unchanged sink attention
  kernel reads the same values in the same order.
  - Query sub-batches are 32 rows (`SPLIT_GATHER_ROWS`).
  - Rows move as 16-byte words; peer words use `__ldcv`.
- **Prefill staging.** A transaction wider than 8 rows copies the peer's committed half of the
  layer's store into a local buffer once per layer, then gathers from that copy. The copy is one
  peer memcpy on the rank's stream. Without it, every query re-read its remote rows over PCIe, so
  a 12k-word prompt paid +20% TTFT.
- **Unchanged.** The indexer keys (`ikvc`) stay replicated. The C128 stores and window rings are
  O(1) or small per token.
- **Snapshot and restore.** Parking reads the logical store from both ranks
  (`split_kvc_to_host`); restoring writes each rank's half back (`split_kvc_from_host`). The
  gates' cache digests and DSpark's cache classes read the logical store too. Before
  `d5b0bbcea` they read the physical rows, which made DSpark's bit-gate FAIL on a store that was
  correct.
- **Default.** The split is on for the TP/EP attention-TP matrix device program, which is the
  served default. `set_c4_split_for_gate` pins it either way. The legacy paths refuse a split
  state.

## Correctness

Everything is bit-identical on the SE pair at `1072d9f40`, the final lane head
(`raw/v5d-1072d9f40/`):

| gate | result |
|---|---|
| `dsv4_kv_split_gate` 3000 / 300 | chunked prefill, 300 eager steps, 300 full-token replay steps and park/restore plus 50 steps: split bit-identical to replicated (logits bits every step) |
| `dsv4_kv_split_gate` at max_seq 1M | the same arms at the served context (`raw/gates-d5b0bbcea/split-1m/`, and `raw/v1-served-f69dbbbbc/split-1m/`) |
| `dsv4_tp_replay_long_gate` 304 | replay bit-identical to eager; tokens sha `112c2fc6...`, unchanged from main |
| `dsv4_rows_gate` TP/EP | every row bit-identical to its solo step across join, leave, row moves, replay re-entry, B-row graphs and sampled draws |
| DSpark TP/EP gate | 14 cells, 77 logit rows, 3206 cache-class comparisons ALL BIT-IDENTICAL; proposal shas `62b368f10f1becea d404da5fe101077f` unchanged |

The split gate's prompt is long enough that the indexer selects from more blocks than its top-k,
so remote rows, recent-ring rows and local rows all reach attention.

## Bytes and capacity

Per-rank session bytes, from the split gate's plan lines at max_seq 1M
(`raw/v1-served-f69dbbbbc/split-1m/gate.log`):

| capacity | replicated bytes/token/rank | split bytes/token/rank |
|---|---|---|
| 65,536 | 14,806 | 9,514 |
| 262,144 | 14,022 | 8,667 |
| 1,048,576 | 13,825 | 8,455 (-38.8%) |

Largest admitted session on a served boot at max_seq 1M. The probe climbs a max_tokens ladder
in 100k steps from 100k to 1M, with a 10-token prompt. It closes each stream after its first
chunk (`raw/v1-served-f69dbbbbc/cap-*`). The next rung above each figure was refused by the
admission memory check:

| route | main `2c5edcb4c` (replicated) | lane (split) |
|---|---|---|
| plain | 800,000 | 1,000,000 (every rung; the ladder stops there) |
| DSpark | 300,000 | 500,000 |

## Served cost

SE pair, one boot per row, order Ls Mn Mn Ls Ls Mn, N=3 per arm, medians. Lane `1072d9f40`
against main `2c5edcb4c`. Every cell is greedy with the same text on every request
(`raw/v5d-1072d9f40/r*`):

| cell | lane agg | main agg | lane decode p50 | main decode p50 | lane TTFT p50 | main TTFT p50 |
|---|---|---|---|---|---|---|
| greedy c1 | 67.58 (67.48..67.58) | 67.87 (67.84..67.91), -0.4% | 70.24 | 70.71 | 176 ms | 175 ms |
| greedy c2 | 92.59 (92.38..92.84) | 93.44 (93.32..93.49), -0.9% | 48.36 | 48.87 | 264 ms | 262 ms |
| greedy c4 | 123.29 (123.21..123.39) | 124.02 (123.82..124.15), -0.6% | 32.54 | 32.77 | 459 ms | 456 ms |
| greedy c2, 1500-word context | 16.97 | 17.17, -1.2% | 41.69 | 42.26 | 11.99 s | 11.88 s (+0.9%) |
| greedy c1, 12000-word context | 4.71 | 4.76, -1.1% | 51.47 | 52.51 (-2.0%) | 49.41 s | 48.88 s (+1.1%) |

Aggregates in tok/s, ranges over the three boots. The prefill staging took the 12000-word TTFT
cost from +19.9% to +1.1%. What remains is the decode step's remote reads, which grow with the
selection: -0.7% decode at c1 on a short prompt, -2.0% at a 12000-word context.

The v1 head (`f69dbbbbc`, no staging, 4-byte gather words) on the same protocol
(`raw/v1-served-f69dbbbbc/r*`):

| cell | lane agg | main agg | lane TTFT p50 | main TTFT p50 |
|---|---|---|---|---|
| greedy c1 | 67.45 | 67.84 (-0.6%) | 176 ms | 174 ms |
| greedy c2 | 92.52 | 93.42 (-1.0%) | 261 ms | 263 ms |
| greedy c4 | 122.79 | 124.30 (-1.2%) | 455 ms | 451 ms |
| greedy c2, 1500-word context | 15.76 | 17.16 (-8.2%) | 13.11 s | 11.88 s (+10.4%) |
| greedy c1, 12000-word context | 4.02 | 4.76 (-15.5%) | 58.62 s | 48.90 s (+19.9%) |

## What stays open

- **Indexer key split (v2).** Splitting `ikvc` too would bring the store to about 7.1 KB per
  token per rank, which is PP-2 parity. Each rank would score its own blocks, take a local
  top-k in the existing total order (score descending, index ascending) and merge the two
  lists. The top-k of the union equals the unsplit top-k element for element. The cost is one
  small exchange per C4 layer, about 21 more cross-rank joins per step. Plain sessions already
  reach the 1M rung, so v2 only buys DSpark context (500k today). It is not taken
  in this lane.
- **Decode read cost.** A decode step still pulls each selected remote row over P2P: up to 256
  rows x 2 KB per C4 layer, -2.0% decode at a 12000-word context. Both ranks compute the same
  selection, since the indexer is replicated, so the owner could push those rows into the peer's
  gather buffer instead. PCIe posted writes are cheaper than non-posted reads. This needs one
  more cross-rank join per C4 layer. It is on the ceiling list (`../ceiling/CEILING.md`).

## Files

- `crates/memra-engine/cu/dsv4_gpu.cu`: `dsv4_c4_split_gather_kernel`,
  `dsv4_c4_split_store_kernel`, and the split store in `memra_dsv4_replay_compressor_emit`.
- `crates/memra-engine/src/dsv4_c4.rs`: `split_gather`.
- `crates/memra-engine/src/dsv4_gpu.rs`: `C4Split`, the byte plans, allocation, attention
  gathers and staging, commit scatter, snapshot/restore, and the digests.
- `crates/memra-engine/src/bin/dsv4_kv_split_gate.rs`: the identity gate.
