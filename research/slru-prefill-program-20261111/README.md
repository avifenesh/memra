# One prefill program for both expert placements (memra#1217)

RTX 5090 Laptop GPU, Ornith-1.5-35B-A3B NVFP4+Q5_K MTP GGUF, prompt `board-2048` (2,048
tokens). Main is ede1b8119f. The fix is this branch. One card, one lease per cell, binaries'
sha256 in `binaries.sha256`. Decision record:
`docs/decisions/MOE-PREFILL-PLACEMENT-INDEPENDENT.md`.

## The defect

Ornith's experts stay on the card (the resident slab) only when free memory at load covers the
experts, the trunk and a 2.0e9-byte reserve. Otherwise the SLRU cache serves them, and
`MEMRA_MOE_RESIDENT=0` forces that placement. #1215's review found that the two placements give
different prefill logits (up to 0.56 at the argmax window) and different greedy served text
(`research/argmax-stored-rows-20261110/expert-placement/`).

## Where the bytes first differ (`localize/`)

Main plus a scratch diagnostic (`tools/diag_patch.py`, not in the branch). It tags which MoE arm
each prefill layer call took and writes layer 0 and 1's MoE input and output bytes. Argmax
window, eager:

| cell | prefill MoE arm | window table | layer 0/1 MoE bytes vs D |
|---|---|---|---|
| A resident (default) | pairs, grouped f16 lane | `0ef88e821faf6264` (main's stored rows) | input equal, output differs |
| B resident, `MEMRA_MOE_F16G=0` | pairs, q8 arm | `85ed6f4403172db6` | equal |
| C resident, `MEMRA_MOE_PAIRS=0` | `moe_ffn_dev` via the cache's layer row | `85ed6f4403172db6` | equal |
| D SLRU (`MEMRA_MOE_RESIDENT=0`) | `moe_ffn_dev` via the cache's layer row | `85ed6f4403172db6` | (reference) |

- On an idle card the SLRU prewarm makes every layer fully cache resident. SLRU prefill then took
  `moe_ffn_dev`, the per-token decode program, serially over all 2,048 tokens.
- A against D at layer 0: the MoE input is byte-equal, and the output differs in 4,171,757 of
  4,171,776 floats (max abs 5.6e-4). `compare.txt` has every t; `dump-sha256.txt` hashes the dumps.
- The only differing op is the grouped f16 lane (`cu/moe_f16_grouped.cu`, the sk visitor with
  direct NVFP4 tiles). It uses f16 activations with a per-row scale and f16 tensor-core MMA with
  f32 accumulation. The per-token and pairs-q8 programs use q8_1 activations and dp4a with the
  same per-pair dot order. Those two are byte-identical to each other (B, C and D agree), and the
  decode columns of all four cells are equal.

## The fix

The resident numerics (the pairs arm, the grouped f16 lane on sm_120a) are the reference for
both placements. The kernels read experts through a `[3 * n_expert]` pointer table. On the SLRU
placement that table points at cache slots instead of the slab (`moe_pairs_y_down_slru`). The
cache lock is held for admission, the table upload and the kernel launches together. The table
comes from:

- `LayerRow`: the cache's existing full-layer row, when all of the layer's blocks are resident;
- `Pinned`: `MoeSlotCache::acquire_pinned` admits the call's active blocks, and eviction skips the
  set's own members while it does;
- `Chunked` (grouped f16 only): contiguous CSR runs of experts that fit the pinnable capacity, one
  pinned admission and one grouped-f16 chain per run, concatenated in CSR order.

## Results

**Placement GPU test** (`crates/memra-engine/tests/moe_slru_pairs_placement_gpu.rs`,
`placement-test/`). A two-layer qwen35_moe fixture (softmax router, NVFP4 experts, 32 experts,
top 2) runs prefill at T = 1, 8, 17, 20, 65, 130 and 1,100. The 1,100-token prefill puts more than
64 rows in every expert group, which selects the 128-row sk tile. It also runs two sessions on one
engine: A primes 40 tokens and decodes 6, and B primes 97 tokens between A's decode steps, then
both decode. Rows are compared bit for bit against the resident arm. The final head (`green.log`
for the grouped f16 lane, `green-q8.log` for the same binary under `MEMRA_MOE_F16G=0`):

| arm | slots | f16 lane: calls (row / pinned / chunked), rows differing | q8: calls, rows differing |
|---|---|---|---|
| SlruRoomy | auto (2,155,365) | 14 / 0 / 0, 0 of 19 | 14 / 0 / 0, 0 of 19 |
| SlruMid | 120 (one layer fits) | 0 / 14 / 0, 0 of 19 | 0 / 14 / 0, 0 of 19 |
| SlruTight | 72 (< 96 per layer) | 0 / 6 / 8, 0 of 19 | 0 / 0 / 0 (per-token), 0 of 19 |
| SlruFloor | 8 | 0 / 0 / 14, 0 of 19 | 0 / 0 / 0 (per-token), 0 of 19 |

The red run on main used the first version of the test (prefill widths up to 130, no mid arm,
18 rows). It had 16 of 18 rows differing in each of the roomy, tight and floor arms
(`red-main.log`). On main, only the two prefills of 16 tokens or fewer match. Every row after the first wider
prefill differs, the decode rows included, because their KV came from the other prefill program.
The red run is main's build with the test's mode counter stubbed to zero.

**Argmax window** (`window/`, `col-shas.txt` hashes the prefill and decode columns separately):

| build | placement | eager table | batch table | SLRU pairs mode |
|---|---|---|---|---|
| main | resident | `0ef88e821faf6264` | `4b0fce7aab35a352` | - |
| main | SLRU | `85ed6f4403172db6` | `ba7e9bd85ce07f39` | - |
| main | SLRU, 500 slots | `85ed6f4403172db6` | | - |
| fix | resident | `0ef88e821faf6264` | `4b0fce7aab35a352` | - |
| fix | SLRU | `0ef88e821faf6264` | `4b0fce7aab35a352` | LayerRow |
| fix | SLRU, 500 slots | `0ef88e821faf6264` | `4b0fce7aab35a352` | Chunked (2 runs per call) |
| fix | SLRU, 2,000 slots | `0ef88e821faf6264` | | Pinned |

The decode columns are the same in every cell of a decode mode: eager `e4d06b87b36b68b8`, batch
`dc415ce6998cc65c`. Decode of 16 rows or fewer keeps main's program on both placements. A decode
wave of 17 to 32 rows (the worker's exact-32 tier) is wider than 16 rows, so on the SLRU placement
it now takes the pairs arm too: the program it already ran on the resident placement. From
reading the code, not measured here: on main that wave was placement-dependent as well.

**Served greedy request** (`served/`, `summary.md`): `/v1/completions`, board-2048, temperature 0,
128 tokens, the default route (MTP speculative decode). Server-reported elapsed time; main and fix
interleaved M F F M M F per placement, N=3:

| placement | main text | fix text | main s (median, N=3) | fix s (median, N=3) |
|---|---|---|---|---|
| resident | `1408f53ea2349977` | `1408f53ea2349977` | 1.718 (N=1) | 1.717 (N=1) |
| SLRU, idle card | `a5f81b0cc03207a1` | `1408f53ea2349977` | 6.533 | 2.202 |
| SLRU, 4,000 slots (spill) | `a5f81b0cc03207a1` | `1408f53ea2349977` | 11.167 | 4.966 |

The fix's text is the same in all three placements, and it is the resident text of the #1215
receipt. Ranges: idle main 6.527-6.538, fix 2.201-2.205; spill main 11.162-11.199, fix
4.958-4.967, so the arms do not overlap. The SLRU remainder over resident (2.20 against 1.72 s) is
the decode side, which this change does not touch: the SLRU placement has no captured draft graph.

**Peak device memory** (nvidia-smi `memory.used`, every 250 ms over boot and request): idle SLRU
21,276 MiB for main and fix; spill 6,332 MiB for both. The pairs workspaces fit inside what the
process already holds. Resident: main 21,884, fix 21,820.

**Two sessions** (`served/itl.md`, spill placement, 4,000 slots). Session A streams 384 greedy
tokens from a short prompt. When A has 48 tokens, session B sends the board-2048 request. The
worker runs B's prime as one step, so A gets no token while it runs. A's longest gap is B's whole
prefill: main 7,976 and 7,977 ms, fix 1,854 and 1,856 ms. A's gaps before B: p50 55 ms, p99 78 ms
in every cell. The cache lock is not what A waits on. The traced run (`fix-itl-trace`,
`MEMRA_MOE_SLRU_PAIRS_TRACE=1`) holds the lock for 1.08 ms per layer call at p50, 1.68 ms at p99
and 18.3 ms at most (the first call), over 480 calls, against B's 1.83 s prefill (480 calls, the boot's spec-cost calibration prefill included). About 555 misses per call. The hold covers
the admission (each miss's H2D is queued on the compute stream; no host wait), the 6 KB table
upload and the kernel launches. It contains no stream synchronize. The window probe's traced
chunked cell shows p50 1.14 ms and p99 7.6 ms per call at 500 slots.

**The final head** (after the review fixes) on Ornith, same card (`window/final/`,
`served-final/`): the SLRU eager table is `0ef88e821faf6264` on the default cache (LayerRow) and
at 500 slots (Chunked). The SLRU served pair returns `1408f53ea2349977` in 2.213 s, peak 21,276 MiB.
The earlier cells ran this branch before the review fixes. Those fixes change only the
observer hooks, the routing call for pruned models (identical for unpruned ones) and the
chunking guard for non-default `MEMRA_MOE_F16G` modes.

## Independent review

A clean-context review (Opus) found no blocker. Its findings and what was done:

- The pairs arm did not feed the host observers (`MEMRA_MOE_TRACE`, `MEMRA_MOE_WEIGHT_TRACE`,
  `MEMRA_MOE_INPUT_TRACE_DIR`, the MoESD capture, the hidden trace). That was already true on the
  resident placement, and this change extended it to SLRU prefill. Fixed: the arm now calls them
  after routing. They are no-ops unless armed and read only the selection.
- The pairs arm routed without the pruned-expert mask. Fixed: it routes with
  `moe_route_cfg(.., m.active_experts)`, which is `moe_route` when nothing is pruned.
- The chunked-run argument holds for the sk visitor only. Fixed: chunking requires
  `MEMRA_MOE_F16G` >= 2 with both GEMM shapes `% 32 == 0` and the sk form not rolled back.
  Otherwise the grouped f16 layer needs its whole active set pinned, like MMA and q8.
- Not fixed (nit): for the MMA and q8 sub-arms, whether a borderline cache fits can change with
  the number of in-flight prefetch slots. That is non-default on sm_120a and documented in the
  decision record.
- Doc scope (the exact-32 decode tier), stale test names and a garbled error string: fixed.
- The test now adds a mid arm (120 slots, pinned), a 1,100-token prefill (128-row tile form
  pinned and chunked), and a `MEMRA_MOE_F16G=0` run of the same binary (`placement-test/green-q8.log`)
  that exercises the q8 sub-arm on pinned slots and the per-token fallback.

## Scope and what is not covered

- The softmax pairs arm only (qwen35_moe family). `gemma4_moe` (memra#1225) and glm5_next's
  grouped prefill (memra#1226) have slab-only prefill arms of the same class and are filed.
- Kept on the per-token program, by design: `MEMRA_MOE_CACHE=0`, a frozen cache (the Hy3
  CPU-expert door), and the MMA or q8 pairs sub-arms when the cache cannot pin a layer's whole
  active set. The q8 sub-arm is byte-identical to the per-token program for NVFP4 (cells B, C and
  D above). On sm_120a every f16g-admitted qtype takes the grouped f16 lane.
- One device (RTX 5090 Laptop GPU) and one model for the served and window cells. The GB10 build
  shares the code path, but its experts fit unified memory and its cells were not run here.

## Reproduce

`tools/cells.sh <probe> <out> name:ENV=V,...` (one lease per cell). `tools/serve-driver.sh` runs
`tools/serve_cell.py` per served cell. `tools/served_summary.py` and `tools/itl_summary.py`
summarize them. `tools/col_shas.py` hashes the window columns. `tools/run_test.sh` runs the
placement test.
