# One MoE program for both expert placements, gemma4_moe (memra#1225)

RTX 5090 Laptop GPU (24 GB), gemma-4-26B-A4B QAT Q4_0 GGUF (`gemma-4-26B_q4_0-it.gguf`, 128
experts, top 8, expert ff 704), prompt `board-2048` (2,103 tokens after the template). Base is
memra#1229's head 8855a38bed, whose gemma code equals main ede1b8119f; the steps below ran the
fix on that base, and the final cells (`window/final*`, `served/final/`) ran the fix rebased on
the landing that carries memra#1229 (b8ba35eb3a) against that landing's own build. One card, one
lease per cell, single runs unless an N is given. Binaries' sha256 in `binaries.sha256`.
Decision record: `docs/decisions/GEMMA-MOE-PLACEMENT-INDEPENDENT.md`.

## The defect

gemma-4-26B-A4B's experts stay on the card (the resident slab) when free memory at load covers
them, the trunk and a 2.0e9-byte reserve: 12.85 + 1.58 + 2.0 GB against about 24.25 GB free on
an idle card. Otherwise, or with `MEMRA_MOE_RESIDENT=0`, base had no device source for them:
`gemma4_moe` has no SLRU path, so every MoE call, prefill AND decode, ran a host-staged loop
(one synchronous H2D per token, expert and projection, then the f32 `qmatvec_f32` chain). The
resident program is the q8_1/dp4a decode and verify-rows kernels, the int8-MMA pairs arm for
prefill, and the layer tail's q8z fusion, all keyed on the slab.

**Argmax window, base** (`window/base/`, `tools/col_shas.py` hashes the prefill and decode
columns separately):

| placement | decode mode | prefill column | decode column | wall s |
|---|---|---|---|---|
| resident | eager | `7a5b4417973d9595` | `9e07f782cbfa39de` | 44 |
| staged (`MEMRA_MOE_RESIDENT=0`) | eager | `5edd0c6a1d481500` | `d97f66781c1a189f` | 1,539 |
| resident | batch | `7a5b4417973d9595` | `5bf867d59cee956d` | 49 |
| staged | batch | `5edd0c6a1d481500` | `57e9e40575e85e78` | 1,437 |

Both columns move, unlike Ornith in memra#1217, whose decode was already one program: the
prefill top-1 flips at row 2096 (27612 to 5140) and the decode top-1 at row 2091 (1005 to
236751); margins move by up to about 0.6.

**Served text, base** (`served/step3/`): the `board-2048` completion (128 tokens) and the same
prompt through the chat template (78 tokens) return the same text on both placements
(`3eb1a6598ed6bf8b`, `442ccca2203294b0`), 115.7 s against 1.87 s end to end. A 256-token greedy
decode from a short prompt does not: resident `9ad77520768940e8`, staged `ceefd021a59bc570`,
split at the first generated token. On base the staged board-2048 request also misses the
server's default 90 s deadline (`served/step1-base/base-slru/`, a `deadline_exceeded` error);
every cell after it raises the ceiling with the `MEMRA_TIMEOUT_MS_MAX` measurement override.

## The fix

The resident program is the reference for both placements. The source is decided once per boot
(`gemma_experts`): the slab, else the SLRU arm, else the staged loop for every call. The SLRU arm
needs the cache to pin the most experts a call below the prefill width can select,
`min(n_expert, 15 * top_k)` = 120 experts, 360 slots. Under one hold of the cache lock that also
covers the launches, its pointer table is the cache's full-layer row (`LayerRow`), one pinned
admission of the call's experts (`Pinned`, memra#1217's `acquire_pinned`; a small-t call reads
its device top-k back once to name them), or, for prefill, contiguous CSR runs of experts that
fit (`Chunked`). Same kernels, same bytes.

The int8-MMA down GEMM reads 36 B past an expert's last row (its ragged k = 704 walked to whole
256-value superblocks) and decodes a block scale there that meets a zero activation. Gemma's cache
slots carry a 144 B tail, zeroed and never written. The resident slab's 144 B tail was stale pool
memory on main (a latent NaN if it ever held an f16 Inf or NaN); `htod_bytes_padded` now zeroes
it. The decision record has the audit of the other expert sources.

## Results

**Placement GPU test** (`crates/memra-engine/tests/moe_slru_gemma_placement_gpu.rs`,
`placement-test/`). A two-layer gemma4_moe fixture with the 26B's expert class: Q4_0 experts,
128 experts, top 8, ragged down k 704, a 512-id vocabulary and a spread router so wide prefills
reach most experts. Forward at T = 1, 8, 15, 16, 40, 130, 600 and 1,000, a prime and decode
steps, and two interleaved sessions: 20 rows, compared bit for bit against the resident arm.

| arm | slots | int8 MMA: calls row / pinned / chunked, rows differing | dp4a (`MEMRA_GEMMA_MOE_MMA=0`): same | base |
|---|---|---|---|---|
| SlruRoomy | auto (194,478) | 40 / 0 / 0, 0 of 20 | 40 / 0 / 0, 0 of 20 | 20 of 20 differ |
| SlruMid | 400 (one layer fits) | 1 / 39 / 0, 0 of 20 | 1 / 39 / 0, 0 of 20 | 20 of 20 differ |
| SlruTight | 360 (the floor) | 0 / 38 / 2 (2 runs each), 0 of 20 | 0 / 38 / 2, 0 of 20 | 20 of 20 differ |
| SlruBelow | 300 | staged for the whole boot (asserted), 20 of 20 differ by design | same | 20 of 20 differ |

`red-main.log` is base with the first version of the test (19 rows, 7 widths); `red.log` is the
final test on base. `green.log` and `green-q8.log` are the final test on the fix before the
review fixes, traced (`MEMRA_MOE_SLRU_PAIRS_TRACE=1`): the chunked calls are the T = 600 and
1,000 prefills of layer 1, 121 and 125 active experts against 120 pinnable. `green-rebased.log`
and `green-q8-rebased.log` are the final head, rebased: the same modes, 0 of 20 rows differing.

**Argmax window, fix** (`window/fix/`). Every cell's table equals the resident one: eager
`9b6a2e863d86864b` (prefill `7a5b4417973d9595`, decode `9e07f782cbfa39de`), batch
`758c421eaf69d8e1` (decode `5bf867d59cee956d`). The resident columns equal base's.

| cell | slots | modes (layer calls, `window/trace-summary.txt`) | lock hold per call p50 / p99 | wall s |
|---|---|---|---|---|
| res-e / res-b | - | - | - | 40 / 38 |
| slru-e / slru-b | auto | LayerRow 63,480 | 6 / 8 us | 36 / 37 |
| slru2000-e | 2,000 | LayerRow 10, Pinned 63,470 | 596 / 1,202 us | 77 |
| slru400-e / -b | 400 | LayerRow 1, Pinned 63,479 | 1,215 / 1,293 us | 121 / 116 |
| slru360-e / -b | 360 | Chunked 245 (2 runs), Pinned 63,235 | 1,215 / 1,294 us | 128 / 127 |

The traced probe logs are committed without their per-call trace lines (about 63k each); the
summary keeps every cell's mode counts and hold percentiles.

**Served** (`served/step3/`, `tools/served_summary.py`). Streamed `/v1/completions` and
`/v1/chat/completions`, temperature 0. Base and fix interleaved on the resident placement, N=3
each; the fix's SLRU cells N=3; the staged cells N=1 (minutes each).

| placement | build | board-2048 text / chat | TTFT s | end to end s | chat end to end s | peak MiB |
|---|---|---|---|---|---|---|
| resident | base | `3eb1a659` / `442ccca2` | 0.552 | 1.868 | 1.981 | 17,168 |
| resident | fix | `3eb1a659` / `442ccca2` | 0.550 | 1.865 | 1.982 | 17,104 |
| staged | base | `3eb1a659` / `442ccca2` | 108.373 | 115.683 | 116.675 | 4,784 |
| SLRU, auto cache | fix | `3eb1a659` / `442ccca2` | 1.057 | 2.370 | 1.980 | 20,560 |
| SLRU, 2,000 slots | fix | `3eb1a659` / `442ccca2` | 0.997 | 3.332 | 5.403 | 7,024 |

**Decode rate** (short prompt, 256 greedy tokens, rate over tokens 2..256):

| placement | build | text | tok/s median (range) | peak MiB |
|---|---|---|---|---|
| resident | base | `9ad77520` | 113.62 (113.58-114.14) | 16,240 |
| resident | fix | `9ad77520` | 113.71 (113.65-114.05) | 16,240 |
| staged | base | `ceefd021` | 17.94 (N=1) | 3,984 |
| SLRU, auto cache (every layer resident in the cache) | fix | `9ad77520` | 113.69 (113.53-113.70) | 20,048 |
| SLRU, 6,000 slots (about half the experts) | fix | `9ad77520` | 96.57 (96.28-96.95) | 10,352 |
| SLRU, 2,000 slots (spill) | fix | `9ad77520` | 56.60 (56.49-56.67) | 6,096 |

What spill costs: a layer that is not fully cache resident reads its top-k back (one stream sync
per layer per token) and admits its misses; at 2,000 slots that halves the decode rate against a
cache that holds every layer, and is still 3.2x the staged program. The resident placement's
decode and prefill are unchanged.

**Memory.** The SLRU arm builds the expert cache the qwen-class SLRU placement already builds
(85% of free memory by default, `MEMRA_MOE_VRAM_FRAC`), so its peak is the cache's, not the
staged loop's three scratch blocks: 20,560 MiB on the idle card, 7,024 MiB at 2,000 slots.

**Final head, rebased** (`window/final/`, `window/final-main/`, `served/final/`, N=1 each; the
rig was shared, so the N=3 timing above is the pre-rebase fix's). The landing's own build gives
the same resident tables as base (eager `9b6a2e86`, batch `758c421e`). The rebased fix gives that
table on the resident placement, on SLRU at the auto cache (LayerRow, eager and batch) and at
360 slots (Pinned, and Chunked in 245 two-run calls). Served: board-2048 `3eb1a659`, chat
`442ccca2` and decode `9ad77520` on every placement; main resident 1.896 s and 112.7 tok/s, fix
resident 1.900 s and 112.5, SLRU 2.448 s and 112.6, 2,000 slots 3.359 s and 56.3. The first
final driver was stopped to yield the rig and restarted with this reduced set, so the cells
file shows an interrupted `res-b`.

## Scope and what is not covered

- gemma4_moe on one device (RTX 5090 Laptop GPU) and one checkpoint. Other gemma4_moe
  quantizations take the same path when their banks are dp4a qtypes with a uniform layout.
- Kept on the staged program, for the whole boot, by design: `MEMRA_MOE_CACHE=0`, a frozen
  cache, a mixed-layout bank, and a cache below the 360-slot floor. The `MEMRA_GEMMA_MOE_FAST=0`
  and `MEMRA_GEMMA_MOE_PAIRS=0` rollbacks apply to both placements alike.
- The grouped f16 lane is env-explicit for gemma (`moe_f16g_gemma_on`); it runs on the SLRU arm
  by the same rules as memra#1217 (runs only where `moe_f16g_runs_exact`, else the floor is the
  whole layer) and was not run here.
- The gemma graph decode programs (bench and gate binaries) and the `MEMRA_GEMMA_ROUND_GRAPH=1`
  round capture refuse a non-resident model; the round falls back to the eager round.

## Reproduce

`tools/cells.sh <probe> <out> name:ENV=V,...` (one lease per cell), `tools/serve_cell.py` per
served cell (driven by `tools/step1-driver.sh`, `tools/staged-driver.sh`,
`tools/step3-driver.sh`), `tools/run_test.sh` for the placement test, `tools/col_shas.py` and
`tools/served_summary.py` to summarize, `tools/assemble.py` to copy the receipts here.

`binaries.sha256`: `-base` is 8855a38bed, `-main` the landing b8ba35eb3a, `-fix0` the fix the
step-3 window and served cells ran (before the review fixes), `-fix1` the review-fixed fix before
the rebase (its cells are not used here), `-fix` the final head rebased. The placement test's
`-red` is the final test on 8855a38bed and `-fix` the final test on the final head; `green.log`
and `green-q8.log` ran an earlier build of the same test source (sha256 4b8ce294...).
