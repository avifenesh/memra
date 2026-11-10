# Verify rows: a speculative verify reads the KV once (GB10 decode round 5, lever 2)

Verdict: shipped. `fa_verify_rows_v4_deep` is bit-identical to the seqs launch it replaces for a verify's rows, on every case of kernel-check's `FA-VERIFY-ROWS` cell on GB10 (both units) and the RTX 5090. On GB10 a speculative round at 131k context goes from 192.3 to 153.5 ms (-20.2%) and at 260k from 294.9 to 221.0 ms (-25.1%); on the RTX 5090 at 28k from 43.3 to 40.2 ms (-7.1%). The served texts are identical and TTFT does not move. The launch is routed by depth: below a measured floor per GQA shape the seqs launch runs, so short contexts and pooled cohorts of short chats are unchanged.

Devices: DGX Spark (GB10, sm_121a), two units, and an RTX 5090 Laptop GPU (sm_120a). Model: Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head, 24 heads over 4 KV heads) with the masked MTP draft, served as the launcher serves it (MTP speculative decode, K=3, so T = 4 verify rows). Base of every B arm: main 53eb765421. Date: 2026-11-10.

The design note this follows is `research/gb10-decode-20261109/MULTIROW-ATTN-DESIGN.md` (written before the code). Its estimate was -55 to -70 ms a round at 131k; the measured -38.8 ms is explained in "What the estimate missed".

## The change

A speculative verify attends T = K+1 query rows against one session's cache. The rows went through `fa_decode_vec_q_seqs_v4_deep`, the batched tick's kernel, with one grid z per row. Each z walked the whole cache, so a round read the KV T times: 15.6 GB a round at 131k and K=3 for 3.89 GB of unique KV.

`fa_verify_rows_v4_deep` runs one block per (KV head, split, row group):

- A row group is a run of consecutive rows of one cache that share the split partition the one-partition law derives from each row's own key bound. Rows of a group share every split's start and differ only in the last split's end. A group holds at most 32 / GQA rows (a block is 32 warps): 5 at the 27B's 24/4. The host forms the groups (`fa_row_groups`) from the positions the kernel reads.
- The block stages each 32-key tile once, with the seqs walk's own staging code, up to the group's bound.
- Every (row, head) warp then runs the seqs walk's score, softmax update and accumulate over the staged tile with its own bound, and writes its row's slot of the seqs partial layout. The seqs combine runs unchanged.

Routes: the eager qwen35 verify's batched-rows arm and the pooled verify (rows of several sessions; a group never spans two sessions). A captured verify graph keeps the seqs launch, because a round's groups depend on its position and a replay cannot rebake them. There is no flag.

## Bit identity

kernel-check `FA-VERIFY-ROWS` compares the new launch with the seqs launch on the same Q, cache bytes and positions: the partial buffers (O, M, L, every word of the launch, from two different pool fills, so an unwritten slot shows) and the combined rows.

| device | GQA shapes | cases per shape | launched the rows walk | bitdiff |
|---|---|---:|---:|---:|
| GB10 unit 2, RTX 5090 (the final head) | 24/4, 16/2, 16/4 (to 260k) | 1,111 | 960 | 0 |
| GB10 unit 2, RTX 5090 (the final head) | 4/4, 8/4, 20/4, 28/4 (to 32k) | 1,079 | 928 | 0 |
| GB10 unit 1 (99048c1afa: the final kernel, an earlier floor for GQA 5 and 6) | the same seven | as above | as above | 0 |

- **Cases:** T = 1..9 rows of one cache at depths inside one tile, on tile edges inside one partition class (a row whose bound ends where a tile starts while its group stages more keys), across changes of the partition (two classes in one round), with a group cut by the 32-warp limit, at 4k, 64k, 131k and 260k; two sessions' rows in one launch.
- **The count that matters is the second one.** A case whose groups are all one row is the seqs launch on both sides (every T=1 case and a few shallow ones). The cell counts the cases whose rows walk really launched and fails if none did. These cases launch the walk at every depth (`fa_verify_rows_v4_unrouted`), below the served floors too.
- **The route check:** the served entry on each side of its depth floor. Below it the seqs launch runs; at it the rows walk launches; both equal the seqs launch bit for bit (`FA-VERIFY-ROWS route` lines, every shape, every device).
- kernel-check's synthetic sections are all green on the three devices (`kernel-check/`).
- **After the rebase on main fcd66f8963** (which brought levers 1 and 3a): the cell again on GB10 unit 2 and the RTX 5090, the same counts, bitdiff 0, every route line OK (`kernel-check/*-rebased.txt`).
- `run-spec` K=1..8 is identical to plain on both roster models on the RTX 5090 (`rows28-5090/run-spec-*.txt`, at the lever's first commit).

## Per attention call

kernel-check's timing rows (`MEMRA_KC_ROWS_TIME=1`): one full-attention layer's launch for T = 4 rows of one session, the 27B's 24/4 shape, N = 24 per arm, interleaved and synchronized. Medians in ms, at the final head. The RTX 5090 run started at 62 C and peaked at 75 C with no thermal slowdown sample (`kernel-check/rtx5090-gpu-1hz.csv`).

| depth (keys) | GB10 seqs | GB10 rows | GB10 rows/seqs | RTX 5090 seqs | RTX 5090 rows | RTX 5090 rows/seqs |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.048 | 0.050 | 1.04 | 0.033 | 0.026 | 0.80 |
| 2,048 | 0.079 | 0.074 | 0.93 | 0.050 | 0.043 | 0.87 |
| 4,096 | 0.149 | 0.134 | 0.90 | 0.085 | 0.079 | 0.92 |
| 8,192 | 0.348 | 0.281 | 0.81 | 0.161 | 0.136 | 0.85 |
| 12,288 | 0.681 | 0.471 | 0.69 | 0.257 | 0.208 | 0.81 |
| 16,000 | 0.956 | 0.644 | 0.67 | 0.341 | 0.273 | 0.80 |
| 17,000 | 0.794 | 0.500 | 0.63 | 0.335 | 0.259 | 0.77 |
| 24,576 | 1.173 | 0.713 | 0.61 | 0.521 | 0.380 | 0.73 |
| 32,768 | 1.569 | 0.970 | 0.62 | 0.747 | 0.537 | 0.72 |
| 131,072 | 6.335 | 3.659 | 0.58 | 2.971 | 2.282 | 0.77 |
| 262,132 | 12.684 | 7.243 | 0.57 | 5.916 | 4.513 | 0.76 |

A round has 16 such calls on the 27B. On GB10 the call falls 42% at 131k and 43% at 260k; on the RTX 5090 23% and 24%.

## Routing

The two launches are bit-identical per row, so choosing between them changes no value. The rows walk shares one staged tile among a group's rows and pays for it with a wider block (rows x GQA warps). It loses where a split holds 8 keys and wins with depth, later for a wider block.

The grid, rows walk over seqs launch per call, by GQA shape and launch shape. The mark after a ratio: `<` rows faster with the two arms' ranges apart, `>` seqs faster, `~` ranges overlap. A pooled launch is routed row by row; its "routed to" column is the shallowest session's.

| GQA | launch | depth (keys) | split keys | GB10 | RTX 5090 | routed to |
|---:|---|---:|---:|---:|---:|---|
| 4 | 1 session x 4 rows | 1,024 | 64 | 0.84 < | 0.72 < | rows walk |
| 4 | 1 session x 4 rows | 2,048 | 64 | 0.71 < | 0.71 < | rows walk |
| 4 | 1 session x 4 rows | 4,096 | 64 | 0.69 < | 0.74 < | rows walk |
| 4 | 1 session x 4 rows | 8,192 | 64 | 0.55 < | 0.67 < | rows walk |
| 4 | 1 session x 4 rows | 12,288 | 64 | 0.45 < | 0.65 < | rows walk |
| 4 | 1 session x 4 rows | 16,000 | 64 | 0.39 < | 0.64 < | rows walk |
| 4 | 1 session x 4 rows | 17,000 | 128 | 0.42 < | 0.63 < | rows walk |
| 4 | 1 session x 4 rows | 24,576 | 128 | 0.41 < | 0.62 < | rows walk |
| 4 | 1 session x 4 rows | 32,768 | 128 | 0.43 < | 0.56 < | rows walk |
| 4 | 1 session x 4 rows | 131,072 | 128 | 0.45 < | 0.55 < | rows walk |
| 4 | 1 session x 4 rows | 262,132 | 128 | 0.45 < | 0.55 < | rows walk |
| 4 | 2 sessions x 4 rows | 300 | 8 | 0.88 < | 0.91 ~ | seqs |
| 4 | 4 sessions x 4 rows | 300 | 8 | 0.94 < | 0.89 < | seqs |
| 4 | 4 sessions x 4 rows | 1,000 | 64 | 0.63 < | 0.68 < | seqs |
| 4 | 4 sessions x 4 rows | 2,000 | 64 | 0.63 < | 0.64 < | rows walk |
| 4 | 4 sessions x 4 rows | 4,000 | 64 | 0.58 < | 0.61 < | rows walk |
| 4 | 4 sessions x 4 rows | 8,000 | 64 | 0.56 < | 0.61 < | rows walk |
| 4 | 4 sessions x 4 rows | 20,000 | 128 | 0.44 < | 0.58 < | rows walk |
| 6 | 1 session x 4 rows | 1,024 | 64 | 1.04 ~ | 0.80 ~ | seqs |
| 6 | 1 session x 4 rows | 2,048 | 64 | 0.93 < | 0.87 < | rows walk |
| 6 | 1 session x 4 rows | 4,096 | 64 | 0.90 ~ | 0.92 ~ | rows walk |
| 6 | 1 session x 4 rows | 8,192 | 64 | 0.81 ~ | 0.85 < | rows walk |
| 6 | 1 session x 4 rows | 12,288 | 64 | 0.69 < | 0.81 < | rows walk |
| 6 | 1 session x 4 rows | 16,000 | 64 | 0.67 < | 0.80 < | rows walk |
| 6 | 1 session x 4 rows | 17,000 | 128 | 0.63 < | 0.77 < | rows walk |
| 6 | 1 session x 4 rows | 24,576 | 128 | 0.61 < | 0.73 < | rows walk |
| 6 | 1 session x 4 rows | 32,768 | 128 | 0.62 < | 0.72 < | rows walk |
| 6 | 1 session x 4 rows | 131,072 | 128 | 0.58 < | 0.77 < | rows walk |
| 6 | 1 session x 4 rows | 262,132 | 128 | 0.57 < | 0.76 < | rows walk |
| 6 | 2 sessions x 4 rows | 300 | 8 | 1.16 > | 1.16 > | seqs |
| 6 | 4 sessions x 4 rows | 300 | 8 | 1.31 > | 1.21 > | seqs |
| 6 | 4 sessions x 4 rows | 1,000 | 64 | 0.85 < | 0.87 < | seqs |
| 6 | 4 sessions x 4 rows | 2,000 | 64 | 0.91 < | 0.88 < | seqs |
| 6 | 4 sessions x 4 rows | 4,000 | 64 | 0.87 ~ | 0.81 < | rows walk |
| 6 | 4 sessions x 4 rows | 8,000 | 64 | 0.89 ~ | 0.84 < | rows walk |
| 6 | 4 sessions x 4 rows | 20,000 | 128 | 0.60 < | 0.78 < | rows walk |
| 8 | 1 session x 4 rows | 1,024 | 64 | 1.16 ~ | 1.32 > | seqs |
| 8 | 1 session x 4 rows | 2,048 | 64 | 1.14 > | 0.85 < | seqs |
| 8 | 1 session x 4 rows | 4,096 | 64 | 1.01 ~ | 0.93 ~ | seqs |
| 8 | 1 session x 4 rows | 8,192 | 64 | 0.97 ~ | 0.98 ~ | seqs |
| 8 | 1 session x 4 rows | 12,288 | 64 | 0.86 ~ | 0.91 < | seqs |
| 8 | 1 session x 4 rows | 16,000 | 64 | 0.87 < | 0.93 ~ | seqs |
| 8 | 1 session x 4 rows | 17,000 | 128 | 0.82 ~ | 0.92 < | rows walk |
| 8 | 1 session x 4 rows | 24,576 | 128 | 0.73 < | 0.83 < | rows walk |
| 8 | 1 session x 4 rows | 32,768 | 128 | 0.70 < | 0.86 < | rows walk |
| 8 | 1 session x 4 rows | 131,072 | 128 | 0.66 < | 0.80 < | rows walk |
| 8 | 1 session x 4 rows | 262,132 | 128 | 0.65 < | 0.81 < | rows walk |
| 8 | 2 sessions x 4 rows | 300 | 8 | 1.33 > | 1.22 > | seqs |
| 8 | 4 sessions x 4 rows | 300 | 8 | 1.35 > | 1.23 > | seqs |
| 8 | 4 sessions x 4 rows | 1,000 | 64 | 0.93 < | 0.90 ~ | seqs |
| 8 | 4 sessions x 4 rows | 2,000 | 64 | 1.00 ~ | 0.94 ~ | seqs |
| 8 | 4 sessions x 4 rows | 4,000 | 64 | 0.95 < | 0.93 < | seqs |
| 8 | 4 sessions x 4 rows | 8,000 | 64 | 0.92 ~ | 0.86 < | seqs |
| 8 | 4 sessions x 4 rows | 20,000 | 128 | 0.81 ~ | 0.82 < | rows walk |

The rule (`fa_rows_floor`, one table for every device): a row joins a group only if its key bound is at least

- **1,024 for GQA up to 4.** The walk is faster at every measured depth from there on both devices.
- **2,048 for GQA 5 and 6** (the 27B). At 1,024 GB10 is flat (1.04, ranges overlap); from 2,048 the median is lower on both devices at every depth.
- **16,384 for GQA 7 and 8** (the 128-key split rung). On the 64-key rung the widest block is within noise of the seqs launch on GB10 (1.16 down to 0.86, ranges mostly overlapping); on the 128-key rung it is lower on both devices.

A row below the floor is a group of one, and a launch whose groups are all one row is the seqs launch itself, with no extra host work. Shapes without a timing row (GQA 1, 2, 5, 7) take their class's floor.

Not measured: an RTX PRO 6000 (188 SMs, its own short split rung). It takes the same table until a probe says otherwise. The probe is one command: `MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 MEMRA_KC_ROWS_TIME=1 kernel-check`, then `tools/tables.py`.

## Served

B = main 53eb765421, T = the lever, one fresh server per cell, B T T B B T, one cell per lease. The signal is the cost of one speculative round inside full bursts (`tools/rounds_by_req.py`); TTFT and the whole request are the no-regression checks. Medians per arm, spread = max - min, N = 3 per arm unless a row says otherwise.

| campaign (the T commit) | request | B ms per round | T ms per round | change | spread B / T | TTFT B / T (s) | total B / T (s) |
|---|---|---:|---:|---:|---:|---|---|
| GB10 unit 2, `rows131-unit2/` (fcdaa08f82) | g131k | 192.28 | 153.46 | -38.81 (-20.2%) | 0.38 / 0.64 | 117.35 / 117.37 | 124.69 / 123.24 |
| | g1k | 88.67 | 88.78 | +0.11 | 0.24 / 0.06 | 1.130 / 1.133 | 5.827 / 5.832 |
| GB10 unit 1, `rows131-unit1/` (31939989cb) | g131k | 190.54 | 155.04 | -35.50 (-18.6%) | 0.86 / 0.79 | 117.77 / 117.82 | 125.05 / 123.76 |
| | g1k | 88.72 | 88.58 | -0.14 | 0.26 / 0.21 | 1.135 / 1.123 | 5.831 / 5.819 |
| GB10 unit 1, `rows131b-unit1/` (99048c1afa), N = 2 | g131k | 190.28 | 153.68 | -36.60 (-19.2%) | 0.22 / 0.68 | 117.78 / 117.79 | 125.05 / 123.68 |
| | g1k | 88.65 | 88.49 | -0.15 | 0.50 / 0.07 | 1.127 / 1.127 | 5.822 / 5.817 |
| GB10 unit 2, `rows260-unit2/` (31939989cb) | g260k | 294.90 | 221.02 | -73.88 (-25.1%) | 0.25 / 1.06 | 312.86 / 312.76 | 329.80 / 325.51 |
| | g1k | 88.69 | 88.69 | 0.00 | 0.17 / 0.16 | 1.131 / 1.130 | 5.827 / 5.830 |
| RTX 5090, `rows28-5090/` (55a92268f6) | g28k | 43.29 | 40.21 | -3.08 (-7.1%) | 1.00 / 0.62 | 25.68 / 25.52 | 28.07 / 27.69 |
| | g1k | 30.58 | 29.47 | -1.11 | 0.55 / 2.30 | 0.815 / 0.816 | 2.542 / 2.514 |

- **Texts:** every request served the same text in every cell of both arms, in every campaign.
- **Separation:** on every deep request every T cell is under every B cell.
- **TTFT does not move:** the lever touches decode only.
- **The commits.** The kernel's walk is the same in all of them. fcdaa08f82 and 55a92268f6 are the first build; 31939989cb added a launch bound that cost about 5% of the call (removed again, see below), which is the difference between the two unit 1 campaigns; 99048c1afa is the final kernel with the route. The first three campaigns ran the rows walk at every depth, so their g1k rows are the walk at 1k (flat); the route now sends that depth to the seqs launch. `rows131b-unit1/` gave up its last two cells to a landing record, hence N = 2.

### Pooled cohorts

A pooled verify of short chats launches at depths under the floors, so it must be unchanged.

- **GB10 unit 2, the 27B, the final head** (`pool-unit2/`; waves of 2, 4 and 8 concurrent requests, greedy and sampled, 400 tokens, B T T B B T, one boot per lease): c2 31.72 (31.67-31.76) against 31.58 (31.51-31.69) tok/s, c4 45.57 (45.57-45.66) against 45.54 (45.46-45.64), c8 47.54 (47.51-47.67) against 47.59 (47.42-47.62). T/B is 0.996, 0.999 and 1.001, and the two arms' ranges overlap in every wave. All 14 streams of every boot are identical to the first B boot's.
- **RTX 5090, Qwen3.5-9B, 99048c1afa** (`pool-5090/routed/`, each boot from a 62 C card with 1 Hz thermal samples): c2 191.9 (184.7-193.6) against 183.8 (182.8-184.5) tok/s, c4 219.9 (213.9-224.2) against 217.4 (214.6-222.0), c8 224.7 (222.4-237.1) against 229.6 (226.5-232.7). All streams identical. The laptop card's own B boots spread 5 to 6%, so this cell cannot resolve less; the GB10 cell above is the parity evidence.
- **What the route fixed.** Before it, the walk ran at every depth and two B/T pairs of the same cell read 2.5 to 9% lower with it (`pool-5090/unrouted/`: B 191/214/207 and 188/221/216 against T 178/196/202 and 182/201/205 tok/s). Per call the pooled shape at depth 300 costs 1.16 to 1.35 times the seqs launch for GQA 6 and 8 (the grid above).

## What the estimate missed

The design note priced the round at the bandwidth floor: 3.89 GB once instead of four times, about 31 ms against 99.65 ms of attention at 131k. The kernel does read the KV once, and the call fell from 6.3 to 3.7 ms a layer (-42%), not to the 2 ms the floor allows. The walk is compute-bound now: a 768-thread block stages one tile and its 24 warps then work on it between two barriers, so the next tile's loads do not overlap this tile's compute the way eight independent 6-warp blocks per SM did. A 2-row block (the KV read twice) is slower than the full group from 32k on both devices.

A launch bound on the kernel (to pin its register budget for a 1,024-thread block) made the compiler spill and cost 5% of the call on GB10. The host cuts a group to the build's own thread limit instead (`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, read once per engine).

Next, not in this change: overlap staging with compute inside the block (a second staged tile, or two half-width blocks per SM over one L2-resident tile), a device-resident group table so a captured verify graph can take the walk, and the same row-group walk for the draft's attention.

## Receipts

- `kernel-check/`: the `FA-VERIFY-ROWS` lines and the summary line per device; the RTX 5090 run's 1 Hz thermal samples.
- `rows131-unit2/`, `rows131-unit1/`, `rows131b-unit1/`, `rows260-unit2/`: GB10 served campaigns (per cell `ab.jsonl`, `guard.log`, `trace.txt`; `rounds-by-request.txt`).
- `rows28-5090/`: the RTX 5090 served campaign and `run-spec`.
- `pool-unit2/`, `pool-5090/`: pooled-cohort runs.
- `tools/`: the cell scripts, drivers and tables. `rows_ab.py` runs next to `research/gb10-decode-20261109/tools/eos_ab.py`; the pooled cells use that directory's `pool_cell.py`. Timed RTX 5090 cells start from a cool card with a 1 Hz thermal sampler (`tools/cool5090.sh`).
