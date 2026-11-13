# The driver-headroom rung reads the allocatable free memory on unified memory (memra#1257)

Date: 2026-11-13. Follows `research/spec-capture-headroom-20261112/` (memra#1221), whose cells found the rung firing on GB10 with about 100 GB available.

**Verdict: adopted.** Admission's driver-headroom rung runs once per admitted request, before its prime. It compared the raw `cuMemGetInfo` reading with its 2,048 MiB floor and trimmed the pool by the shortfall. On GB10 that reading leaves out page cache, so with the page cache full it trimmed on a host that was not short.

The rung now reads `unified_memory::allocatable_mem_info`: on unified memory the larger of the driver's reading and `MemAvailable`, on a discrete card `cuMemGetInfo` as before. With a full page cache, main's rung fired in all 4 cells and the fix's in none of 4. Every request completed in both arms with identical texts and no OOM line, including a 260k cold prime that started at MemFree 1.6 GB with no trim (main's raw reading in the same state was 1.5-1.9 GB). With `MemAvailable` itself under the floor and the prefix cache off, the fix's rung crosses its threshold on the allocatable reading (section 3). That shows the crossing, not protection: it released nothing in either arm, and with the served prefix cache admission refuses first.

## Conditions

- One DGX Spark (GB10, sm_121a), Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head) with the masked MTP draft, 262,144 context, served as memra#1221's `tools/serve.sh` serves it. One cell per lease, one boot per cell.
- Arms, both `memra-server` for sm_121a, differing only in `worker.rs`:
  - B = main d6ae4e525d;
  - T = this change (fix/1257-headroom-unified).
  - Each cell's `guard.log` has the binary's sha256.
- Cells reuse memra#1221's harness (`research/spec-capture-headroom-20261112/tools/`): a fresh server, a page-cache fill once it is ready, then requests greedy to EOS with their own nonces.
  - **full-cache**: g1k, then g260k (259,946 tokens).
  - **cold-first**: g260k first, right after the fill, so the cold prime meets the low reading; then g1k (`tools/specpub_ab_first.py`, `tools/specpub_cell_first.sh`).
- **The fill.** memra#1221's single-reader `fill_cache.sh` left MemFree at 32 GB on this unit after its 120 s cap (cell `01-B`: no rung line in main either; most of its first files were already cached).
  - `tools/fill_cache2.sh` runs four readers and reaches MemFree under 3 GB in under a minute: MemFree 1.6-2.9 GB at the requests, `MemAvailable` about 101 GB.
  - Cells `21-B` and `22-T` use `tools/fill_cache3.sh`, which also keeps one reader going while the server lives and logs MemFree at 1 Hz (`memfree.log`). It did not hold MemFree under 5 GB past the first minutes, so the cold-first cells answer the 260k question instead.
- `trace.txt` holds the server's `[admit-trim]`, `[spec]`, `[device-trim]`, `[admission]`, `[admit-cal]` and OOM lines; `tools/tab.py` and the table below read them.

## 1. Full page cache

| cell | MemFree after the fill (MB) | rung lines | raw reading / released (MiB) | g260k TTFT (s) | g1k TTFT (s) | lowest MemAvailable (MB) |
|---|---|---|---|---|---|---|
| 01-B (one-reader fill) | 31,502 | 0 | - | 313.7 | 1.185 | 81,050 |
| 11-B | 1,912 | 1 | 1,929 / 0, before g1k | 315.7 | 1.180 | 80,953 |
| 12-T | 2,850 | 0 | - | 314.3 | 1.163 | 81,109 |
| 21-B | 1,876 | 1 | 1,916 / 0, before g1k | 317.0 | 1.159 | 81,110 |
| 22-T | 1,820 | 0 | - | 316.6 | 1.179 | 81,106 |

## 2. Cold request first

| cell | MemFree after the fill (MB) | rung lines | raw reading / released (MiB) | g260k TTFT (s) | g1k TTFT (s) | lowest MemAvailable (MB) |
|---|---|---|---|---|---|---|
| 31-B | 2,031 | 2 | 1,830 / 0 before g260k; 1,511 / 512 before g1k | 314.0 | 1.205 | 81,619 |
| 32-T | 1,604 | 0 | - | 313.5 | 1.203 | 81,094 |
| 33-T | 2,869 | 0 | - | 314.5 | 1.205 | 81,693 |
| 34-B | 2,355 | 1 | 1,677 / 352 before g1k | 313.9 | 1.209 | 81,696 |

- Main's rung fired in every cell (5 times), at a raw reading of 1,511-1,929 MiB with `MemAvailable` about 100 GB. It released 0-512 MiB.
- The fix's rung never fired. In `32-T` the 260k prime started right after a fill to MemFree 1,604 MB, so its driver-side allocations (cuBLAS workspaces, graph instantiation and the capture arenas of speculative decoding) ran with MemFree near 1.6 GB, where main's raw reading sat under the floor, and nothing was trimmed. They succeeded: no OOM line, no refused capture, and the same server line classes as main's cells apart from `[admit-trim]`.
- All 9 page-cache cells, 01-B included, answered both requests with the same texts (g260k sha256 `5275df37...`, g1k `26ca98d4...`, 134 and 127 tokens, `stop`). With the two admitted short cells, all 20 completed requests in this record match by hash (9 g260k, 11 g1k); the two refused ones hash as empty. The g260k TTFT spread is 313.5-317.0 s across both arms.

## 3. MemAvailable itself short

memra#1221's truly-short cell (`short_cell.sh`, `mem_holder.py`): a fresh server with its memory guard lowered to 1 GiB, then `mem_holder.py` takes `MemAvailable` under the rung's floor (target 1,800 MB; it holds in 256 MB steps), then one g1k request.

| cell | prefix cache | MemAvailable / MemFree before the request (MB) | admission | rung line | request | kernel OOM lines |
|---|---|---|---|---|---|---|
| 01-T | served default | 1,790 / 670 | refused: `available 0MB`, HTTP 400 | - | refused | 0 |
| 02-B | served default | 2,022 / 819 | refused: `available 0MB`, HTTP 400 | - | refused | 0 |
| 11-T | off (`MEMRA_PREFIX_CACHE_MB=0`) | 2,006 / 827 | admitted | `reading=allocatable`, 1,993 MiB under 2,048, 0 released | 127 tokens, `stop`, same text | 0 |
| 12-B | off | 2,024 / 789 | admitted | raw 765 MiB, 0 released | 127 tokens, `stop`, same text | 0 |

- With the served prefix cache, admission refuses before the rung in both arms. On unified memory it books the cache's allowed growth as committed, and with `MemAvailable` this low nothing is left (`available 0MB`). So at a short arrival, admission is the guard and the rung is not reached.
- With the prefix cache off, admission counts `MemAvailable` plus the pool's cached blocks and admits the request. The fix's rung then crosses its threshold on the allocatable reading (1,993.7 MiB against 2,048), and main's crosses on the raw reading (765 MiB).
- These cells show the threshold crossing, not protection:
  - neither arm's rung released anything (the pool's 149 MB of cached blocks did not trim);
  - `mem_holder.py` gave back 1,024 MB during the g1k request in both 11-T and 12-B (held 96,512 then 95,488 MB, `holder.log`), so the request did not run against the held floor.
  - Both requests completed with the same text and no OOM line, in the server or the kernel. Whether a request that really needs those bytes ends well when nothing gives memory back is not measured here.

## Tools

memra#1221's harness (`research/spec-capture-headroom-20261112/tools/`: `serve.sh`, `serve_short.sh`, `specpub_cell.sh`, `specpub_ab.py`, `short_cell.sh`, `short_req.py`, `mem_holder.py`, `tsrun.py`) with these additions in `tools/`:
- `fill_cache2.sh`: the four-reader fill;
- `fill_cache3.sh`: the same plus a reader that follows the server and a MemFree log;
- `specpub_ab_first.py` and `specpub_cell_first.sh`: the cold-first order;
- `short_cell_noprefix.sh`: the short cell with the prefix cache off;
- `tab.py`: the tables;
- `assemble.py`: copies a cell's receipts here, home paths written as `~`.

## CPU

- `driver_headroom_rung_reads_the_allocatable_free_memory` (memra-server): memra#1221's full-page-cache reading (raw 1,882 MiB, `MemAvailable` 100 GB) trims nothing; a short `MemAvailable` (1,500 MiB) trims the shortfall; the rung's body reads `allocatable_mem_info` and calls no `.mem_get_info()`.
- Red arm: with the raw reading restored in the rung (test kept), the test fails.
- On main d6ae4e525d: server library suite 1,235 passed; engine `unified_memory` and `env_audit` 11 passed; clippy (engine and server, all targets) and fmt green.
