# Three more readers of the free memory on unified memory (memra#1307)

Date: 2026-11-14. Follows `research/admit-headroom-unified-20261113/` (memra#1257),
`research/pooled-verify-memcap-20261113/` (memra#1266) and
`research/spec-capture-headroom-20261112/` (memra#1221). Decision record:
[`docs/decisions/UNIFIED-MEMORY-READERS.md`](../../docs/decisions/UNIFIED-MEMORY-READERS.md).

**Verdict: two adopted, one kept.** On a unified-memory device (GB10) `cuMemGetInfo` free is the
host's `MemFree` and leaves out page cache. Three decisions still read it:

| reader | decision | why |
|---|---|---|
| pooled DFlash2 verify width (`spec.rs` `pooled_verify_headroom`, read in `dflash.rs`) | the allocatable reading (`pooled_verify_reading`: the larger of `cuMemGetInfo` free and `MemAvailable`, plus the pool's cached blocks) | a level, sized like admission's: what one pooled forward may allocate beyond one serial round |
| per-session draft-state measurement (`spec.rs`, the capture bracket) | `MemFree + Buffers + Cached + SReclaimable` at both ends (`unified_memory::delta_free_bytes`); a failed capture's appetite floor keeps the raw entry reading | a delta: the raw reading and `MemAvailable` both move by hundreds of MiB while the page cache fills or is reclaimed, and the result is a high-water that sticks |
| DeepSeek-V4 memory door (`dsv4_gpu.rs` `stage_memory`, `dsv4_serve.rs`) | raw reading kept, documented | the route never loads on unified memory: `Dsv4Gpu::load` takes exactly two CUDA devices and the weights fill about 83 GiB of each 96 GB card |

Discrete cards: `allocatable_mem_info` and `delta_free_bytes` return `cuMemGetInfo` free unchanged,
so every decision there reads the value it read before.

What the cells show (Qwen3.8-27B on two DGX Sparks, main against this branch):

- **Draft-state delta.** Booting under a filling page cache, main recorded a per-session charge of
  55 and 46 MB, the head 34 and 35 MB (a quiet boot: 33 to 34). `MemAvailable` fell 433 MiB across
  one such bracket, which rules it out. The delta reading is not exact: while the kernel reclaims,
  3 of 13 boots that met a full, idle page cache read 2, 18 and 22 MiB for the same capture, on
  every host reading, though the driver's own figure for the process rose by its usual 32 MiB
  (section 5). Each later session's capture parks 1.7 MiB, so the low value still covers a session.
- **Width.** With the page cache full main's width stopped 7 of 757 planned pooled forwards (3 of
  308 for cohorts of two, 4 of 449 for cohorts of four) with about 95 GB available; the head
  stopped none. With `MemAvailable` itself at 2 GB the head's headroom fell from 92,948 to
  4,065 MiB and the cohort finished without an OOM line.
- **Texts.** Every request wrote the same text in both arms: 30 MTP requests, 24 DFlash2 requests
  in cohorts of two, 32 in cohorts of four, 48 decoded plain four at once, 8 in the short-memory
  cells. No OOM line in any cell that booted.
- **Two boots died at `Engine::new`** with `CUDA_ERROR_OUT_OF_MEMORY` and about 122 GB available,
  one per arm and unit, each 15 s after a cell whose readers were probably still filling the page
  cache. Not this change's code: memra#1316.

## Conditions

- Two DGX Sparks (GB10, sm_121a, driver 580.178.04), one campaign each: unit A (sections 1 to 4)
  and unit B (sections 5 to 7). Arms are compared only within a unit.
- Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head). Two servings (`tools/serve1307.sh`):
  - **mtp**: the masked MTP draft, 262,144 context, 32 sessions, as memra#1221's `serve.sh` serves
    it. The draft-state measurement runs on this route's draft-graph captures.
  - **dflash**: the DFlash2 drafter, 32,768 context, 4 sessions, prefix cache and reuse pool off so
    every request primes cold. The pooled DFlash2 verify runs here.
- Arms, both `memra-server` for sm_121a, built on the unit from one tree; each cell's `guard.log`
  has the binary's sha256:
  - B = main bbc6f83aea's `spec.rs`, `dflash.rs`, `unified_memory.rs`, `dsv4_gpu.rs`;
  - T = this branch's.
  - Both carry the `[diag1307]` lines of `tools/diag1307.py` (stderr only, no decision reads
    them): one line per planned pooled DFlash2 forward and one per draft-state bracket that
    captured, with the raw, `MemAvailable` and free-plus-reclaimable deltas.
  - D (section 5 only) = T plus `diag1307.py --deep`. Its rows are never compared.
- Page cache states (`tools/cell1307.sh`): **quiet** (`drop_caches` before the cell), **live** (a
  four-reader fill to MemFree under 3 GB once the server is ready, then two readers kept going
  through the requests), **liveboot** (the same fill and readers started before the server, so the
  boot's calibration runs under them). A cell that follows a live cell boots with the page cache
  full and idle.
- Requests (`tools/client1307.py`): greedy, non-streaming, each with its own nonce. mtp: three 1k
  prompts, 128 tokens. dflash: two pairs sent together, 3.7k prompts, 384 tokens. Each row has the
  sha256 of the reasoning and content text.
- The first campaign's client read `reasoning_content`, which this server does not send in a
  non-streaming message, so every hash was the hash of an empty text and compared nothing. Its
  cells are in `gb10/unit-a/first/` for their bracket lines only. Every text comparison below
  comes from the rerun with the fixed client, which marks a 200 row without text as an error.

## 1. The instrument: what a delta may read

`tools/probe_alloc.py` brackets a device allocation (`cuMemAlloc`, `cuMemsetD8` over it,
synchronize) with every reading a delta could use. Unit A, no server running. Deltas in MiB,
median (min to max); a reading that tracks the allocation reads its size.

`gb10/unit-a/probe` (8 samples per row; this first run logged `MemFree` and `Cached` only, so its
last column is their sum):

| state | size | raw (`cuMemGetInfo`) | `MemAvailable` | `MemFree + Cached` |
|---|---:|---|---|---|
| quiet | 34 | +34.6 (+26.8 to +34.8) | +34.6 (+26.8 to +34.7) | +34.6 (+26.8 to +34.7) |
| quiet | 256 | +260.2 (+252.2 to +260.5) | +260.2 (+252.2 to +260.5) | +260.2 (+252.2 to +260.5) |
| full | 34 | +34.6 (+34.5 to +40.6) | +34.6 (+34.5 to +35.5) | +34.6 (+34.5 to +34.6) |
| full | 256 | +258.5 (+258.2 to +258.9) | +258.5 (+258.2 to +258.9) | +258.5 (+258.2 to +258.9) |
| live | 34 | +40.4 (+15.4 to +45.0) | +34.6 (+34.6 to +37.6) | +34.8 (+34.6 to +36.9) |
| live | 256 | +280.6 (-673.9 to +303.2) | +260.4 (+258.5 to +879.8) | +260.1 (+258.4 to +260.6) |

`gb10/unit-a/probe2` (16 samples per row, MemFree 1.0 to 8.3 GB before each):

| state | size | raw (`cuMemGetInfo`) | `MemAvailable` | free + reclaimable |
|---|---:|---|---|---|
| full | 34 | +36.0 (-15.9 to +40.6) | +35.0 (+33.7 to +42.3) | +34.8 (+33.7 to +38.8) |
| full | 256 | +180.2 (-108.5 to +286.0) | +261.9 (+258.0 to +877.7) | +258.7 (+256.7 to +262.7) |
| full | 1024 | +739.5 (-365.6 to +857.4) | +1050.2 (+1025.2 to +1188.2) | +1042.5 (+1021.2 to +1044.9) |
| live | 34 | +41.4 (+35.0 to +43.4) | +34.8 (+33.4 to +39.0) | +34.7 (+33.4 to +36.7) |
| live | 256 | +218.7 (-499.5 to +308.3) | +259.6 (-368.0 to +285.4) | +259.7 (+251.1 to +261.0) |
| live | 1024 | +33.3 (-837.8 to +386.6) | +1055.5 (+1030.1 to +1624.6) | +1041.6 (+1028.3 to +1049.3) |

- The raw reading is `MemFree` (the two, read a moment apart, differ by at most 15 MiB). With the
  page cache full it missed a 256 MiB allocation by -930 to +52 MiB and a 1 GiB one by -1,862 to
  -167 MiB.
- `MemAvailable` holds while nothing reclaims and missed by up to 624 MiB either way when reclaim
  ran (+877.7 and -368.0 MiB for 256 MiB).
- Free plus reclaimable read 33.4 to 38.8 MiB for 34, 251.1 to 262.7 for 256 and 1,021.2 to
  1,049.3 for 1,024 over the second run's 96 samples (the first run's `MemFree + Cached` sums fall
  inside those ranges). Without `SReclaimable` one 1 GiB sample read 992.9 MiB; with it,
  1,028.3 MiB.

## 2. The draft-state bracket in serving (unit A)

Each boot's spec calibration captures the draft graphs; the first bracket that captures records the
draft-state high-water admission charges every speculative session. A quiet boot parks 33.8 to
34.9 MiB there on every reading. `gb10/unit-a/mtp` and `gb10/unit-a/first/mtp`, 16 boots:

| page cache at the boot | cell | arm | MemFree at the bracket, MB | first bracket: raw / `MemAvailable` / free + reclaimable, MiB | charge recorded, MB |
|---|---|---|---:|---|---:|
| dropped or mostly free | first 05-B, 06-T; 01-B, 05-B, 06-T | both | 61,485 to 67,249 | 33.9 to 34.1 on all three | 33 or 34 |
| filling (liveboot) | 21-B | main | 1,394 | 54.5 / 37.9 / 35.9 | 55 |
| filling (liveboot) | 24-B | main | 1,471 | 46.5 / 35.2 / 35.2 | 46 |
| filling (liveboot) | 22-T | head | 1,348 | 46.2 / 32.3 / 34.3 | 34 |
| filling (liveboot) | 23-T | head | 1,254 | 66.0 / 35.2 / 35.2 | 35 |
| full, idle | first 01-B | main | 2,209 | 34.5 / 34.5 / 34.5 | 34 |
| full, idle | first 04-B | main | 1,630 | 33.8 / 33.8 / 33.8 | 33 |
| full, idle | 04-B | main | 1,833 | 37.4 / 34.3 / 34.3 | 37 |
| full, idle | first 02-T | head | 2,186 | 34.9 / 34.3 / 34.3 | 34 |
| full, idle | first 03-T | head | 2,505 | 34.0 / 38.0 / 34.0 | 34 |
| full, idle | 02-T | head | 1,438 | -60.3 / 433.2 / 2.2 | 2 |
| full, idle | 03-T | head | 1,446 | 21.9 / 21.9 / 21.9 | 21 |

- **Under a filling page cache main over-charges and the head does not.** Main recorded 55 and
  46 MB; the head 34 and 35 MB, while its raw reading moved 46.2 and 66.0 MiB. The charge is a
  high-water, so main's value stays for the life of the server.
- **`MemAvailable` is ruled out by cell 02-T.** It fell 433.2 MiB across that bracket. As the
  delta's reading (this branch's first commit) that would have been the charge of every later
  speculative session.
- **Two head boots read low** (02-T: 2 MB, 03-T: 21 MB), both with the page cache full and idle.
  In 03-T all three readings agree, so no page-cache movement explains it. Main's reading has the
  same hole: in 02-T it read -60.3 MiB, and 6 of the 58 GB10 boots in memra#1221's and memra#1257's
  traces read 0 MiB at this bracket. Section 5 looks at what the driver holds in such a boot.
- **The session brackets.** Each request's own capture parks 1.7 MiB in a quiet cell. Over the 56
  such brackets taken under the readers: raw -218.1 to +22.6 MiB (median 7.3), `MemAvailable`
  -30.3 to +5.7 (median 1.7), free plus reclaimable -11.2 to +4.8 (median 1.7). The one -11.2 is a
  bracket where all three fell 11 to 15 MiB together.
- All 30 requests of the 10 rerun cells wrote the same texts in both arms (m1 and m2 `efa0605b9b82`,
  m3 `34ab48fa38c3`; 128 tokens, 432 to 435 characters), 0 OOM lines.

## 3. The pooled DFlash2 verify width, cohorts of two (unit A)

`gb10/unit-a/dflash`. The served spec gate admits speculation for two sessions at most
(`MEMRA_SPEC_GATE_LOW=2`), so a pooled forward has two members. One checkpoint column is
149.6 MiB on this model; the second member's need was up to 911 MiB.
"Needed" and "above the floor" are the server's own cap line, in MiB.

| cell | arm | state | pooled forwards planned with 2 live rounds | stopped by the width | headroom, MiB | raw free / `MemAvailable`, MiB | rows | OOM lines |
|---|---|---|---:|---:|---|---|---|---:|
| 11-B | main | live | 154 | 2 (759 needed, 700 above the floor) | 456 to 1,592 | 649 to 1,513 / 94,615 to 95,539 | 4/4 | 0 |
| 14-B | main | live | 154 | 1 (607 needed, 467 above the floor) | 427 to 2,505 | 718 to 2,188 / 94,177 to 95,509 | 4/4 | 0 |
| 13-T | head | live | 167 | 0 | 94,768 to 95,138 | 756 to 1,371 / 94,626 to 95,451 | 4/4 | 0 |
| 15-B | main | quiet | 154 | 0 | 75,238 to 75,710 | 74,960 to 75,747 / 93,312 to 94,007 | 4/4 | 0 |
| 16-T | head | quiet | 154 | 0 | 93,849 to 94,730 | 75,388 to 76,053 / 93,671 to 94,363 | 4/4 | 0 |
| 12-T | head | live | did not boot (below) | | | | 0/0 | |

- With the page cache full main's width stopped 3 of 308 planned forwards, at a raw reading of 831
  and 854 MiB with 95 GB available; those rounds verified serially. The head stopped none.
- The four requests wrote the same texts in all five cells (`741d776aeef2`, `286b5068e3a3`,
  `88d9941cdd49`, `909d65540627`; 384 tokens), and the 48 per-burst acceptance lines of each cell
  are the same set. The forward count differs between boots (154 or 167; 16-T, a head cell, has 154) with
  how the two requests' bursts interleave.
- **12-T did not boot.** `[server] FATAL: worker init failed: Engine::new failed:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`, 0.2 s after the start line, 15 s after
  the previous cell's server exited, with `MemAvailable` at 121.9 GB and the previous cell's
  readers probably still running. MemFree and the GPU's process list at the failure were not
  captured. `Engine::new` runs no code this change touches. Filed as memra#1316; the cell's whole
  `server.log` is kept.

## 4. Four requests at once under the served gate (unit A)

`gb10/unit-a/dflash4`, six cells. At a wave of four the gate turns speculation off
(`[spec-k] ... K=0 source=concurrency`), the requests decode plain, and no pooled forward is
planned. All 48 requests completed with the same texts in both arms, 0 OOM lines. These cells say
nothing about the width.

## 5. What a low first bracket is (unit B)

`gb10/unit-b/bootfull`: arm D, boots only. Two start from a dropped page cache; six start with the
page cache filled just before the boot and idle during it, the state of section 2's two low
readings. At the entry and the end of the first bracket arm D logs the driver's own figure for this
process (`nvidia-smi --query-compute-apps`, 1 MiB steps) and the `/proc/meminfo` fields. The
nvidia-smi call runs inside the production bracket, so arm D's own recorded charge (37 to 42 MB)
is not evidence; the three deltas below are read inside the dump calls.

| cell | page cache at the boot | MemFree at the bracket, MB | driver's figure for the process, MiB | raw / `MemAvailable` / free + reclaimable, MiB | `Cached` over the dump window, MiB |
|---|---|---:|---|---|---:|
| 51-D | dropped | 67,388 | 34,322 to 34,354 (+32) | 34.1 / 34.1 / 34.1 | 0.0 |
| 52-D | dropped | 67,346 | +32 | 34.5 / 34.5 / 34.5 | +0.1 |
| 53-D | full, idle | 1,269 | +32 | 37.4 / 34.1 / 34.1 | +3.6 |
| 54-D | full, idle | 1,238 | +32 | 34.3 / 35.8 / 33.8 | +0.4 |
| 55-D | full, idle | 1,982 | +32 | -127.0 / 17.8 / 17.9 | -288.6 |
| 56-D | full, idle | 2,040 | +32 | 37.7 / 34.2 / 34.2 | +3.6 |
| 57-D | full, idle | 1,774 | +32 | 38.1 / 34.7 / 34.7 | +3.6 |
| 58-D | full, idle | 1,627 | +32 | 37.4 / 33.9 / 33.9 | +3.6 |

- **The capture is the same size in every boot.** The driver's figure rose by 32 MiB in all eight,
  55-D included.
- **55-D reproduces the low reading** (17.9 MiB on the delta reading, 17.8 on `MemAvailable`,
  -127.0 on the raw one). Over the window of the two dump calls, which is slightly wider than the
  bracket's own readings, the kernel reclaimed 288.6 MiB of page cache, `MemFree` rose 259.1 MiB,
  and `MemFree + Buffers + Cached` fell 29.4 MiB (`SReclaimable` did not move): most of the
  missing 16 MiB was in the counters a moment later.
- So a low first bracket is the host's counters read while reclaim is moving pages, not a smaller
  capture. No host-wide reading is exempt: in section 2's 03-T all three read 21.9 MiB. Across
  both units 3 of the 13 boots that met a full, idle page cache read low on the delta reading
  (2.2, 17.9 and 21.9 MiB); none of the 4 boots under a filling page cache and none of the quiet
  ones did.
- What it costs: the charge is 2 to 22 MB instead of 34 for that server's life. Every later
  session's own capture parks 1.7 MiB (section 2), so the low value still covers what a session
  parks.

## 6. The pooled DFlash2 verify width, cohorts of four (unit B)

`gb10/unit-b/dflash4low`: the same two groups of four requests with the spec gate's band raised
(`MEMRA_SPEC_GATE_LOW=4`, `MEMRA_SPEC_GATE_HIGH=5`, an operator setting), page cache live. A pooled
forward takes consecutive rounds while their rows fit 16 and the width allows; the members beyond
the first needed up to 1,369 MiB.

| cell | arm | pooled forwards planned with 2+ live rounds | members taken | stopped by the width | headroom, MiB | raw free / `MemAvailable`, MiB | rows | OOM lines |
|---|---|---:|---|---:|---|---|---|---:|
| 61-B | main | 221 | 98 x 2, 95 x 3, 28 x 4 | 3 | 722 to 5,715 | 775 to 3,570 / 92,110 to 94,516 | 8/8 | 0 |
| 64-B | main | 228 | 107 x 2, 94 x 3, 27 x 4 | 1 | 1,043 to 4,008 | 795 to 2,588 / 92,432 to 94,518 | 8/8 | 0 |
| 62-T | head | 217 | 90 x 2, 97 x 3, 30 x 4 | 0 | 93,925 to 94,858 | 808 to 2,734 / 92,198 to 94,494 | 8/8 | 0 |
| 63-T | head | 228 | 103 x 2, 104 x 3, 21 x 4 | 0 | 93,660 to 95,670 | 764 to 2,369 / 92,909 to 96,033 | 8/8 | 0 |

- Main's width stopped 4 of 449 planned forwards (1,215 to 1,369 MiB needed against 722 to
  1,221 MiB above the floor, with 94 GB available); the head stopped none of 445. The 16-row limit,
  not memory, sets most forwards' width in both arms.
- The eight requests wrote the same texts in all four cells, and the same texts the plain decode of
  section 4 wrote for the same prompts on the other unit (`741d776aeef2`, `909d65540627`,
  `f3c4baa2f2b1`, `b897ef9e30f5`, `b180229f86d0`). The 96 per-burst acceptance lines of each cell
  are the same set.
- `gb10/unit-b/dflash`: the pair cell repeated on this unit. 17-T (head) planned 154 forwards,
  stopped none, and wrote section 3's four texts. 18-B (main) did not boot: the same
  `Engine::new failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` as section 3's
  12-T, 15 s after the previous cell's server exited, `MemAvailable` 122.4 GB. That is main's
  engine code on the other unit (memra#1316).

The effect of the width's reading is small in these cells: the pool's cached blocks (1.1 to
1.7 GB) keep the raw headroom above most members' needs, so main serialized 7 of 757 pooled
forwards across sections 3 and 6.

## 7. A cohort decoding while MemAvailable is taken down to 2 GB (unit B)

`gb10/unit-b/dflashshort`: the band raised as in section 6, four requests asked to copy their
records (3,072 tokens each). Once the server logs its first cohort, memra#1221's `mem_holder.py`
takes `MemAvailable` from 93 GB to 2.1 GB in 40 s and holds it there for the remaining 135 s of
decode (89,344 and 89,600 MB held; it gave nothing back in either cell). Planned forwards with two
or more live rounds, by `MemAvailable` at the forward:

| cell | arm | `MemAvailable` | forwards | members taken | stopped by the width | headroom, MiB | raw free, MiB | pool's cached blocks, MiB |
|---|---|---|---:|---|---:|---|---|---|
| 41-T | head | over 10 GB | 171 | 65 x 2, 87 x 3, 19 x 4 | 0 | 12,226 to 92,948 | 739 to 75,189 | 1,109 to 3,562 |
| 41-T | head | 3 to 10 GB | 17 | 17 x 2 | 0 | 5,533 to 12,020 | 708 to 1,085 | 3,498 to 3,576 |
| 41-T | head | 2.0 to 3.0 GB | 788 | 723 x 2, 65 x 3 | 0 | 4,065 to 5,695 | 693 to 826 | 3,561 to 5,193 |
| 42-B | main | over 10 GB | 204 | 102 x 2, 80 x 3, 22 x 4 | 0 | 2,471 to 74,854 | 668 to 75,281 | 1,109 to 3,402 |
| 42-B | main | 3 to 10 GB | 22 | 21 x 2, 1 x 3 | 0 | 2,517 to 2,693 | 700 to 876 | 3,338 to 3,353 |
| 42-B | main | 2.0 to 3.0 GB | 827 | 820 x 2, 7 x 3 | 0 | 2,484 to 4,188 | 666 to 831 | 3,337 to 4,969 |

- The head's headroom follows the shortage: 92,948 MiB before the holder, 4,065 to 5,695 MiB with
  `MemAvailable` at 2 to 3 GB (that reading plus the pool's cached blocks, less the floor).
- Neither arm's width stopped a forward. The pool's cached blocks alone (3.3 to 5.2 GB) cover the
  largest need (1,369 MiB), so these cells show the reading following the shortage and the cohort
  finishing; they do not show the cap protecting anything.
- The four requests wrote the same texts in both arms (`3f8bc637f152`, `4704ffe8a314`,
  `6c48f62717b3`, `b5558d3b40e0`; 3,072 tokens, `length`). 0 OOM lines in the server, 0 in the
  kernel log of either window; lowest `MemAvailable` 2,086 and 2,059 MB.
- The holder's memory is shared anonymous memory, which `/proc/meminfo` counts in `Cached`: at the
  end of 41-T `Cached` read 93.7 GB with `MemAvailable` at 2.1 GB. That is why a level never takes
  the delta's reading.

## Excluded

- A discrete card. Nothing changes there: both readings return `cuMemGetInfo` free (the code
  path, and the tests' discrete rows). No cell.
- The DeepSeek-V4 door on GB10. The route does not load on one device.
- `MemAvailable` short at a request's arrival. Admission refuses first there (memra#1257's short
  cells); section 7 takes the memory after the cohort is admitted.
- Any model but Qwen3.8-27B, and any timing claim. These cells compare decisions and texts, not
  speed.

## Tools

`tools/`, with memra#1221's `tsrun.py` and `mem_holder.py` and memra#1257's `fill_cache2.sh`:

- `probe_alloc.py`: the instrument probe;
- `diag1307.py`, `prep_arms.sh`, `build1307.sh`: the arms' diagnostic lines, their source trees and
  their builds;
- `serve1307.sh`, `cell1307.sh`, `live_fill.sh`, `client1307.py`, `campaign1307.sh`: the servings,
  one cell, the fill with readers, the requests, the campaigns;
- `tab1307.py`: the tables; `assemble1307.py`: copies receipts here, home paths written as `~`;
- `dd-checks.sh`, `dd-suites.sh`: the CPU checks and red arms.

## CPU

`cpu/`, on this branch's head at main bbc6f83aea (`checks-summary.txt` and `suites-summary.txt`
start with the engine files' sha256):

- `pooled_dspark_verify_headroom_reads_the_allocatable_free_memory`: memra#1221's full-page-cache
  reading (raw 1,882 MiB, 100 GB available) leaves 346 MiB of headroom on the raw reading and
  98,864 MiB on the allocatable one; the width's body reads `pooled_verify_reading`, which reads
  `allocatable_mem_info`.
- `draft_state_delta_reads_free_plus_reclaimable_and_the_oom_floor_the_raw_one`: two of the
  probe's 256 MiB samples read 258.8 and 256.7 MiB on the delta reading, 0 and 226.8 on the raw
  one, 0 and 877.7 on the allocatable one; the bracket takes the delta reading at both ends and
  every failed capture records the raw entry reading.
- `free_plus_reclaimable_reads_each_field_from_its_own_line` (`unified_memory.rs`).
- Red arms, tests kept, one production line changed, each fails (`cpu/R*.log`):
  - R1: the width reads `mem_get_info` again;
  - R2: the bracket end reads `mem_get_info` again (main);
  - R3: the bracket end reads `allocatable_mem_info` (this branch's first commit);
  - R4: one failed capture's floor takes the delta's entry reading.
  A red arm whose edit does not apply is reported as such and not run: R4's first version asserted
  the wrong anchor count, left the source unchanged and passed.
- Engine library suite 782 passed, server library suite 1,263 passed; clippy (engine and server,
  all targets) and fmt green.
