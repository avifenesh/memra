# On unified memory a level reads the allocatable free memory and a delta reads free plus reclaimable (2026-11-14)

**Status:** adopted, memra#1307. Receipts:
[`research/unified-readers-20261114/`](../../research/unified-readers-20261114/). Follows
[SPEC-CAPTURE-HEADROOM-UNIFIED.md](SPEC-CAPTURE-HEADROOM-UNIFIED.md) (memra#1221, memra#1257) and
[POOLED-VERIFY-MEMORY-CAP.md](POOLED-VERIFY-MEMORY-CAP.md) (memra#1266).

## Question

On a unified-memory device (GB10) `cuMemGetInfo` free is the host's `MemFree`: it leaves out page
cache the kernel hands to the driver on demand. Admission, the prefix budget, the pre-capture guard,
admission's driver-headroom rung and the pooled MTP verify's cap already read
`unified_memory::allocatable_mem_info` (the larger of the raw reading and `MemAvailable`). Three
readers still took the raw value: the pooled DFlash2 verify's width, the per-session draft-state
measurement, and the DeepSeek-V4 route's memory door. Which reading does each need?

## Chosen

- **The pooled DFlash2 verify's width reads the allocatable figure.** `pooled_verify_headroom`
  takes `pooled_verify_reading`, the reading the pooled MTP verify's cap takes: the larger of
  `cuMemGetInfo` free and `MemAvailable`, plus the pool's cached blocks. Its rule is unchanged: a
  member joins while what the forward needs beyond one serial round fits the reading less the
  1,536 MiB transient floor. It is a level, sized like admission's.
- **The draft-state measurement reads free plus reclaimable at both ends.** It is a delta: two
  readings around a draft-graph capture, recorded as a high-water that admission charges every
  speculative session. On unified memory both ends take `unified_memory::delta_free_bytes`:
  `MemFree + Buffers + Cached + SReclaimable` from one read of `/proc/meminfo`. Page-cache filling
  and reclaim move pages between its terms, so they cancel in it. A level never takes this
  reading: not every cached page is reclaimable.
- **A failed capture's appetite floor keeps the raw entry reading.** It records "a capture failed
  with this much free" in the same high-water. The raw reading is at most what the capture could
  get, so the bound is proven; an estimate above it would stick as an over-charge.
- **The DeepSeek-V4 memory door keeps the raw reading.** The route takes exactly two CUDA devices
  (`Dsv4Gpu::load`) and its weights fill about 83 GiB of each 96 GB card, so it does not load on a
  GB10. `stage_memory` says what the door needs if it ever does: the allocatable figure for its
  level and the delta reading for the `occupied` deltas behind its calibrated terms.
- **Discrete cards read what they read.** `allocatable_mem_info` and `delta_free_bytes` return the
  `cuMemGetInfo` value there.
- **"Unified?" is the engine's answer, read once when it is built** (`Engine::unified_memory`).
  Both ends of a delta ask the engine, so a delta is never taken across two readings, and a
  discrete card pays no device attribute call per reading. Asking the device again goes through
  `CudaContext::attribute`, which first returns any stored driver error; an end that read
  "discrete" against a unified entry would record tens of GB as the charge.

## Rejected

- **The allocatable figure for the delta** (this branch's first commit). `MemAvailable` is the
  kernel's estimate from its LRU lists and moves by hundreds of MiB while reclaim runs. Across two
  served boot brackets that park 34 MiB it fell 433 and 625 MiB (cells `02-T` and `72-Tc`), which
  the high-water would have kept as the charge of every later speculative session. In the allocation probe a 256 MiB
  allocation moved it by -368 to +878 MiB.
- **The raw reading for the delta** (main). It is `MemFree`: page cache filling lowers it and
  reclaim raises it. Booting under a filling page cache, main recorded 55, 46 and 16 MB where a
  quiet boot records 33-34; the head recorded 34, 35 and 20. With reclaim running it reads zero: 6 of 58
  boots in the GB10 traces of memra#1221 and memra#1257 read 0 MiB at the first bracket. In the
  probe a 256 MiB allocation moved it by -500 to +308 MiB.
- **`MemFree + Cached` without the slab.** One of 96 probe samples read 993 MiB for a 1 GiB
  allocation; with `SReclaimable` it read 1,028 MiB. `Buffers` changed nothing in the samples and
  belongs to the same pool.
- **The pool's own counters plus the capture's driver allocations** (memra#1307's candidate for
  the delta). The bracket already records the larger of the parked delta and the pool's
  capture-time peak, so the pool's counters are its floor. A pool-only delta cannot see what the
  driver holds outside the pool, which is 32 MiB of the first capture's 34, and the allocations
  graph instantiation makes inside the driver are not the engine's to count.
- **Dropping the DFlash2 rule's transient floor**, the other half of what POOLED-VERIFY-MEMORY-CAP
  left open. Nothing here measured it. The floor stays.

## Measured

Two DGX Sparks, one campaign each, Qwen3.8-27B (NVFP4 trunk, Q5_K output head), main against this
branch, both arms carrying the same diagnostic lines. Tables and conditions are in the write-up.

- **The instrument.** A device allocation of 34, 256 and 1,024 MiB bracketed by each reading, page
  cache full or filling, 96 samples: free plus reclaimable read 33.4 to 38.8, 251.1 to 262.7 and
  1,021.2 to 1,049.3 MiB; the raw reading -15.9 to +43.4, -499.5 to +308.3 and -837.8 to
  +857.4 MiB; `MemAvailable` 33.4 to 42.3, -368.0 to +877.7 and 1,025.2 to 1,624.6 MiB.
- **The draft-state charge in serving.** A quiet boot records 33 to 34 MB. Booting under a
  filling page cache, three boots per arm: main 55, 46 and 16 MB, the head 34, 35 and 20 MB (the
  third pair is the folded head against main with landing 24). Over 56 session brackets taken
  under the readers, where a quiet one reads 1.7 MiB: raw -218.1 to +22.6 MiB, `MemAvailable`
  -30.3 to +5.7, free plus reclaimable -11.2 to +4.8 (median 1.7).
- **The width.** With the page cache full main's width stopped 8 of 462 planned pooled forwards
  in cohorts of two and 4 of 449 in cohorts of four (the spec gate's band raised to four), at raw
  readings of 669 to 1,543 MiB with about 95 GB available. The head stopped none of 933. With
  `MemAvailable` taken to 2 GB under a decoding cohort the head's headroom fell from 92,948 to
  4,065 MiB; no forward was stopped in either arm and the cohort finished.
- **Texts.** The same in both arms in every request (156 across the routes); no OOM line in any
  cell that booted.
- **The landing code.** The last pair of each kind above ran main with landing 24 against the
  head after the review's fold (the write-up's section 8); the rest ran before the fold changed
  code.

## Limits

- **The delta reading removes the page cache's share, not every host effect.** With reclaim
  running, 4 of 19 boots that met a full page cache (3 of 13 with it idle, 1 of 6 with it filling)
  read 2.2, 17.9, 20.4 and 21.9 MiB at the first bracket, where the driver's own figure for the
  process (`nvidia-smi --query-compute-apps`) rose by 32 MiB in all eight boots that logged it. In one of them all three
  host readings agreed on the low value; in another the missing part was in the counters a moment
  later. The charge then stays at 2 to 22 MB for that server's life. Each later session's capture
  parks 1.7 MiB, so the low value still covers a session; main's raw reading has the same hole (6
  of 58 earlier boots read 0).
- **The driver's own per-process figure would be exact** and was not taken: it needs NVML or an
  `nvidia-smi` call inside the capture bracket, and NVML stalls under graph capture and large
  allocations on Blackwell (`health.rs`). It reads in 1 MiB steps.
- **Other processes.** Any host-wide reading moves when another process allocates or frees
  anonymous memory during the bracket. The cells ran with nothing else on the unit.
- **The width's effect is small here.** The pool's cached blocks (1.1 to 1.7 GB) keep the raw
  headroom above most members' needs, so main serialized 12 of 911 forwards. No cell shows the cap
  protecting a forward when memory is short: at 2 GB the cached blocks alone covered the need.
- **This record covers the three readers memra#1307 named.** The deepseek-v4.1 route is outside
  it: it loads on one device by default and has its own raw readers (`dsv41_gpu/experts.rs`
  `device_memory`, `dsv41_gpu/decode.rs` `device_mem`: the driver's free memory plus the pool's
  cached blocks, read as levels and as the two ends of deltas). Only its size keeps it off a GB10
  today. If it loads there, its levels need the allocatable figure and its deltas the delta
  reading.
- One model on GB10. A discrete card reads as before and has no cell. Timing was not measured.
- `/proc/meminfo` is host-wide; the cgroup caveat of
  [SPEC-CAPTURE-HEADROOM-UNIFIED.md](SPEC-CAPTURE-HEADROOM-UNIFIED.md) applies to both readings.
