# On unified memory the pre-capture guard reads the allocatable free memory (2026-11-12)

**Status:** adopted, memra#1221 (part: the boundary pause's trim-driven share; the plane copies
stay). Receipts:
[`research/spec-capture-headroom-20261112/`](../../research/spec-capture-headroom-20261112/).
Follows [`research/spec-boundary-publish-20261110/`](../../research/spec-boundary-publish-20261110/)
(memra#1208).

## Question

memra#1221 sized the one pause a long cold speculative request takes after its first burst, when
the publisher copies the prompt's KV planes into the prefix cache. On GB10 with Qwen3.8-27B and
its MTP draft, main's profile gives 57.5 ms at a 131k prompt (38.6 ms of plane copies, 15.5 ms of
device idle while the host waits in `cuMemAllocAsync`) and 272.0 ms at 260k (76.8 ms of copies,
191.4 ms of idle; the pool grows 2.5 GB inside the publisher). The 260k cells split in two: 261 to
272 ms where the pre-capture guard had trimmed the pool, 129.6 ms where it had not. Why does the
trim fire, and is the pool growth it causes avoidable?

## What the cells showed

- **The idle time is the pool growing.** The first planes come from cached blocks in microseconds;
  the last ones each grow the pool by 96-288 MB and hold the scheduler thread 8-77 ms. At 260k
  the publisher grew the pool 2,400 MB in every cell where the pre-capture trim fired.
- **The trim fires on a full page cache, not on a short device.** On GB10 `cuMemGetInfo` leaves out
  page cache, which the kernel hands to the driver on demand. With the page cache filled (MemFree
  2.4-5.1 GB, MemAvailable about 101 GB) the raw reading at the cold request's capture was
  2,008-2,785 MB against 3,139-3,141 MB required, so the guard released the 1,696 MB the prime
  had just left cached and the publisher grew it back one burst later.
- **The same reading refuses short requests' captures.** In 7 of the 8 filled-cache cells of
  the narrow-trim campaign, and 3 of 3 main cells of the record, the 1k request sent right
  after the fill had its draft-graph capture refused before any attempt and drafted eagerly for
  the whole request, with about 100 GB available. The head refuses none (0 of 3).
- **With the trim skipped at that reading the capture succeeds.** Arm N skipped the trim at a raw
  reading of 2,259 and 2,785 MB: no capture warning, no out-of-memory line, server exit 0, lowest
  MemAvailable 81 GB, the same text.
- **Trimming only the shortfall is not enough.** Of the 789-1,397 MB the narrow trim keeps, about
  800 MB is used again before the boundary (main grows it back during the prime's tail, the
  capture and burst 1); the publisher's growth fell by 576, 32 and 0 MB in three cells, and the
  publisher's time did not follow (151-181 ms main, 153-168 ms narrow trim).

The record campaign, unit 1, filled page cache, probe arms, medians with spread:

| | main, 260k | head, 260k | main, 131k | head, 131k |
|---|---:|---:|---:|---:|
| cells | 3 | 3 | 2 | 2 |
| guard fired on the cold request | 3 trims (1,696 MB each) | 0 | 2: one refusal of the 131k capture, one trim (896 MB) | 0 |
| pool growth inside the publisher, MB | 2,400 (0) | 1,408 (0) | 544, 768 | 544, 544 |
| publisher on the scheduler thread, ms | 152.7 (8.7) | 124.2 (24.5) | 54.1, 74.5 | 57.8, 56.7 |
| first burst boundary over its rounds, ms | 165.7 (4.9) | 142.7 (24.8) | 69.5, 79.9 | 87.5, 51.3 |
| ms per round | 187.22 (0.44) | 187.26 (0.49) | 137.88, 137.36 | 138.95, 137.25 |
| TTFT, s | 313.236 (1.288) | 313.270 (0.317) | 117.495, 117.785 | 117.874, 117.615 |
| 1k request's capture refused | 3 of 3 | 0 of 3 | 2 of 2 | 0 of 2 |

At 260k every head cell is under every main cell on the publisher (-28.5 ms, -18.7%) and on the
boundary (-23.0 ms). Texts and burst structures are identical across arms; no out-of-memory line in
any cell. At 131k main's guard fired in both cells: it refused the 131k request's own capture in
one (effective 3,046 MB < 3,141, so that request drafted eagerly), trimmed 896 MB before it in the
other, and in that cell's boot calibration trimmed 15,808 MB with the page cache already full. The
head's guard fired in none.

**Truly short** (`gb10/short-u1`): a holder process (`oom_score_adj` 1000, capped at 300 s) took
`MemAvailable` to about 2 GB during a 131k prompt's prime. Main's guard refused the capture
(raw 1,100 MB + pool 1,070 MB < 3,140 MB). The head's guard also refused it (allocatable 1,970 MB
+ pool 1,070 MB = 3,040 MB < 3,140 MB). Both requests were served eagerly with the same text.
There was no out-of-memory line and no kernel OOM in either window. With the memory short at a
request's arrival, admission refuses the request (HTTP 400) on both arms before any capture.

**The band this change opens** (`gb10/band-u1`, B V V B): the raw reading under the requirement
and `MemAvailable` above it. The holder kept `MemAvailable` near 3.5 GB through a 131k prompt's
prime. Main's guard refused the capture in both cells (driver 1,389 and 1,408 MB + pool 1,070 MB
< 3,140 MB, with `MemAvailable` at 3,429 and 3,476 MB). The head's guard let it go ahead without
a trim at `MemAvailable` 3,463 and 3,470 MB, and the capture succeeded in both: no capture
warning, no out-of-memory line, no kernel OOM, `MemAvailable` no lower than 3,225 MB to the
request's end. All four cells served the same text in the same rounds.

## The change

`capture_headroom_refusal` (`spec.rs`) reads `unified_memory::allocatable_mem_info` instead of
the raw `cuMemGetInfo`:

- **Discrete cards:** unchanged. The guard reads `cuMemGetInfo`, trims to zero and logs as main
  does; `capture_headroom_plan` returns `TrimAll` there.
- **Unified memory** (`CU_DEVICE_ATTRIBUTE_INTEGRATED`): the larger of the raw reading and
  `/proc/meminfo` `MemAvailable`, the reading admission, the prefix-cache budget and the MoE
  residency decision already size with. An unreadable `/proc/meminfo` keeps the raw reading.
- **The requirement is unchanged:** twice the capture's measured appetite plus two
  `CAPTURE_HEADROOM_FLOOR`s (3,072 MiB). That floor is the margin over `MemAvailable`, which
  already leaves out the kernel's own reserve.
- **The trim stays as the backstop**, narrowed on unified memory only: when even the allocatable
  reading is short and the pool holds cached blocks, the pool releases the shortfall (`required - free`) through
  `cuMemPoolTrimTo` and keeps the rest (`capture_trim_keep`), as admission's driver-headroom rung
  does. It used to release every cached block. A discrete card keeps the trim to zero: the narrow
  trim measured flat on GB10 and has no cell on a discrete card. The decision is one pure
  function (`capture_headroom_plan`: `Go`, `Refuse`, `TrimAll`, `TrimShortfall`) with a table
  test over the GB10 readings and discrete rows.
- Unchanged: the 256 MiB graph-launch floor and admission's driver-headroom rung keep the raw
  reading (`unified_memory.rs` says why).

## Rejected

- **Reserving the publisher's planes earlier** (memra#1221's lever), on the scheduler thread at
  the prime's end (P) or on a helper thread at burst 1's second commit (H). Both take the
  publisher to 1-13 ms and both move the growth into the first burst: at 131k, same unit,
  three cells each, H took the stall from 55.8 to 42.5 ms and added 123 ms to burst 1 and
  2.2 ms to each of burst 2's rounds, and it grew the pool by 3.3 GB because the prime's
  transients were still alive (6.6 GB at 260k). P put 117 ms (131k) and 641 ms (260k) before
  the first token. Neither shortens the request; both hold more memory. Their code is not on the
  branch (`tools/probe_h.py`, `tools/probe_p.py` in the receipts rebuild them).
- **The narrow trim alone** (this branch's first commit, kept as the backstop): no
  measurable pause gain, see above.
- **Never trimming** (arm N as a rule). It removes the protection where memory is short. The
  allocatable reading keeps the guard and removes the false trigger.
- **A margin constant on top of `MemAvailable`.** The requirement already carries two 1,536 MiB
  floors over twice the appetite, and `MemAvailable` already leaves out the kernel's reserve; a
  second margin would have no measurement behind it.

## Limits and open items

- The plane copies stay: about 40 ms at 131k and 77 ms at 260k of device time, and the pool
  still grows inside the publisher when the entry is larger than what the pool holds (1,408 MB at
  260k at the head). Removing them needs a publication that does not copy (memra#1221 part 2).
- The publisher's time per GB of growth depends on how fast the kernel reclaims page cache: the
  same 2,400 MB cost 151-181 ms on unit 1 and 261 ms on unit 2. Cells compare only within a unit.
- Admission's driver-headroom rung (`MEMRA_ADMIT_DRIVER_HEADROOM_MB`, 2,048 MiB) and the
  graph-launch floor still read the raw value. On a full page cache the rung fired before the
  prime in 8 of the 14 filled-cache 260k cells (10 times) at a raw reading of 1,882-2,002 MB and
  released 0-160 MB each time: small next to the capture trim's 1,696 MB, and left for its own item (memra#1257).
- One model (Qwen3.8-27B with its MTP draft) on GB10. On a discrete card the guard is main's; on
  the RTX 5090 neither the trim nor a false refusal fired at 28k or 60k prompts.
- `/proc/meminfo` is host-wide. Inside a container or a memory-cgroup-limited service,
  `MemAvailable` can overstate what the process may take, at least for its page cache; whether
  GB10 driver allocations are charged to the cgroup is not established. Admission, the prefix
  budget and the MoE residency decision already read it this way; this change extends it to the
  capture guard. The cells here ran on the host, outside any memory cgroup limit.
- The band raw < required <= `MemAvailable` was measured in two head cells and two main cells
  at 131k, with a holder that gives back 256 MB when the server's use pushes `MemAvailable` under
  its target (it did so once per cell, on both arms). The capture's appetite there is 2 x 34 MB
  under a 3,072 MB floor, so these cells show the head's capture succeeding in the band; they do
  not show how an over-ask ends when nothing gives memory back. On unified memory that can be the
  kernel's OOM killer rather than a `CUDA_ERROR_OUT_OF_MEMORY`.
