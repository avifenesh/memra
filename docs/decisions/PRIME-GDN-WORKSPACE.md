# The prime's GDN scan planes are one workspace, parked between primes above admission's floor (2026-11-14)

**Status:** adopted for dense GDN trunks on the mma scan route (memra#1258, part 2). Measured on an
RTX 5090 Laptop (sm_120a) and on GB10 (sm_121a, both DGX Spark units) with Qwen3.8-27B NVFP4 +
Q5_K and DFlash2. No env seam: `MEMRA_ADMIT_DRIVER_HEADROOM_MB` is the floor the park reads.

## Question

A prime call allocated the chunked GDN scan's planes fresh in every GDN layer and freed them in the
same layer: about 520 MB per layer at 4,096 rows on Qwen3.8-27B, the largest of them a 192 MiB
`ssnap16` block. Where that block landed made the first call's pool draw differ from prime to prime,
and under memory pressure the call failed with the bytes it needed free inside the pool, in pieces.
Where should those planes live?

## Decision

One workspace (`GdnScanWs`) holds the planes for a prime. `prime_layers` lends it to the engine for
the walk. At the prime's end it is parked on the engine, still mapped, and the next prime takes the
same bytes back. Parked bytes count as cached, a trim frees them only when the cached blocks fall
short of its target, admission charges only the workspace's rows past the call's, and the boot
calibration leaves the parked bytes out of its measured peak.

The workspace is parked only while the device's free reading (admission's own:
`allocatable_mem_info`) holds the floor admission keeps free in front of every request, 2,048 MiB by
default. Under the floor it is dropped at the prime's end and its bytes are cached blocks again; the
next prime allocates a new one. The reason is what parked bytes are: only a prime or a trim can use
them, while the steps that follow a prime allocate from the pool's cached blocks and then from the
driver.

## Evidence

`research/prime-hoist-20261113/`.

- **RTX 5090 Laptop, ~5.7k tokens, margin sweeps** (a cold DFlash prime beside a decoding keeper,
  placed by a ballast at the gate's `required` plus a margin):
  - main fails inside the prime's first call on prompt 5,774 at margins of 49 and 100 MiB, with
    1.3 GB unused inside the pool;
  - with the workspace no row fails: 21 rows, among them that prompt at 48 MiB and a zero-margin
    row, all seven rows of a sweep on the boot's own charge.
- **Per-call draw on that card:** the pool grows 704-736 MiB per prime at 5.7k against main's
  800-992, and main's 192 MiB step is gone. At ~8.1k the second call draws 1,103-1,105 MiB, the same
  in both boots, against main's 1,616-2,255.
- **~8.1k tokens under a 100-150 MiB margin:** both arms fail in the prime's second call at the same
  rate (2 of 10 rows on a boot's own charge for main, 2 of 8 with the workspace). The workspace
  neither removes nor adds to that failure.
- **GB10, both units:** the first call draws 1,201-1,204 MiB in every boot against main's
  1,474-1,477, with no trim and no pool growth inside a prime. Under a full page cache (MemFree
  1.5-2.9 GB, `MemAvailable` about 120 GB) the workspace still parks.
- **Outputs:** `run-gen`, `run-spec` K=1..8 and `dspark_q38_gate` are identical to main's on the RTX
  5090 and on GB10; the served answer texts equal main's in the 8.1k cells on the 5090 and in the
  GB10 cells.
- **Required memory:** 2,320 MB against main's 2,353 on the 5090 at 5.7k (reserve 1,543 against
  1,576); within 17 MB of main's on GB10.
- No timing was measured; the capacity cells read memory only.

## Rejected

- **The workspace resident in the prime slab from boot.** About 480 MB that admission counted as
  used. The 5090 margin sweeps could not place their margins (56-60 MiB seen for targets of 75-150),
  and at 8k tokens 12 of 14 rows were served plain where main served 12 of 14 on spec.
- **Freed to the pool at every prime's end, on every device.** On GB10 the next prime's draw then
  depended on what other allocations left: 1,571 MiB in one boot and 1,603 in another, both above
  main's 1,474-1,477.
- **Parked on every device, whatever its free memory.** Admission counts parked bytes as cached, and only a prime
  or a trim can use them. On the RTX 5090 at zero margin the prime completed with 94 MB of driver
  free and 480 MB parked, and the request's next step ran out of memory, where main ran the same
  prompt clean at 1 MiB.
- **An env seam for the park.** The floor is admission's existing one, and `0` (the rung off) parks
  always.
