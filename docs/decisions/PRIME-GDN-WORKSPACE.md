# The prime's GDN scan planes are one workspace, kept mapped only above a free-memory floor (2026-11-14)

**Status:** adopted for dense GDN trunks on the mma scan route (memra#1258, part 2), with the review
fold of 2026-11-14; some cells of the fold are still owed (listed in the record). Measured on an
RTX 5090 Laptop (sm_120a) and on GB10 (sm_121a, both DGX Spark units) with Qwen3.8-27B NVFP4 +
Q5_K and DFlash2. No env seam: the floor is the one admission keeps free
(`MEMRA_ADMIT_DRIVER_HEADROOM_MB`), and 2,048 MiB when that is 0 or there is no admission.

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

The workspace stays mapped only while the device's free reading (admission's own:
`allocatable_mem_info`) holds a floor of 2,048 MiB by default, the one admission keeps free in
front of every request. The floor is asked at every point the workspace could stay mapped:

- **at a prime's end:** under the floor it is dropped, its bytes are cached blocks again, and the
  next prime allocates a new one;
- **between two calls of one prime:** a prime of more than one call yields between its calls and
  other sessions run. Under the floor the workspace is dropped at the end of a call that has more
  to come and the next call allocates it again. Held there, its bytes are used bytes: no trim
  frees them and the sessions in the gap cannot take them;
- **later:** a workspace parked on a roomy device is dropped in front of the next admitted request
  once the device no longer holds the floor.

A walk that fails, a chunk that fails and a request retired between two calls of its prime all
release the workspace, on every engine the model owns.

The reason for the floor is what the workspace's bytes are. Parked, only a prime or a trim can use
them; held between calls, nothing can. The steps that run meanwhile allocate from the pool's
cached blocks and then from the driver.

## Evidence

`research/prime-hoist-20261113/`.

- **RTX 5090 Laptop, ~5.7k tokens, margin sweeps** (a cold DFlash prime beside a decoding keeper,
  placed by a ballast at the gate's `required` plus a margin; the change before the review fold):
  - main fails inside the prime's first call on prompt 5,774 at margins of 49 and 100 MiB, with
    1.3 GB unused inside the pool;
  - with the workspace no row fails: 21 rows, among them that prompt at 48 MiB and a zero-margin
    row, all seven rows of a sweep on the boot's own charge.
- **Per-call draw on that card:** the pool grows 704-736 MiB per prime at 5.7k against main's
  800-992, and main's 192 MiB step is gone. At ~8.1k the second call draws 1,103-1,105 MiB, the same
  in both boots, against main's 1,616-2,255.
- **~8.1k tokens under a 100-150 MiB margin, before the fold:** both arms fail in the prime's
  second call in 2 rows (of 10 for main, of 8 with the workspace), but not on the same prompts.
  Main's failures (prompts 8,234 and 8,215) pass with the workspace. The workspace's two failures
  are one scenario that main passes: prompt 8,173 as the first placed row of a boot. Repeated in
  fresh boots at 100 MiB, it ran out of memory in 3 boots of 4 where main ran clean in 6 of 6: the
  workspace's 481 MiB stayed held across the gap between the prime's two calls, the keeper's
  bursts in the gap grew the pool by 64 MB from the driver, and the second call had that much less.
- **The same scenario with the fold:** 2 fresh boots of 2 clean, pool used after the first call
  20,560 MB as on main, no growth in the gap. Two boots; the remaining boots and the other 8.1k
  and 5.7k cells of the fold are owed.
- **GB10, both units:** the first call draws 1,201-1,204 MiB in every boot against main's
  1,474-1,477, with no trim and no pool growth inside a prime. Under a full page cache (MemFree
  1.5-2.9 GB, `MemAvailable` about 120 GB) the workspace still parks. One boot per arm of the fold
  reads the same (1,201-1,204 against 1,506-1,509 on the later main).
- **Outputs:** `run-gen`, `run-spec` K=1..8 and `dspark_q38_gate` are identical to main's on the RTX
  5090 and on GB10 before the fold; the served answer texts equal main's in the 8.1k cells on the
  5090 and in the GB10 cells, the fold's GB10 boot included. Exactness on the fold's binary is owed.
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
- **The floor asked only at a prime's end.** The form this change first had. It held the workspace
  across the gap between a prime's two calls on a short device (the 8,173 rows above), and it left
  a workspace parked on a roomy device when the device got short later.
- **The driver-headroom rung's target counting parked bytes**, as the way to reclaim a parked
  workspace later. The rung would free the workspace only after the other cached blocks ran out.
  Dropping it in front of the rung makes its bytes the cached blocks the gate already counted.
- **`MEMRA_ADMIT_DRIVER_HEADROOM_MB=0` meaning "always park".** That value switches the rung off;
  it would also have switched on the rejected always-park form, in the server and in every engine
  binary. The park keeps its own 2,048 MiB default.
- **An env seam for the park.** The floor is admission's existing one.
