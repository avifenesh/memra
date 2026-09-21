# Write-combined contract destinations: mechanism, code sites, measurement plan (day 16)

Status: a FINDING of day 15 (`DAY15.md` finding 1) with its first measured cell on day 16, reported as the
first cell of the `MEMRA_KV_HOST_CONTRACTS` decide-by review (2026-10-05), not as a verdict. The allocation
flag is engine territory (`crates/memra-engine/src/tier_transfer.rs`) and is NOT changed by this lane. Every
`file:line` is in tree `25891baa9`; `core.rs` is cudarc 0.19.8 `src/driver/safe/core.rs`.

## Mechanism

| Allocation | Call | Flags | Used by |
|---|---|---|---|
| The contract's pinned destinations (every `HostPlaneBytes::Contract` lease under the door) | `CudaTransfers::alloc_host` (`tier_transfer.rs:210-245`) calls `CudaContext::alloc_pinned` (`core.rs:1412-1424`) | `cuMemHostAlloc(bytes, CU_MEMHOSTALLOC_WRITECOMBINED)` (`core.rs:1417-1420`); cudarc exposes no flag parameter and `PinnedHostSlice`'s fields are private, so a different flag needs either a cudarc change (an external dependency: refused for this lane) or an engine-owned pinned backing type inside `PinnedAllocation` (`tier_transfer.rs:35-40`) | Option B's D2H destinations (day 15), Option C's H2D sources (day 16, twins of the same leases) |
| The OFF tier's planes | `PinnedHostBuf::new` (`crates/memra-engine/src/pinned_host.rs:186-201`) | `cuMemHostAlloc(len, 0)` (`pinned_host.rs:190`): cached, page-locked | every OFF-arm plane, every handoff import |
| The startup arena | `PinnedHostArena` (`pinned_host.rs:102-104`) | `cuMemHostAlloc(.., CU_MEMHOSTALLOC_PORTABLE)`: cached | the `MEMRA_GLM5_TP_KV_HOST` arena path, refused with the door at boot |

Write-combined host memory is what the CUDA driver documents for buffers the CPU WRITES and the device
READS: CPU stores are combined and posted, the DMA engine reads it across PCIe as fast as or faster than
cached pinned memory, and CPU LOADS bypass the cache hierarchy (each load is an uncached read; sequential
reads run at a fraction of cached-DRAM bandwidth). So the flag is right for one direction of one party and
wrong for the other: the DMA in both directions is unaffected (a D2H writes it, an H2D reads it), and every
CPU read of a plane pays.

## The CPU reads of a contract plane over its lifetime (door ON)

| When | Read | Where | Bytes per demote or promote of one ~160 MB entry |
|---|---|---|---|
| Demote, at D2H completion | the engine's completion checksum, `checksum(item.host.bytes())` | `tier_transfer.rs:722-724` (`progress`) | every plane once (34 items on the spec surface) |
| Demote, at bind | `bind_tier_image`'s `StateBundle` checksum per segment, compared with the receipt | `worker.rs` `bind_tier_image` (`checksum(bytes)` per `Role::Key`/`Value`/`Draft`) | every plane again (day 15 finding 2: two hashes per plane, not three) |
| Promote (Option C), at H2D completion | the engine's completion checksum of the H2D SOURCE, `checksum(item.host.bytes())`: the same `progress` code, the source twin is the WC lease | `tier_transfer.rs:722-724` | every plane once more; this is the byte attestation `Completion::require` checks against the D2H receipt before publication |
| Promote, OFF program under ON (days 13 to 15) | none: `htod_u8_into(&bytes[..kb])` is a DMA read of the WC pointer | `worker.rs plane_up` | 0 (the day-15 promote delta is the inline demote inside the promote's timing window, `HOSTPREFIX-DOOR.md` "The promote's timing window") |
| `MEMRA_KV_HOST_FAULT=flip-demote` | `flip_first_byte`: read, flip, `write` | `worker.rs HostPlaneBytes::flip_first_byte` | diagnostic only, gate box |

Under OFF none of these reads exist (no receipt, no bind hash: `bind_tier_image` runs only with a tier), so
the OFF arm has zero CPU reads of a plane at demote and zero at promote. The door's byte attestation costs
CPU reads by design; the flag decides whether they run at cached or write-combined speed.

## What a decision needs (the decide-by review, 2026-10-05)

1. **The wall-time cost on the target card, N>=5, both orders, one window** (this day's cell, below): the
   demote and promote lines OFF versus ON for the same entry. The demote delta is the two WC hashes plus the
   ticket lifecycle; the promote delta under Option C is the engine's WC hash of the source plus the ticket
   lifecycle, separated from the inline demote by pairing each promote line with the demote line printed
   inside its window.
2. **A hash-speed micro-cell** (not today): SHA-256 GB/s on this host over a cached pinned buffer versus a
   write-combined one of the same size, so the WC share of the delta is a measured number rather than the
   whole delta minus the ticket lifecycle.
3. **The engine's arm**: a cached-pinned `alloc_host` (an engine-owned pinned backing with `cuMemHostAlloc(..,
   0)` or `CU_MEMHOSTALLOC_PORTABLE` inside `PinnedAllocation`), measured the same way, both rigs (the
   per-hardware rule). Alternatives the review weighs against it: one hash per plane at demote (bind takes
   the engine's receipt as its bundle checksum instead of recomputing, which removes one WC read at the cost
   of bind's independent recomputation), or the documented cost with the flag kept (WC is the documented
   choice for the H2D source direction).

None of these is this lane's call; this file carries the mechanism and the numbers.

## Cell design (day 16, `pro-single-day16/wc-pair/`, script `wc-cell.sh`, analysis `wc-pair.py`)

One collector cell (`tools/tier-battery.py --rig pro-single`, `/tmp/memra-gpu.lock` held once for the whole
cell, 250 ms `nvidia-smi` telemetry), four server boots in the order OFF, ON, ON, OFF (both orders of the
pair), the same binary, `MEMRA_PREFIX_CACHE_MB=256` (the device holds one ~160 MB entry), `MEMRA_KV_HOST_MB=
8192`, the gates' default spec environment (every entry draft-bearing, 34 items), `MEMRA_KV_HOST_VERIFY`
unset (the production promote shape). Per boot seven requests over two prompts: r1 P_A seeds E_A; r2 P_B
seeds E_B and evicts E_A (the first demote); r3..r7 alternate P_A and P_B, each a host hit that promotes one
entry and whose insert evicts the other (a demote inside the promote's window). Observations per arm per
order: demotes r2..r6 (N=5), promotes r3..r7 (N=5), from the `[prefix-host] demote: .. in X ms` and `promote:
.. in X ms` lines; the promote's inline demote is the `demote:` line printed inside the same request, so
`promote_excl = promote_ms - inline_demote_ms`. Reported: per-order medians (N=5), pooled medians (N=10),
min and max, and the telemetry regime (temperature, power draw, SM clock) over the cell and per arm window
(`marks.tsv`). Single card, one window, executed-not-qualified: the first cell, not a verdict.

## Results

(filled after the cell runs; see the section appended below)
