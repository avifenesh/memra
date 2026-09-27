# Write-combined contract destinations: mechanism, code sites, measurement plan (day 16)

Status: a FINDING of day 15 (`DAY15.md` finding 1) with its first measured cell on day 16, reported as the
first cell of the `MEMRA_KV_HOST_CONTRACTS` decide-by review (2026-10-05), not as a verdict. The allocation
flag is engine territory (`crates/memra-engine/src/tier_transfer.rs`) and is NOT changed by this lane. Every
`file:line` is in tree `25891baa9`; `core.rs` is cudarc 0.19.8 `src/driver/safe/core.rs`.

**Superseded on the target card class (day 17, closed by pointer).** The decision this file asked for
was taken by lane A: `docs/decisions/PINNED-DESTINATIONS.md` (lead ruling 23). `PinnedKind::for_device`
returns `Cached` for the RTX PRO 6000 Blackwell class (day-13 receipt: bind hash 77.6 against 1711 ms
write-combined at 160 MiB, D2H not above at every pair) and `WriteCombined` for the RTX 5090 class and
every unrecognized name (its cell inconclusive under the rule as pre-registered); `alloc_host` takes the
resolved kind, no environment variable exists, `alloc_host_kind` is the gate's measurement seam. So
sections "Mechanism" and "The CPU reads of a contract plane" below describe the destinations as they were
in tree `25891baa9`; on the target card class the contract's destinations are now cacheable and the
"Results" numbers (demote 37.8 against 169.2 ms and the rest) are the WRITE-COMBINED cost, kept as the
record, not the door's current cost there. What the door's decide-by review still owes (`HOSTPREFIX-
DOOR.md` Status): the same pair cell on a binary carrying `for_device` (the cost with cached destinations)
and the hash-speed micro-cell; items 2 and 3 of "What a decision needs" are otherwise answered by lane A's
record (item 3 by the decision, item 2 by the pinned-ab cell's engine-hash and bind-hash columns).

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

## Results (day 16, `pro-single-day16/wc-pair2-retry3/` capture, `wc-pair2/ev/` evidence; replay `wc-pair.py`: `WC PAIR REPLAY: PASS`, 12 checks)

The first cell of the decide-by review, not a verdict. One RTX PRO 6000 Blackwell Server Edition at its
600 W limit, binary `cd8e9c11...` (tree `25891baa9`), one collector lock hold (the runner waited three
120 s retries behind lane B's campaign, then held the lock for the whole 64 s window), four boots OFF, ON,
ON, OFF, seven requests each, `MEMRA_KV_HOST_VERIFY` unset, default spec environment (34 items per batch).
Regime over the window (257 samples at 250 ms, `command.gpu.csv`): temperature 37 to 51 C, power draw at
most 492 W under the 600 W cap, SM clock 180 (idle between boots) to 2422 MHz. `executed-not-qualified`.

| Line | OFF, order 1 (N=5) | OFF, order 2 (N=5) | ON, order 1 (N=5) | ON, order 2 (N=5) | pooled OFF (N=10) | pooled ON (N=10) |
|---|---|---|---|---|---|---|
| `demote:` ms, r2..r6 | median 37.8 (6.1 to 42.9) | 37.8 (6.1 to 42.7) | 169.2 (136.1 to 174.6) | 169.2 (135.7 to 174.3) | **37.8** | **169.2** |
| `promote:` ms, r3..r7 (the window contains the inline demote) | 12.2 (10.5 to 47.4) | 12.3 (10.6 to 47.2) | 172.2 (169.6 to 207.3) | 171.2 (168.9 to 206.7) | **12.2** | **171.7** |
| promote minus its inline demote, ms | 4.5 (4.4 to 4.5) | 4.5 (4.5 to 4.6) | 33.4 (32.7 to 33.7) | 33.2 (32.4 to 33.4) | **4.5** | **33.2** |

Raw sequences, verbatim from the logs (ms), the two orders agree to within 0.5 ms on every position:

| Arm | demotes r2..r7 | promotes r3..r7 |
|---|---|---|
| o1-off | 37.8, 41.8, 42.9, 6.1, 7.7, 6.1 | 46.3, 47.4, 10.5, 12.2, 10.5 |
| o2-off | 37.8, 42.2, 42.7, 6.1, 7.8, 6.2 | 46.8, 47.2, 10.6, 12.3, 10.7 |
| o1-on | 169.2, 171.7, 174.6, 136.1, 139.5, 135.9 | 205.1, 207.3, 169.7, 172.2, 169.6 |
| o2-on | 169.2, 171.4, 174.3, 135.7, 138.8, 135.5 | 204.6, 206.7, 169.0, 171.2, 168.9 |

What the sequences show, stated as observations:

1. **A first-touch step in BOTH arms, the same size.** The first three demotes of every boot (r2, r3,
   r4) cost about 35 ms more than the last three (r5, r6, r7): 38 to 43 against 6 to 8 ms OFF, 169 to
   175 against 136 to 140 ms ON. The host tier grows by one fresh 160 MB pinned region per demote until
   r4 (A, B, then A's replacement is allocated before the old twin drops); from r5 the replacement drops a
   region the next `cuMemHostAlloc` can reuse. The step is the cost of page-locking and zero-filling fresh
   host pages (`PinnedHostBuf::new` OFF, `alloc_host` ON), equal in both arms: not a door effect and not a
   WC effect. The medians above straddle it (three slow, two fast in r2..r6); the steady-state pair is the
   r5..r7 rows.
2. **The steady-state demote delta is about 130 ms, at first touch also about 130 ms.** ON minus OFF:
   136 to 140 against 6 to 8 (r5..r7), 169 to 175 against 38 to 43 (r2..r4). The delta is what the door
   adds at demote: the two CPU SHA-256 passes over the write-combined destination (the engine's completion
   checksum, bind's bundle checksum), the ticket lifecycle (fence, submit, per-item event sync, take,
   require, retire, acknowledge) and the receipt line. This cell does not split those; the hash-speed
   micro-cell does.
3. **The promote's own share is 4.5 ms OFF against 33 ms ON.** OFF's `htod_u8_into` is an asynchronous
   `cuMemcpyHtoDAsync` with no completion observation, so 4.5 ms is the allocations and the launch; the
   DMA completes inside the following inline demote's window (its D2H synchronizes the same stream). ON's
   33 ms includes the observed completion of the H2D (`synchronize(&ticket)`, about 160 MB over PCIe), the
   engine's SHA-256 over the write-combined source at completion, `require`, `ready_view` per item and the
   ticket's retirement. So the two numbers are not the same quantity; the honest comparison is the whole
   promote line with its inline demote, 12.2 against 171.7 ms (median, N=10), of which the inline demote
   carries 6 to 43 against 136 to 175.
4. **The gate cells' larger numbers (day 15 and this day's identity gate: demote about 150 and 278 ms,
   promote 266 and 423 ms) carry `MEMRA_KV_HOST_VERIFY=1`**, which digests the device entry at demote and
   again at promote (a D2H readback plus a hash of 160 MB each, in both arms); this cell runs without it,
   the production promote shape.

For the decide-by review, from this cell: the door's cost at demote is about 130 ms per 160 MB entry
(N=10, one card, one window) and at promote about 29 ms on the promote's own share plus the same demote
cost inside its window; a cached-pinned `alloc_host` arm would remove the WC share of the two demote-side
hashes and the one promote-side hash, and only the micro-cell says how much of the 130 that is.
