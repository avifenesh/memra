# Pinned destinations of the contract path: cached or write-combined, per device (2026-09-21)

**Status:** decided and landed as a per-device naked default (lane `lane/spill-a-20260919` day 14,
lead ruling 22 in `research/spill-lead-20260919/INTEGRATION-DAY12.md`). `PinnedKind::for_device`
(`crates/memra-engine/src/tier_transfer.rs`) returns `Cached` for the RTX PRO 6000 Blackwell class
and `WriteCombined` for the RTX 5090 class and for every class without a receipt. No environment
variable selects the arm; the transfer gate's `alloc_host_kind` is the measurement seam. Receipts:
`research/spill-a-20260919/DAY13.md` (target card) and `DAY14.md` (local 5090), census
`research/spill-a-20260919/PINNED-FLAGS.md`.

## Question

The host-tier contracts door (`MEMRA_KV_HOST_CONTRACTS=1`, decide-by 2026-10-05) demotes KV planes
into pinned host leases the engine allocates through `CudaTransfers::alloc_host`, the one pinned
allocation site of the contract path. Until day 13 that site took cudarc's `alloc_pinned`, which
hard-codes `cuMemHostAlloc(.., CU_MEMHOSTALLOC_WRITECOMBINED)`. Write-combined pages suit CPU
stores and DMA in both directions and make every CPU READ of the bytes uncached, and the door reads
every demoted plane on the CPU twice at demote (the engine's completion checksum, the bind
checksum) and, under lane C's Option C, once more at promote. Every other pinned allocation in the
engine chooses its flags by hand and already follows the rule "cached for CPU reads, write-combined
for H2D-only streams" (`PINNED-FLAGS.md` section 1). Which attribute should the contract path's
destinations carry, and on which cards?

## The rule (the lead's, pre-registered before any run)

"the arm wins on the host-read and D2H medians at every pair in both orders with byte exactness in
all cells; otherwise inconclusive." Operationalized in `DAY13.md`: byte exact in every roundtrip,
warm-ups included; the driver's write-combined bit equals the arm's at every roundtrip; cached
`bind_hash_ms` below write-combined at every one of the 10 pairs; cached `d2h_ms` not above
write-combined at every pair; the per-order medians of both. Cell: `tier-transfer-gate pinned-ab
--bytes 167772160 --pairs 5` (160 MiB, the entry class lane C measured), one process, one CUDA
context, one collector lock hold, one untimed warm-up per arm, then A B x 5 and B A x 5 (A =
write-combined, B = cached), N=5 per arm per order, N=10 pooled, the collector's 250 ms telemetry;
`wc-ab.py` replays the rule offline from the mirrored log and must agree with the binary.

## Measured: two card classes, each on its own receipts

The two cards' numbers are never divided into each other; each verdict stands on its own cell.

### RTX PRO 6000 Blackwell (one card, 600 W; `DAY13.md`, `pro-single-day13/pinned-ab-160m-s2`)

Regime: 282 samples at 250 ms, 35 to 36 C, 87 to 91 W under the 600 W limit, SM 2347 to 2362 MHz;
host loadavg 1.00 before, 1.03 after. Pooled medians (N=10), write-combined against cached: alloc
24.33 against 29.31 ms; D2H 3.01 against 3.00; engine hash 1710.20 against 77.61; bind hash (the
host-read) 1711.12 against 77.58; byte compare 649.79 against 6.47; H2D 2.98 against 2.98; H2D
source hash 1710.50 against 77.60. Per pair: bind hash cached below 10/10, D2H cached not above
10/10 and strictly below 10/10, H2D not above 9/10; byte exact 22/22; driver flags 6 and 2.

Verdict line, verbatim: `PINNED-AB rule byte_exact_all=true driver_flags_honoured=true
bind_hash_cached_below_wc=10/10 engine_hash_cached_below_wc=10/10 d2h_cached_not_above_wc=10/10
d2h_cached_strictly_below_wc=10/10 h2d_cached_not_above_wc=9/10 medians_both_orders=true
cached_arm=wins-on-this-card cached_arm_strict_d2h_reading=wins-on-this-card`.

Stated beside it: a first sitting of the same cell (`pinned-ab-160m`) was `inconclusive` on the
D2H clause alone, 8/10, two pairs 4 us and 1 us the other way on a 3.0 ms DMA; the host-read
clause held 10/10 at the same factor of 22 in both sittings.

### RTX 5090 (one RTX 5090 Laptop GPU, no power limit reported; `DAY14.md`, `rtx5090-day14/pinned-ab-160m`)

Regime: 240 samples at 250 ms, 55 to 57 C, power draw 28 to 29 W (`power.limit` `[N/A]`,
`power.max_limit` 175 W), SM 1590 to 1627 MHz; host loadavg 1.12 before, 1.25 after; the sitting
inside `systemd-run --scope -p CPUQuota=1200% -p MemoryMax=28G`. Pooled medians (N=10),
write-combined against cached: alloc 31.91 against 40.79 ms; D2H 7.27 against 7.34; engine hash
1446.33 against 36.77; bind hash 1449.02 against 37.53; byte compare 578.58 against 10.12; H2D
6.04 against 6.04; H2D source hash 1447.36 against 36.73. Per pair: bind hash cached below 10/10,
D2H cached not above 5/10 (order 1 medians 7.28 against 7.46, order 2 7.27 against 7.33: cached
above in both), H2D not above 6/10; byte exact 22/22; driver flags 6 and 2.

Verdict line, verbatim: `PINNED-AB rule byte_exact_all=true driver_flags_honoured=true
bind_hash_cached_below_wc=10/10 engine_hash_cached_below_wc=10/10 d2h_cached_not_above_wc=5/10
d2h_cached_strictly_below_wc=5/10 h2d_cached_not_above_wc=6/10 medians_both_orders=false
cached_arm=inconclusive cached_arm_strict_d2h_reading=inconclusive`.

The 16 MiB context cell on the same card (`pinned-ab-16m`, N=5 per arm per order) sharpens the
D2H clause: cached D2H 0.74 against write-combined 0.70 ms with non-overlapping ranges (0.73 to
0.75 against 0.69 to 0.72), 0/10 pairs; bind hash 3.55 against 144.72 ms, 10/10. On this host the
DMA into cacheable pinned pages costs about 5 % at 16 MiB and about 1 % at 160 MiB (inside that
size's 6.66 to 7.78 ms spread); on the EPYC host of the target card it cost nothing measurable.
The host-read clause holds on both cards by a factor of 22 (target card) and 39 (this card).

## The decision

`PinnedKind::for_device(name)`: `Cached` where the card class carries a receipt that meets the rule
(the RTX PRO 6000 Blackwell class, day 13), `WriteCombined` on the RTX 5090 class (its cell is
inconclusive under the rule as pre-registered) and on every class without a receipt. The key is the
device name exactly as `parallel::HardwareTarget::from_device_name` keys the product shape
(`"RTX PRO 6000"` with `"Blackwell"`; `"RTX 5090"`), so the two tables cannot drift; compute
capability alone cannot separate the two classes (both are 12.0). `CudaTransfers::new` resolves it
once from the owner context's device name; `alloc_host` takes it; `alloc_host_kind` stays the
gate's explicit arm. The flag bits per kind are unchanged (4 and 0); the copy program is unchanged
(the same `memcpy_dtoh` and `memcpy_htod` on the same owner stream); byte exactness is checked on
every roundtrip in both arms.

Cells: `pinned_kind_per_device_default_resolves_by_card_class` (CPU: the receipted class resolves
to `Cached`, the RTX 5090 class and unknown names to `WriteCombined`, the elsewhere arm is the
enum's `Default`), `pinned_kind_default_is_todays_write_combined_flag_bits` (the bits per kind),
`alloc_host_delegates_with_the_default_kind_and_no_other_pinned_allocation_remains` (source text:
one resolution site, no environment read), `pinned_kind_arm_is_honoured_by_the_driver` (GPU: the
default lease reads back the card's resolved arm from `cuMemHostGetFlags`), and
`tier-transfer-gate conformance` and `roundtrip` through the new default on both cards, which print
`PINNED-DEFAULT device=".." kind=.. flags=..` once and `PINNED-DEFAULT roundtrip bytes=.. kind=..
driver_flags=..` per size beside every `byte_exact=true` line (`DAY14.md`).

## Rejected

- **An environment door.** The door hygiene rule: a winner is a naked default and a per-device
  question is answered by detection, not by a flag; the gate's arm argument is the measurement
  seam and the rollback is `git revert`.
- **One global flip to cached.** The RTX 5090 class does not carry a receipt under the rule, and
  its 16 MiB cell shows a real D2H cost for cached pages on that host. One-rig evidence sets at most
  one rig's default.
- **Keying on compute capability.** Both classes are sm_120; the name is the attribute the engine
  already keys product shape on.

## What would reverse or extend it

- **RTX 5090 class to `Cached`:** a cell on that class that meets the rule (the D2H clause at
  every pair), or a lead ruling that weighs the D2H clause against the host-read for the door's
  demote (at 160 MiB on this card one demote under the door pays one D2H and two host reads:
  write-combined 7.27 + 2 x 1449 ms, cached 7.34 + 2 x 37.5 ms; the rule as pre-registered does not
  trade one clause for the other, and it is applied as written).
- **RTX PRO 6000 Blackwell class back to `WriteCombined`:** a D2H or H2D regression on that class
  in a later sitting of the same cell, or a served long-entry cell under the door that shows the
  cached arm losing on a whole demote.
- **The door's decide-by review (2026-10-05):** if the door is removed, the contract path's CPU
  reads go with it and the leases are DMA-only again, the direction write-combined suits; the
  default is then re-examined, not carried.
