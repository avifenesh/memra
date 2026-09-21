# WP-A day 13: the write-combined contract destinations, a typed arm and the target-card A/B

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `8302f6b0a` (= origin, merged to
`main` by #597), merged `origin/main` `1b354be59` (`c2f13f6cc`, pushed through the whole hook
battery). Census `febd54201` (`PINNED-FLAGS.md`), seam `33405a186`; the receipts commit that
completes this file is the branch tip. Lane C's finding that opens the day:
`research/spill-c-20260919/WC-DESTINATIONS.md` and `DAY16.md` "The WC pair" (the contract's pinned
destinations are write-combined, so every CPU read of a demoted image runs uncached; demote median
37.8 ms OFF against 169.2 ms ON, steady state 6 to 8 against 136 to 140 ms, N=5 per arm per order,
one lock hold, one RTX PRO 6000 Blackwell).

## Pre-registration (written and committed before any run on the card)

**Cell.** `tier-transfer-gate pinned-ab --bytes 167772160 --pairs 5` (160 MiB, the ~160 MB entry
class C measured) on one RTX PRO 6000 Blackwell through `tools/tier-battery.py --rig pro-single`
(one `/tmp/memra-gpu.lock` hold for the whole cell, the collector's 250 ms `nvidia-smi` sampler),
one process, one CUDA context, the same governor. Arm A = `PinnedKind::WriteCombined` (today's
default, `cuMemHostAlloc` flags 4), arm B = `PinnedKind::Cached` (flags 0). One untimed warm-up
roundtrip per arm (A then B), then order 1: A B, A B, A B, A B, A B; order 2: B A, B A, B A, B A,
B A. N=5 per arm per order, N=10 pooled.

**The roundtrip** (`tier_transfer_gate.rs pinned_roundtrip`, the engine's own calls in the order
the door runs them; the device source is staged outside every timed phase):

| Phase | What is timed | Host read of the pinned bytes? |
|---|---|---|
| `alloc_ms` | `alloc_host_kind`: `cuMemHostAlloc` with the arm's flags, the tracking event, the zero fill (CPU stores) | stores only |
| `d2h_ms` | `d2h(CopyOp)` submit to the owner stream's `synchronize()`: the DMA into the pinned destination | no |
| `engine_hash_ms` | `synchronize(&ticket)` on the already complete copy: `progress`'s SHA-256 over the destination (the engine's completion checksum, the first read the door makes at demote) | yes |
| `bind_hash_ms` | `checksum(host.bytes())` over the taken destination: the read `bind_tier_image` does (the second) | yes: **the host-read the rule names** |
| `compare_ms` | `host.bytes() == pattern` (the byte-exactness check, a memcmp) | yes, reported, not in the rule |
| `h2d_ms` | `h2d(CopyOp)` submit to `synchronize()`: the DMA out of the pinned source | no (DMA read) |
| `source_hash_ms` | `synchronize(&ticket)` on the complete H2D: the engine's checksum over the SOURCE (Option C's promote-side read) | yes, reported |
| `driver_flags` | `cuMemHostGetFlags` on the lease's pointer | (the driver's record of the arm) |
| `byte_exact` | D2H destination equals the pattern and its checksum, AND the H2D destination read back equals the pattern | |

**Rule (the lead's, verbatim):** "the arm wins on the host-read and D2H medians at every pair in
both orders with byte exactness in all cells; otherwise inconclusive."

**Operationalized, before the run.** The cached arm WINS ON THIS CARD iff all of:

1. `byte_exact=true` in every roundtrip, warm-ups included (22 of 22).
2. `driver_flags` equals the arm's flag bits in every roundtrip (4 for A, 0 for B).
3. At every one of the 10 pairs (5 per order), cached `bind_hash_ms` < write-combined
   `bind_hash_ms`.
4. At every one of the 10 pairs, cached `d2h_ms` <= write-combined `d2h_ms`. The DMA is
   predicted flag-independent (the attribute is the CPU's, `PINNED-FLAGS.md` section 4), so the
   D2H clause is a no-regression clause: a cached regression there is a loss. The strict reading
   (cached `d2h_ms` < write-combined at every pair) is evaluated and printed beside it as
   `cached_arm_strict_d2h_reading`, so the lead can apply either reading to the same numbers.
5. In each order, the cached median (N=5) of `bind_hash_ms` is below the write-combined median,
   and the cached median of `d2h_ms` is not above it.

Otherwise INCONCLUSIVE. Comparisons are at the timers' 1 us resolution; no rounding before the
comparison. Every median is printed with its N; the regime is the collector's sampler
(temperature, power draw, SM clock, the 600 W limit) plus host `loadavg` before and after and
`tmux ls` before the sitting. The binary prints one `PINNED-AB rule ...` line and one `RESULT`
JSON with every sample; `wc-ab.py` (offline) recomputes the medians and the rule from the
mirrored `command.log` and must agree with the binary.

**What the verdict is and is not.** The target-card cell of the `MEMRA_KV_HOST_CONTRACTS`
decide-by review (2026-10-05). A win here moves no default: the owner law needs the balanced A/B on
each card class the default applies to (the local RTX 5090 cell is the follow-up, stated), and
the lead rules on the door. The seam stays at today's flag bits until then.

**Context cells, not part of the rule.** `pinned-ab --bytes 16777216 --pairs 5` (16 MiB, a
small-plane class); `tier-transfer-gate conformance` and `roundtrip` (the default arm through the
engine-owned backing: every `PASS` line and every `byte_exact=true` line as on day 12, the
refactor's regression check); `cargo test --release -p memra-engine --lib tier_transfer --
--ignored` (`pinned_kind_arm_is_honoured_by_the_driver` on the card).
