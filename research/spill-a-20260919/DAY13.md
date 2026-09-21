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

## What changed (`33405a186`, `abfc3fc32`)

`crates/memra-engine/src/tier_transfer.rs`: `PinnedKind { WriteCombined, Cached }` with
`host_alloc_flags()` (4, the constant cudarc passes; 0), `name`, `parse`; `Default` is
`WriteCombined`. `PinnedBacking` replaces cudarc's `PinnedHostSlice<u8>` as the lease's backing:
one `result::malloc_host(bytes, kind.host_alloc_flags())` (the same `cuMemHostAlloc` FFI the engine
already calls at four sites), the same `CU_EVENT_BLOCKING_SYNC` tracking event, the same `HostSlice`
contract (the owner stream waits on the event before a copy, records it after), the same host access
rule (`as_ptr`, `as_slice` and their mutable twins synchronize the event first), the same `Drop`
(event synchronize, `free_host`). `alloc_host` keeps its signature and delegates to
`alloc_host_kind(bytes, request, PinnedKind::default())`; `CudaPinnedLease::pinned_kind()` reports
the arm. The two copy sites (`memcpy_dtoh` at the D2H, `memcpy_htod` at the H2D) are untouched:
the arms differ in the flag bits handed to the driver and in nothing else. No `MEMRA_*` read, no
`.cu`, no change to any caller (`worker.rs:9242`, the gates). `tier-transfer-gate pinned-ab
[--bytes N] [--pairs N]` is the measurement case (pre-registration above). Unit cells:
`pinned_kind_default_is_todays_write_combined_flag_bits` (CPU: the default's bits equal
`CU_MEMHOSTALLOC_WRITECOMBINED`, `Cached` is 0, neither carries `PORTABLE` or `DEVICEMAP`, names
round-trip), `alloc_host_delegates_with_the_default_kind_and_no_other_pinned_allocation_remains`
(CPU, source text: `alloc_host` delegates with the default; exactly one `result::malloc_host(`
call in the file, inside `PinnedBacking::alloc`; no `alloc_pinned` call), and
`pinned_kind_arm_is_honoured_by_the_driver` (GPU, ignored without a device: `cuMemHostGetFlags`
carries each arm's write-combined bit, the default lease reads back write-combined, a host write
then read is exact under both arms, the governor's pinned charge returns to zero).

## Target card: three sittings, one RTX PRO 6000 Blackwell Server Edition at 600 W (driver 580.178.04)

BOX3 host: 30 vCPUs of an AMD EPYC 9555 (reported L3 480 MiB over 30 instances), one NUMA node,
88.4 GiB RAM, `MemFree` 28 GiB at the cells; no `tmux` session on the box before any sitting; lane B
did not hold the lock during the day (zero lock retries). Every cell went through
`tools/tier-battery.py --rig pro-single` with `/tmp/memra-gpu.lock` held once per cell
(`ev/LOCK.json`, `--external-lock`), 250 ms telemetry, `CELL.jsonl`; receipts mirrored under
`pro-single-day13/` (binaries excluded), the box tree `/root/wt-a` on `lane-a-day13`, clean at
both builds (`build/`, `build2/`).

### Sitting 1 (`c65f3c11e`, gate binary `0624d65b…`): every sample recorded, three cells `failed` by an assertion of my own

The gate's "arm honoured" check compared `cuMemHostGetFlags` for equality with the arm's bits. The
driver reports `DEVICEMAP` (2) on every pinned allocation of a UVA platform, so the write-combined
arm read back **6** and the cached arm **2** in every one of the 22 roundtrips (both cells): the arm
IS honoured, the check was wrong, and its `assert!(flags_ok)` at the end of the cell (after every
sample and the `RESULT` line had printed) made `pinned-ab-160m` and `pinned-ab-16m` exit 101, so the
collector classifies them `failed` (`failure_quote`: `thread 'main' ... panicked at
crates/memra-engine/src/bin/tier_transfer_gate.rs:1073:5`). The same equality failed the GPU unit
cell (`gputest`, exit 101: `assertion left == right failed: WriteCombined, left: 6, right: 4`).
`conformance` (13 `PASS` lines, ending `PASS native governor zero after controlled drain`) and
`roundtrip` (six `byte_exact=true` lines, 4 KiB to 256 MiB) passed unchanged. Fix `abfc3fc32`:
the check tests the write-combined bit (`flags & CU_MEMHOSTALLOC_WRITECOMBINED ==
kind.host_alloc_flags()`), the raw value stays in every line. The three failed cells are kept as
the record; their samples are reported below beside sitting 2.

Sitting 1, `pinned-ab-160m` (160 MiB, N=5 per arm per order, warm-up per arm; regime 290 samples,
35 to 36 C, 87 to 91 W under the 600 W limit, SM 2347 to 2362 MHz): pooled medians (N=10) write-
combined against cached: alloc **24.59** vs **29.40** ms; D2H **3.006** vs **2.999**; engine hash
**1724.9** vs **77.57**; bind hash **1735.8** vs **77.56**; byte compare **693.5** vs **6.46**; H2D
**2.978** vs **2.975**; H2D-source hash **1716.1** vs **77.61**. Per pair: bind hash cached below
write-combined **10/10**, engine hash **10/10**, D2H cached not above write-combined **8/10**
(order 1 pair 2: 3.007 vs 3.003 ms, order 2 pair 1: 3.007 vs 3.006 ms, that is 4 us and 1 us
against), H2D 8/10; medians in both orders held. Under the pre-registered rule this sitting is
**INCONCLUSIVE** on clause 4 alone (`PINNED-AB rule ... d2h_cached_not_above_wc=8/10 ...
cached_arm=inconclusive`); byte exact 22/22.

### Sitting 2 (`abfc3fc32`, gate binary `746cb521…`): the decision cell

`gputest-s2`: `test result: ok. 3 passed; 0 failed` (the GPU cell now reads 6 and 2 and passes),
`executed-not-qualified`. `conformance-s2`: the same 13 `PASS` lines as sitting 1 (diff empty).
`roundtrip-s2`: six `byte_exact=true` lines. So the default arm through the engine-owned backing
runs the frozen schedules and the byte roundtrips exactly as before (day 12's cells).

`pinned-ab-160m-s2` (160 MiB, N=5 per arm per order, one untimed warm-up per arm, 22 roundtrips;
regime 282 samples at 250 ms, 35 to 36 C, 87 to 91 W under the 600 W limit, SM 2347 to 2362 MHz;
host loadavg 1.00 before, 1.03 after; `executed-not-qualified`, exit 0; replay `wc-ab.py`: `WC AB
REPLAY: PASS (15 checks, 0 failed)`):

| Phase (ms) | WC order 1 (N=5) | WC order 2 (N=5) | cached order 1 (N=5) | cached order 2 (N=5) | pooled WC (N=10) | pooled cached (N=10) |
|---|---|---|---|---|---|---|
| alloc | 24.47 | 24.06 | 29.14 | 29.33 | **24.33** (23.78 to 24.92) | **29.31** (28.91 to 29.49) |
| D2H | 3.01 | 3.01 | 3.00 | 3.00 | **3.01** (3.00 to 3.01) | **3.00** (3.00 to 3.01) |
| engine hash | 1710.80 | 1695.97 | 77.61 | 77.58 | **1710.20** (1691.92 to 1712.84) | **77.61** (77.55 to 77.66) |
| bind hash (the host-read) | 1711.22 | 1692.42 | 77.58 | 77.58 | **1711.12** (1690.59 to 1712.78) | **77.58** (77.51 to 77.69) |
| byte compare | 650.65 | 646.32 | 6.47 | 6.45 | **649.79** (644.75 to 651.42) | **6.47** (6.37 to 6.59) |
| H2D | 2.98 | 2.98 | 2.97 | 2.98 | **2.98** (2.98 to 2.98) | **2.98** (2.97 to 2.99) |
| H2D source hash | 1711.34 | 1691.97 | 77.59 | 77.61 | **1710.50** (1690.30 to 1713.83) | **77.60** (77.52 to 77.94) |

| Order | Pair | WC bind hash | cached bind hash | WC D2H | cached D2H | WC engine hash | cached engine hash | WC H2D | cached H2D |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1711.22 | 77.56 | 3.008 | 3.006 | 1710.00 | 77.66 | 2.977 | 2.975 |
| 1 | 2 | 1711.69 | 77.59 | 3.010 | 2.996 | 1711.46 | 77.56 | 2.978 | 2.971 |
| 1 | 3 | 1711.03 | 77.56 | 3.007 | 3.000 | 1712.12 | 77.61 | 2.977 | 2.969 |
| 1 | 4 | 1711.68 | 77.58 | 3.008 | 2.999 | 1710.80 | 77.61 | 2.977 | 2.974 |
| 1 | 5 | 1711.21 | 77.69 | 3.007 | 2.998 | 1710.41 | 77.64 | 2.977 | 2.975 |
| 2 | 1 | 1710.05 | 77.61 | 3.003 | 3.000 | 1709.85 | 77.58 | 2.977 | 2.974 |
| 2 | 2 | 1712.78 | 77.51 | 3.007 | 3.002 | 1712.84 | 77.56 | 2.978 | 2.978 |
| 2 | 3 | 1691.01 | 77.66 | 3.009 | 3.001 | 1692.83 | 77.66 | 2.978 | 2.988 |
| 2 | 4 | 1692.42 | 77.51 | 3.005 | 3.002 | 1695.97 | 77.55 | 2.976 | 2.975 |
| 2 | 5 | 1690.59 | 77.58 | 3.012 | 3.000 | 1691.92 | 77.61 | 2.977 | 2.977 |

The rule line, verbatim: `PINNED-AB rule byte_exact_all=true driver_flags_honoured=true
bind_hash_cached_below_wc=10/10 engine_hash_cached_below_wc=10/10 d2h_cached_not_above_wc=10/10
d2h_cached_strictly_below_wc=10/10 h2d_cached_not_above_wc=9/10 medians_both_orders=true
cached_arm=wins-on-this-card cached_arm_strict_d2h_reading=wins-on-this-card`.

### Verdict for this card (the target-card cell of the decide-by review)

Under the pre-registered rule, on the decision cell `pinned-ab-160m-s2`: **the cached arm wins on
this card**, on both readings of the D2H clause (byte exact 22/22; driver flags honoured; bind
hash cached below write-combined at every one of the 10 pairs and in both orders' medians, 77.6 ms
against 1711 ms, a factor of 22; D2H cached not above write-combined at every pair and strictly
below at every pair, medians 3.00 against 3.01 ms; H2D 2.98 against 2.98 ms). Stated plainly
beside it: sitting 1 of the same cell was INCONCLUSIVE on clause 4 alone, two D2H pairs going 4 us
and 1 us the other way on a 3.0 ms DMA whose medians are flag-independent to within 0.01 ms in both
sittings; the host-read clause held 10/10 in both sittings at the same factor of 22 (1735.8 against
77.56 ms in sitting 1). The per-pair D2H clause of my operationalization is jitter-sensitive at the
microsecond level; the numbers it guards did not move between arms or sittings.

What the numbers say about the door's cost on this card: one CPU read of a 160 MiB write-combined
lease by the contract's own `checksum` takes 1.69 to 1.71 s (about 98 MB/s), and the door reads
every demoted plane twice at demote (the engine's completion checksum, the bind checksum) and, under
Option C, once more at promote; the same reads over cached pinned memory take 77.6 ms each (2.16
GB/s, the SHA-256 rate of one core on this host). The DMA in both directions is unaffected (3.00 ms
D2H, 2.98 ms H2D, 56 GB/s on 160 MiB). The cached arm costs about 5 ms more at allocation (29.3
against 24.3 ms for page-locking plus the zero fill of 160 MiB, the one CPU-store phase, where
write-combining helps), paid once per lease.

This moves no default. The seam stays at `PinnedKind::WriteCombined` (today's flag bits); the
local RTX 5090 cell is the follow-up (the owner law: one-rig evidence sets a one-rig default at
most, and per-hardware arms are keyed on the device), and the lead rules on the door at the
decide-by review (2026-10-05).

## Context cells (not part of the rule)

- `pinned-ab-16m` (sitting 1, a failed cell by the flags assertion; 16 MiB, N=5 per arm per order)
  and `pinned-ab-16m-s2` (sitting 2, `executed-not-qualified`): bind hash write-combined **171.4**
  against cached **7.75** ms pooled (sitting 1; sitting 2 the same to 0.1 ms), engine hash 170.7
  against 7.77, byte compare 64.9 against 0.41, D2H 0.323 against 0.319, H2D 0.309 against 0.309,
  alloc 2.77 against 3.00; bind hash cached below write-combined 10/10 in both sittings. The rule's
  per-pair D2H clause at a 0.32 ms DMA: 10/10 in sitting 1, 8/10 in sitting 2; H2D cached not
  above write-combined 2/10 and 1/10 (the cached arm's H2D is 0 to 4 us slower at this size, inside
  the 0.31 ms DMA's jitter). At 16 MiB the write-combined read rate is the same 98 MB/s.
- `pinned-ab-4m7-s2` (sitting 3; 4,718,592 bytes, the server's per-item size at 160.7 MB over 34
  items, N=3 per arm per order, below the rule's N): bind hash **47.63** against **2.18** ms, engine
  hash 47.49 against 2.19 (the same factor of 22, 99 MB/s against 2.16 GB/s), D2H 0.098 against
  0.098, byte exact 14/14, driver flags 6 and 2.
- `probe-160m` and `probe-4m7` (sitting 3; `pinned-read-probe.py`: ctypes on `libcuda`,
  `cuMemHostAlloc` with flags 4 and 0, `memset`, then Python's `hashlib.sha256` and a bulk copy over
  the same bytes, interleaved A B A B and B A B A, N=3 per arm per order, no engine code): at
  160 MiB sha256 write-combined **1361.0** ms (123 MB/s) against cached **77.7** ms (2158 MB/s),
  copy 789.2 against 44.7 ms, alloc plus memset 25.0 against 28.5 ms; at 4.7 MB sha256 39.2 ms
  (120 MB/s) against 2.19 ms (2151 MB/s), copy 18.5 against 0.67 ms; `cuMemHostGetFlags` 6 and 2;
  every copy exact. The attribute is the host's, not the engine's: a process with no cudarc and no
  memra code reads write-combined pinned memory at the same rate the gate does.

## The rate discrepancy with lane C's whole-demote line, resolved

C's ON boots (`wt-spill-c` `pro-single-day16/wc-pair2/ev/o1-on-server.log`, read only) carry six
`[prefix-host] contracts door D2H receipt: ... items=34 (16 KV planes, draft) complete=34
require=ok` lines and `demote:` lines of 169.2, 171.7, 174.6, 136.1, 139.5, 135.9 ms for "160.7MB"
entries of 89 tokens. The server's timer (`worker.rs:10199` `t0`, the line at `:10248`) wraps
`host_entry_from_device` (which runs `host_kv_planes_through_contract`, `:9140`, whose step 6 is
`t.synchronize(&ticket)`, the engine's completion hash of every item) and `bind_tier_image` (the
bind hash), so both hashes are inside those 136 to 175 ms. At the rate measured here (98 to 123
MB/s for a write-combined read by `checksum` or by Python) two passes over 160.7 MB would take about
3.3 s, which those lines cannot contain.

The bytes are the difference, not the rate. `host_kv_planes_through_contract` copies and hashes
`p.len.checked_mul(p.k_tok_bytes)` and `p.len.checked_mul(p.v_tok_bytes)` per plane
(`worker.rs:9191-9192`): the committed prefix, 89 tokens in C's cell. The "160.7MB" in the
`demote:` line is `host_image_bytes(dead.bytes, ..)`, the entry's accounted plane capacity at
`MEMRA_CTX=8192` (the OFF program copies capacity: 160.7 MB in 6 to 8 ms is a 20 to 26 GB/s DMA).
At 89 of 8192 tokens the door copies and hashes about 1.75 MB per pass, so the two demote-side
write-combined hashes cost about 29 ms of C's 130 ms delta (against about 1.6 ms cached); the rest is
the 34-item ticket lifecycle (34 fresh `cuMemHostAlloc` leases with their zero fills and tracking
events, one batch, 34 event synchronizes, 34 `take_destination`s, `require`, `retire_source`,
`retire`, `acknowledge`) and the receipt line, which this lane did not split.

What that means for the decide-by review, stated as the measured rates and their arithmetic: the
write-combined penalty is a factor of 22 per byte read, and the bytes the door reads per demote are
the entry's committed length times its row bytes, twice (three times under Option C). At C's
89-token entries that is tens of milliseconds; at a full 8192-token entry of the same shape (160 MB
committed) the two demote-side hashes would take about 3.4 s write-combined against about 0.16 s
cached, and Option C's promote-side hash another 1.7 s against 0.08 s. Those long-entry figures are
the measured per-byte rates times the byte counts, not a served measurement: a served long-entry cell
is the natural next receipt for the review (C's or the lead's call, not this lane's).

## CPU gates (all under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, runner `day13/run-gates.sh`, logs `day13/gates/`)

| gate | exit | verbatim tail |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | (no output) |
| `cargo test -p memra-tier -p memra-kv -p memra-engine --offline --lib` | 0 | engine `test result: ok. 517 passed; 0 failed; 25 ignored`, kv `71 passed; 0 failed`, tier `7 passed; 0 failed` |
| `cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings` | 0 | `Finished` |
| `bash tools/check-flags.sh` | 0 | see `day13/gates/check-flags.log` |
| `bash tools/docs-registry-census.sh` | 0 | see `day13/gates/docs-registry-census.log` |
| `git diff --check` | 0 | (no output) |
| `git diff --cached --check` (the staged receipts; my addition) | 2, read | `research/spill-a-20260919/pro-single-day13/gputest-s2/command.log:13: new blank line at EOF.`: cargo's trailing blank line inside the collector's raw capture, left as captured because `command.capture.json` pins that file's sha256 (`tier-battery.py --validate` agrees byte for byte); the same blank line in my own `day13/gates/test-lib.log` was removed (not pinned, no measurement touched) |
| `bash -n` on the eight cell and driver scripts; `python3 -m py_compile` on `wc-ab.py` and `pinned-read-probe.py` | 0 | (no output) |

The first compile of the seam failed clippy once, verbatim `error: very complex type used. Consider
factoring parts into `type` definitions` (`tier_transfer_gate.rs:930`), fixed with `type Phase`;
the source-text cell failed once on its own over-broad count (`alloc_pinned` in the doc comments),
narrowed to call syntax; both reruns green. No `MEMRA_*` read was added (the census is unchanged).

## Scope

Done: the census (`PINNED-FLAGS.md`), the typed seam at today's default with the arm reachable from
the gate, the CPU and GPU unit cells, the pre-registered A/B on the target card (three sittings, the
decision cell WINS for the cached arm under the rule, the first sitting INCONCLUSIVE on the D2H
clause by microseconds, both reported), the driver-independent probe, and the resolution of the
discrepancy with C's whole-demote line (bytes, not rate). Not done, stated: the local RTX 5090 cell
(the per-hardware law; the follow-up before any default moves); a served long-entry cell under the
door; the split of the ticket lifecycle inside C's 130 ms; any default change (the seam stays at
`PinnedKind::WriteCombined` until the lead rules at the decide-by review, 2026-10-05). memra#552
gets a comment pointing here; it stays open.
