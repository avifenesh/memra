# WP-A day 14: the 5090 cell and the per-device pinned destination default (ruling 22)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `446252336` (day 13; reached `main`
through integ20, #606, `main` `2a589903d`), merged `origin/main` `2a589903d` (`2b2191afe`). Lead
ruling 22 (`research/spill-lead-20260919/INTEGRATION-DAY12.md`), verbatim: "Cached pinned
destinations for the contract path on the target card class, pending the 5090 cell. The rule was
pre-registered and met 10/10 in both orders with byte exactness; the effect is a mechanism
(write-combined memory is slow to read back on the host, and the door reads every image back) and
22x. Per the one-rig rule the default flips for the RTX PRO 6000 class first: lane A day 14 runs the
same cell on the local 5090 and lands the per-device default (`PinnedKind::Cached` where the card
class has a receipt, `WriteCombined` elsewhere) with the FLAGS and decision records; the door's
decide-by review reads the door's cost again after that."

## Pre-registration of the 5090 cell (written and committed before any run on the card)

**Cell.** The day-13 decision cell, same shape: `tier-transfer-gate pinned-ab --bytes 167772160
--pairs 5` (160 MiB) through `tools/tier-battery.py --rig rtx5090` (one `/tmp/memra-5090.lock` hold
for the whole cell, the collector's 250 ms `nvidia-smi` sampler, `--external-lock` with the lock
proof in `ev/LOCK.json`), one process, one CUDA context. Arm A = `PinnedKind::WriteCombined`
(today's default, `cuMemHostAlloc` flags 4), arm B = `PinnedKind::Cached` (flags 0). One untimed
warm-up roundtrip per arm (A then B), then order 1: A B x 5; order 2: B A x 5. N=5 per arm per
order, N=10 pooled. The roundtrip's phases are the day-13 table (`DAY13.md` "The roundtrip"),
unchanged; the gate binary is built from this tree at `2b2191afe` (the merge commit; the commit that adds
this section changes no engine source: this file and the cell scripts only; `rtx5090-day14/build/`,
sha256 in every cell's `ev/binary.sha256`).

**Card.** The local RTX 5090 Laptop GPU (compute capability 12.0, 24463 MiB, driver 595.84). This
laptop reports no power limit (`nvidia-smi` `power.limit` is `[N/A]`; the collector records it as
unknown, never zero). The thermal regime is therefore the collector's sampler (temperature, power
draw, SM clock) with the missing limit stated, plus host `loadavg` before and after and `tmux ls`
before the sitting. Host: 24 CPUs of an Intel Core Ultra 9 275HX, 61 GiB RAM, one NUMA node. The
whole sitting runs inside `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G` (the
owner's local-rig cap; the gate is one process with one hashing thread, far below the quota).
Lane B may run cells on this card: the runner retries a busy lock 15 x 120 s and never kills a
holder.

**Rule (the lead's, verbatim, the same as day 13):** "the arm wins on the host-read and D2H medians
at every pair in both orders with byte exactness in all cells; otherwise inconclusive."
Operationalized exactly as `DAY13.md` "Operationalized, before the run" clauses 1 to 5 (byte exact
in every roundtrip, warm-ups included; the driver's write-combined bit equals the arm's at every
roundtrip; cached `bind_hash_ms` below write-combined at every one of the 10 pairs; cached `d2h_ms`
not above write-combined at every pair, the strict reading printed beside it; the per-order medians
of both). Comparisons at the timers' 1 us resolution, no rounding. The binary's `PINNED-AB rule`
line is the verdict; `wc-ab.py` recomputes it offline from the mirrored `command.log` and must
agree.

**What the verdict is and is not.** This is the RTX 5090 class cell of ruling 22. If the cached arm
WINS ON THIS CARD under the rule, the per-device default lands with the RTX 5090 class on
`PinnedKind::Cached` beside the RTX PRO 6000 Blackwell class (day 13's receipt). If INCONCLUSIVE,
the RTX 5090 class stays on `PinnedKind::WriteCombined` in the same per-device default, the
receipt is recorded, and what would move it is stated. Either way the RTX PRO 6000 Blackwell class
moves to `Cached` today on its own receipt (the lead's ruling), and every other card class stays
on `WriteCombined`. No cross-box timing comparison: two card-class verdicts, each on its own
receipts; the numbers of the two cards are never divided into each other.

**Context cells, not part of the rule.** `pinned-ab --bytes 16777216 --pairs 5` (16 MiB);
`tier-transfer-gate conformance` and `roundtrip` (today's default arm through the engine-owned
backing on this card: every `PASS` line and every `byte_exact=true` line); `cargo test --release
-p memra-engine --offline --lib tier_transfer -- --include-ignored` (the arm-honoured cell on this
card's driver: the `DEVICEMAP` bit is expected on every UVA platform).

**Then (task 2), after the verdict.** The default becomes a function of the device class, keyed on
the device name the way `parallel.rs` `HardwareTarget::from_device_name` keys the product shape
(`"RTX PRO 6000"` and `"Blackwell"`; `"RTX 5090"`): `Cached` where the class carries a receipt,
`WriteCombined` elsewhere, resolved once in `CudaTransfers::new` from the owner stream's context;
`alloc_host` takes it; no `MEMRA_*` read; `alloc_host_kind` stays the gate's measurement seam. The
second sitting on each card is `conformance` and `roundtrip` through the new default
(`byte_exact=true` on every roundtrip), with the resolved default printed by the gate.
