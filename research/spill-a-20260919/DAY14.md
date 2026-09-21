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

## The 5090 cell: one sitting, one RTX 5090 Laptop GPU (driver 595.84), `cached_arm=inconclusive`

Host regime: 24 CPUs of an Intel Core Ultra 9 275HX, 61 GiB RAM (`MemAvailable` 52 GiB), one NUMA
node; loadavg 1.12 before and 1.25 after the decision cell; seven unrelated `tmux` sessions on the
laptop before the sitting (none holding the card: zero compute processes, 15 MiB used, the lock
free; `driver.log`); zero lock retries. Every cell went through `tools/tier-battery.py --rig
rtx5090` with `/tmp/memra-5090.lock` held once per cell (`ev/LOCK.json`, `--external-lock`), 250 ms
telemetry, `CELL.jsonl`, all five captures `executed-not-qualified`, exit 0, `--validate` rc=0
(`day14/collector-validate-*.log`). The card reports no power limit: the collector's
`gpu_power_limits` records `power.limit` `[N/A]` and `power.max_limit` 175 W (`ev/card.csv`: default
limit 95 W, enforced limit 175 W as `nvidia-smi` prints them; the limit is unknown to the sampler,
never zero). Gate binary `94a1c8e05fc0c33b…` built at `2b2191afe` (the pre-default tree,
`rtx5090-day14/build/`), sitting tree `42f66f8d2`.

Context cells first: `gputest` `test result: ok. 3 passed; 0 failed; 0 ignored` (the arm-honoured
cell on this driver reads 6 and 2, as on the target card); `conformance` 13 `PASS` lines ending
`PASS native governor zero after controlled drain`; `roundtrip` six `byte_exact=true` lines (4 KiB to
256 MiB), today's write-combined default through the engine-owned backing on this card.

`pinned-ab-160m` (160 MiB, N=5 per arm per order, one untimed warm-up per arm, 22 roundtrips;
regime 240 samples at 250 ms, 55 to 57 C, power draw 28 to 29 W, limit `[N/A]`, SM 1590 to
1627 MHz; `wc-ab.py` replay `day14-wc-ab-160m.log`):

| Phase (ms) | WC order 1 (N=5) | WC order 2 (N=5) | cached order 1 (N=5) | cached order 2 (N=5) | pooled WC (N=10) | pooled cached (N=10) |
|---|---|---|---|---|---|---|
| alloc | 31.09 | 32.74 | 42.65 | 38.94 | **31.91** (30.30 to 33.54) | **40.79** (36.85 to 44.97) |
| D2H | 7.28 | 7.27 | 7.46 | 7.33 | **7.27** (7.17 to 7.78) | **7.34** (6.66 to 7.77) |
| engine hash | 1443.67 | 1446.54 | 36.84 | 36.69 | **1446.33** (1441.60 to 1450.10) | **36.77** (36.07 to 38.57) |
| bind hash (the host-read) | 1449.67 | 1447.83 | 36.57 | 38.28 | **1449.02** (1444.90 to 1455.39) | **37.53** (36.40 to 41.44) |
| byte compare | 578.24 | 579.65 | 10.14 | 10.09 | **578.58** (570.87 to 580.08) | **10.12** (9.36 to 11.13) |
| H2D | 6.04 | 6.04 | 6.04 | 6.04 | **6.04** (6.03 to 6.06) | **6.04** (6.03 to 6.04) |
| H2D source hash | 1446.47 | 1448.25 | 36.66 | 36.79 | **1447.36** (1438.77 to 1450.85) | **36.73** (36.34 to 40.83) |

| Order | Pair | WC bind hash | cached bind hash | WC D2H | cached D2H | WC engine hash | cached engine hash | WC H2D | cached H2D |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1455.39 | 36.41 | 7.279 | 7.462 | 1443.67 | 36.84 | 6.050 | 6.044 |
| 1 | 2 | 1449.67 | 36.57 | 7.551 | 6.661 | 1450.10 | 37.70 | 6.036 | 6.041 |
| 1 | 3 | 1453.07 | 37.75 | 7.779 | 7.192 | 1441.60 | 36.26 | 6.030 | 6.045 |
| 1 | 4 | 1449.54 | 39.17 | 7.223 | 7.540 | 1446.79 | 37.59 | 6.042 | 6.035 |
| 1 | 5 | 1446.53 | 36.40 | 7.181 | 7.768 | 1443.21 | 36.65 | 6.063 | 6.034 |
| 2 | 1 | 1447.83 | 41.44 | 7.270 | 7.357 | 1444.48 | 36.69 | 6.040 | 6.030 |
| 2 | 2 | 1445.27 | 40.69 | 7.175 | 7.170 | 1446.54 | 37.04 | 6.046 | 6.036 |
| 2 | 3 | 1444.90 | 37.32 | 7.609 | 7.239 | 1446.12 | 38.57 | 6.038 | 6.037 |
| 2 | 4 | 1449.40 | 37.12 | 7.441 | 7.425 | 1448.38 | 36.07 | 6.027 | 6.030 |
| 2 | 5 | 1448.64 | 38.28 | 7.201 | 7.327 | 1447.28 | 36.38 | 6.027 | 6.041 |

The rule line, verbatim: `PINNED-AB rule byte_exact_all=true driver_flags_honoured=true
bind_hash_cached_below_wc=10/10 engine_hash_cached_below_wc=10/10 d2h_cached_not_above_wc=5/10
d2h_cached_strictly_below_wc=5/10 h2d_cached_not_above_wc=6/10 medians_both_orders=false
cached_arm=inconclusive cached_arm_strict_d2h_reading=inconclusive`.

The replay, verbatim: `per pair: bind 10/10 cached<wc; engine 10/10 cached<wc; d2h 5/10 cached<=wc,
5/10 cached<wc; h2d 6/10 cached<=wc`, `cached arm on this card: INCONCLUSIVE (pre-registered rule);
strict D2H reading: INCONCLUSIVE`, `ok   replay agrees with the binary's verdict: inconclusive`, and
its summary `WC AB REPLAY: FAIL (15 checks, 2 failed)`: the two "failed" checks are the replay's
rule clauses 4 and 5 (`FAIL rule 4: cached D2H not above write-combined at every pair: 5/10`,
`FAIL rule 5: medians in both orders (bind below, D2H not above)`), which is the inconclusive
verdict restated, not an integrity failure; all 13 integrity and agreement checks passed (`ok`).
The script's summary line counts rule clauses as checks; it was written on day 13 and is not
changed after the run.

### Verdict for this card class

Under the pre-registered rule, on `pinned-ab-160m`: **the cached arm is INCONCLUSIVE on this card**,
on both readings of the D2H clause. Byte exact 22/22; driver flags honoured (6 and 2); the host-read
clause holds at every one of the 10 pairs and in both orders' medians (bind hash 37.5 against
1449 ms pooled, a factor of 39; this host's one-core SHA-256 runs at 4.5 GB/s over cached pinned
memory and at 116 MB/s over write-combined). The D2H clause fails: cached D2H is not above
write-combined at 5 of 10 pairs and the cached median is above in both orders (7.46 against 7.28 ms
in order 1, 7.33 against 7.27 in order 2; pooled 7.34 against 7.27, on a per-pair spread of 6.66 to
7.78 ms). The 16 MiB context cell makes the same clause sharper on this host: cached D2H 0.74
against write-combined 0.70 ms with non-overlapping ranges (0.73 to 0.75 against 0.69 to 0.72),
0/10 pairs, while its bind hash is 3.55 against 144.72 ms (10/10) and H2D 0.61 against 0.63
(`pinned-ab-16m`, regime 26 samples, 57 C, 28 to 29 W, SM 1590 to 1597 MHz; `day14-wc-ab-16m.log`,
its rule line `... d2h_cached_not_above_wc=0/10 d2h_cached_strictly_below_wc=0/10
h2d_cached_not_above_wc=9/10 medians_both_orders=false cached_arm=inconclusive ...`). So on this
host the DMA into cacheable pinned pages costs about 5 % at 16 MiB and about 1 % at 160 MiB, inside
the larger size's jitter; on the target card's host it cost nothing measurable (day 13, both
sittings, medians flag-independent to 0.01 ms). Two card-class verdicts on their own receipts; no
number of one card is divided into a number of the other.

What follows under ruling 22, as pre-registered: the RTX 5090 class stays on
`PinnedKind::WriteCombined` in the per-device default; the RTX PRO 6000 Blackwell class moves to
`PinnedKind::Cached` on its day-13 receipt; every other class stays on `WriteCombined`. What would
move the RTX 5090 class is in `docs/decisions/PINNED-DESTINATIONS.md` (a cell on the class that
meets the D2H clause, or a lead ruling that weighs the clause against the host-read: at 160 MiB one
demote under the door pays one D2H and two host reads, write-combined 7.27 + 2 x 1449 ms against
cached 7.34 + 2 x 37.5 ms on this card; the rule does not trade one clause for the other and it is
applied as written).

## The per-device default (`4488d83fe`)

`crates/memra-engine/src/tier_transfer.rs`: `PinnedKind::for_device(name) -> PinnedKind` returns
`Cached` for `parallel::HardwareTarget::RtxPro6000Blackwell` and `WriteCombined` for
`HardwareTarget::Rtx5090` and for every name `HardwareTarget::from_device_name` refuses (made
`pub(crate)` in `parallel.rs`, the one name table: `"RTX PRO 6000"` with `"Blackwell"`, `"RTX
5090"`). `CudaTransfers` carries `pinned_default`, resolved once in `new` from
`owner.context().name()` (an unnamed device resolves to the elsewhere arm); `pinned_default()`
reports it; `alloc_host` delegates with it. `alloc_host_kind` is unchanged (the gate's arm). The
enum's `Default` stays `WriteCombined` (the arm for a class with no receipt) and the bits per kind
are unchanged (4 and 0). No `MEMRA_*` read; no `.cu`; the copy sites untouched; the production
caller (`worker.rs`, the contract D2H leases) is untouched and takes the resolved default through
`alloc_host`. Compute capability was not used as the key: both classes are 12.0, so it cannot
separate them, and the name is what the engine already keys product shape on (`parallel.rs`,
`hybrid.rs`).

Unit cells: `pinned_kind_per_device_default_resolves_by_card_class` (CPU: three RTX PRO 6000
Blackwell edition names resolve to `Cached` with flags 0; two RTX 5090 names, H100, B200, an Ada
"RTX PRO 6000" and the empty name resolve to `WriteCombined` with flags 4; the elsewhere arm equals
the enum's `Default`); `pinned_kind_default_is_todays_write_combined_flag_bits` (unchanged bits);
`alloc_host_delegates_with_the_default_kind_and_no_other_pinned_allocation_remains` (source text:
`alloc_host` delegates with `self.pinned_default`, exactly one `PinnedKind::for_device(` call in the
non-test body, the constructor's resolution line present, no `std::env::` or `env::var` anywhere in
the non-test body, one `result::malloc_host(` site); `pinned_kind_arm_is_honoured_by_the_driver`
(GPU: `pinned_default()` equals `for_device` of the context's name, the default lease's
`pinned_kind` and its `cuMemHostGetFlags` write-combined bit match it, then both arms as before).
`tier-transfer-gate` prints `PINNED-DEFAULT device="<name>" kind=<arm> flags=<bits>` once per
`setup()` (conformance constructs two engines, so twice) and `PINNED-DEFAULT roundtrip bytes=N
kind=<arm> driver_flags=<raw>` before each roundtrip size's `PASS` line; the `PASS` and `byte_exact`
lines are byte-identical in format to day 13; the `RESULT` scope string names the decision record.

### Second sittings through the new default (both cards, tree `4488d83fe`)

Local RTX 5090 Laptop GPU (`rtx5090-day14/*-s2`, gate `1360288006606f29…`, each cell one lock hold,
`--validate` rc=0): `gputest-s2` `test result: ok. 4 passed; 0 failed; 0 ignored`;
`conformance-s2` `PINNED-DEFAULT device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined
flags=4`, 13 `PASS` lines identical to the first sitting's; `roundtrip-s2` the same header line,
then `PINNED-DEFAULT roundtrip bytes=<n> kind=write-combined driver_flags=6` and `byte_exact=true`
at every one of the six sizes (`PASS` lines identical to the first sitting's). This class resolves
to write-combined, so the second sitting is the refactor's regression check here.

Target card, one RTX PRO 6000 Blackwell Server Edition at 600 W, driver 580.178.04
(`pro-single-day14/`, BOX3 `/root/wt-a` on `lane-a-day14` at `4488d83fe`, clean; build
`pro-single-day14/build/`, gate `8cbeda3a0991c7d1…`; no `tmux` session, zero compute processes,
the lock free, zero retries; each cell one `/tmp/memra-gpu.lock` hold; mirrored without the
binary; `--validate` rc=0, `day14/collector-validate-box-*.log`): `gputest-s2` `test result: ok. 4
passed; 0 failed; 0 ignored` (the default lease reads back cached, driver record 2, on the card the
default now applies to); `conformance-s2` `PINNED-DEFAULT device="NVIDIA RTX PRO 6000 Blackwell
Server Edition" kind=cached flags=0`, 13 `PASS` lines identical to day 13's `conformance-s2`;
`roundtrip-s2` the same header, then `PINNED-DEFAULT roundtrip bytes=<n> kind=cached
driver_flags=2` and `byte_exact=true` at every one of the six sizes, `PASS` lines identical to day
13's. So the cached default runs the frozen schedules and the byte roundtrips on the target card
exactly as the write-combined default did, with the driver's record confirming the arm at every
size.

## Records

`docs/decisions/PINNED-DESTINATIONS.md` (the question, the two cards' cells with N and regime, the
rule as landed, the rejected arms, what would reverse or extend it) and its row in
`docs/decisions/README.md`; `docs/TESTING.md`'s `tier-transfer-gate pinned-ab` paragraph rewritten
(the rule, both cards' verdicts, the per-device default and its cells, the `PINNED-DEFAULT` lines);
`research/INDEX.md` row `spill-a-20260919/day14`; `STATE.md`. `docs/FLAGS.md` gets no row: there is
no environment read (the flags census is unchanged, `day14/gates/check-flags.log`). The door
pointer for the decide-by review is a comment on memra#552 (lane C owns
`research/spill-c-20260919/HOSTPREFIX-DOOR.md`).

## CPU gates (all under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, runner `day14/run-gates.sh`, logs `day14/gates/`)

| gate | exit | verbatim tail |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | (no output) |
| `cargo test -p memra-tier -p memra-kv -p memra-engine --offline --lib` | 0 | engine `test result: ok. 518 passed; 0 failed; 25 ignored`, kv `71 passed; 0 failed`, tier `7 passed; 0 failed` |
| `cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings` | 0 | `Finished` (the two `warning:` lines in the log are the engine build script's nvcc notices, not lints) |
| `bash tools/check-flags.sh` | 0 | `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)` |
| `bash tools/docs-registry-census.sh` | 0 | `docs-registry-census: flags-table-census: docs/FLAGS.md tables=58 rows=905, every row matches its header` |
| `git diff --check` | 0 | (no output) |
| `git diff --cached --check` (the staged receipts, at each receipts commit) | 2, read | `research/spill-a-20260919/rtx5090-day14/gputest/command.log:13: new blank line at EOF.`, then the same for `rtx5090-day14/gputest-s2/command.log:13` and `pro-single-day14/gputest-s2/command.log:14`: cargo's trailing blank line inside the collector's raw capture, left as captured because `command.capture.json` pins each file's sha256 (`--validate` rc=0 on every cell); the same blank line in my own `day14/gates/test-lib.log` was removed (not pinned, no measurement touched) |
| `bash -n` on the eleven cell and driver scripts (both rigs); `python3 -m py_compile` on `wc-ab.py` and `pinned-read-probe.py` | 0 | (no output) |

The first `cargo fmt --check` after the seam's tests were written failed on two over-long
`assert_eq!` lines (`build2/fmt.log`, kept); `cargo fmt` applied, the release gate and test
binaries rebuilt (`build2/`), and every cell of both second sittings ran on the rebuilt binary.
No `MEMRA_*` read was added.

## Scope

Done: the 5090 cell under the pre-registered rule (INCONCLUSIVE on the D2H clause, the host-read
clause 10/10 at a factor of 39, byte exact 22/22, the 16 MiB context cell sharpening the D2H
finding to 0/10 with non-overlapping ranges); the per-device default under ruling 22 (`Cached` on
the RTX PRO 6000 Blackwell class, `WriteCombined` on the RTX 5090 class and elsewhere, keyed on the
device name, resolved once, no env read); the CPU and GPU unit cells; conformance and roundtrip
through the new default on both cards with the driver's record of the arm at every size; the
decision record, TESTING, INDEX and STATE. Not done, stated: a served long-entry cell under the
door (the lead's or C's call); the split of the ticket lifecycle inside C's 130 ms; any change to
the RTX 5090 class's arm (it moves only on a cell that meets the rule or a lead ruling on the D2H
clause). The door's decide-by review (2026-10-05) reads the door's cost again with the target card
on cached destinations.
