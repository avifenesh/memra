# Session C day 31: the RTX 5090 pair, the whole-budget failure arm on the RTX 5090, the packet's two missing rows

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees engine files in the range; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; nothing here is a
qualification claim. No commit on main, no PR. No engine change today: research scripts and records only. Today's
card is the LOCAL RTX 5090 Laptop GPU with the 9B artifact (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`), lock
`/tmp/memra-5090.lock`; every number here is this card's own and is compared to nothing measured on the target card.

## Merge (first action)

`gh pr list --head lane/spill-integ41-20260922 --state all` printed `[]` and `origin/lane/spill-integ41-20260922`
does not exist on `origin` (`git branch -r`: nothing; `git log origin/lane/spill-integ41-20260922`: `unknown
revision`), so neither arm of the first-action rule matched a remote ref. The lead's integ41 branch exists in this
clone as the local branch `lane/spill-integ41-20260922` at `e14c270a6` (its worktree `wt-spill-integ41`, not
touched), 11 commits ahead of `origin/main` `88d3dfd49` (#643, integ40) and 0 behind, carrying this lane's day-30
tip `20a23f4d5`, A's day-25 tip `483425d83`, the lead's union merges, the rebuilt owed-cell table (`34810e918`)
and the integ41 record (`e14c270a6`). That branch merged `--no-ff` as `5cef58f09`, CLEAN: no conflict, so
`HOSTPREFIX-DOOR.md` is the integ41 side by construction (the lead's rebuilt table, ruling 35).
`check-conflict-markers: OK`. Pushed in the announced development mode (`20a23f4d5..5cef58f09`). Stated: the
merged commits are the lead's local, not-yet-pushed commits; their hashes are what the lead's push will carry
unless the lead rewrites them first, in which case the integ42 union resolves the duplicate.

The integ41 record (read from the local branch with `git show lane/spill-integ41-20260922:research/spill-lead-20260919/INTEGRATION-DAY12.md`)
carries ruling 35 (a union across two lanes that both branched from an older main is checked against THAT base)
and ruling 36 (A's proposal 1 approved for A day 26 as specified, with the refusal's typed line naming the pin and
the entry, and the hit gate's ON-arm route counts on both cards unmoved; proposal 2 accepted: the seam stays, 0.4 ms
is the recorded price, Move 2 owed item 3 closes on that receipt). A's `DAY25.md` verdict lines are quoted verbatim
in the packet (Task 3).

## Build (local, this worktree's `target/`)

`cargo build --release -p memra-server` under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`
(`day31-cpu/build-local.log`, `rc=0`, tree `5cef58f09`, `MEMRA_CUDA_ARCH auto-detected 120a`); `memra-server`
SHA-256 `efa7119b0973feef...`. The engine test binary (`cargo test -p memra-engine --lib --release --no-run`,
`day31-cpu/build-engine-tests.log`, `rc=0`) and the transfer gate (`cargo build --release -p memra-engine --bin
tier-transfer-gate`, `day31-cpu/build-transfer-gate.log`; the first attempt named the target with underscores and
failed `rc=101`, the second `rc=0`; SHA-256 `01c2d4c49db8c49b...`) built the same way. Another session's cargo
build ran from `wt-spill-integ41` during mine (seen in `pgrep` only; not touched).

## Task 1, pre-registration: the door's demote and promote PAIR on this card (committed before the cell runs)

**What is owed.** `HOSTPREFIX-DOOR.md` section D item 5 and the packet's section 4 RTX 5090 table: "The door's
demote and promote cost as a pair (this class stays write-combined under `PinnedKind::for_device`): not run".
Every pair cell so far ran on the target card (A day 15 cached, C day 16 write-combined on the pre-`for_device`
engine); the 5090 class has the hash micro-cell (`rtx5090-day18/hashmicro/`: `wc_ms=1431.613` per 160 MiB pass)
and the D2D receipt price (`rtx5090-day28/price/`), no pair.

**The cell (`day31-wc-cell.sh` under `day31-local-run.sh`; receipts `rtx5090-day31/pair/`).** The day-16
`wc-cell.sh` shape (`pro-single-day16/wc-cell.sh`, replay `wc-pair.py`), adapted only in what this card and model
need: the 9B artifact, `MEMRA_PREFIX_CACHE_MB=64` (the gates' budget on this card since day 23: one 53.8 MB
64-token entry fits the 67,108,864 B budget, two do not, so the second insert evicts and demotes),
`MEMRA_KV_HOST_MB=8192`, `MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_COMPAT=openai`, port 18131 (the gates' 18119
left to the lead's battery), `MEMRA_KV_HOST_VERIFY` and `MEMRA_SERVE_SPEC` unset as on day 16 (the default spec
boot; on the 9B the entries are `insert (spec-boundary): 64 tokens, 53.8MB`, draft-bearing, demoted as `54.8MB, 18
items`, `rtx5090-day26/identity-default-on/ev/host-on-server.log`). Four boots in ONE collector hold
(`tools/tier-battery.py --rig rtx5090 --external-lock`, lock `/tmp/memra-5090.lock`, the collector's 250 ms
sampler into `command.gpu.csv`, plus the cell's own 1 s `nvidia-smi` sampler into `ev/card.during.csv`;
`card.{before,after}.csv` and `compute-apps.{before,after}.csv` around the hold): `o1-off, o1-on, o2-on, o2-off`.
Per boot the same seven requests as day 16 (`P_A, P_B, P_A, P_B, P_A, P_B, P_A`, `max_tokens=48`, greedy): r1
seeds E_A; r2 seeds E_B and evicts E_A (demote 1); r3 to r7 are host hits that promote and whose insert evicts the
other entry (a demote inside the promote window). Observations: demotes r2 to r6 (N=5 per boot) and promotes r3 to
r7 (N=5 per boot), pooled per arm over the two orders (N=10 per arm). Bounded wait before the hold: the runner
waits for no compute app and at least 20000 MiB free and retries a refused lock, 15 waits of 120 s in total (30
minutes); if that ends busy the cell is recorded NOT RUN with the card's last snapshot (`pair/NOT-RUN.txt`) and
nothing on the card is inspected beyond `nvidia-smi`'s listing or signalled.

**What the cell reads (fixed now; `day31-wc-reading.py`, dry-tested on the day-16 receipts where it reproduced
`wc-pair.py`'s `37.8 / 169.2` demote and `4.5 / 33.2` promote-minus-inline pooled medians; `wc-pair.py` itself is
also run on the receipts for its `WC PAIR REPLAY` line).**

- The server's own lines, as day 16: `[prefix-host] demote: N tokens, X MB in Y ms` (the `in` figure: from the
  eviction's `t0` to the host insert, which under the door spans the submit, the copy-stream batch, the tick-top
  poll, the two host hashes and the publish; under OFF the synchronous D2H and its hash) and `[prefix-host]
  promote: N tokens, X MB in Y ms` (from the probe's `t0` to the device insert; under the door it spans the park
  and the re-admission; the promote window contains the inline demote of the evicted entry in both arms, so
  `promote minus inline demote` is printed beside it, as day 16 did).
- Beside them, what day 16's tree did not print and today's does: the door's `demote published off the tick:
  ticket seq=S complete after P poll(s), C ms from submission to completion` and `promote published off the tick:
  ... C ms from submission to completion` figures (N=6 and N=5 per ON boot), the `request parked` count per boot
  (the day-29 double park would read 2 per promote on this tree if the restore route also parks the hit; on the
  9B default boot the entries are draft-bearing and A day 23's draft-bearing restore is on `main`, so the route
  is reachable), and the `restore submitted off the tick` count.
- Pooled medians with N, min, max and IQR per arm; `on_minus_off` per quantity with `unc = sqrt(IQR_on^2 +
  IQR_off^2)`, `isolated` when `|value| > unc`, else `under_resolution`. One `DAY31 PAIR VERDICT:` line.
- Regime: the collector's `command.gpu.csv` (250 ms) and the cell's `card.during.csv` (1 s): temperature, power
  draw, memory used and SM clock ranges; `card.{before,after}.csv` quoted. This card reports `power.limit [N/A]`.

**Admissibility, per boot (an inadmissible boot decides nothing and is reported as such).** Exactly 6 `demote:`
and 5 `promote:` lines; the ON boots carry 6 `contracts door D2H receipt` and 5 `H2D receipt` lines and the boot
line `[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): ...`, the OFF boots none of the three; every
promote has an inline demote in its window; zero `refused`, `DISABLED` or `dropped` lines; the ON boots carry 6
demote and 5 promote `published off the tick ... from submission to completion` lines.

**That this card's destinations are write-combined, and how that is verified.** `PinnedKind::for_device`
(`crates/memra-engine/src/tier_transfer.rs`) resolves the arm from the device name through
`parallel::HardwareTarget`: `RtxPro6000Blackwell` to `Cached`, `Rtx5090` and every unknown name to
`WriteCombined`; the production allocator takes it once per `CudaTransfers` (`pinned_default =
PinnedKind::for_device(&owner.context().name())`, line 446), and the CPU table test
`pinned_kind_per_device_default_resolves_by_card_class` pins `"NVIDIA GeForce RTX 5090 Laptop GPU"` to
`WriteCombined` with flag bits `CU_MEMHOSTALLOC_WRITECOMBINED`. The server's boot lines do NOT print the arm:
the door's boot line (`[prefix-host] contracts door ON ...: ... KV plane D2H through the transfer engine on the
pageable tier (Option B); KV plane H2D through the same engine on promote (Option C)`) proves the door's engine
is constructed for this boot, and the OFF tier's `[prefix-host] on: budget 8590MB pinned cacheable host RAM`
line names the OFF program's own cached destinations, not the door's. So the verification is three receipts in
the same hold, quoted in the run section: (1) the device name in the cell's `card.before.csv` and the ON boots'
door line present; (2) `tier-transfer-gate roundtrip` run first inside the hold (its header says the collector
only, and the collector holds the lock here), which prints from the production allocator `PINNED-DEFAULT
device="<name>" kind=<arm> flags=<bits>` and per size `PINNED-DEFAULT roundtrip bytes=N kind=<arm>
driver_flags=<cuMemHostGetFlags>` (6 = write-combined plus the UVA device-map bit; 2 = cached), into
`ev/pinned-default.log`; (3) the table test above, already green on the CPU. The reading's admissibility includes
`kind=write-combined` on the `PINNED-DEFAULT device=` line; if it reads `cached` the premise of the row is wrong
and the numbers say so.

**Expected readings, stated before the run (shape, not numbers).** Under OFF the demote is one synchronous D2H
of about 54.8 MB and its hash on cached pinned memory (day 26's identity-plain-off boot on this card read `in
48.7ms` and `57.3ms` single lines, not a pooled figure); under ON the `in` figure carries the copy-stream batch,
the tick-top poll and the two host hashes over write-combined memory (the day-18 micro-cell's 1431.6 ms per 160
MiB pass would put about 490 ms per pass on 54.8 MB, but day 26's single ON lines on this card read `in 157.4ms`
and `338.8ms`, so the census question of section D item 6 is open and the cell reads what it reads). Whichever
way the numbers fall, they are quoted with N and regime and compared to nothing from the target card.

**What I will NOT do.** No tuning after a result; a second attempt happens only for a lock refusal, an idle wait
that ends busy, or a boot failure, and is reported as an attempt. No change to the seven-request shape, the
budgets, or the reading rules after the run. No third lock name; no bare GPU run (every boot and the transfer
gate inside the collector's hold); no touch of other sessions' processes (the lead's battery and hit-gate runs
share this card and take the same lock); nothing under other lanes' worktrees.

## Task 2, pre-registration: the whole-budget failure arm on this card (committed before the cells run)

**What is owed.** Section D item 9 and the packet's correctness table: "Failure, whole-budget arm
(`MEMRA_KV_HOST_TENANT_PCT=100`) | RTX 5090 | none | not run | not run | stated missing". The 27B receipt stands
for the arm (`pro-single-day23-gates/cells/failure-default-pct100-{off,on}`, day 22's ON run); this card never
ran it (days 23 and 26 ran the share-cap arm at the server default 50).

**The cells (`day31-cell.sh` under `day31-local-battery.sh`; receipts `rtx5090-day31/gates/`).**
`tools/kv-host-spill-failure-gate.sh` unchanged, with `MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_KV_HOST_TENANT_PCT=100`,
four cells in one sitting: `failure-default-pct100-off`, `failure-default-pct100-on` (`MEMRA_KV_HOST_CONTRACTS=1`),
`failure-plain-pct100-off` (`MEMRA_SERVE_SPEC=0`), `failure-plain-pct100-on`. The gate takes the canonical lock
itself (`LOCK.json` owner `internal-canonical`); the battery waits, bounded (15 x 120 s), for an idle card before
each cell and records `NOT RUN` with the snapshot if the wait ends busy; a refused lock is retried 15 x 120 s by
the cell driver. The day-26 driver's shape (`CELL.txt`, `card.{before,after}.csv`, `compute-apps.*`, `gate.log`,
`verdict.txt`, `gate.exit`), plus a census of the pool-full boot's demote lines into `poolfull-demote-lines.txt`.

**What is read.** The verdict line (`KV-HOST-SPILL FAILURE GATE: ALL GREEN` or `N FAILURE(S)`) and the `ok:` /
`FAIL:` counts from `gate.log` (the whole-budget arm asserts 14 checks: the share-cap arm's `prefix_host_tenant_rejects`
check is not in this arm); the gate's arm line `pool-full refusal arm: whole host budget (MEMRA_KV_HOST_TENANT_PCT=100
disarms the share cap; insert-path skip demote after the copy)`; the server's typed line `[prefix-host] skip
demote: entry X MB > host budget 1MB` verbatim with this model's entry size; on the ON cells the door's lines that
precede it in the same boot (`demote submitted off the tick ... 18 items`, `contracts door D2H receipt ...
require=ok`, `demote published off the tick ...`), as the target card's day-23 ON cell showed the refusal firing
AFTER the whole Move 1 contract ran. The digest-mismatch and alloc-refusal cells are the gate's own and are read
from the verdict only.

## Task 1, the run (tree `934a6da3a`, binary `efa7119b0973feef...` built at `5cef58f09`; receipts `rtx5090-day31/pair/`)

**Hold.** `day31-local-run.sh`: the card was idle at the first probe (no compute app, 23970 MiB free; `pair/waits.log`
absent), the collector took the lock at once (`pair/wc-pair/lock.json` `{"rig": "rtx5090", "lock":
"/tmp/memra-5090.lock", "acquired": true, "owner": "collector", "mechanism": "inherited-flock-same-open-description",
...}`, `ev/LOCK.json` the same inode, no `lock-retries`), one attempt, `wc-pair.exit` 0, `CELL.jsonl` `status:
executed-not-qualified`, window 12:47:30Z to 12:48:06Z (35 s; the 9B boots in about 5 s on this card). Inside the
hold, before the boots, `tier-transfer-gate roundtrip` (`ev/pinned-default.log`, `rc=0`) printed verbatim:
`PINNED-DEFAULT device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined flags=4`, then `PINNED-DEFAULT
roundtrip bytes=4096 kind=write-combined driver_flags=6` and the same `kind=write-combined driver_flags=6` at 65536,
1048576, 16777216, 67108864 and 268435456 bytes (6 = the write-combined bit plus the UVA device-map bit): the premise
of the row holds on this card. The ON boots' door line present, the OFF boots' absent (`ev/*-boot-lines.txt`; the
ON line: `[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): 1 model program identities, tenant salt per
pool namespace, server governor ledger pinned/pageable 17180MB device 201MB in-flight 198; host tier armed; KV plane
D2H through the transfer engine on the pageable tier (Option B); KV plane H2D through the same engine on promote
(Option C)`; `[server] build: memra-0.138.0-70d2fdd3fdb6 (id: source-tree, git: 5cef58f09790)`).

**Regime.** Collector `command.gpu.csv`: 192 samples at 250 ms, 54 to 74 C, 9.5 to 169.9 W, `power.limit [N/A]`,
memory used 15 to 9753 MiB, SM clock 180 to 2227 MHz. The cell's `ev/card.during.csv`: 36 samples at 1 s, 55 to 74
C, 27.7 to 169.3 W. `card.before.csv`: `15 MiB` used, 54 C, 9.48 W, P8; `card.after.csv`: `15 MiB`, 62 C, 30.62 W,
P0; `compute-apps.{before,after}.csv` empty.

**Admissibility.** Every clause of the pre-registration held in every boot (`reading.log`, 23 `ok`, 0 `FAIL`;
`wc-pair-replay.log` `WC PAIR REPLAY: PASS (12 checks)`): 6 demotes and 5 promotes per boot; receipts D2H=6 H2D=5
in the ON boots, 0 in the OFF; the door boot line as the arm requires; every promote with an inline demote; zero
`refused`, `DISABLED` or `dropped` lines; 6 demote and 5 promote completion lines per ON boot; the pinned arm
printed write-combined. Entries `64 tok/54.8 MB` in every boot.

**The lines, verbatim (`reading.log`).**

`o1-off: 6 demotes, 5 promotes, receipts D2H=0 H2D=0, parked=0, restore_submitted=0, refused=0, disabled=0, dropped=0, door_on_line=0`
`  raw demote in-ms (r2..r7):  [23.1, 23.8, 21.8, 5.1, 4.8, 5.3]`
`  raw promote in-ms (r3..r7): [27.6, 25.3, 8.6, 8.7, 8.8]  inline demote ms: [23.8, 21.8, 5.1, 4.8, 5.3]`

`o1-on: 6 demotes, 5 promotes, receipts D2H=6 H2D=5, parked=10, restore_submitted=5, refused=0, disabled=0, dropped=0, door_on_line=1`
`  raw demote in-ms (r2..r7):  [57.8, 60.4, 58.9, 39.7, 39.5, 39.7]`
`  raw promote in-ms (r3..r7): [42.4, 38.4, 21.5, 20.5, 20.1]  inline demote ms: [57.8, 60.4, 58.9, 39.7, 39.5]`
`  door demote submission-to-completion: median 26.5 (N=6, min 18.0, max 37.4, IQR 19.2)  raw [34.8, 37.4, 37.2, 18.3, 18.0, 18.1]`
`  door promote submission-to-completion: median 15.5 (N=5, min 14.9, max 15.9, IQR 0.8)  raw [15.9, 15.5, 15.5, 15.0, 14.9]`

`o2-on: 6 demotes, 5 promotes, receipts D2H=6 H2D=5, parked=10, restore_submitted=5, refused=0, disabled=0, dropped=0, door_on_line=1`
`  raw demote in-ms (r2..r7):  [57.7, 58.8, 56.9, 39.4, 38.5, 38.7]`
`  raw promote in-ms (r3..r7): [40.2, 38.1, 20.6, 20.4, 20.1]  inline demote ms: [57.7, 58.8, 56.9, 39.4, 38.5]`
`  door demote submission-to-completion: median 26.7 (N=6, min 17.3, max 36.3, IQR 18.3)  raw [35.3, 36.3, 35.5, 18.1, 17.3, 17.5]`
`  door promote submission-to-completion: median 14.8 (N=5, min 14.6, max 15.4, IQR 0.6)  raw [14.8, 14.6, 14.6, 15.4, 14.9]`

`o2-off: 6 demotes, 5 promotes, receipts D2H=0 H2D=0, parked=0, restore_submitted=0, refused=0, disabled=0, dropped=0, door_on_line=0`
`  raw demote in-ms (r2..r7):  [18.2, 24.1, 22.6, 5.4, 6.5, 17.6]`
`  raw promote in-ms (r3..r7): [27.7, 26.0, 9.5, 9.9, 21.3]  inline demote ms: [24.1, 22.6, 5.4, 6.5, 17.6]`

`pooled OFF (o1-off + o2-off):` `demote:  median 20.0 (N=10, min 4.8, max 24.1, IQR 18.0)`; `promote: median 15.6 (N=10, min 8.6, max 27.7, IQR 17.6)`; `promote minus inline demote: median 3.5 (N=10, min 3.4, max 4.1, IQR 0.4)`

`pooled ON (o1-on + o2-on):` `demote:  median 57.3 (N=10, min 38.5, max 60.4, IQR 19.3)`; `promote: median 21.1 (N=10, min 20.1, max 42.4, IQR 18.5)`; `promote minus inline demote: median -19.3 (N=10, min -37.4, max -15.4, IQR 7.4)`; `door demote submission-to-completion: median 26.5 (N=12, min 17.3, max 37.4, IQR 18.1)`; `door promote submission-to-completion: median 14.9 (N=10, min 14.6, max 15.9, IQR 0.7)`

`DAY31 PAIR VERDICT: demote off 20.0 (N=10) on 57.3 (N=10) on_minus_off +37.3 unc 26.4 isolated; promote off 15.6 (N=10) on 21.1 (N=10) on_minus_off +5.4 unc 25.6 under_resolution; promote_minus_inline off 3.5 (N=10) on -19.3 (N=10) on_minus_off -22.9 unc 7.4 isolated; parked per boot [0, 10, 10, 0]; pinned=write-combined; admissible=True`

**What the cell read (nothing tuned; this card's figures, compared to nothing from the target card).**

- The raw lists step in both arms: the first three demotes of every boot read 18 to 24 (OFF) and 57 to 60 (ON), the
  later ones 5 to 7 (OFF; o2-off's r7 17.6 excepted) and 38 to 40 (ON), which is A day 17's first-touch step on this
  card too; the pooled medians straddle that step, hence IQRs of 18 to 20 and an `unc` of 26 on the demote's +37.3
  (`isolated` at that resolution) and of 25.6 on the promote's +5.4 (`under_resolution`).
- The door's own completion figures: demote `from submission to completion` 34.8 to 37.4 on the first three of a boot
  and 17.3 to 18.3 after (N=12), promote 14.6 to 15.9 (N=10, IQR 0.7). The demote's `in` figure minus its completion
  figure is about 21 to 23 ms on every demote (57.8 against 34.8, 39.7 against 18.3): the part of the ON demote that
  runs after the copy landed (the tick-top poll, the two host hashes, the publish). The micro-cell's 1431.6 ms per 160
  MiB write-combined pass would put about 490 ms per pass on 54.8 MB; the whole ON demote reads 38.5 to 60.4, so the
  two hashes are not reading write-combined memory at that rate on this tree. Which memory they read is section D
  item 6's question and is not attributed by this cell.
- `promote minus inline demote` is negative on the ON arm because the inline demote publishes at a later tick top than
  the promote (log order per hit: `demote submitted`, `promote published`, `D2H receipt`, `demote published`); day
  16's subtraction assumed a demote that completes inside the promote's window and does not isolate the ON promote on
  this tree. The figure to read beside the OFF promote's steady 8.6 to 9.9 is the door's own `promote published ...
  14.6 to 15.9 from submission to completion`, with the promote's `in` figure 20.1 to 21.5 steady (the park and the
  re-admission at a tick top).
- Every ON boot reads `parked=10` and `restore_submitted=5` for its 5 promotes, with `restore landed off the tick ...
  33.1ms to 33.4ms to re-admission`: the day-29 double park reproduces on this card, on draft-bearing entries (A day
  23's draft-bearing restore on `main`). A day 26's approved proposal 1 is the named change; nothing here changes it.
- The write-combined premise is a receipt now (the `PINNED-DEFAULT` lines above), not a reading of the table.

## Task 2, the run (tree `934a6da3a`; receipts `rtx5090-day31/gates/`)

`day31-local-battery.sh`: no wait (the card idle before every cell, `battery.log`), four cells 12:49:0xZ to
12:50:11Z, no lock retry, `compute-apps.{before,after}.csv` empty in every cell, card 54 to 61 C at the snapshots.
Verbatim, one row per cell (`verdict.txt`; `ok:` and `FAIL:` counted in `gate.log`; the arm line and the refusal from
`gate.log` and `poolfull-demote-lines.txt`):

| Cell | Environment | Verdict | ok / FAIL | The pool-full boot's demote lines (verbatim, elided digests) |
|---|---|---|---|---|
| failure-default-pct100-off | `MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_KV_HOST_TENANT_PCT=100` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 14 / 0 | `[prefix-host] skip demote: entry 54.8MB > host budget 1MB` x2 |
| failure-default-pct100-on | the same, `MEMRA_KV_HOST_CONTRACTS=1` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 14 / 0 | `[prefix-host] demote submitted off the tick: 64 tokens, 54.8MB, ticket seq=3, 18 items on the contracts door's copy stream (model gate)`; `[prefix-host] contracts door D2H receipt: ticket issuer=2 seq=3 epochs=0/1/1 items=18 (8 KV planes, draft) complete=18 require=ok ... retired acknowledged`; `[prefix-host] demote published off the tick: ticket seq=3 complete after 1 poll(s), 36.2ms from submission to completion (tick-top poll)`; `[prefix-host] skip demote: entry 54.8MB > host budget 1MB`; then seq=5 the same (`23.3ms`) and the second refusal |
| failure-plain-pct100-off | `... MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 14 / 0 | `[prefix-host] skip demote: entry 54.6MB > host budget 1MB` x2 |
| failure-plain-pct100-on | `... MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 14 / 0 | `demote submitted off the tick: 64 tokens, 54.6MB, ticket seq=2, 16 items ...`; `D2H receipt: ticket issuer=2 seq=2 ... items=16 (8 KV planes) complete=16 require=ok ...`; `demote published off the tick: ticket seq=2 complete after 1 poll(s), 24.9ms from submission to completion (tick-top poll)`; `skip demote: entry 54.6MB > host budget 1MB`; then seq=4 (`14.7ms`) and the second refusal |

Every cell printed the gate's arm line `pool-full refusal arm: whole host budget (MEMRA_KV_HOST_TENANT_PCT=100
disarms the share cap; insert-path skip demote after the copy)` and `ok: pool-full refusal is LOUD and named (skip
demote: entry X MB > host budget B MB)`. As on the target card's day-23 ON cell, the door's whole Move 1 contract runs
(submit, receipt `require=ok`, publish at the tick top) and the whole-budget refusal fires after it; the OFF arm
refuses after its synchronous copy. The digest-mismatch and alloc-refusal cells are the gate's own and are inside the
14 ok. `executed-not-qualified`.

## Task 3: the packet (`DOOR-DECISION-PACKET.md`)

Section 3's RTX 5090 whole-budget row and section 4's RTX 5090 pair row are filled from `rtx5090-day31/` (the pair
as three rows: demote, promote with promote-minus-inline, the verdict line; every figure copied from `reading.log`,
the failure lines from `poolfull-demote-lines.txt` and `gate.log`). Section 5 item 3 carries A's `DAY25 DOUBLE-PARK`
stall and e2e lines for both orders and both `DAY25 DECOMPOSITION` lines verbatim (read from `A/DAY25.md` as merged
through integ41), A's reading and proposal 1, ruling 36's approval for A day 26, and today's `parked=10` census on
this card; item 4 carries both `DAY25 RETIRE-SETTLE` lines verbatim, the refutation of day 24's "host wait" reading,
proposal 2 and ruling 36's acceptance (the seam stays, 0.4 ms recorded, Move 2 owed item 3 closed). Section 2's
"what still runs on the tick" keeps the retire settle as a statement and no longer carries it as a cost (0.4 ms, the
settle's fixed cost). Item 7 and section 6's first outcome no longer list the two 5090 cells as missing (the census
question of item 6 stays, with today's arithmetic beside it). The header notes A day 24 on `main` since #643 and A
day 25 on the integ41 branch. No recommendation added; the three outcomes stay as they were written.

## Task 4: records and checks

`STATE.md` rewritten; `research/INDEX.md` row `spill-c-20260919/day31`; `HOSTPREFIX-DOOR.md` section C's RTX 5090
whole-budget row filled, section D items 5 and 9 RESOLVED with this card's cost rows and the four verdicts. Checks:
`shellcheck -S warning` clean on the four drivers; `bash tools/check-flags.sh`; `bash
tools/check-conflict-markers.sh`; `git diff --check`; `.gitattributes` `*.log -whitespace` in `rtx5090-day31/`; zero
em dashes in every file added or edited today (results in the closing commit's message).

## Pushes

`5cef58f09` (the merge), `934a6da3a` (the pre-registration, the drivers, the reading script), then the closing commit
(this record's run sections, the receipts, the packet, the door table, STATE, INDEX), each in
`MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`,
logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

Local RTX 5090 only today (BOX3 not touched). Every boot inside the collector's hold or the gate's own lock; the
only processes stopped were this cell's own servers and its sampler, by pid; compute apps empty at every snapshot.
The lead's cargo build from `wt-spill-integ41` and the lead's integ41 worktree were seen in `pgrep` and `git
worktree list` only, never touched. No `/tmp` scratch left (the collector writes under the receipt dir); no bundle.
The clone's `target/` holds the three binaries built today.

## Budget

About 2.6 agent-hours against 4: reading and the merge 0.6, the two pre-registrations and drivers 0.7, the builds and
the two runs (35 s and 70 s of card time, polled) 0.3, the packet and door-table edits with the receipt cross-check
0.6, the records and checks 0.4. Blockers: none (the card was idle at both probes; no bounded wait was consumed).
