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

## Task 3 (the packet) and Task 4 (records): written after the cells, from their receipt files

The packet's section 4 RTX 5090 table gains the pair row and section 3 gains the failure row (or `not run` with
the quoted reason), section 5 items 3 and 4 carry A's day-25 verdict lines verbatim and ruling 36, and section 2
drops the retire-settle wait as a cost. Every number added is re-read from the file named beside it. No
recommendation.
