# batch-oom-recover on the RTX PRO 6000 Blackwell Max-Q: pre-registration

Written 2026-11-04, before any Max-Q run of this cell. Nothing here changes after a Max-Q number is seen. A clause
that fails is recorded as it reads; a revision is a new dated file written before its run.

This text is the body of the DAY49 addendum F that the scope asks for
(`research/spill-b-20260919/DAY49.md`, section 1 rule: registered and pushed before the cell). The kit does not
commit; whoever lands the lane copies it into DAY49.md and pushes it before the run.

## 1. The question

`docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md` makes the batched-chunk OOM retry the default on the full-power
RTX PRO 6000 Blackwell Workstation Edition only, because its deciding cell (DAY49 addendum D, sections 2.3 and 2.4)
ran on that card. Its "What would change it" line: "A cell on the Max-Q or Server Edition adds that edition."
This cell is that Max-Q cell. It decides one thing: whether `batch_oom_recover_default` also returns true for
"NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" (`maxq-batch-oom-recover.patch`).

## 2. The rule, quoted from the Workstation pre-registration

DAY49.md 1.8 (addendum D), the gate's arm j, verbatim:

> `j-ctrl`: no fault, the control digests. `j`: `MEMRA_STEP_OOM_FAULT=batch:1`. Green: exactly one `fired: this
> batched decode chunk` line naming at least 2 sessions, one `retrying` line naming the same count as untouched, one
> `retried (ok)`, all three streams `200` with the control's digests, no 5xx, no panic. `j-red`: `batch:2`, so the
> retry is faulted too. The green assertion must fire (`retry failed`, and the chunk's sessions end with the error
> event).

DAY49.md 1.8, the serving shape, verbatim:

> The serving shape runs again with `MEMRA_STEP_OOM_FAULT=batch:1`, arms `off` and `on` in both orders, on two
> routes: the spec default, and `MEMRA_SERVE_SPEC=0`. A boot counts only if its fired line names at least 2
> sessions. On `on` every request ends `200`; `off` is the before reading (the chunk's sessions end with the error
> event). If no fire reaches a batched chunk on the spec-default route, that reads as `not reached` for that route,
> with the boot's fired count quoted.

The code that applies these clauses is the registered code, unchanged:

- `tools/health-fault-gate.sh` arm j prints `HFG (j) ... -> PASS|FAIL` and `HFG (j-red) ... -> PASS|FAIL`
  (the only change at 2ccb038d1c against 8926ccfb3 is the source of "recovery on": the card default and its boot
  line instead of the deleted env door; DEVIATIONS.md D1, D14).
- `research/spill-b-20260919/day49d-read.py serve`: an `on` boot reads PASS iff
  `status.get(200,0)==len(rows) and errors==0 and ok==retry==1` (day49d-read.py:57); an `off` boot reads
  `READING (the before)`; a boot without a batched fire of >= 2 sessions reads `NOT REACHED`.

The Workstation reading (DAY49 2.3, 2.4, and the decision record) was: arm j PASS and j-red PASS on every gate unit,
j-vmm green and red as registered, every `on` boot PASS on both routes and both orders, every boot reached.
`maxq-verdict.py` applies the same reading and prints it as one line. On the Workstation receipt it prints
(`dryrun-reader.sh`, `DRYRUN.txt`):

```
VERDICT batch-oom-recover card=pro6000 ... gate_j_pass=2/2 gate_j_red_pass=2/2 j_vmm=green=PASS,red=PASS serve_on_pass=4/4 serve_off_read=4/4 ... routes_reached=plain:4/4,spec-default:4/4 -> ON
```

## 3. The Max-Q cell as registered here

- **Arms by binary** (the env door is retired at the kit's source and the boot audit refuses it):
  `off` = 2ccb038d1c16358edde65582597e7b840e3f63fb as is, which prints
  `[admit-oom] batch OOM recovery: OFF` on a Max-Q; `on` = the same commit plus `maxq-batch-oom-recover.patch`,
  which prints `[admit-oom] batch OOM recovery: ON`. Hashes in PROVENANCE.txt.
- **Gate units:** j1 and j2, each `tools/health-fault-gate.sh` with `HFG_ARMS=j HFG_READY_WAIT_S=900` on the `on`
  binary and the 27B: j-ctrl, j (`batch:1`), j-red (`batch:2`), `MEMRA_SERVE_SPEC=0`, three concurrent 48-token
  streams with the registered prompts.
- **j-vmm is not part of the Max-Q cell.** On a Max-Q the on-demand KV planes are off (kv_vmm.rs:77 keys them on the
  Workstation Edition), `vmm_reap_for` returns at `!kv_vmm::armed()`, and `MEMRA_KV_ALLOCATOR` is retired, so the
  reap the j-vmm pair reads cannot run. It is recorded as `n/a`, not as a pass.
- **Serving boots:** `day49d-run.sh` with `MODEL_KEY=q38 BOOT_CTX='' NO_SCOPE=1 FAULT=batch:1
  CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64"`, in the registered order
  O1-off, O1-on, O2-on, O2-off (spec default), then P1-off, P1-on, P2-on, P2-off (`MEMRA_SERVE_SPEC=0`). Each boot is
  8 burst requests plus one probe (N=9 rows). The prompts are cut from the deciding tree's `docs/SERVING.md`
  (sha256 022b1d50620f5b34bb6e6127be05ff79184ce3aa294aca618a31a0dd73887cff, the bytes the Workstation boots
  recorded in `inputs.sha256`), so they are the Workstation's prompts.
- **Order:** the gates, then the boots, in one sitting (chain.sh's order).

## 4. The verdict

`maxq-verdict.py` prints one line, `VERDICT batch-oom-recover card=pro6000-maxq ... -> <word>`, into
`out-*/VERDICT.txt`:

- **ON** (the recovery becomes the Max-Q default; land the patch): j1 and j2 both read `HFG (j) ... -> PASS` and
  `HFG (j-red) ... -> PASS`; all 8 boots present and reached; all 4 `on` boots read PASS; and section 5 holds.
- **OFF** (the Max-Q keeps the chunk's error arm): section 5 holds and a registered clause read FAIL (a gate's
  `-> FAIL` other than "boot failed", or an `on` boot's `-> FAIL`). Record it as it reads.
- **NOT-READ** (no decision; rerun in another window): a gate or boot is missing, did not boot, SKIPped, or did not
  reach a batched chunk of >= 2 sessions; or section 5 fails.

The `off` boots are the before reading. They are quoted (status counts) and never decide the verdict. On the
Workstation they read 8 of 9 `503` on the spec default and 5 of 9 on the plain route; the Max-Q's chunk sizes may
differ, and the rule only needs >= 2 sessions per fired chunk.

## 5. Max-Q specifics (kit clauses; they make a run valid, they never turn a FAIL into a PASS)

- **The card** is chosen at run time (`CARD=<nvidia-smi index>`) by the session that holds the window, pinned with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD`. `run.sh` refuses unless `nvidia-smi -i $CARD` reads
  "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", under 2000 MiB used and no compute app. `card.csv`
  records index, uuid, bus id, power.limit, default and max limit, driver and clocks.
- **Power:** the <box> caps read 250, 250, 280 and 280 W (cards 0 to 3) on 2026-11-04, against a 300 W default and max;
  the owner has not changed them. The Workstation cell ran at 600 W. Every clause is pass/fail on HTTP status,
  retry lines and digests, so the verdict does not depend on power; power.limit is recorded, not judged.
- **Arm identity:** every `on` boot and each gate's j-ctrl print `batch OOM recovery: ON`; every `off` boot prints
  `OFF`. A mismatch is NOT-READ (the wrong binary ran).
- **Pin:** every process of this kit seen in `nvidia-smi --query-compute-apps` holds the pinned card's uuid
  (`pins.csv`). A kit process on another card stops the run: NOT-READ.
- **Thermal:** the owner's Max-Q rule. Stop at >= 95 C, or at >= 92 C with the fan at >= 95 % on 3 consecutive
  250 ms samples (`THERMAL-STOP.txt`): NOT-READ.
- **Box-wide canary:** the server's gpu-watch reads every card's `nvidia-smi` and the kernel's Xid log
  (health.rs:1232-1260). A `[gpu-watch] CRITICAL` line in any of this cell's server logs may come from a training
  card, so it makes the run NOT-READ, not OFF.
- **Deadline:** `TOTAL_TIMEOUT_S` (default 5400 s). A unit cut by it is missing: NOT-READ.

## 6. Not timed

No clause is timed. Wall time, telemetry and the 250 ms samples are recorded for the corpus only. The cell runs on
one card while the box's other cards train (AGENTS.md, Evidence discipline: "short untimed pass/fail cells"), so
there is no `run-timed.sh`.
