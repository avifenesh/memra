# park-compact on the RTX PRO 6000 Blackwell Max-Q: pre-registration of the Max-Q reading

Written 2026-11-04, before any Max-Q boot, from `research/spill-b-20260919/DAY38.md` at the fix-stack tip
`2ccb038d1c16358edde65582597e7b840e3f63fb`. The cell is DAY38 addendum D (1.11), the deciding cell of the
park-compaction default (`docs/decisions/KV-PARK-COMPACT-DEFAULT.md`), rerun on one Max-Q card. Nothing in this
file changes after a number is seen. A failed clause is recorded as it reads.

DAY38 1.1 requires the pre-registration "committed and pushed before any day-38 code and before any day-38 boot",
and the scope note for this rerun names it DAY38 1.12, addendum E. This kit commits nothing: committing
sections 1 to 6 below as addendum E before the first boot is owed by the session that runs the kit.

## 1. Subject (DAY38 1.1, unchanged)

Quoted: "The door acts on the plain continuation pool only (spec and DSpark pools park live engine sessions, out of
scope by design). Every boot of this cell therefore runs: `MEMRA_SERVE_SPEC=0` ...; `MEMRA_PREFIX_CACHE_MB=0` ...;
`MEMRA_REUSE_POOL` at its default (2 per model and namespace); every conversation runs in its own cache namespace
(`cache_salt`) ...; greedy decoding, ... the 27B at the checkpoint's 262,144 on the target card."

Model: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, sha256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`,
15,705,922,304 bytes, served as `q38`, `MEMRA_CTX` unset (262,144). `MEMRA_TIMEOUT_MS_MAX=3600000`,
`MEMRA_ADMIT_PREDICT_SHADOW=1`, N=5.

## 2. Workload (DAY38 1.2 as amended by 1.11, unchanged)

`day38-client.py` (sha256 `dda4ae2f...`, byte-identical to the Workstation run's). Shape Xp: every turn sends
`max_ctx = L + 512`; turn 2 = turn 1's ids, turn 1's completion tokenized, stream ids to 64 new tokens,
`max_tokens=1`; turn 3 likewise with `max_tokens=32`; each turn's cold twin in a fresh namespace. Shape A: the
affinity rewind with an `x-session-id` header, cold twins without it. Main boots: `--lengths 6144,30720,122880
--shapes Xp,A`. Fault boots: `--lengths 6144 --shapes Xp --turn4`. The prompt text is `docs/SERVING.md` from the
Workstation run's tree `3476ea4dca` (sha256 `022b1d50...`, the same file box-d's `inputs.sha256` records).

## 3. Arms and boots

Arms are binaries (DEVIATIONS.md D1): `off` = the tip unpatched, where the Max-Q parks whole; `on` = the tip plus
`patches/maxq-arm.patch` (`HardwareTarget::is_pro6000_maxq` and one more arm in `park_compact_for_device`, the
candidate default itself); `red` = `on` plus `day38-red.patch` (`retire_may_park` ignores `errored`, unchanged).

Boots, by name as the reader reads them:

| boot | binary | env beyond 1.1 | client |
|---|---|---|---|
| main-O1-off | off | none | main |
| main-O1-on | on | none | main |
| main-O2-on | on | none | main |
| main-O2-off | off | none | main |
| fault-batch | on | `MEMRA_STEP_OOM_FAULT=1` | fault |
| fault-nobatch | on | `MEMRA_STEP_OOM_FAULT=1 MEMRA_SERVE_BATCH=0` | fault |
| fault-nobatch-red | red | `MEMRA_STEP_OOM_FAULT=1 MEMRA_SERVE_BATCH=0` | fault |

The interaction pair (1.5, vmm-off and vmm-on) does not run: its door is deleted at the tip and it is outside the
rule. Parts for windows of about 4 h: part 1 = main-O1-off, main-O1-on, fault-batch, fault-nobatch,
fault-nobatch-red; part 2 = main-O2-on, main-O2-off. The reading is taken over the union of the boots of one
complete part 1 and one complete part 2 (`out-combined-*/`); each order's pair always runs inside one part.

## 4. Clauses and rule (quoted from DAY38.md)

P1' (1.11): "(a) Door identity: every X' turn 2 and 3 that resumed on both arms (`cached_tokens > 0` on both) has
equal digests on `on` and `off`; every shape-A turn 2 has equal digests on `on`, `off` and its cold twin (the grid
law). (b) The resumes happened: at least 80% of the X' turns 2 and 3 resume on both arms (a tokenization miss
misses on both and is counted), every `on` resume prints `[kv-reuse] park-compact grow:`, and every shape-A turn 2
prints `[worker] plain-affinity: rewound to` on both arms. (c) No row answers non-200. Reading, no bound: X'
resumed rows against their cold twins per arm (flips under the near-tie contract), with the resume row counts."

P2' (1.11): "`MEMRA_STEP_OOM_FAULT`'s non-batching injection point fires only on a session that is `prefill_done`,
so the forged failure lands on a decode step after the prime (the batched point is already a decode chunk).
Clauses as in addendum A: the errored session writes no `park-compact` line (the boot's count equals its HTTP 200
rows) and its conversation's next turn runs cold (equal to its cold twin, no pool hit). The red arm ... is
expected to read one extra `park-compact` line and a resumed next turn; if it does not, the red arm has not
exercised the defect and that is recorded."

Rule (1.6, re-registered by 1.11): "**PROMOTE-ELIGIBLE** when P1 (a) to (c) and P2 pass. P3 and P4 go with it. ...
**FAIL (no reading)** when P1 or P2 fails: the cause is quoted, and the door is revised under a new
pre-registration before the decide-by or recorded as failed at it." "Neither card's reading is compared with the
other's, and no timing crosses cards."

The reader is `day38d-read.py`, unchanged, run with the card label `pro6000-maxq`. On the Workstation it printed
(DAY38.md 2.3, box-d `SUMMARY.txt`), for example:
`DAY38D P1 card=pro6000 order=O1 xp_turns=30 resumed_both=30 frac=1.00 xp_door_differ=[] prompt_mismatch=[] a_turn2=15 a_differ=[] on_resumed=30 park_compact_grow=30 affinity_rewound={'main-O1-off': 15, 'main-O1-on': 15} non200=[] faults=0 -> PASS`
and the verdict "The rule of 1.6 on the target class: PROMOTE-ELIGIBLE (P1' and P2' pass; P3 and P4 go with it)".

## 5. Max-Q specifics

- The card index is chosen at run time (`CARD`). The runner refuses unless `nvidia-smi -i $CARD` reports exactly
  "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" with under 2,000 MiB used and no compute app. It
  records the card's uuid, PCI bus id, `power.limit`, `enforced.power.limit`, default and max limits in
  `card.json` and `card.csv`. At kit time the <box> limits read 250/250/280/280 W (default 300 W; max 325, 300, 300,
  325 W), and the owner has not changed them. The rule does not depend on the power limit.
- The server is pinned with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD`; every boot records its
  server pid and the card's compute apps at ready.
- Regime: GPUs 0, 1 and 3 train beside the cell (all <box> GPU pairs are NODE on one root complex). P1' and P2'
  are untimed (digests, counts of log lines and parks, HTTP status, greedy decoding), so they fit the
  AGENTS.md exception for untimed pass/fail cells beside other work. P3 and P4 from these runs are co-tenant and
  are not readings for the record; `run-timed.sh` takes them in a quiet slot (the four main boots).
- The owner's Max-Q thermal guard wraps every boot: stop at >= 92 C with fan >= 95 % on 3 samples 10 s apart, or
  at >= 95 C. A boot it stops is not read.
- At the tip, the other defaults keyed on the full-power Workstation (batch-OOM recovery, `w_release_for_device`,
  kv-vmm placement, admission by memory, handoff, tier transfer, MoE cache) stay off on the Max-Q. At `a803d3080`
  none of them existed, so the Max-Q boots run the regime the Workstation cell ran in. In particular fault-batch
  takes the batched chunk's error arm as in DAY38D; if batch-OOM recovery is later extended to the Max-Q, that
  boot must be re-specified.

## 6. The VERDICT line and what it decides

`maxq-verdict.py` prints the registered reader's lines and then one line:

`VERDICT DAY38D card=pro6000-maxq rule=DAY38-1.6 P1'=[O1:..,O2:..] P2'=[fault-batch:..,fault-nobatch:..] red=[fault-nobatch-red:..] -> <outcome>`

with outcome `PROMOTE-ELIGIBLE (P1' and P2' pass; P3 and P4 go with it)` (the Workstation's wording),
`FAIL (no reading: ...)`, `INCOMPLETE (absent boots: ...)` (a part not yet run) or `INVALID (...)`. INVALID means
a kit validity check failed and nothing is concluded: a boot ran a binary other than its arm's, an off boot wrote
a park-compact line or an on boot none, a boot's card was not the Max-Q, the server pid was not on the pinned
card at ready, the model or kit files did not match their hashes, or the thermal guard stopped a boot. The red arm
is reported and not ruled on. On the Workstation receipts the same script prints `-> PROMOTE-ELIGIBLE`
(DRYRUN.txt).

- PROMOTE-ELIGIBLE: the Max-Q joins the default with `patches/maxq-arm.patch` (its test moves the Max-Q to the
  positive list of `park_compact_follows_the_receipted_cards`), plus the record changes the scope lists:
  `docs/decisions/KV-PARK-COMPACT-DEFAULT.md` (title, Chosen, the Rejected bullet, a Max-Q row, Scope),
  `docs/SERVING.md` 2276-2283, `docs/FLAGS.md` 1730-1732, `docs/rigs/rtx-pro-6000-blackwell.md` 12-14, and
  DAY38.md 2.5 with the receipts.
- FAIL: the Max-Q keeps parking whole; the failing clause and its quoted cause go to DAY38.md 2.5 and to the
  decision record's Rejected list.

## 7. Expected wall time (a projection, not a clause)

Workstation (600 W): main boots 37.4 min each, 30.7 min of it at the 122,880-token length (cold primes about
52 s each); fault boots about 1 min each (box-d run.log, client.jsonl). Max-Q at 280 W: two readings put a
300 W Max-Q 1.15 to 1.43 times slower on cold primes than a 600 W card (`research/newbox-bench-20260811`, 4k
cold TTFT, PP-2) and q38 at 2,450 prime tok/s at 24.6k tokens on a Max-Q (darklanes
`dspark-boundary-capture-6000-class-green`, 2026-09-06) against about 3,340 tok/s at 30.7k on the Workstation
(box-d), 1.36x. Projection: 1.4x to 1.6x, so 52 to 60 min per main boot; part 1 about 110 to 125 min, part 2
about 105 to 120 min. Upper bound if primes scale with board power (600/280 = 2.14x): 80 min per main boot, part
1 about 166 min, part 2 about 160 min, which still fits a 4 h window. The runner's first projection is that
upper bound; after the first main boot it uses the measured time.
