# moe-prefetch on the RTX PRO 6000 Blackwell Max-Q: pre-registration (2026-11-04, before any Max-Q cell)

Written 2026-11-04 19:25Z, before the cell. The kit's `run.sh` records this file's SHA-256 in `out-*/provenance.log`, so
a change after the run shows.

## Why this cell

The owner ruled on 2026-11-04 that the RTX PRO 6000 target is the Max-Q. The current default
(`moe_cache::in_token_prefetch_for_device`, fix4/gates `2ccb038d1c`) turns the in-token MoE prefetch on for the
full-power Workstation Edition only (`pro6000-ws-receipt`), because its deciding cell (DAY59.md section 2) ran on a
rented Workstation card at 600 W. The Max-Q reads `no-receipt-default` (off). This cell reruns the same registered
cell on one Max-Q card and decides whether the default extends to it.

## What is replayed, unchanged

DAY59.md section 1 and 1a, quoted in full below: the same binaries' source tree (`c60` from `da649107c`), the same
artifact (`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, SHA-256 `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`),
the same arms (OFF: `MEMRA_MOE_PREFETCH` unset; PF: `MEMRA_MOE_PREFETCH=1`; one binary serves both), the same gates
(G1, G2, G3), the same two timing shapes (`pftime`, `pfnaked`), the same orders (order 1 (OFF, PF) x 5, order 2
(PF, OFF) x 5, N=10 per arm), `MEMRA_NGEN=32`, prompt `55 88 13`, the seven G3 prompts at 48 tokens, 250 ms telemetry,
and the same reader (`day59-pf.py`, unchanged) with the same rule. The cell scripts are the Workstation sitting's
(`f0e77b8be`) with portability edits only (`PORTABILITY.diff`, listed in `DEVIATIONS.md`).

## Max-Q specifics

- Card: one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition on <box>. The index is chosen at run time
  (`CARD=<idx>`), pinned with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD`, and every idle check and the
  250 ms telemetry read that card only (`nvidia-smi -i $CARD`). The window plan is GPU 0.
- Power: <box>'s caps read 250, 250, 280, 280 W on GPUs 0 to 3 at kit time (2026-11-04 19:10Z); the owner has not
  changed them. The runner records the chosen card's `power.limit`, `power.default_limit` and `power.max_limit` at
  start and end, every card's limit at start, and the collector records the limit in every telemetry row. The
  reading is the Max-Q at that recorded limit, not at 300 W.
- Box: the timed cells run with no compute app on any <box> card (AGENTS.md, one scored campaign per box). The runner
  refuses to start if any card has a compute app, and skips a timed cell (its verdict then reads `void`) if one
  appears before it.
- Binaries: rebuilt on the laptop from `da649107c` (CUDA 13.1, `MEMRA_CUDA_ARCH=120a`, release, default features),
  as the 5090 queue v9 did (DAY59.md section 3). Their hashes differ from the Workstation's (`57995815...`,
  `cf0cc23d...`, `7ddb8187...`) because the build path is part of the binary; the source tree is the named one.
  Hashes are in `PROVENANCE.txt` and are checked by the runner before any cell.

## The rule the Max-Q reading is judged by (quoted, DAY59.md:44-49)

> `noise` the larger IQR of the two arms. Per shape: `pf_wins` iff `median(PF) < median(OFF) - noise` in order 1, in
> order 2 and pooled; `pf_loses` iff `median(PF) > median(OFF) + noise` in both orders; `pf_flat` otherwise.

> PF becomes that card's naked default iff G1, G2 and G3 pass there, `pftime` reads `pf_wins` and `pfnaked` does
> not read `pf_loses`

Gate clauses (DAY59.md:28-36, implemented in `day59-pf.py`): G1 every run exit 0 and `MATCH`, one `tokens:` tape per
shape across both arms; G2 `=== SELF-CONSISTENCY PASS ===` and 8 of 8 K lines `self-consistency: PASS (identical to
plain target)`; G3 all seven responses of the PF boot byte-equal to the OFF boot's, no request error. "any failure
voids that card's timing verdict".

The DAY59.md section 2 note on `noise=0.0000` applies unchanged: `run-gen` prints gen-only seconds to three decimals,
so ten identical values give an IQR of 0 and the rule reads it as registered.

## The verdict line

`day59-pf.py` prints its per-cell lines exactly as on the Workstation (`DAY59 G1 ...`, `DAY59 G2 ...`, `DAY59 G3 ...`,
`DAY59 TIME ...`, `DAY59 VERDICT rig=pro-maxq shape=pftime integrity=... -> ...`, the same for `pfnaked`). The kit's
`maxq-verdict.py` applies the rule above to those lines and prints one line, the last line of `out-*/VERDICT.txt`:

`DAY59 VERDICT rig=pro-maxq card=<idx> power_limit=<W> g1=.. g2=.. g3=.. pftime=.. pfnaked=.. -> qualifies|does-not-qualify|void`

- `qualifies`: G1, G2, G3 PASS, `pftime` `pf_wins`, `pfnaked` not `pf_loses`, both timing cells `integrity=ok`.
- `does-not-qualify`: a gate failed (which voids the timing verdict, so the rule is not met), or the rule is not met.
- `void`: a reading is missing (a skipped or killed cell) or a timing cell's integrity failed. No decision; rerun.

Run on the Workstation receipt (`pro-single-day61/`), the readers reproduce its four `reading.log` files byte for
byte and the combiner prints `... g1=PASS g2=PASS g3=PASS pftime=pf_wins pfnaked=pf_flat -> qualifies`.

## What each outcome leads to

- `qualifies`: the Max-Q gets its own arm in `in_token_prefetch_for_device` (`(true, "pro6000-maxq-receipt")` for a
  `RtxPro6000Blackwell` name containing `Max-Q`; the Server Edition stays `no-receipt-default`), with the test,
  docstring, `docs/decisions/MOE-INTOKEN-PREFETCH-DEFAULT.md`, `docs/FLAGS.md` and a new DAY59.md section 4 carrying
  this receipt. A HEAD boot on the Max-Q then prints `[moe-cache] in-token prefetch: on (card pro6000-maxq-receipt)`;
  that boot line checks the arm and its source only, not the timing. The change is its own commit.
- `does-not-qualify`: the Max-Q keeps `no-receipt-default` (off); the receipt is recorded in DAY59.md section 4.
- `void`: nothing changes; the cell is rerun in another quiet window.

The decision record's "Owed" item (a second MoE family on the legacy slot-cache spill path) stays owed on the Max-Q.

## The registration, quoted in full (DAY59.md lines 22-70; identical at the sitting tree `f0e77b8be` and at `2ccb038d1c`)

```text
## 1. Pre-registration

**The binary.** One `run-gen`, `run-spec` and `memra-server` from the tree at the cells' commit (named in the sitting
before it runs), used by both arms: OFF (the variable unset, the naked default) and PF (`MEMRA_MOE_PREFETCH=1`). The
artifact is the MoE door's approved one (`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, `df27a780...7adf`).

**Correctness gates, per card (pass/fail; any failure voids that card's timing verdict).**
- G1, the tape: `run-gen` OFF and PF at two shapes, the pressure shape (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32
  MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`, day 18's) and a heavy one (`MEMRA_MOE_SLOTS=512`, the same otherwise):
  every run `MATCH`, one `tokens:` tape per shape across both arms.
- G2, speculative self-consistency: `run-spec` PF at the pressure shape, K=1..8: `=== SELF-CONSISTENCY PASS ===` and
  every K `self-consistency: PASS (identical to plain target)`.
- G3, the serving shape: `memra-server` booted OFF then PF on the pressure MoE environment, the same greedy requests
  in each boot (three sequential, then four concurrent, 48 tokens each, `/v1/completions`): every response of the PF
  boot byte-equal to the same request's in the OFF boot, no request error (`day59-serve-cell.sh`).

**The timing cells, per card.** Two shapes, each its own collector hold, 20 runs, order 1 (OFF, PF) x 5 and order 2
(PF, OFF) x 5, N=5 per arm per order, 250 ms telemetry, the runner under the 1200% CPU cap (12 pinned cores on a box
without systemd):
- `pftime`, the pressure shape (where misses exist);
- `pfnaked`, the naked shape (`MEMRA_NGEN=32`, prompt `55 88 13`, no MoE variable: the default residency on that card).

**Readings and the rule** (`day59-pf.py`). Gen-only decode seconds (the day-18 observation), the steady window beside
it. `noise` the larger IQR of the two arms. Per shape: `pf_wins` iff `median(PF) < median(OFF) - noise` in order 1, in
order 2 and pooled; `pf_loses` iff `median(PF) > median(OFF) + noise` in both orders; `pf_flat` otherwise.

**What follows (the flags doctrine, per card).** PF becomes that card's naked default iff G1, G2 and G3 pass there,
`pftime` reads `pf_wins` and `pfnaked` does not read `pf_loses`; the promotion itself is the owner's call, and a
default flip is its own change (the variable keeps `=0` as the rollback seam). `pf_loses` or `pf_flat` at the pressure
shape on a card: the in-token prefetch is not that card's default, and if it holds on both cards the door is deleted
by the lane that measured it (the flags doctrine: losing or flat arms are deleted), unless the MoE slot cache door
(which carries its own prefetch, `DAY50.md`) is deleted first, in which case the decision stands for the legacy alone.

**Decide-by** 2026-10-04 (the MoE slot cache door's date, the two are read together), in the row.

**What each card can decide.** Each card its own default; the RTX 5090 (which needs its reset first) and the target
card are read separately.

## 1a. The binaries, named before any cell

One build label, `c60`, serves this day and `DAY60.md`: `run-gen-c60`, `run-spec-c60` and `memra-server-c60`, all
from one tree, the lane's tip at the sitting (engine of `fec3c582f`, which adds `DAY60.md`'s log-only
`--moe-dispatch-clock`; no arm here sets it, and without it the slot cache keeps no clock). `day59-cell.sh` names
those three binaries (it named `-final` before this section; the day-51 `-final` binaries in the local binary
directory are a different tree and stay untouched for `DAY51.md`'s queued cells). The exact commit is recorded in
each cell's `tree.sha` and `binary.sha256` and in the sitting's build log.

Pinned (`DAY61.md` section 2b, before any cell): the lane's tip now carries DAY61's I11 and I12, so the label `c60`
is built from `da649107c`, the commit whose engine is `fec3c582f`'s as this section names.
```
