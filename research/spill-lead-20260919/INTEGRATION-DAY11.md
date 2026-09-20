# Integration: day 11 (`lane/spill-integ8-20260921`, from `main` `30e9c7a38` = #576 merged)

## State at open (21:15Z)
Mac session finished (owner note). #573 (old session integ6) and #576 (integ7: C day 10, records, TESTING.md rewrite)
are on `main`. Other sessions on this rig: other Claude sessions (`wt-fix-379-suffix`, `wt-api-lockstep-router`,
`wt-mtp-*`) and codex PRs. Conflict resolution done on request for #566 (`docs/ROUTER.md`, union),
#557 (`research/INDEX.md`, main's spill-lead row) and #569 (`INDEX.md` + boundary allowlist, union); each pushed with
a comment; #566 pushed under its own `MEMRA_RELEASE_QUALIFICATION_MODE=development` (that branch retires the perf
skip), #557 with the logged `MEMRA_SKIP_PERF_CI=1`. None merged by this session.

## Lane B day 11 (`d1844b3c0`, pushed by the lead with the logged override; engine files in range are `main`'s plus
## the gate binary)
`--reclaim-cycles N` on `kv-tier-gate` (cli, `reclaim_contract::classify_cycles`, `active::Roundtrip`/`write_cycles`,
gate loop), tier suite `tests/reclaim/{main,day11}.rs`, `verify-day11.py`, TESTING.md line. Series verdicts verbatim:
```text
pro-single/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
rtx5090/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
RECLAIM-CYCLES: class=one-time-driver-mapping-metadata cycles=5 granule=2097152 residual_first=2097152 residual_last=2097152 g1_reclaim_qualified=false
pro-single/cycles-8192: ACTIVE-8K reclaim-cycles control not executed: REFUSED: diagnostic could not re-reserve original VMM address
rtx5090/cycles-8192: ACTIVE-8K reclaim-cycles control not executed: REFUSED: diagnostic could not re-reserve original VMM address
```
Per cycle on both cards: `free_after_restore == free_before` exactly, released 905,969,664 B, residual 2,097,152 B,
baseline drift 0, mapped-VA probe deltas 0, restored prefix bit-identical. PRO 600/600 W; the laptop 5090 reports
power limit `[N/A]` (max 175 W). Cross-box timing not compared. Lead code read: `classify_cycles` requires every cycle
`bounded_no_leak`, identical `free_before`, and residual exactly one granule for the one-time class; one cycle is
`unclassified`; growth in either the residual or the unreturned series is `growing-residual`; the per-cycle G1 line
and tightening (e) are untouched (`g1_reclaim_qualified` folds with AND). Refusals fail closed (count < 2, junk,
duplicate, no diagnostic, pooled, baseline case, second route), unit-tested.

## Lead rulings, day 11
6. **Lift tightening (e) for one class, under the series condition only.** The decision record pre-registered the
   lift: "lifting (e) for a specific class needs a lead ruling backed by evidence on both card classes". Both card
   classes now carry `one-time-driver-mapping-metadata` from a 5-cycle series with drift 0. Ruling:
   `g1_reclaim_qualified=true` with a nonzero residual is allowed only when all of: the run is a `--reclaim-cycles N`
   series with N >= 5; `residual_series_class = one-time-driver-mapping-metadata`; (a) to (c) hold in every cycle;
   the restored prefix is bit-identical in every cycle; baseline drift is 0. The label becomes
   `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, N cycles)`. A single roundtrip with a nonzero
   residual, a series shorter than 5, any other class, and any pooled run stay `not G1 PASS` / `false` /
   `not-applicable-pooled`. Criteria (a) to (d) are unchanged; (e) stays in force for every other shape. B day 12
   lands the gate line, `verify-day12.py`, the decision-record paragraph (evidence table, this ruling), the
   TESTING.md line, and reruns the 32k series on both cards to produce the label; the label exists only when the
   rerun prints it.
7. **8k cycles control.** The mapped-VA probe refused to re-reserve the original VMM address for the small 8k planes
   (K 5 granules, V 3) on both cards, deterministically; it freed the stray reservation and refused, as designed. No
   probe or CLI change. The 8k evidence stays the single-roundtrip `ACTIVE-8K G1 PASS` with residual 0 on both
   cards (day 10 PRO, day 9 5090). The limitation goes into the door doc as an open item; revisited only if the door
   is promoted at decide-by.
8. **Lane A local flake.** `crates/memra-tier/tests/storage/day4.rs` returns `Err(Busy)` at :143, :240, :309, :471
   on this rig only (3 of 53 on a quiet release rerun; 276/276 on the PRO). A day 10 task: reproduce under a CPU
   quota, decide race versus rig, fix or label; no engine change without a receipt. Dispatched after C day 11
   finishes (two-agent cap).

## Lane C day 11 (`f5c5f592e`, pushed by the lead with the logged override plus a boundary pin)
Installer catalog from the model plan: new `crates/memra-gguf/src/expert_banks.rs` (`expert_bank_catalog(plan, contract,
census)` selects the three bank requirements per MoE block by semantic id `LayerTensor::MoeExpert{Gate,Up,Down}Bank`,
trunk by position and MTP by depth, binds through `TensorContract::bind`; no tensor name spelled in the installer, no
architecture allowlist, no new env read). `native.rs` takes the contract from the model pack and the census from
`census_from_gguf`, refuses with `REFUSED: experts-via-tier expert catalog refused: <detail>` exit 2 (no MoE
projections, missing or ambiguous contract entry, contract `Missing`/`DuplicateCensusName`/`ShapeMismatch`, undeclared
or unconsumed scale planes, non-GGUF dialect, plan/model layer disagreement), prints
`[experts-via-tier] catalog blocks= banked= projections= catalog_sha256= records= records_sha256=`. Nits landed:
`expert_bank_cli` splits at `=` and matches keys exactly (look-alikes are usage errors), `SLOT_TAIL_PAD_BYTES` shared by
`gpu_slot_bytes` and `moe_cache::hard_slot_bytes` (10 literals, all 8, `slru-synthetic.json` re-pinned with identical
rows), `#[doc(hidden)] pub mod banked_residency` with no crate-root re-exports. Parity: `catalog_sha256`
`2204b15974f6c5e7794d6f4912af1ec6961f52bf53f6f82325f94c123bdefde8` (41 blocks, 123 projections) in all three cells and
recomputed by `verify-day11.py` from memra's own `tensor-census.tsv` with the day-10 literal spelling; `installed
artifact_sha256=df27a780...7adf host_slots=16 max_expert_bytes=860160` byte-identical to day 10. Cells (BOX3, 600 W,
N=1, `executed-not-qualified`, `pro-single-day11/`):
```text
(a) run-gen default:            prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH   [expert-gpu-slru] slots=90705 allocated_bytes=78021538440 evictions=0
(b) run-spec gpu-bytes=6881344: === SELF-CONSISTENCY PASS ===   [expert-gpu-slru] slots=8 allocated_bytes=6881344 evictions=674569
(c) run-spec gpu-bytes=6021176: REFUSED: experts-via-tier GPU bank budget cannot hold the eight-slot minimum (requested 6021176, minimum 6881344, ceiling 77968398729)   exit 2, collector refused
--validate: cells: 3, failed_commands: 0, refused_commands: 1, qualification: false
```
Push note: the pre-push public-boundary gate flagged the two gzipped 52 MB SLRU trace logs of cell (b)
(`provider_name_aws`, "first hit line 1810" of the compressed byte stream); the decompressed text has zero
hits for that provider-name pattern (the rule text is not quoted here because the gate fires on it), so the lead pinned both `(path, sha256)` in `tools/public-boundary-allowlist.jsonl`
with that reason (`f5c5f592e`). Door doc item 3: first half landed, scale admission still pending.

## Lane A day 10 dispatched (ruling 8)
Local storage `Err(Busy)` flake: reproduce 5x under the CPU quota, classify race versus test timing versus rig, fix only
what the evidence supports, no GPU.

## Lanes
- B day 11 sealed and pushed (`d1844b3c0`); B day 12 dispatched (ruling 6).
- C day 11 sealed and pushed (`f5c5f592e`); merged into integ8.
- A day 10 running (ruling 8). D sealed. E docs folded into #576. F idle.
