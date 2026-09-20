# KV physical reclaim: VMM-backed planes, not pool trim (2026-09-20)

**Status:** gate-only door (`kv-tier-gate --kv-allocator vmm`, default `pooled`), decide-by 2026-10-04.
Not a serving default. Lane: `research/spill-b-20260919/{DAY8.md,DAY9.md,RECLAIM-DESIGN.md}` (rented RTX 5090)
and `DAY10.md` (one RTX PRO 6000 Blackwell).

**G1 as of 2026-09-20:** `ACTIVE-8K G1 PASS` on the rented RTX 5090 and on one RTX PRO 6000 Blackwell.
32k on both cards: bit-identical, one-granule residual unclassified, not G1 PASS. The 8k pooled control on
the PRO card reports `not-applicable-pooled`. B's verifier prints the 32k label with a dash; the wording here
follows the writing rule, the content is unchanged.

## Question
When the tiered-KV materializer demotes a prefix (D2H via `memra_engine::tier_transfer::CudaTransfers`)
and the governor drops the device charge, does device memory actually come back?

## Measured (rented RTX 5090, Qwen3.8-27B NVFP4, `kv-tier-gate --case active`, collector-locked)
- **Pooled planes (today's `Cache::new`)**: `ACTIVE-8K copy/restore bit-identical, no reclaim, not G1 PASS`.
  32 demotes / 32 reloads, 243,269,888 B of sources dropped (all 32 `Rc` owners 1 to 0, destructor reached,
  registry 0), pool *used* fell by that amount, pool *reserved* unchanged, `cuMemGetInfo` free unchanged at
  17,934,516,224 B. `cuMemPoolTrimTo(pool, 0)` returned nothing. Verdict line: `RECLAIM-DIAG: freed but
  not observable`. So the naive fix (trim after demote) is **rejected by measurement**, not by argument.
- **VMM planes** (`cuMemAddressReserve` fixed VA + `cuMemCreate`/`cuMemMap`/`cuMemSetAccess` at the
  2,097,152 B granularity; demote = unmap + release whole chunks inside the demoted range; restore = map +
  H2D to the same VA): `ACTIVE-8K G1 PASS`. Released = reacquired = **201,326,592 B**, driver free VRAM
  moved by exactly that (17,613,651,968 to 17,814,978,560 to 17,613,651,968), all 32 planes restored at their
  original virtual addresses, tokens/logits/state byte-identical to the frozen 8k baseline. Kernels and the
  numeric program are untouched: same addresses, same bytes.
- **32k**: bit-identical and 903,872,512 B observed vs 905,969,664 B released, exactly one granule short on
  both demote and restore; free VRAM restored exactly. 16k shows a 0 B residual; freeing a spare VA
  reservation and `cuCtxSynchronize` return nothing. Residual **unclassified**, so the label stays
  `ACTIVE-32K physical reclaim/restore bit-identical, one-granule residual unclassified, not G1 PASS`.

## Measured on the target card (one RTX PRO 6000 Blackwell, 96 GB, 600/600 W; `DAY10.md` on `lane/spill-b-20260919`)
- New 8k and 32k baselines frozen on the card (`BOX3-BASELINES.json`); never interchangeable with the
  RTX 5090 bundles. Every cell N=1, `--rig pro-single`, `/tmp/memra-gpu.lock`, 250 ms telemetry.
- **8k VMM**: `ACTIVE-8K G1 PASS`, released = reacquired = 201,326,592 B, residual 0, class `none`, free VRAM
  85,863,301,120 to 86,064,627,712 to 85,863,301,120 B, all 32 planes at their original VA.
- **32k VMM**: bit-identical, 903,872,512 B observed vs 905,969,664 B released, residual **2,097,152 B**,
  class `unclassified`, not G1 PASS. The mapped-VA probe on the actual demoted planes reads
  `mapped_va_release_delta_bytes=0`, `mapped_unmap_delta_bytes=0`, `mapped_va_roundtrip_equal=true` on all
  32 rows: the granule does not return on VA free or on unmap, so a `va-reservation-page-table` class is
  refuted rather than assumed. Neither card supplies the required classification.
- **8k pooled control**: `g1_reclaim_qualified=not-applicable-pooled`, zero observed reclaim, zero trim release.

## Rejected alternative
Paged KV block pool (reclaim = pool-available bytes): needs kernel-side indirection, i.e. a numeric-program
change; recorded in `RECLAIM-DESIGN.md`, not implemented.

## G1 reclaim criterion (fixed before the 32k label, applies to every run)
(a) observed free-VRAM rise >= released chunk bytes minus one allocation granule; (b) `free_restored ==
free_before` exactly; (c) demote and restore deltas identical; (d) any residual is *classified* by a
diagnostic, never inferred. **Tightening (e), day 10:** `g1_reclaim_qualified=true` additionally requires
`residual_bytes=0` (`kv_tier_gate/active.rs`: `reclaimed = vmm_granularity != 0 && reclaim_observed &&
observation.residual == 0`; `research/spill-b-20260919/verify-day10.py` enforces the same). A classified
nonzero residual is recorded with its class and bytes but does not qualify; lifting (e) for a specific class
needs a lead ruling backed by evidence on both card classes. The gate prints the raw equality field and the
residual bytes and class (`kv_tier_gate/reclaim_contract.rs`); a pooled run publishes
`g1_reclaim_qualified=not-applicable-pooled` and can never carry the label. (e) is a tightening, never a
relaxation, of (a) to (d). Lead ruling 6 (Day 11 to 12 below) lifts (e) for exactly one series shape.

## Day 11 to 12: the residual classified by its series, and ruling 6

Day 11 (`DAY11.md`, source `1566d4f81`) added `--reclaim-cycles N`: N demote/restore roundtrips in ONE
process on ONE cache under the same tokenwise program, with the per-cycle receipt set under `cycle-<k>/`
and the series in `reclaim-cycles.tsv` / `reclaim-cycles.txt`. Both card classes ran the 32k series with
N=5 (`--case active --context 32768 --tiers host --same-program --kv-allocator vmm --reclaim-diagnostic
--reclaim-cycles 5`, N=1 cell each, collector-locked, 250 ms telemetry; no timing compared between the
cards). Bytes verbatim from `DAY11.md`:

One RTX PRO 6000 Blackwell (96 GB, 600/600 W):

| Cycle | Free before demote | Free after demote | Free after restore | Released chunks | Reclaim delta | Reacquired | Residual | Restore residual | (a)-(c) | per-cycle G1 line |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 2 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 3 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 4 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 5 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |

RTX 5090 Laptop GPU (local development rig, `power.max_limit 175.00 W`, no frozen bundle of its own):

| Cycle | Free before demote | Free after demote | Free after restore | Released chunks | Reclaim delta | Reacquired | Residual | Restore residual | (a)-(c) | per-cycle G1 line |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 2 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 3 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 4 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |
| 5 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | 2,097,152 | 0 | true | `false` |

On both cards `free_before_drift_bytes` is 0 in every cycle, the mapped-VA probe reads
`mapped_va_release_delta_bytes=0`, `mapped_unmap_delta_bytes=0`, `mapped_va_roundtrip_equal=true` in every
cycle, every cycle's `restored-prefix-state.tsv` equals the suspended `prefix-state.tsv`, and the series class
is `one-time-driver-mapping-metadata`. Day-11 verdicts, verbatim:

```text
pro-single/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
rtx5090/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
```

**Lead ruling 6 (day 11, `research/spill-lead-20260919/INTEGRATION-DAY11.md`), verbatim:** "Lift tightening
(e) for one class, under the series condition only. The decision record pre-registered the lift: 'lifting (e)
for a specific class needs a lead ruling backed by evidence on both card classes'. Both card classes now
carry `one-time-driver-mapping-metadata` from a 5-cycle series with drift 0. Ruling:
`g1_reclaim_qualified=true` with a nonzero residual is allowed only when all of: the run is a
`--reclaim-cycles N` series with N >= 5; `residual_series_class = one-time-driver-mapping-metadata`; (a) to
(c) hold in every cycle; the restored prefix is bit-identical in every cycle; baseline drift is 0. The label
becomes `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, N cycles)`. A single roundtrip
with a nonzero residual, a series shorter than 5, any other class, and any pooled run stay `not G1 PASS` /
`false` / `not-applicable-pooled`. Criteria (a) to (d) are unchanged; (e) stays in force for every other
shape. B day 12 lands the gate line, `verify-day12.py`, the decision-record paragraph (evidence table, this
ruling), the TESTING.md line, and reruns the 32k series on both cards to produce the label; the label exists
only when the rerun prints it."

Implementation (day 12, `kv_tier_gate/reclaim_contract.rs::series_verdict`, `active.rs::write_cycles`, the
gate status line; offline `research/spill-b-20260919/verify-day12.py`; CPU red arms
`crates/memra-tier/tests/reclaim/day12.rs`): the per-cycle line is untouched; the series
`g1_reclaim_qualified` and the printed label come from the series verdict alone; `reclaim-cycles.txt` records
`series_min_cycles=5` and `series_label` (the label, or `not-printed`). The `K` tag follows the committed
context, so a 16k series cannot print `32K`. Criteria (a) to (d) are unchanged and tightening (e) stays in
force for every other shape: a single roundtrip with a nonzero residual, a series shorter than 5, any other
class, a drifting baseline, a differing restore and any pooled run stay `false` / `not-applicable-pooled`.

**Lead ruling 7 (8k cycles control), open item.** The 8k series control refused on both cards in cycle 1,
after the full prefill, inside the mapped-VA probe (`KvPlane::probe_demoted_va_release`): for the small 8k
planes (K 5 granules, V 3) the driver returned a different address when asked to re-reserve the original
base, the probe freed the stray reservation and refused, as designed, verbatim
`REFUSED: diagnostic could not re-reserve original VMM address` (PRO twice, laptop once; deterministic).
Fail-closed: no token executed against the suspended cache, no `ACTIVE.txt`, no cycle row. No probe or CLI
change. The 8k evidence stays the single-roundtrip `ACTIVE-8K G1 PASS` with residual 0 on both cards (day 10
PRO, day 9 rented RTX 5090). Revisited only if the door is promoted at decide-by.

## Scope
The gate constructs the cache directly with VMM planes (`Cache::new_with_allocator(…, KvAllocator::Vmm)`;
34 VMM / 0 pooled planes at position 0 on the 8k rerun, `construction=direct`, `empty_plane_swap=false`); the
earlier empty-plane swap is retained only as history in `DAY9.md`. No footprint, admission, or serving claim.
`unsafe` is confined to the driver FFI plus one documented `upgrade_device_ptr` inside a private `KvPlane`
owner whose `Drop` always `leak`s the slice before unmap/release/address-free. **Surface this door owns
(what the decide-by promotes or deletes):** `KvAllocator` + `Cache::new_with_allocator` in `memra-kv`,
`KvDev::alloc_vmm_u8` on `Engine`, `KvPlane` (`crates/memra-kv`), the `vmm` arm and its receipts in
`kv_tier_gate/active.rs`, and the lane cells. Containment is a call-site policy (only the gate constructs
VMM planes today), not a type-level guarantee.

## Decide-by 2026-10-04
Promote to the naked default for the tiered materializer only with the 32k residual classified on the
target card and the serving-shape gates run; otherwise delete the door (env read, dispatch arm, gate cells)
per the door-hygiene rule. Until then no relaxation of (a) to (d).
