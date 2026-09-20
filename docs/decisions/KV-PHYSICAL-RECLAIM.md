# KV physical reclaim: VMM-backed planes, not pool trim (2026-09-20)

**Status:** gate-only door (`kv-tier-gate --kv-allocator vmm`, default `pooled`), decide-by 2026-10-04.
Not a serving default. Lane: `research/spill-b-20260919/{DAY8.md,DAY9.md,RECLAIM-DESIGN.md}` (rented RTX 5090)
and `DAY10.md` on `lane/spill-b-20260919` (one RTX PRO 6000 Blackwell, pending integration).

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
diagnostic, never inferred. The gate prints the raw equality field and the residual bytes and class
(`kv_tier_gate/reclaim_contract.rs`, `kv_tier_gate/active.rs`); a pooled run publishes
`g1_reclaim_qualified=not-applicable-pooled` and can never carry the label.

## Scope
The gate swaps freshly allocated, still-empty planes for VMM planes before the first token; bootstrap still
goes through ordinary `Cache::new`, so the pooled reservation stays. No footprint, admission, or serving
claim. `unsafe` is confined to the driver FFI plus one documented `upgrade_device_ptr` inside a private
`KvPlane` owner whose `Drop` always `leak`s the slice before unmap/release/address-free. Lane B's day 10
branch adds direct construction (`Cache::new_with_allocator`, 34 VMM / 0 pooled planes at position 0, 8k
G1 PASS rerun); it is lane evidence pending integration, not a change to this tree.

## Decide-by 2026-10-04
Promote to the naked default for the tiered materializer only with the 32k residual classified on the
target card and the serving-shape gates run; otherwise delete the door (env read, dispatch arm, gate cells)
per the door-hygiene rule. Until then no relaxation of (a) to (d).
