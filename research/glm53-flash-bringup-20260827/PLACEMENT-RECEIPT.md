# GLM-5.3-Flash NVFP4 — placement receipt

**Lane** `lane/glm53-flash-bringup` · **date** 2026-08-28 · **status** modeled + gated, NOT yet run
on the real artifact.

**Question.** The minted NVFP4 artifact is 190.7 GB and the serving box is 2×96 GB. Can it fit and
serve, using memra's existing expert-residency machinery rather than new invention?

**Answer.** Yes, and with more room than the framing suggested. Two premises in the brief were
wrong (§1, §2), the residency machinery already covers this model with essentially no wiring (§4),
and the real risk is not capacity — it is that **two defaults will each grab the memory the 1M KV
plane needs, before that plane exists** (§6).

Arithmetic is reproducible: `placement-arith.py` in this directory (reads `mint-receipts/
nvfp4-config.json` + `glm-index.json`). Units: **GB = 10⁹ bytes** (the mint log's unit — index
`total_size` 328,326,771,576 = 328.3 GB); **GiB = 2³⁰** (nvidia-smi's unit).

---

## 1. The byte split — VERIFIED, and it is not what the brief assumed

| tier | bytes | share |
|---|---:|---:|
| routed experts (43 layers × 288 × 3) | **175.31 GB** (163.27 GiB) | 92.0% |
| shared experts (43 × 1 × 3) | 0.61 GB | 0.3% |
| **all expert mass** | **175.91 GB** (163.83 GiB) | **92.3%** |
| non-expert (KDA, MLA, indexer, mHC, routers, norms, embed, lm_head, vision, MTP) | **14.67 GB** (13.66 GiB) | **7.7%** |
| **modeled total** | **190.58 GB** | vs mint receipt **190.7 GB** (Δ −0.12 GB, 0.06%) |

**The brief estimated ~156 GB experts / ~35 GB non-expert. Both halves were wrong.**

- *Experts, 156 → 175.3 GB.* 156 GB is the 4.0-bit calculation (311.65 G params × 0.5 B). NVFP4 is
  **4.5 bits/element**: the FP8 per-16 group-scale plane adds 1 byte per 16 elements (+12.5% of the
  packed size, +19.5 GB). Confirmed against the repack stride the loader actually produces:
  `row_bytes = in_f/64 × 36` = 36 B per 64 elements = 0.5625 B/elem exactly.
- *Non-expert, 35 → 14.7 GB.* The quantization `ignore` list is 628 entries: **almost nothing
  outside the experts was quantized**, so the non-expert tier is mostly BF16 — but it is also much
  smaller than assumed. The largest term is the 34 KDA layers at 9.37 GB; embed + lm_head are
  1.27 GB each (not tied); vision 1.09 GB; all 12 MLA layers together only 1.25 GB.

Three independent cross-checks, all exact:

1. **Census identity.** 12,384 expert tensors ÷ 288 experts = **43 MoE layers** (42 sparse decoder
   + the MTP/NextN layer). 37,152 routed + 129 shared + 48 MLA (12 layers × q_a/q_b/kv_a/o) + 9
   dense-MLP = **37,338 quantized**, exactly the mint receipt's count. 38,770 − 37,338 = **1,432
   kept**, exactly the receipt's count.
2. **Source identity.** 76,108 source entries = 38,770 output tensors + 37,338 `weight_scale_inv`
   siblings, exactly.
3. **Bottom-up.** Summing every non-expert tensor from config dims lands at 14.67 GB, and the
   grand total reproduces 190.7 GB to 0.06%.

### The unit trap in "190.7 against 192"

That comparison mixes units and **understates the headroom**. 190.7 GB (decimal) = **177.6 GiB**.
A "96 GB" Blackwell reports 97,887 MiB = 95.6 GiB usable, so the box is **~191.2 GiB**. The weights
alone therefore fit with ~13.6 GiB to spare — not ~1 GB. The conclusion is unchanged (a 1M KV plane
is 37.6 GiB and does not fit in 13.6), but the margin is real and it is what makes §5 comfortable.

## 2. `MEMRA_MOE_CACHE` is already default-ON — there is no flag to flip

The brief states the flag's current default is OFF. It has been **ON since 2026-07-08**:

- `Engine::moe_cache_enabled()` is `std::env::var("MEMRA_MOE_CACHE") != Ok("0")` — enabled unless
  explicitly disabled (`lib.rs:4103`).
- `docs/FLAGS.md:390` spells the row `MEMRA_MOE_CACHE=0` | stage-every-token | **default-on
  2026-07-08**. `=0` is the *rollback*, not the default.

The source of the confusion was a **stale module comment**: `moe_cache.rs:15` read "Gated behind
`MEMRA_MOE_CACHE` (default off => current stage-every-token behavior)" — the pre-2026-07-08 state,
stale for seven weeks, and quoted as live in the placement brief. **Fixed in this change** (it now
states the default, the rollback direction, and why).

## 3. KV and recurrent state — quoted from `crates/memra-kv/src/lib.rs`

`LatentKvLayer`: latent width = `kv_lora_rank + rope_head_dim` = 512 + 0 (NoPE) f32 = **2 KiB/token/
layer**; the DSA indexer plane is `2 × index_head_dim` = 256 f32 = **1 KiB/token/layer**; resident
pool keys are `max_ctx/pool × index_head_dim` f32. 12 MLA layers (11 trunk + 1 in the MTP layer, per
`index_share_for_mtp_iteration`). KDA recurrent state is context-independent.

| ctx | latent | indexer | pool keys | KDA state | **total/seq** |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 0.81 | 0.40 | 0.05 | 0.15 | **1.41 GB** (1.31 GiB) |
| 131,072 | 3.22 | 1.61 | 0.20 | 0.15 | **5.19 GB** (4.83 GiB) |
| 262,144 | 6.44 | 3.22 | 0.40 | 0.15 | **10.22 GB** (9.52 GiB) |
| 1,048,576 | 25.77 | 12.88 | 1.61 | 0.15 | **40.42 GB** (37.64 GiB) |

The source's own 1M figures ("12.9 GB" indexer, "1.6 GB" pool keys) reproduce exactly. It also
documents an unimplemented **tail-ring** reduction that would replace the 12.88 GB indexer plane
with a few MiB at no numeric cost — the single largest available KV saving, and the thing that
would move the 1M row in §5 from 81% to ~89% resident.

## 4. Does `moe_cache` support this model? — Yes, and it needed no wiring

Honest assessment, from reading the code rather than the header:

**Structurally a good fit — better than the header's wording implies.**

- **The cache is quant-format agnostic.** There is no `qtype`, block size, or scale anywhere in
  `moe_cache.rs`. A slot is `max_block_bytes` of opaque bytes keyed by `BlockId{layer,proj,ex}`.
  The header's "the same *GGUF* block bytes" is about the internal repacked layout, not the file
  format.
- **NVFP4 arrives as one contiguous block.** `repack_modelopt_to_gguf` fuses modelopt `weight`
  (U8 e2m1) + per-16 `weight_scale` (F8_E4M3) into memra's `block_nvfp4` layout at load. For every
  glm5 projection this is **4,718,592 B** — gate/up `[in 4096, out 2048]` and down `[in 2048, out
  4096]` give the *same* stride, so the SLRU runs as **one size class** (no `MEMRA_MOE_SIZE_AWARE`
  needed).
- **The macro scale cannot be moved by residency.** `weight_scale_2` rides `HostExps::macros` (host
  metadata) and is folded post-matmul — it is not in the cached bytes. This is what makes §B.3 hold
  for NVFP4 rather than merely *seem* to.
- **The safetensors NVFP4 MoE loader already exists** and is exercised by other checkpoints
  (Step-3.7-Flash-NVFP4, the unsloth 35B-A3B ST class), including the `.memra-repack` on-disk tier
  and the pinned-host tier.
- **Names resolve with no new mapping.** `resolve_ggml` handles per-expert MoE names generically;
  `SafetensorsSource::find` maps `<stem>.scale` → `weight_scale_2` / `weight_global_scale`; and
  `lookup()` carries the `model.` ↔ `model.language_model.` prefix fallback that glm5's VL wrapper
  needs. All three go through the same fallback, so the macro resolves for the wrapped names too.
- **Routing and dispatch are already correct.** `cfg.sigmoid_router()` returns `Some((2.5, true))`
  unconditionally for glm5 (fixed earlier in this lane), and 288 experts / 8 active / 1 shared need
  nothing special.

**One consequence worth stating plainly, because it is a serving fact, not a defect:** glm5_next is
denied *every* fused and device-dispatch MoE arm by construction — `moe_ffn_pairs`, `moe_ffn_dev`
and the grouped-decode pair all require `sigmoid_router().is_none()` (glm5 is sigmoid `noaux_tc`)
and `!swiglu_clamped_at(il)` (glm5's PRE-clamped SwiGLU has no fused twin), and the macro-carrying
banks are denied again by `no_exp_macros`. **glm5 therefore serves on the per-expert sequential
`qmatvec_view` loop.** That is the correct and only numerically-sound arm today, it is why the
residency comparison in §7 is unusually clean (provenance-only), and it is the obvious future perf
lever — a PRE-clamp fused epilogue plus macro folding would unlock gdec for this family.

**What was wired:** only the stale-comment fix in `moe_cache.rs` (§2) — plus the gate in §7.
Nothing else was needed, and nothing was invented. That is the intended outcome of "use the
existing machinery", not a shortcut.

**Flag decision (house flag law).** **Reuse `MEMRA_MOE_CACHE`; glm5 inherits the fleet default-ON;
no new flag, therefore no new `docs/FLAGS.md` row.** Reasons, recorded:
1. It is already the fleet default (2026-07-08) — this is not a new decision for glm5 to make.
2. For glm5 it is not an optimization but **the product path**: at the 1,048,576-token context this
   model's whole claim rests on, fully-resident experts cannot coexist with the KV plane (§5).
3. The identity property is now gated for this model's exact class (§7).
4. Rollback is unchanged and documented: `MEMRA_MOE_CACHE=0`.

## 5. Resident footprint on a 2×96 GB box

Box usable ≈ **191.2 GiB** (95.6 GiB/card; re-measure with `nvidia-smi` on the target).

| configuration | weights | +KV @1M | +KV @128k | verdict @1M |
|---|---:|---:|---:|---|
| fully resident (all experts) | 177.5 GiB | **215.1** | 182.3 | **does not fit** |
| SLRU 15% hot (5,572 slots) | 38.7 | 76.4 | 43.5 | fits |
| SLRU 30% hot (11,145 slots) | 63.2 | 100.8 | 68.0 | fits |
| SLRU 50% hot (18,576 slots) | 95.9 | 133.5 | 100.7 | fits |

Taking KV out **first** and solving for slots is the useful form:

| ctx | KV | free for experts | slots | resident share | host-only |
|---:|---:|---:|---:|---:|---:|
| 131,072 | 5.2 GB | 176.8 GB | 37,152 | **100%** | 0.0 GB |
| 262,144 | 10.2 GB | 171.8 GB | 36,409 | 98% | 3.5 GB |
| 1,048,576 | 40.4 GB | 141.6 GB | 30,009 | **81%** | 33.7 GB |

(assumes 8 GB across both cards for CUDA context + activations + workspace)

**So the hot-mass question is far less load-bearing than expected.** The brief's ~15–20% hot mass
was the reason to believe this would work; in fact the box holds **81% of all routed blocks even at
1M**, and 100% at 128k. The SLRU is not being asked to find a small hot set — it is being asked to
choose which ~19% to leave home. Miss rates should be very low, but that is a projection.

## 6. Recommended configuration — and the two defaults that will hurt you

**Neither relevant default is safe here, and both fail the same way: they size themselves against
free VRAM *before the KV plane is allocated*.**

1. **`MEMRA_MOE_RESIDENT=0`** — pin it. The resident planner (`hybrid.rs::should_reside`) budgets
   `free − trunk_bytes − headroom(2 GB)` and **has no KV term at all** (read at `hybrid.rs:286-305`).
   Per card under PP-2 that is roughly 102.6 − 7.3 − 2 ≈ **93 GB budget against ~88 GB of expert
   mass → it answers RESIDENT**, slabs the experts, and leaves ~7 GB for a KV plane that needs ~20.
   *This is not hypothetical:* the gate in §7 failed on exactly this on its first run, where the
   planner slabbed a 0.00 GB expert set and no SLRU was ever built.
2. **`MEMRA_MOE_SLOTS=<N>`** — pin it explicitly. Left to `auto`, the SLRU takes
   `MEMRA_MOE_VRAM_FRAC` (0.85) of free VRAM, measured at the same pre-KV moment. Use the §5 table:
   ~15,000 slots/card for 1M, and cap rather than fill for shorter contexts.

Also recommended:

- **Keep the host bank pinned** (`MEMRA_MOE_PINNED` is auto-on with the cache). The full 175.31 GB
  bank is host-resident; check the box's RAM and `MemAvailable` first — the M3 lesson (2026-07-07)
  is that pinning only pays when the *unpinned* remainder still fits page cache.
- **Shared experts stay resident** (0.61 GB, every token uses them). They are not SLRU-placed.
- **The MTP layer's expert bank is 4.08 GB** and can stay host-only while MTP speculation is off —
  a free 4 GB if the 1M row is tight.
- **One SLRU size class** — leave `MEMRA_MOE_SIZE_AWARE` off; all three projections share a stride.
- Re-check placement balance under PP-2: MoE layers do not split 21.5/21.5, and the MLA layers
  (which carry all the KV) are every 4th layer, so per-stage KV is not automatically even.

## 7. The gate

`crates/memra-engine/tests/glm5_moe_residency_gpu.rs` — pins MOE-SLRU-PLAN §B.3 for the **NVFP4
macro-carrying** expert class.

There was already a §D.2 gate, and it is worth being precise about what it did and did not cover:
`src/bin/kernel_check.rs:8262` (`d2-cache-bit-identity`) stages one expert of a real 35B GGUF into
a scratch and into a slot and compares a single `qmatvec_view`. Its dtype arm accepts
`IQ3_S | IQ4_XS | Q6_K | Q8_0` and skips everything else — **NVFP4 has always taken the skip
arm** — and it needs a multi-GB checkpoint at a hardcoded path, so it does not run in CI. The new
gate is complementary, not a replacement: end-to-end rather than one block, the NVFP4
macro-carrying class rather than k-quants, and CI-reachable on a fixture.

Two arms, fresh `Engine` + fresh model load each (the resident-vs-SLRU decision and
pinned-vs-paged host buffers are both taken at load time): **`FullyResident`** (device-resident
slabs) vs **`SlruResidency`** (`MEMRA_MOE_CACHE=1`, `MEMRA_MOE_RESIDENT=0`, `MEMRA_MOE_SLOTS=8`
against 24 live routed blocks, so evictions are forced). Fixture is the real glm5 routing program
(sigmoid `noaux_tc`, 2.5 scaling, `norm_topk_prob`, 1 shared expert, PRE-clamped SwiGLU), with
banks minted as **real NVFP4 blocks** (`f32_to_nvfp4`) plus a **non-unit macro plane**
(`<stem>.scale`). `moe_intermediate_size` is 64, not 32, because NVFP4 needs `in_f % 64 == 0` and
down's `in_f` is that dimension.

Because glm5 is denied every fused arm, both arms run the *same* per-expert kernels and differ in
exactly one thing — whether the block was already resident. Equality is asserted **bit-exact**
(`f32::to_bits`), not within a tolerance.

**Results.** Tree `5a51cb06865e0b4cb9d1246ed865164a3ca61978` ("glm53-flash: indexer performance
measured on serving-class hardware") plus this lane's uncommitted changes; laptop RTX 5090.
Invocation:

```
flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
  cargo test -p memra-engine --test glm5_moe_residency_gpu -- --ignored --test-threads=1 --nocapture
```

```
[residency] slots=8 hits=243 misses=558 staged=2571264B (live routed blocks = 24, slots pinned to 8 to force eviction)
[residency] 11 rows bit-identical: fully-resident slabs vs SLRU hot-set residency
[mutation] one byte inside a cached NVFP4 block     -> 7/11 rows differ
[mutation] one per-expert weight_scale_2 macro      -> 7/11 rows differ
[mutation] caught 2/2 mutants
```

Correctness only — **no timing is reported from this rig** (rig law). The whole bar sweep was
re-run at this SHA after a mid-session HEAD move by another lane in this shared checkout, so the
numbers above and the gate table below all attribute to one tree:

| gate | plain | `--ignored` |
|---|---|---|
| `glm5_kpool_indexer_gpu` | 4 | 8 |
| `kda_quant_operand_gpu` | 0 | 4 |
| `glm5_routed_router_gpu` | 5 | 3 |
| `swiglu_preclamp_gpu` | 3 | 7 |
| `hyper_connections_gpu` | 1 | 6 |
| `mla_gpu_forward` | 0 | 5 |
| `kda_fixture_gpu` | 3 | 0 |
| `mla_fixture_forward` | 3 | 0 |
| `mla_fixture_load_gpu` | 0 | 1 |
| **`glm5_moe_residency_gpu`** (new) | **1** | **2** |

`cargo check --workspace --tests` clean; `cargo fmt --all` applied; libs at
memra-gguf 162 / memra-reference 22 / memra-kv 15.

Non-vacuity is asserted, not assumed, and it **earned its keep** — the first run failed on it (§6).
The gate requires `hits > 0`, `misses > 0`, `slots < live blocks`, and `stats == None` on the
resident arm. A third, CPU-only test (`the_fixture_is_the_nvfp4_macro_carrying_shape`, runs without
CUDA) pins the fixture's shape, that the macro plane is non-unit (an all-ones plane makes
`stacked_macros` return `None` and would blind the macro mutation), and that each mutation
perturbs exactly one tensor.

The **macro mutation is the one that matters**: `weight_scale_2` is the only part of an expert's
value residency does *not* carry, and dropping it is a ~3×10⁴ error that is fluent and invisible
(the measured-garbage class, 2026-07-16). A gate that could not see it would be blind to exactly
this class's characteristic failure.

## 8. What remains unproven

1. **The real artifact has never been loaded.** It is not on this rig (`~/models/glm53-nvfp4`
   absent). Everything in §1 and §5 is *modeled* from config + census — validated to 0.06% against
   the mint receipt, but not measured. First action on the serving box: load it and diff the actual
   per-tier bytes against `placement-arith.py`.
2. **Name resolution is verified by reading, not by executing.** The `.scale` → `weight_scale_2`
   path through the `model.language_model.` prefix fallback is correct in source, and the gate
   proves macros are *read and folded* — but no synthetic-safetensors harness exists, so the
   wrapped-name resolution itself is unproven until a real load. **This is the top silent-failure
   risk**: a macro that fails to resolve defaults to 1.0 *silently*, and the result is fluent
   garbage. Assert non-unit macros in the first real load's startup log.
3. **Hot-expert mass is borrowed.** The ~15–20% figure comes from other models; for a 288-expert
   sigmoid router it is unmeasured. §5 shows the answer matters less than expected (81% resident at
   1M), but the miss rate is a projection. Measure with `MEMRA_MOE_TRACE`.
4. **No timing anywhere.** Rig law: this laptop 5090 is lock-serialized correctness only. The
   4.87 GB/token cold-miss figure in `placement-arith.py` is labeled arithmetic, not a measurement.
   Every throughput and PCIe number must come from the serving-class box.
5. **Card-class qualification.** Per the card-keyed-defaults law, a 5090 green does not qualify a
   96 GB Blackwell. The slot budget, pinned-host tier, and PP-2 placement all need a boot
   output-sample gate on the serving card class.
6. **Card capacity is assumed** (95.6 GiB from the 97,887 MiB the 96 GB Blackwell reports). Re-run
   §5 against real `nvidia-smi` free bytes.
7. **PP-2 balance is not modeled.** §5 is a box aggregate; MoE and MLA layers do not split evenly,
   so per-card slot counts need the actual stage cut.
8. **1M has never been exercised end to end** on this model — the KV table is arithmetic from the
   plane definitions, and chunked-prefill workspace at 1M is a guess (the 8 GB overhead term).
