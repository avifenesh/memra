# DeepSeek-V4-Flash loader lane — receipts (2026-08-18)

Lane 1 of dsv4-flash support: artifact loading + quantization decode truth. Ends at the
census gate; no GPU kernels, no forward pass. Branch `lane/dsv4-flash-loader` (off v0.91.0
`022d84814`), NOT merged. Artifact: `nvidia/DeepSeek-V4-Flash-NVFP4` staged on the hyperscaler bench
box at `/home/ubuntu/models/dsv4-flash-nvfp4` (157 GB, 46 shards, 135,235 tensors).

## What landed

| commit | content |
|---|---|
| `1fa804855` | `Arch::DeepSeekV4` + `DeepSeekV4Config` (config.rs), `dsv4.rs` census + 3 CPU decoders, Python oracle |
| `ff9bf0a68` | `dsv4-census` gate binary (memra-gguf bin) |
| `d51cb17a1` | compressor census fix: two shape classes (the gate's first-run catch) |
| `37e266532` | drop unused muts (zero-warning build) |

- **A — arch config**: `ModelConfig::from_config_json` now parses `model_type: deepseek_v4`
  into `Arch::DeepSeekV4` + `ModelConfig.dsv4: Option<DeepSeekV4Config>` carrying every
  forward-pass field: `scoring_func` (sqrtsoftplus), `topk_method` (noaux_tc),
  `routed_scaling_factor` 1.5, `norm_topk_prob`, `n_shared_experts`, `num_hash_layers` 3,
  `hc_eps`/`hc_mult`/`hc_sinkhorn_iters` (1e-6 / 4 / 20), MLA block (`head_dim` 512,
  `num_key_value_heads` 1, `q_lora_rank` 1024, `qk_rope_head_dim` 64, `o_lora_rank` 1024,
  `o_groups` 8), indexer block (`index_n_heads` 64, `index_head_dim` 128, `index_topk` 512),
  `compress_ratios` (len 44 = 43 trunk + 1 MTP), `compress_rope_theta` 160000,
  `sliding_window` 128, `swiglu_limit` 10.0, yarn block (factor 16, orig ctx 65536,
  beta 32/1), `num_nextn_predict_layers` 1. Missing required fields refuse BY NAME
  (from_gguf's expect style). Unknown fields tolerated by the parser's construction.
  `carried_prime_batch_eligible` exhaustive match: denied.
- **B — tensor map**: `dsv4::expected_census` derives all 135,235 tensors (names, dtypes,
  shapes, byte lengths) from config math; unit test pins the total and spot shapes.
  Splits are derived, never hardcoded: compressor ⟺ `compress_ratios[il] != 0` (41),
  indexer ⟺ ratio == 4 (21), hash routing ⟺ `il < num_hash_layers` (3), MTP layer =
  trailing `compress_ratios` entries (ratio 0 → no compressor/indexer, has gate.bias).
  `verify_census` refuses loudly per named tensor on missing/extra/dtype/shape/bytes.
- **C — CPU decoders** (`dsv4.rs`): `dequant_nvfp4_expert` (reuses the qwen-lane
  `fp8_e4m3_to_f32`; unit-pinned equal to `nvfp4_repack::dequant_modelopt_row` × scale_2),
  `dequant_mxfp4_expert` (new: e2m1 × E8M0-per-32), `dequant_fp8_blk128` (new: e4m3 ×
  E8M0 128×128 grid). `e8m0_to_f32` PROPAGATES the 0xFF NaN code (a NaN scale must fail
  the gate, not silently zero).
- **D — gate**: `cargo run -p memra-gguf --bin dsv4-census <dir>`; final output banked at
  `CENSUS-GATE.txt` (this dir; run @ `37e266532` on the box against the real artifact).
  **PASS: 135,235 tensors verified, 0 failures, 7.6 s.** The refusal path's receipt is
  `CENSUS-GATE-run1-FAIL.txt`: the first run refused with 60 named per-tensor mismatches
  (the compressor find below) and exit 1.
- **E — oracle**: independent numpy decode (`dsv4_decode_oracle.py`, this dir; stdlib
  safetensors reader, no shared code with the Rust path) vs `dsv4-census --dump`,
  sha256 of the little-endian F32 outputs — **bit-exact on all three recipes**:

| tensor | recipe | sha256 (Rust == Python) |
|---|---|---|
| `layers.20.ffn.experts.7.w1` | NVFP4 | `4f1739e443d15311ebae13e1ccc202859978ef4fa51574aae6c2f70f2d031304` |
| `mtp.0.ffn.experts.7.w1` | MXFP4 | `13ed7d5d49bf3b14c7c317ba7a00f8c578ae316513189217738ae6cadefb12e6` |
| `layers.20.attn.wq_a` | FP8+E8M0 | `9e5f3a2f9bf6acdfcf6a1dd7c5c396f45d62dee63d4a826f1494ee947b6574b7` |

## Measured quant geometry (from bytes, not ancestry)

Three recipes coexist in the one artifact:

1. **Trunk routed experts — modelopt NVFP4** (declared in hf_quant_config.json:
   NVFP4/group-16 per `layers.{0..42}.ffn.experts`, verified by the gate):
   `weight` U8 [out, in/2] (two e2m1/byte, elem 2i → low nibble), `weight_scale` F8_E4M3
   [out, in/16] (groups along IN), `weight_scale_2` F32 scalar, `input_scale` F32 scalar
   (activation-side, unused for weight decode).
   `w = e2m1 × e4m3(scale) × scale_2`. Cast provenance evidence: the artifact is a
   LOSSLESS MXFP4→NVFP4 cast (its own `cast_mxfp4_to_nvfp4.log`: 33,792/33,792 tensors,
   17.7e9/17.7e9 blocks lossless) — measured on-box: every effective scale (e4m3 ×
   scale_2) is an EXACT power of two (fraction-err 0 over 524,288 scales/tensor) and
   adjacent 16-group scale pairs are IDENTICAL (32-group ancestry). The gate hard-checks
   both on 5 sampled experts; a native NVFP4 sibling would legitimately relax this.
   Measured `scale_2`: 2^-12/2^-11/2^-13 by tensor; effective scales 2^-8..2^-4.
2. **MTP routed experts (`mtp.*`) — OCP MXFP4**, excluded from the cast
   (`exclude_modules: ["mtp.*", ...]`): `weight` I8 [out, in/2] (same nibble packing;
   dtype label differs), `scale` F8_E8M0 [out, in/32], `w = e2m1 × 2^(b-127)`.
   Measured E8M0 byte range 119..122 (2^-8..2^-5), zero 0xFF codes.
3. **All other quantized linears** (attn wq_a/wq_b/wkv/wo_a/wo_b, shared experts,
   indexer wq_b, MTP e_proj/h_proj) — **FP8 E4M3 + 128×128 E8M0 block scales**
   (config.json quantization_config: fmt e4m3, scale_fmt ue8m0, weight_block_size
   [128,128]): `weight` F8_E4M3 [out, in], `scale` F8_E8M0 [out/128, in/128].
   All dims are exact multiples of 128. Measured scale bytes 115..116 on wq_a.

Group-axis proof: every scale grid's shape factors uniquely against its logical tensor
(NVFP4 (2048,256) vs (2048,4096) forces 16-along-IN; MX (2048,128) forces 32-along-IN;
(8,32) vs (1024,4096) forces 128×128), and the pow2/pair-share structure pins the
indexing order.

BF16: embed/head/norms/gate.weight/compressor wkv+wgate+norm/indexer weights_proj.
F32: hc_*, attn_sink, gate.bias, compressor ape. I64: tid2eid.

## Findings that corrected the ground truth

1. **The banked tensor-census.txt was WRONG on the compressor** (it collapsed shapes per
   name pattern). Gate run #1 refused with 60 named mismatches; measured truth:
   - fine layers (ratio 4, the 21 indexer layers): `wkv/wgate [1024, 4096]`,
     `ape [4, 1024]`
   - coarse layers (ratio 128, 20 layers): `wkv/wgate [512, 4096]`, `ape [128, 512]`
   - i.e. **ape = [compress_ratio, latent]** (one positional row per position in a
     compression block); latent = 2×head_dim (fine) vs head_dim (coarse); norm =
     [head_dim] on both. Run #2 after the fix: PASS.
2. **"First 3 layers are dense" (task brief) is FALSE.** Layers 0–2 are HASH-ROUTED MoE:
   full 256-expert banks + `ffn.gate.tid2eid` I64 [129280, 6] (token-id → 6 expert ids,
   all entries verified in [0, 256)) and no `gate.bias`. `num_hash_layers: 3` is the
   config key. All 43 trunk layers carry 256 experts (43×256×3 quantized projections).
3. **The task brief's "MoE experts are I8-packed with F8_E8M0 scales" conflated the two
   expert recipes**: that describes only the MTP experts; trunk experts are U8 +
   E4M3-per-16 + F32 scale_2 (see geometry above).
4. hc widths: layer-level base/fn rows = 24 (= 4! = hc_mult!), scale = 3 (= hc_mult-1);
   head-level rows = 4 (= hc_mult), scale = 1; fn width = hc_mult×hidden = 16384 both
   levels. Single-artifact evidence, encoded as formulas so a different hc_mult
   re-verifies through the gate.
5. Aux tensor magnitudes (bounds evidence): gate.bias absmean grows 8.99 (layer 3) →
   27.4 (layer 42); hc bases have long tails (min −40.7); hc_head_fn absmean 1.5e-3.
   attn_norm/q_norm gains sit far below 1 (0.068) — unusual but consistent across
   layers, likely folded normalization; forward lane should not "fix" it.

## Process notes

- **Box flow**: lane branch pushed to the box's repo clone
  (`git push <box1 ssh destination — redacted; box terminated 2026-08-19, conditions banked in darklanes>:/home/ubuntu/memra-src lane/dsv4-flash-loader`; detach
  HEAD on the box first — non-bare clone refuses pushes to the checked-out branch), then
  `git checkout lane/dsv4-flash-loader && cargo build --release -p memra-gguf --bin
  dsv4-census`. CPU-only; GPUs untouched.
- **Pre-push overrides used, knowingly**: (1) `MEMRA_SKIP_PERF_CI=1` — the push range
  touches `crates/memra-engine/src/parallel.rs` only to add `dsv4: None` to a test
  fixture (no kernel/default change); the hook's own text sanctions this override for
  the generic gate. The perf-board `--check` gate ran normally and passed. (2) The nsys
  blob guard fired on HISTORICAL blobs when pushing to a fresh ref (URL pushes have no
  tracking refs to exclude, so the hook scanned all history); resolved by seeding the
  branch at the shared base on the box first so only the new commits are inspected — no
  override needed, and the new commits add no blobs.
- Gate exit code is 1 on failure (`tee` masks it in a bare pipe — use `set -o pipefail`
  when scripting it).
- Workspace-wide `cargo check --workspace --tests` on the box: see
  `WORKSPACE-CHECK.txt` (this dir).

## Open risks for the forward-pass lane

1. **Nibble order within a byte is stats-invisible.** Both decoders agree on
   elem-2i→low-nibble (modelopt convention, same as the proven qwen NVFP4 lane), but a
   swapped order is a permutation that no distribution check can catch. The first
   per-layer paired-forward oracle (dequantized CPU reference vs official outputs)
   is the real acceptance for it.
2. **wkv carries no rope sub-block** (out = 512 = kv_lora, vs V3's kv_lora+rope 576).
   Where k_pe comes from (wq_b's 512-wide heads? the compressor?) is a forward-semantics
   question the loader cannot answer. Same for wo_a/wo_b's grouped o_lora factorization
   (`o_groups` 8 × `o_lora_rank` 1024) and `attn_sink` (64,) per layer.
3. **Compressor semantics**: the two shape classes + ape=[ratio, latent] strongly suggest
   block-wise KV compression with intra-block positional encodings, alternating
   fine(4)/coarse(128) layers; `sliding_window: 128` and `compress_rope_theta` interact
   with it. No reference implementation was consulted for bring-up yet — the
   `inference/` dir in both staged repos should be read against these shapes.
4. **sqrtsoftplus scoring + tid2eid hash routing have no engine arm**; tid2eid rows
   index the FULL 256-expert bank, so any expert-parallel placement must honor
   token-id-keyed routing on layers 0–2.
5. **hc (Sinkhorn hyper-connections)**: 24 = 4! rows is an interpretation
   (permutation-basis) — verify against the reference implementation before writing
   kernels; the census only pins shapes.
6. **MTP experts are MXFP4, not NVFP4** — the drafter lane needs the second decode path
   (group-32 E8M0) end to end; do not reuse the trunk expert kernel unchecked.
7. e4m3 NaN→0.0 (modelopt convention) is applied to WEIGHTS; measured zero NaN codes in
   sampled tensors, but a full-artifact NaN-code sweep only covered the samples.

---

# Lane 3 — CPU paired-forward + fixture gate (2026-08-18)

Lane 3 of dsv4-flash support: the full f32 CPU forward in Rust, gated per-array against
the lane-2 oracle fixtures (darklanes research/deepseek-flash-20260818/fixtures/,
byte-identical copies staged on the box at /home/ubuntu/dsv4-lane2-fixtures/). Branch
`lane/dsv4-flash-loader`, NOT merged.

## What landed

| commit | content |
|---|---|
| `3a5c500cb` | `dsv4_forward.rs` (full forward: hc Sinkhorn, shared-latent attention, fine/coarse compressors, DSA indexer, sqrtsoftplus router, MTP) + `dsv4-forward` gate binary |
| `e6e2d2fce` | `dsv4-mtp-probe` binary (NextN token-shift probe) |
| `6f310ea71` | lint pass, no arithmetic change — both gates re-run at this tip, numerically IDENTICAL to the banked logs (timing lines aside): determinism across builds confirmed |

Structure: per-mechanism public functions (`hc_pre/hc_post/hc_split_sinkhorn/hc_head`,
`CompressorW::forward`, `IndexerW::forward`, `AttnW::forward`, `MoeW::forward`,
`BlockW::forward`, `precompute_freqs_cis`/`apply_rope`, `act_quant`/`fp4_act_quant`/
`hadamard`/`e2m1_rne`/`pow2_ceil`) — the gate compares per-fixture-array, not one
monolith. All geometry derived from `DeepSeekV4Config` (compress_ratios,
num_hash_layers, hc_mult, o_groups…), zero hardcoded counts. 10 pure-math unit tests
(RNE ties, Sylvester order + involution, Sinkhorn double-stochasticity, rope inverse
round-trip, index-builder enumerations, npz reader on a synthetic zip).

Precision doctrine: every dot/long reduction accumulates in f64 and rounds once to f32,
so |rust − torch| ≤ torch's own blocked-f32 rounding error; elementwise transcendentals
stay f32. QAT round-trips (pow2-ceil scales, e4m3/e2m1 RNE) are exact given equal
inputs; the gate carries a ≤2-element one-grid-step flip allowance on quantized-kv
arrays (unused — zero flips observed anywhere).

## Contract decision: the engine targets the ARTIFACT variant (clamp-only)

**Decision: `dsv4_fixtures_artifactvariant` (kernel.py:88 clamp-only) is the engine's
serving contract for this artifact.** Evidence:

1. The NVFP4 artifact ships its own `inference/` (model.py IDENTICAL to the reference,
   verified by diff on the box); its kernel.py differs by exactly one line —
   kernel.py:88 casts the act_quant inplace value through `out_dtype` (bf16) instead of
   `FP8`, i.e. the FP8 rounding of the window/compressor-KV QAT sim is disabled.
2. NVIDIA quantized with modelopt v0.44.0 and published the near-lossless eval table
   (GPQA Diamond, AA-LCR, τ²-Bench Telecom, SciCode, IFBench — artifact README §
   Evaluation) for THIS artifact with THIS inference stack: the shipped-and-evaluated
   semantics of the weights we serve are the clamp-only kernel.
3. The weights memra loads ARE this artifact (157 GB NVFP4); the BF16 reference weights
   are not staged and not served. The reference law variant remains implemented behind
   `ActQuantVariant::RefFp8Round` and green (below) for any future BF16-reference lane.

Both variants were gated in SEPARATE invocations against their own fixture sets —
never mixed. The 5.1-max-abs logits fork between the two contracts (lane 2) is what the
0.5 logits threshold is sized to catch.

## Gate runs (cert lines)

Binary built on the box at `e6e2d2fce` (rust 1.97.1, release). Both PASS, exit 0.
Fixture integrity: every npz payload sha256 verified against its JSON before compare
(npz byte-identical rig↔box: artifactvariant 36167185…, ref 8d2e8078…).

```
dsv4-forward /home/ubuntu/models/dsv4-flash-nvfp4 /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json
  -> banked: research/dsv4-flash-loader-20260818/dsv4-forward-gate-artifactvariant.log  (PASS, 15 arrays, 117.0s)
dsv4-forward /home/ubuntu/models/dsv4-flash-nvfp4 /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_ref.json
  -> banked: research/dsv4-flash-loader-20260818/dsv4-forward-gate-ref.log             (PASS, 15 arrays, 116.2s)
```

### CONTRACT gate — artifactvariant (clamp-only): PASS, 15/15

| array | shape | max-abs | threshold | verdict |
|---|---|---|---|---|
| embed_out | [1,32,4096] | 0.0 | 0 (bit-exact) | PASS |
| layer0_out | [1,32,4,4096] | 1.526e-5 | 2.789e-3 | PASS |
| layer0_attn_out | [1,32,4096] | 5.960e-6 | 9.546e-4 | PASS |
| layer2_out | [1,32,4,4096] | 1.526e-5 | 1.495e-2 | PASS |
| layer2_attn_out | [1,32,4096] | 4.172e-6 | 1.342e-3 | PASS |
| layer2_compressor_kv | [1,8,512] | 1.431e-6 | 1.045e-3 | PASS (0 flips) |
| layer2_indexer_kv | [1,8,128] | 0.0 | 1.000e-3 | PASS (FP4 grid, exact) |
| layer2_index_score | [1,32,8] | 7.153e-7 | 1.013e-3 | PASS (−inf causality mask exact) |
| layer3_out | [1,32,4,4096] | 1.907e-5 | 1.503e-2 | PASS |
| layer3_attn_out | [1,32,4096] | 6.199e-6 | 4.597e-3 | PASS |
| final_logits_last | [129280] | 3.862e-5 | 0.5 | PASS (top1 2581==2581, top5 EXACT, top20 20/20) |
| mtp_logits_last | [129280] | 3.099e-5 | 0.5 | PASS (top1 2581==2581, top5 EXACT, top20 20/20) |
| c160_layer3_out_last | [1,4,4096] | 6.557e-7 | 1.873e-4 | PASS |
| c160_layer3_attn_out_last | [1,4096] | 4.292e-6 | 2.278e-3 | PASS |
| c160_layer3_compressor_kv | [1,1,512] | 7.153e-7 | 7.999e-4 | PASS (active coarse r=128) |

### Cross-receipt gate — ref (FP8-round): PASS, 15/15

Same table shape; notable rows (full table in the banked log):
final_logits_last max-abs 3.944e-1, mtp 4.583e-1 (thr 0.5; top1/top5/top20 all exact);
c160_layer3_attn_out_last 4.545e-4; everything shallow ≤ 2e-5.

**Why ref-logits diverge 4 orders more than clamp-only (3.9e-1 vs 3.9e-5), measured not
guessed:** the two runs share every code path except the FP8 RNE rounding inside
act_quant. Rounding is a discontinuous quantizer: ~1e-6-relative upstream skew
(torch-vs-Rust reduction order) occasionally lands on a code boundary and flips one
e4m3 code (a ~2⁻³-relative jump in one kv element), and 43 layers of window+compressor
QAT accumulate those flips. Clamp-only is continuous, so skew stays ~1e-5 through the
whole trunk. The id-level gates (top1/top5 exact) are the semantic acceptance either
way; the contract variant is also numerically tight.

## MTP token-shift convention: RESOLVED — V3 NextN alignment confirmed

Lane-2 open item. The fixtures replay model.py:826 (same ids to trunk and MTP) and
cannot pin the drafter alignment; a decisive counting probe can:

```
dsv4-mtp-probe /home/ubuntu/models/dsv4-flash-nvfp4 clamp-only "<52 ids: chat prompt 'Count from 1 to 30…' + assistant '</think>1 2 3 … 11 12'>"
  -> banked: research/dsv4-flash-loader-20260818/dsv4-mtp-probe-clamp-only.log
```

Tokenizer facts making it can't-hallucinate: each number is two tokens (223 = space,
907 = "13", 929 = "14"), so t+1 and t+2 after "…12" are near-deterministic and DISTINCT.

| stream | top-1 (logit) | reading |
|---|---|---|
| trunk next token | 223 (28.2) | " " — as forced |
| MTP, UNSHIFTED ids (826 shape) | 223 (27.9) | predicts t+1, trunk-like (mis-aligned use) |
| MTP, SHIFTED ids[1..]+greedy | **907 (51.0, next candidate 33.2)** | predicts t+2 — V3 convention |

**Conclusion: the drafter consumes the one-ahead embedding stream — MTP position i
fuses trunk h_i with Emb(t_{i+1}) and predicts t_{i+2}** (DeepSeek-V3 NextN). The
18-logit margin under the shifted alignment vs the trunk-magnitude confidence under the
unshifted one shows the head is trained for the shift. Spec-decode wiring for the
drafter lane: input_ids to mtp = trunk ids shifted left by one with the sampled next
token appended; hc-state stays position-aligned. (The banked mtp_logits_last fixture
remains a pure numeric gate at the 826 call shape.)

## Process notes

- Box flow unchanged from lane 1 (detach HEAD on box, push, checkout, release build).
  `MEMRA_SKIP_PERF_CI=1` used knowingly on both pushes: the range touches only
  memra-gguf oracle/gate code + Cargo.lock (sha2 dep, already in the workspace
  lockfile) — no kernels, no engine defaults; the hook's own text sanctions this.
- Gate runtime 117 s per variant on the 48-core box (torch fixture generator: 510 s for
  both variants) — CPU-only, GPUs untouched.
- tid2eid duplicate-expert probe (would make torch's `y[idx] += v` last-wins semantics
  ambiguous): 0 duplicate rows / 129280 on all 3 hash layers; the Rust MoE refuses
  loudly if a future artifact ships one.

## Open for the next lanes

1. **Engine integration**: this lane's forward is the CPU oracle, not a serving path —
   the GPU pipeline (bf16 activations, FP8/FP4 GEMMs) needs its own arms with
   interleaved A/B + output-sample gates against these banked arrays' semantics.
2. **End-to-end template/think-mode arm** (encoding_dsv4): fixtures exist in the
   artifact (`encoding/tests/`), untouched by this lane.
3. **Official-API greedy cross-check** (SEMANTICS.md §9.2): the contract decision here
   is from artifact provenance; a can't-hallucinate greedy comparison vs the DeepSeek
   API would additionally pin which variant DeepSeek itself serves (informational for
   the BF16 reference, not blocking for serving the NVFP4 artifact).
4. Drafter lane: wire the CONFIRMED V3 shift into the spec-decode loop; acceptance-rate
   cells on the 2-card box.

---

# Lane 4 — GPU trunk bring-up on the 2-card box (2026-08-18)

Lane 4 of dsv4-flash support: the trunk forward on the 2× RTX PRO 6000 Blackwell 96GB
box, correctness-gated against the lane-2/3 banked CPU oracle (artifactvariant contract,
lane-3 decision). NOT a perf lane: any timing below is informational, single-run, not a
claim. Branch `lane/dsv4-flash-loader`, NOT merged.

## Placement math (written BEFORE loading — this section is the plan of record)

Artifact byte census by class, measured from the 46 shard headers on the box
(2026-08-18, python over `model.safetensors.index.json` + safetensors headers):

| class | stored | tensors |
|---|---|---|
| trunk routed experts (NVFP4) | 145.125 GiB | 132,096 |
| MTP routed experts (MXFP4) | 3.188 GiB | 1,536 |
| MTP other (FP8/BF16/F32) | 0.159 GiB | 39 |
| embed (BF16) | 0.986 GiB | 1 |
| head (BF16) | 0.986 GiB | 1 |
| tid2eid (I64) | 0.017 GiB | 3 |
| everything else | 6.248 GiB | 1,559 |
| TOTAL | 156.71 GiB | 135,235 |

### Quant rungs (explicit, per the lane mandate)

**bf16-dequant of the whole model does NOT fit**: ~291B params × 2 B ≈ 542 GiB ≫
192 GB. The experts must stay quantized on GPU. Chosen rungs:

1. **Trunk routed experts: resident AS-STORED NVFP4** (weight U8 nibble-pairs +
   e4m3 per-16 scales + f32 scale_2), **dequantized on the fly per activated expert**
   into a ~50 MB reused bf16 scratch by a new kernel, then bf16×bf16→f32 cuBLASLt GEMM.
   The dequant is EXACT in bf16: e2m1 (1-bit mantissa) × e4m3 (3-bit mantissa) needs
   ≤5 significand bits ≤ bf16's 8; scale_2 is asserted power-of-two at load (lane-1
   measured fact, now a refusal), so it shifts exponents only.
2. **FP8-blk128 linears** (attn wq_a/wq_b/wkv/wo_a/wo_b, shared experts, indexer wq_b):
   **host dequant at load** with the lane-1 bit-proven `dequant_fp8_blk128` → cast bf16
   (EXACT: e4m3 3-bit mantissa × e8m0 pow2), resident bf16 (~10.9 GiB total).
   wo_a resident bf16 is additionally reference-lawful (model.py:539 runs the einsum BF16).
3. **BF16-stored tensors**: resident as-is, EXCEPT promotions to f32 where SEMANTICS
   §7.2 marks f32 islands: gate.weight, compressor + indexer-compressor wkv/wgate/norm,
   all norms (bf16→f32 is exact).
4. **F32-stored** (hc_*, ape, attn_sink, gate.bias): f32; hc fn matrices on GPU, hc
   base/scale + gate.bias + tid2eid host-side (the scalar math runs on host, below).
5. **MTP (optional per lane brief)**: not loaded for the trunk gates; its GPU path
   would ride rung 1 with an e8m0/32 (MXFP4) dequant kernel.

### Split across the 2 cards

Engine idiom: the only *executing* multi-GPU scheme in memra is **PP layer-split**
(pp.rs; Step-3.7-Flash 105 GB serves PP-2 on this same card class — docs/PERFORMANCE.md
"PP-2-or-nothing"). TP/EP exist only as an unwired planning contract (parallel.rs, zero
call sites). PP needs exactly ONE boundary crossing per forward — the hc state
[s,4,4096] f32 ≈ 13 MB at s=192 — vs per-layer collectives for TP. Chosen: layer split,
derived from config (`n_layer`, per-layer class from `compress_ratios`), not hardcoded.

Per-layer resident bytes (derived from config math + measured class totals):
experts as-stored 145.125/43 = 3.375 GiB every layer; + FP8→bf16 0.262 GiB (fine) /
0.246 GiB (coarse/window-only); + f32-island promotions 0.046 / 0.021 GiB; + hc/norms
~0.004 GiB → **fine ≈ 3.687 GiB, coarse ≈ 3.646 GiB, window-only ≈ 3.630 GiB**.

| stage | content | planned resident |
|---|---|---|
| GPU0 | embed + layers 0..21 (2 window-only, 10 fine, 10 coarse) | 0.99 + 80.59 = **81.58 GiB** |
| GPU1 | layers 22..42 (11 fine, 10 coarse) + head + trunk hc_head/norm | 77.02 + 0.99 = **78.01 GiB** |

Card capacity 97,887 MiB = 95.59 GiB; CUDA context + modules ≈ 0.6–1.0 GiB →
~94.5 GiB usable. **Headroom: ≈12.9 GiB (GPU0), ≈16.5 GiB (GPU1).** Transients are
small at gate lengths (s ≤ 208): expert dequant scratch 50 MB + activations/q/o/kv
< 250 MB + cuBLASLt workspace 64 MB → < 0.5 GiB/card, ~25× inside headroom.
Boundary copy via host bounce for bring-up (peer-copy + mempool-access grants are a
perf-lane step; pp.rs:1527's mempool footgun deliberately avoided here).

### Numeric contract + gate threshold derivation (bf16-vs-f32, not guessed)

GPU pipeline: **weights contribute ZERO error** (all dequants exact in bf16, above).
f32 islands preserved exactly as SEMANTICS §7.2: norm internals, the entire compressor
(incl. its GEMMs, f32 kernels), hc (f32 mix GEMM on GPU; Sinkhorn on host = the
oracle's own f32 code), router scoring/selection/renorm (f32 gate GEMM kernel; scalar
math on host = oracle code), the three QAT sims (f32 kernels, bit-exact grid math),
attention softmax/accum (f32 scores, f64 accumulators), logits GEMM f32, residual hc
state f32 end-to-end. **bf16 enters ONLY as activation-input rounding at the
non-island GEMMs** (wq_a, wq_b, wkv, wo_a, wo_b, indexer wq_b, expert w1/w3/w2, shared
experts): unit roundoff u = 2⁻⁸.

Propagation: each of the D = 2·43 = 86 residual sub-blocks injects ≈2 sequential
bf16-rounded GEMM hops on its branch path; hc comb is doubly-stochastic (row sums 1,
non-expanding) and rmsnorm re-normalizes scale, so independent roundings accumulate
random-walk: ε ≈ 2u·√D ≈ 0.072 relative at the head.

**Per-array threshold: thr = 2u·√d · amax_fixture(array)**, d = sub-block depth of the
array (layer l out: d = 2l+2; attn_out: d = 2l+1; final logits: d = 86). For
final_logits_last (amax 35.51): thr ≈ 2.57. **top-1 / top-5(set) / top-20(set) id
agreement is mandatory and threshold-free.** Discrete-risk watchlist (analyzed, not
hidden, if hit): expert-selection near-ties on score layers, greedy argmax near-ties,
e2m1 QAT code flips on the indexer path (selection-only, and at gate lengths ≤208
tokens every completed block is selected → flip-invisible).

### Gate protocol (banked before running)

- **(pre) expert-dequant sub-gate**: GPU NVFP4-dequant kernel output (bf16→f32) must be
  BIT-EXACT vs the lane-1 host decoder on sampled experts across layers/projections —
  the bf16-exactness proof makes any mismatch a kernel bug (nibble order, scale grid).
- **(a) output-sample gate**: full 15-array fixture compare (dsv4_fixtures_
  artifactvariant, one variant per invocation), thresholds per the derivation above +
  mandatory id gates on final logits.
- **(b) greedy continuation ≥160 tokens**: GPU generates greedily from the fixture
  32-token prompt by re-prefill per step (O(n²) is the accepted bring-up rung; decode
  caching is the perf lane). CPU reference = lane-3 forward, teacher-forced in ONE
  192-token prefill with per-position argmax over positions 31..190: greedy is
  causal-deterministic, so per-position agreement ⇒ the CPU greedy continuation equals
  the GPU sequence exactly, and the first disagreement position IS the first greedy
  divergence (identical prefix), where both logit rows get banked and analyzed.
- **(c) VRAM/stability**: per-GPU bytes at load / after warmup / peak, no OOM, no CUDA
  errors, token-sequence + final-logits determinism across 2 full greedy runs.

### Gate-policy corrections after run 1 (banked before the rerun; measured values did not move)

Run 1 (log: `dsv4-gpu-gate-run1.log`, this dir) ran the full trunk end-to-end on both
cards: placement within 0.7 GiB of the plan (dev0 82.21 / dev1 78.64 GiB used vs 81.58 /
78.01 planned), expert-dequant sub-gate 5/5 BIT-EXACT, zero CUDA errors, 10/13 trunk
arrays PASS, logits top-1 and top-5 EXACT. Three policy defects in MY GATE FORMULAS
(not in the forward) surfaced and are corrected with derivations:

1. **Extreme-value factor was missing from the max-abs threshold.** The banked formula
   bounded max-abs by 2u·√d·absmax; the per-element error is a zero-mean random walk
   with σ ≈ u·√d·scale and the gate takes a MAX over n elements, which carries the
   factor √(2·ln n) (≈4.85 at n=129280) — the ad-hoc 2× under-modeled exactly that.
   Corrected: thr = u·√d·√(2 ln n)·absmax. Measured drift sits at 0.3–0.7× the
   corrected bound at every banked depth (final logits: 4.35 measured vs 6.2 bound;
   quantitative depth-consistency: rel drift 1.5e-2 at d=8 → ≈5e-2 at d=86, ×√(86/8),
   exactly the random-walk prediction — no discrete-jump/bug signal).
2. **indexer_kv one-step flip bound**: a 0 ↔ ±0.5·s e2m1 flip has |diff| ==
   max(|got|,|ref|); run 1's 0.5× bound rejected legitimate one-step flips. Corrected
   to 1.0× (necessary adjacency condition), budget unchanged (5%). Run 1 measured 3
   flips / 1024 elements ≈ the predicted u/grid-step ≈ 1/64 rate.
3. **index_score was modeled as continuous — wrong for the FP4 path.** Its inputs carry
   e2m1 QAT; one flipped kv/q element perturbs a head dot by ≤ 2s_kv·6s_q against a
   128-term sum — a few % of the score scale. Corrected: exceeders budgeted 5% of
   elements, each ≤ 0.05·absmax. Selection semantics are unaffected at gate lengths
   (top-min(512, nb) selects ALL completed blocks; −inf causality mask compared
   EXACTLY and matched in run 1). Run 1 measured 2 exceeders at ≤1.8% of absmax.
4. **top-20 rule operationalized as ≥18/20 overlap** — the SAME rule the lane-3 CPU
   gate uses for "top-20 id agreement" (dsv4_forward_gate.rs:30), with raw overlap and
   the 20/21 boundary gaps always printed. Physics note: at σ_logit ≈ u√86·|logit| ≈
   0.5–0.9, a 0.27-gap 20/21 boundary cannot be set-stable under ANY correct bf16
   implementation; the greedy-continuation gate is the semantic instrument. Run 1:
   18/20 with the two misses at the boundary (ref gap 0.2684).

## Gate (a) — GPU output-sample gate: PASS (run 2, corrected policy)

```
target/release/dsv4-gpu-gate /home/ubuntu/models/dsv4-flash-nvfp4 /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json 0,1
  run 1 -> banked: research/dsv4-flash-loader-20260818/dsv4-gpu-gate-run1.log       (FAIL — 3 gate-formula defects, corrected above)
  run 2 -> banked: research/dsv4-flash-loader-20260818/dsv4-gpu-gate-run2-PASS.log  (PASS, 14 arrays, 1 explicit SKIP, exit 0)
```

Engine binary built on the box at the lane tip (rust 1.97.1, release, MEMRA_CUDA_ARCH=120a,
CUDA 13.2). Variant: clamp-only (the lane-3 contract), never mixed.

| array | shape | max-abs | threshold (u·√d·√(2 ln n)·absmax) | verdict |
|---|---|---|---|---|
| embed_out | [1,32,4096] | 0.0 | 0 (bit-exact) | PASS |
| layer0_out | [1,32,4,4096] | 5.064e-2 | 7.906e-1 | PASS |
| layer0_attn_out | [1,32,4096] | 1.832e-2 | 1.810e-1 | PASS |
| layer2_out | [1,32,4,4096] | 4.946e-2 | 2.936e0 | PASS |
| layer2_attn_out | [1,32,4096] | 1.904e-2 | 2.276e-1 | PASS |
| layer2_compressor_kv | [1,8,512] | 6.061e-3 | 1.332e-1 | PASS (0 flips — clamp-only is identity) |
| layer2_indexer_kv | [1,8,128] | 5.000e-1 | 1.164e-1 | PASS (3 one-step e2m1 flips / 1024 el, ≈ predicted 1/64 rate) |
| layer2_index_score | [1,32,8] | 7.195e-2 | 1.055e-1 | PASS (−inf causality mask exact; flip budget unused) |
| layer3_out | [1,32,4,4096] | 7.474e-2 | 3.410e0 | PASS |
| layer3_attn_out | [1,32,4096] | 2.305e-2 | 9.225e-1 | PASS |
| final_logits_last | [129280] | 4.349e0 | 6.714e0 | PASS (top1 2581==2581, top5 set EXACT, top20 18/20; misses at the 0.2684-gap 20/21 boundary — physics note in the corrections) |
| c160_layer3_out_last | [1,4,4096] | 1.318e-3 | 3.648e-2 | PASS |
| c160_layer3_attn_out_last | [1,4096] | 1.382e-2 | 3.841e-1 | PASS |
| c160_layer3_compressor_kv | [1,1,512] | 1.361e-3 | 1.081e-1 | PASS (active coarse r=128) |
| mtp_logits_last | — | — | — | SKIPPED (MTP GPU path optional for this lane; a skip is never a PASS) |

**Expert-dequant sub-gate: 5/5 BIT-EXACT** (GPU NVFP4 kernel vs the lane-1 host decoder,
samples across layers 0/2/20/22/42 and all three projections, incl. the lane-1 oracle pin
tensor layers.20.experts.7.w1).

**Placement (gate c, measured at the checkpoints; plan hit within 0.7 GiB):**

| checkpoint | dev0 used / free (GiB) | dev1 used / free (GiB) |
|---|---|---|
| post-load | 82.21 / 12.77 | 78.64 / 16.33 |
| post-warmup (32-tok fwd) | 82.24 / 12.74 | 78.67 / 16.30 |
| post-gate (incl 160-tok fwd) | 82.33 / 12.64 | 78.67 / 16.30 |

Split at layer 22 (derived from per-layer byte mass, matching the plan-of-record table:
planned 81.58 / 78.01 GiB resident). No OOM, no CUDA errors. Load 154 s from page-cached
artifact; 32-token full-trunk forward ≈ 1.4 s (informational, single-run, not perf claims).

## Gate (b) — greedy continuation, 160 tokens: PASS (CPU greedy == GPU greedy exactly)

```
target/release/dsv4-gpu-greedy /home/ubuntu/models/dsv4-flash-nvfp4 /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json /home/ubuntu/dsv4-greedy-out 160 2 0,1
  -> banked: research/dsv4-flash-loader-20260818/dsv4-gpu-greedy.log + dsv4-gpu-greedy.json
target/release/dsv4-greedy-verify /home/ubuntu/models/dsv4-flash-nvfp4 /home/ubuntu/dsv4-greedy-out/gpu_greedy.json /home/ubuntu/dsv4-greedy-out
  -> banked: research/dsv4-flash-loader-20260818/dsv4-greedy-verify-PASS.log + dsv4-cpu-verify.json (exit 0)
```

- GPU: 160 greedy tokens from the fixture 32-token prompt, re-prefill per step (the
  banked O(n²) bring-up rung). First token 2581 ("We") — matches the fixture greedy
  sanity pin.
- CPU reference: the lane-3 oracle forward, teacher-forced in ONE 192-token prefill,
  per-position argmax over the 160 predictive positions (equivalence argument banked in
  the plan of record — greedy is causal-deterministic).
- **RESULT: 160/160 positions agree; first divergence: NONE.** The CPU greedy
  continuation IS the GPU sequence, token for token. No divergence-analysis branch was
  needed.
- Robustness margins (measured from the banked CPU logits matrix): CPU top1-vs-top2
  margin over the 160 positions — min 0.0061, p10 0.762, median 9.36, max 21.38. The
  min-margin position still agreed (GPU/CPU drift is heavily correlated between
  neighboring candidates). GPU-vs-CPU logit-row max-abs over sampled positions: 4.349
  (same magnitude as gate (a)'s final-logits row, consistent with the bf16 drift model).
- Full logits banked: cpu_logits_all.bin sha256 c36c01d99bb0c168e93985c94a3ad2861a
  1dc892f4530997d99d3effb7eaf41c, gpu_logits_run{0,1}.bin sha256 928c1dc9cd67e473d3c5
  8220fa1dd2a215b17a9c0a9da8421ca238236414342c (runs 0 and 1 BYTE-IDENTICAL). The 83MB
  bins live rig-side at ~/dsv4-lane4-bins/ (not committed — repo blob discipline) and
  on the box at /home/ubuntu/dsv4-greedy-out/.

## Gate (c) — VRAM / stability: PASS

- Placement table: gate (a) section above (post-load / post-warmup / post-gate; peak
  during greedy: dev0 82.36 GiB, dev1 78.77 GiB — from dsv4-gpu-greedy.log vram lines).
- No OOM, no CUDA errors across load + gate + 2×160-step greedy runs (~330 forwards).
- Determinism: 2 independent greedy runs produced IDENTICAL token sequences and
  BYTE-IDENTICAL 82.7 MB logits streams (sha256 equal). Single-stream-per-stage
  launches, fixed-tree/sequential reductions, no atomics in accumulation paths, and
  ascending-expert-order MoE accumulation are what make this hold by construction.
- Timing, informational only, single-run, NOT perf claims: load 154 s (page-cached),
  full-trunk prefill ≈ 1.1 s/step at s≈190 under re-prefill greedy (176 s / 160 steps),
  CPU teacher-forcing verify 241 s on 48 cores.

### Run-3 correction (banked before rerun): derived top-20 boundary-band rule + MTP path

MTP GPU path taken (trunk landed with box time to spare, per the lane brief): MXFP4
expert slabs (recipe DETECTED from stored dtypes, refused on surprise; E8M0 0xFF NaN
code refused at load), MtpDev on the last stage (pp idiom), shared trunk head,
`mtp_logits_last` at the fixture call shape (model.py:826). Run 3
(`dsv4-gpu-gate-run3.log`): MXFP4 sub-gate samples BIT-EXACT (incl. the lane-1 MXFP4
oracle pin mtp.0.experts.7.w1); mtp_logits_last max-abs 4.504 vs thr 6.332 PASS, top-1
(2581) exact, top-5 set exact — but top-20 overlap 16/20 against the fixed ≥18/20 floor.

Measured cause (fixture mtp_top20 logits): ranks 18-21 are a near-tie cluster with gaps
0.0258 / 0.0373 / 0.0646 — four ids inside a band an order of magnitude smaller than the
derived per-logit drift σ = u·√d·|logit| ≈ 0.88. A fixed overlap floor measures cluster
geometry, not correctness. **Corrected top-20 rule (fully derived, replaces the floor):
every id in the symmetric difference of the two top-20 sets must lie within
band = 3·√2·u·√d·|ref rank-20 logit| of the reference boundary** — an id further above
the boundary must never drop out, one further below must never climb in; either is a
real bug. Top-1 exact and top-5 set-exact stay strict. Raw overlap + boundary gaps are
always printed. (Trunk run-1 evidence re-read under this rule: 2 set-diff ids at a
0.2684-gap boundary, both in-band.)

## MTP GPU path — gate: PASS (run 4, all 15 fixture arrays, 0 skips)

```
target/release/dsv4-gpu-gate /home/ubuntu/models/dsv4-flash-nvfp4 /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json 0,1
  run 3 -> banked: dsv4-gpu-gate-run3-mtp.log            (FAIL only on the fixed top-20 floor; analysis above)
  run 4 -> banked: dsv4-gpu-gate-run4-PASS-15arrays.log  (PASS, 15 arrays, 0 skipped, exit 0)
```

- **mtp_logits_last: max-abs 4.504 vs thr 6.332 PASS; top-1 (2581) exact; top-5 set
  exact; top-20 boundary-band rule: 8 set-diff ids, 0 out-of-band** (band ±3.729; the
  16/20 raw overlap is the measured rank-18..21 near-tie cluster, gaps 0.026/0.037/0.065).
- final_logits_last under the same derived rule: 4 set-diff ids, 0 out-of-band (band
  ±3.604), raw overlap 18/20.
- Expert-dequant sub-gate now 7/7 BIT-EXACT across BOTH recipes (5 trunk NVFP4 + 2 MTP
  MXFP4, incl. both lane-1 oracle pin tensors).
- MTP resident on dev1 (pp idiom): dev1 post-load 78.64 → 82.14 GiB (+3.50 GiB, matching
  the 3.19 GiB MXFP4 slabs + bf16 e/h_proj + block); headroom 12.8 GiB. Run-3 vs run-4
  mtp max-abs identical (4.504e0) — deterministic across process runs.
- The MTP row is gated at the FIXTURE call shape (model.py:826, same ids to trunk and
  MTP). The V3 NextN drafter shift (lane-3 finding) is wired knowledge for the
  spec-decode lane, not exercised here.

## Lane 4 summary — all mandated gates PASS

| gate | verdict | evidence |
|---|---|---|
| (pre) expert-dequant sub-gate | PASS 7/7 bit-exact | run-4 log |
| (a) output-sample vs artifactvariant fixtures | PASS 15/15 arrays (incl. optional MTP) | dsv4-gpu-gate-run4-PASS-15arrays.log |
| (b) greedy continuation, 160 tokens | PASS 160/160, zero divergence | dsv4-greedy-verify-PASS.log, dsv4-cpu-verify.json |
| (c) VRAM/stability/determinism | PASS | placement tables; 2 greedy runs byte-identical; no OOM/CUDA errors |

Support claim per the house law: **(DeepSeek-V4-Flash, NVFP4 artifact, clamp-only
contract) trunk+MTP forward CORRECTNESS on 2× RTX PRO 6000 Blackwell 96GB is gated**
— at the bring-up rung only (bf16 dequant-on-the-fly GEMMs, prefill-only, re-prefill
greedy, host-bounce boundary). This is NOT a serving or perf claim; no perf number in
this lane is a claim.

## Open for the next lanes (perf/headline + drafter)

1. **Decode path**: KV caching (window ring + compressed-block cache + pending-block
   buffers + indexer cache), batched expert dispatch, CUDA-graph capture — the O(n²)
   re-prefill greedy rung becomes a real decode loop. Interleaved A/B protocol applies
   to every perf claim; none were made here.
2. **Native quant GEMMs**: replace dequant-to-bf16 scratch with the engine's NVFP4
   MMQ class (w4a8/f8f4) + a fused MXFP4 arm — each needs its own output-sample +
   greedy gates (numeric class change; this lane's thresholds do not transfer).
3. **PP-2 serving integration**: fold the placement into the engine's PpNRt door
   (MEMRA_PP_STAGES=2), peer-copy boundary (mempool-access grants, pp.rs:1527),
   admission/KV plumbing; hybrid_forward arm or dedicated dsv4 path decision.
4. **Drafter lane**: wire the lane-3-confirmed V3 NextN shift into spec-decode;
   acceptance-rate cells on this box (MTP forward is now gated on GPU).
5. Template/think-mode arm (lane 5, in flight elsewhere) gates any standard-surface
   exposure; no product surface without a separate owner call (PLAN.md).

Process notes: box flow = rsync lane files to /home/ubuntu/memra-src + release build
(rust 1.97.1, MEMRA_CUDA_ARCH=120a, CUDA 13.2); all gate outputs pulled to this dir
immediately after each run (spot discipline); 83 MB logits bins rig-side at
~/dsv4-lane4-bins/ (not committed). Box /tmp gate logs cleaned at lane close.

### Determinism-across-builds rerun at the pushed tip

Branch pushed to the box repo (f78a0cd944..d18fcf6524) and rebuilt from the committed
tree; gate rerun at the tip: **PASS, 15 arrays, gate table numerically IDENTICAL to
run 4** (banked: `dsv4-gpu-gate-tip-d18fcf6524-PASS.log`; `diff` of the two tables is
empty). Pre-push overrides, knowingly (lane-1 precedent): `MEMRA_SKIP_PERF_CI=1` — the
range adds a NEW self-contained TU + modules + bins; no existing kernel, dispatch
default, or serving path is touched (build.rs/lib.rs/Cargo.toml changes are additive;
config.rs change is JsonObj visibility). The perf-board `--check` and public-boundary
hooks ran normally and passed. Box /tmp gate logs deleted after banking (tmp hygiene).

---

# Lane 6 — true decode path (incremental KV, no re-prefill) on the 2-card box (2026-08-18)

Lane 6 of dsv4-flash support: the O(n) decode loop for the lane-4 GPU trunk, gated by
equivalence against the lane-4 re-prefill path. Branch `lane/dsv4-flash-loader`, NOT
merged, NOT pushed. Semantic law: darklanes SEMANTICS.md §1-2 + the reference model.py
decode branches (M:255-276 get_window_topk_idxs, M:344-377 Compressor decode state
machine, M:405-433 Indexer decode, M:521-534 Attention decode) — read directly on the
box before this design was written. A prior lane-6 attempt died in recon and committed
nothing (worktree verified at 3ce248aecf); its only box residue was a set of dead nvcc
temporaries in /tmp (tmpxft_*, qmatvec module, 09:38Z) — deleted before work started.

## Decode cache design (banked BEFORE building — this section is the plan of record)

### Per-layer state, mirroring the reference cache geometry (M:473-474, M:491)

ONE attention kv cache buffer per layer, `[win + cap_blocks, 512] f32`, exactly the
reference layout: slot `p % win` = window row of absolute position p (M:530 ring), slot
`win + j` = compressed block j (M:491 `compressor.kv_cache = kv_cache[:, win:]`; decode
index offset `win`, M:509). Rows store what prefill stores: kv_norm→rope(pos)→act_quant
values (window) and pooled→norm→rope(j·ratio)→QAT values (blocks). Under the clamp-only
contract act_quant is an identity (448·pow2ceil(amax/448) ≥ amax and pow2 division is
exact), so window/compressor rows carry no grid; the indexer store DOES sit on the
e2m1×pow2 grid (FP4 QAT is real in both kernel variants).

Layer classes (derived from config `compress_ratios`, never hardcoded):

| class | count (dev0/dev1) | ring | compressed store | pending | indexer |
|---|---|---|---|---|---|
| window-only (r=0) | 2/0 | [128,512] | — | — | — |
| fine (r=4) | 10/11 | [128,512] | [s/4, 512] grows | kv+score [8, 1024] ×2 | store [s/4, 128] grows + pend [8, 256] ×2 |
| coarse (r=128) | 10/10 | [128,512] | [s/128, 512] grows | kv+score [128, 512] ×2 | — |

### Pending-block state (the fine-overlap subtlety)

A fine (coff=2) block j pools over 8 positions spanning blocks j−1 and j: prev block
through latent dims [0:512], current through [512:1024] (M:295-314). The reference keeps
`kv_state/score_state [coff·ratio, coff·d]` with rows [0:ratio] = prev block's RAW
wkv/wgate outputs and rows [ratio:2ratio] = current block's, shifting cur→prev on each
emission (M:344-370). This lane keeps the same state machine with ONE deliberate
difference: pending rows store the RAW score (ape NOT pre-added; the reference pre-adds
ape into score_state). Reason: the lane-4 prefill pooling kernel
(`memra_dsv4_compressor_pool`) adds `ape[p]` inside the kernel, and ape depends only on
the in-block position p — adding it at pool time is algebraically identical AND lets
decode emission reuse the prefill kernel VERBATIM: lay pending as [prev(ratio) rows,
cur(ratio) rows] and call the kernel with nb=2, overlap=1, taking output block 1 (its
index arithmetic then reads rows 0..ratio as prev via dims [0:d] and rows ratio..2ratio
as cur via dims [d:2d] — exactly the emission pooling), so the decode-emitted block is
computed by the SAME kernel expression order as prefill emission: bit-identical given
bit-identical pending rows. Coarse: pending [ratio, latent], kernel nb=1, overlap=0,
direct. Block-0-at-decode (no prev block): pend_score initialized to −inf, pend_kv to 0
— (−inf + ape) = −inf reproduces the reference/oracle j==0 masking bit-exactly.
Emission triggers: (pos+1) % ratio == 0 (M:344); block index pos/ratio; rope at block
start pos+1−ratio (M:366). The indexer's compressor runs the same machine at d=128,
latent=256, with Hadamard+FP4 after rope, and is invoked BEFORE indexer scoring so a
block completed this step is scored and attendable (M:415, nb = (pos+1)//4 — no causal
mask needed at decode, the store is causal by construction).

### Prefill→decode handoff

Prefill runs the lane-4 path with cache population hooks: (i) window ring gets the last
min(s,128) post-QAT kv rows at slots p%win (M:524-527 semantics); (ii) compressed
stores get the pooled blocks the prefill already computed, at [win+j]; (iii) pending
gets raw wkv/wgate GEMM row slices: fine = rows of the last COMPLETE block into
prev-slots + the s%4 remainder rows into cur-slots (M:346-352); coarse = the s%128
trailing rows; indexer pending likewise. The prefill compressor is extended to return
its raw kv/score device buffers (it always computed them; they were dropped) and to
compute them even when s < ratio (the reference does — remainder-only prefills must
seed pending; the pooled output stays None, so lane-4 gate behavior is unchanged).

### Index construction at decode (host, mirrors M:255-276 exactly)

Window part: fixed width 128; t < 127 → [0..t, −1 pad]; t ≥ 127 → ring slots ordered
[sp+1..128, 0..sp] (sp = t%128) = ascending ABSOLUTE position — the same slot order as
prefill rows, so attention accumulation order is identical (the −1 pads that prefill
lacks are skipped by the kernel and change nothing). Compressed part: coarse = blocks
0..nb−1 ascending + win offset; fine = host topk over the decode index_score (value
desc, index asc — the oracle's exact ordering), k = min(512, nb), + win offset, no −1
(decode sets are causal by construction, M:431-433). Top-512 selection becomes
non-trivial at nb > 512 ⇔ s ≥ 2052.

### One kernel signature change

`memra_dsv4_indexer_score` gains `lim0`: the causal limit is (t+1)/ratio with LOCAL t,
which is 0 at decode (everything would mask); decode passes the explicit block count,
prefill passes −1 (behavior unchanged). All other decode compute reuses the lane-4
kernels unmodified at s=1 (their per-row arithmetic is s-invariant by construction —
grid = one block per row, fixed-tree reductions).

### VRAM math — cache bytes as f(seq)

Growth: fine layer = 512 B/token (attn, 2 KiB per 4 tokens) + 128 B/token (indexer) =
640 B/token; coarse = 16 B/token. Fixed: ring 256 KiB/layer; pending fine 64+16 KiB,
coarse 512 KiB. Rope tables (lane-4 load, scale with max_seq): 512·max_seq B/device.

| | dev0 (10f+10c+2w) | dev1 (11f+10c) |
|---|---|---|
| per-token growth | 6,560 B | 7,200 B |
| fixed (rings+pending) | 11.3 MiB | 11.1 MiB |
| s=4k (+rope 2 MiB) | 39 MiB | 41 MiB |
| s=32k (+rope 16 MiB) | 232 MiB | 252 MiB |
| s=128k (+rope 64 MiB) | 895 MiB | 975 MiB |
| s=1M (+rope 512 MiB) | 6.92 GiB | 7.55 GiB |

Lane-4 measured free after full load: dev0 12.77 GiB, dev1 12.80 GiB (with MTP). 1M
decode state FITS on both devices with ≥5.2 GiB margin. VRAM ceiling (dev1 binds,
7,712 B/token incl. rope): ≈1.7M tokens — the model's own 1M ctx binds FIRST, so the
box ceiling for decode is the model, not memory. Two honest caveats: (i) reaching long
s requires DECODING there — the lane-4 prefill path carries O(s) transients (q/o
buffers ≈ 262 KiB/token ⇒ ~2.6 GiB at s=10k) so long PROMPTS need a chunked-prefill
lane; (ii) the correctness-rung indexer scoring kernel is one thread per block-column
(64×128 sequential f64 per block) — at 1M (262k blocks) it would be seconds/step; fine
at gate lengths, a perf-lane rewrite for long ctx. Stores are allocated at max_seq
capacity up front (reference `register_buffer` shape; deterministic addresses); the
gate binary sizes max_seq to the probe (2304), so allocated cache ≈ 26/28 MiB/device.

## Equivalence doctrine — decode vs lane-4 re-prefill (banked BEFORE running)

Weights identical (same resident tensors). Per component, decode arithmetic vs
re-prefill arithmetic at the same position:

**Class I — genuinely identical (bit-exact given identical inputs):** embed row gather;
every f32-island kernel (rmsnorm/headrms, dots_f32 GEMMs: hc mixes, gate scores,
compressor wkv/wgate, head logits — one fixed-tree block per output element, s-invariant),
rowsq_scale, rope, hadamard, act_quant/fp4 grids, compressor pooling (decode reuses the
prefill kernel with the 2-block pending layout — identical expression order), swiglu,
hc collapse/post, host Sinkhorn/routing/topk; sink-attention given identical slot order
(proven identical above); decode-emitted cache rows vs prefill-emitted rows.

**Class II — accumulation order legitimately differs (derived threshold):** the
cuBLASLt bf16 GEMMs (wq_a, wq_b, wkv, wo_a, wo_b, indexer wq_b, weights_proj, expert
w1/w2/w3, shared experts, e/h_proj) run m=1 at decode vs m=s at re-prefill — the
heuristic plan is keyed on (dev,m,n,k), so the f32 reduction order of the SAME exact
products (bf16×bf16 is exact in f32: ≤16 significand bits) may differ (tiling/split-k).
Reordering a k-term f32 sum perturbs the output by σ ≈ u32·√(k/2)·‖p‖₂ per path
(u32 = 2⁻²⁴; partial-sum random-walk model), ‖p‖₂ ≈ |y| under cancellation ⇒ per-hop
relative u_ord = 2·u32·√k_max = 2·2⁻²⁴·√8192 ≈ 1.08e-5 (k_max = 8192 = wo_b; the
leading 2 covers both paths + model slack). Folded through the lane-4 depth/extreme-
value doctrine: **thr(logits row) = u_ord·√86·√(2·ln 129280)·absmax_ref ≈
4.9e-4·absmax_ref** (≈0.017 at absmax 35.5) — ~400× tighter than the lane-4 CPU-vs-GPU
bound because bf16 input rounding is common to both paths here. If measured max-abs is
0.0 at every checkpoint, cuBLASLt is row-stable across m for these shapes and the gate
reports BIT-EXACT (measured, not assumed); otherwise it must sit under thr.

**Class III — discrete channels riding class-II drift (watchlist, analyzed if hit,
never hidden):** (a) e2m1 code flips in indexer q/kv (P ≈ drift/grid-step ≈ 1e-6/elem;
selection-only, and selection is all-blocks until s ≥ 2052); (b) router selection
near-ties (a flip = O(1) logits jump → gate fails loudly, banks both rows + biased-score
margins); (c) fine topk order swaps at score near-ties (changes f64 summation order
only); (d) greedy argmax near-ties (top1−top2 margins printed at every checkpoint;
lane-4 measured min 0.0061 over 160 steps, ~13× above thr).

## Gate protocol (banked before running; cert lines carry binary+invocation+banked output)

New binary `dsv4-gpu-decode-gate` (memra-engine). One long greedy decode from the
fixture 32-token prompt to s=2208 (2176 incremental steps, max_seq 2304), run TWICE in
one process, plus lane-4 re-prefill compares:

- **(a) step-equivalence**: at 63 checkpoint lengths s ∈ {33..72 (40 consecutive — many
  fine-block completions and pending straddles), 126,127,128 (coarse block 0 completes
  at pos 127),129 (first window wrap),130,131,132,160,192,255,256,257 (coarse block 1),
  384,512,768,1024,1025,2048,2052 (top-512 saturation: nb=513>512 at pos 2051),2053,
  2056,2176,2208}: re-prefill the decode run's first s tokens through gpu.forward()
  (the untouched lane-4 path) and compare the full 129,280-logit row: max-abs vs thr,
  argmax equality (mandatory), top-5 set + top-20 boundary-band (lane-4 derived rule),
  top1−top2 margin printed. **Top-512 saturation IS reachable and covered** (s ≥ 2052).
- **(b) greedy identity**: decode tokens[0..160] == lane-4 banked
  `dsv4-gpu-greedy.json` tokens_run0, 160/160 (greedy is prefix-stable so the longer
  run does not disturb the first 160).
- **(c) long-context probe**: the same s=2208 run passes ≥16 coarse completions, 544
  fine blocks, ≥16 window wraps, saturation; ≥6 of the (a) checkpoints sit past
  s=1024. Token-id agreement + logit deltas banked per checkpoint.
- **(d) determinism**: run 2 repeats run 1's token sequence exactly and the FULL
  2176-row logits stream sha256 matches byte-identically.
- **(e) VRAM**: mem_get_info at post-load/post-alloc/post-prefill/post-run checkpoints;
  measured cache allocation vs the capacity math above; zero OOM/CUDA errors.
- **informational only, single runs, NOT perf claims**: mean ms/step over s∈[190,210]
  and s∈[1014,1034], against lane-4's ~1.1 s/step re-prefill at s≈190 — the O(n²)→O(n)
  shape, nothing more.

## Run-1 (smoke, n_new=200) + bisect findings: the reference is not realization-stable —
## gate-policy corrections (banked BEFORE the rerun, lane-4 process)

Smoke run (`lane6/decode-gate-smoke-run1-FAIL.log`, `lane6/decode-gate-smoke-run1.json`):
determinism (d) BYTE-IDENTICAL across 2×200 steps, cache-math (e) MATCH, ~101 ms/step at
s≈200 (vs lane-4 ~1100 ms/step re-prefill) — but (a) FAILed everywhere: decode-vs-
reprefill logits max-abs 0.63–3.81 vs the banked thr ≈ 0.02, and (b) diverged from the
lane-4 banked sequence at step 15. The bisect that followed (dsv4-decode-probe,
`lane6/probe-banked.log`) established, in order:

1. **The decode path's per-row inputs are class-I clean.** At the first decode step,
   layer-0 post-attn-norm x is BIT-IDENTICAL to the re-prefill's (max-abs 0.0); q/kv/o
   sit at 1e-6/1e-7 (pure GEMM noise). The drift enters at the wo projection
   (o 1.2e-6 → attn_out 1.0e-3: measured Jacobian gain ~×300–800) and then compounds.
2. **Pure cuBLASLt m-sensitivity is tiny and my Class-II model was right**: the same
   row through the same bf16 weight at m=1 vs m=32 vs m=33 differs by ≤ 1.2e-7 abs
   (wq_a and wkv shapes measured; rows within one launch bit-identical).
3. **The pure lane-4 path is NOT realization-stable — measured control**: prefill@32 vs
   prefill@33 (zero lane-6 code), SAME position 31: attn_out drifts 3.4e-4 at layer 0,
   h-state 1.46 at layer 21, 4.76 at layer 42 (position 16: 13.1). The trunk amplifies
   per-hop 1e-7 reorder noise ~×1.3/layer until rmsnorm/hc anchoring saturates it at a
   decorrelation floor.
4. **The logits m-floor of the pure lane-4 path**: appending ONE token moves the SAME
   row's logits by 0.18–3.08 max-abs (15 rows measured, median ≈ 0.57). The smoke run's
   decode-vs-reprefill drift (0.63–2.88 at those rows) is the SAME distribution: the
   decode path adds nothing beyond the reference's own realization noise.
5. **The CPU oracle endorses the decode trajectory at lane-4's own standard**
   (`lane6/cpu-verify-smoke-197of200.json`, teacher-forced verify of the decode
   200-token sequence): 197/200 positions agree; ALL 3 divergences are near-ties with
   CPU margins 0.2106 / 0.1819 / 0.3121 — an order of magnitude inside the lane-4
   derived noise band 3·√2·u·√d·|top1| ≈ 6.1 (u = 2⁻⁸, d = 86, |top1| ≈ 40), the same
   physics note lane 4 banked for its top-20 boundary ("a 0.27 gap cannot be set-stable
   under ANY correct bf16 implementation" — the argmax at a 0.21 margin cannot be
   either). Lane-4's own greedy run crossed a 0.0061-margin near-tie and got the CPU
   token by correlated luck; two GPU realizations (decode vs re-prefill) are less
   correlated and flip such positions.

**Why the banked (a) threshold was wrong**: the u_ord model treated the trunk as
norm-preserving (comb doubly-stochastic + rmsnorm re-anchoring ⇒ √d random walk). That
holds for the AGGREGATE bf16 noise lane 4 modeled (its absmax-anchored thresholds
implicitly carry the growth), but per-hop 1e-7 noise is amplified by measured per-layer
Jacobian gains (×1.3/layer through wo/wq_b compositions and softmax-gate/e2m1-flip
discrete channels) until it SATURATES at the model's decorrelation floor — the same
floor for any sub-bf16 perturbation, measured ≈ 0.2–3 absolute on logits (points 3–4).
No decode implementation on this kernel stack can match re-prefill tighter than that
floor, because the reference does not match ITSELF across m.

### Corrected gate policy (derived, banked before the rerun)

- **(a1) decode vs re-prefill** (the brief's comparison, thresholds corrected): both
  paths are lawful realizations of the same arithmetic; each is bounded against the CPU
  oracle by the lane-4 derived bf16 bound thr₄ = u·√86·√(2 ln n)·absmax (lane-4
  measured 4.35 vs 6.71), so their mutual distance is bounded by the triangle
  inequality: **thr_a1 = 2·thr₄**. Argmax disagreement, top-5 and top-20 set changes
  are governed by the pair band = **2× the lane-4 boundary band** (3·√2·u·√86·|ref
  boundary logit|): any disagreeing id must be in-band of the reference row. Raw
  overlaps and margins always printed. The measured m-floor rows (consecutive
  checkpoints) are printed alongside as the realization-noise context.
- **(a2) decode vs the CPU ORACLE (the semantic instrument, factor 1)**: the lane-4
  teacher-forcing protocol on the DECODE trajectory: one CPU prefill over
  [prompt + decode tokens]; at every checkpoint the decode logits row must sit within
  thr₄ of the CPU row, top-1 must agree except in-band near-ties (band =
  3·√2·u·√86·|top1|), top-5/top-20 by the lane-4 band rule. This holds the decode path
  to EXACTLY the standard lane-4's re-prefill was held to.
- **(b) greedy identity, corrected**: raw decode-vs-lane-4-banked identity is a
  realization coin-flip at in-band near-ties (measured: first divergence step 15,
  CPU margin 0.2106). Corrected verdict: (i) the FIRST divergence vs the banked lane-4
  sequence must be at an in-band CPU margin (else it is a real bug); (ii) the CPU
  teacher-forcing of the full decode trajectory must agree at every position whose CPU
  margin is out-of-band (in-band flips reported with margins, never hidden). Raw
  agreement counts always printed.
- (c)/(d)/(e) unchanged.

## Run-2 crash: a latent lane-4 kernel launch-limit bug, found by the first long prefill

The full run's re-prefill leg crashed at checkpoint s=1024 with `fp4 qi rc=10001`
(cudaErrorInvalidValue): `dsv4_fp4_act_quant` (and `dsv4_act_quant`) launched groups on
grid.x and ROWS on grid.y — the indexer-q FP4 QAT at prefill has rows = s·64 index
heads = 65,536 at s=1024, one past CUDA's 65,535 grid.y ceiling. Lane 4 never prefilled
past 208 tokens, so the bug was invisible until this lane's long-context probe. Fix:
pure index swap (rows ride grid.x, limit 2³¹−1); per-group arithmetic untouched — and
the lane-4 output-sample gate rerun at this lane's tip is numerically IDENTICAL to the
banked lane-4 run-4 table (below). Crash log banked:
`lane6/decode-gate-run2-CRASH.log`. The decode runs themselves (rows ≤ 64/launch) had
already completed 2×2176 steps byte-identically before the crash.

## Lane 6 gate runs — full probe, s = 32 → 2208 (run 3, all gates)

```
target/release/dsv4-gpu-decode-gate /home/ubuntu/models/dsv4-flash-nvfp4 \
  /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json \
  /home/ubuntu/dsv4-greedy-out/gpu_greedy.json /home/ubuntu/dsv4-decode-out/full 2176 0,1
  -> banked: lane6/decode-gate-run3-PASS.log + lane6/decode-gate-run3.json  (exit 0)
target/release/dsv4-greedy-verify /home/ubuntu/models/dsv4-flash-nvfp4 \
  /home/ubuntu/dsv4-decode-out/full/decode_seq_for_verify.json /home/ubuntu/dsv4-decode-out/full/cpu-verify
  -> banked: lane6/cpu-verify.log (2164/2176; raw exit 1 = divergences found, adjudicated below)
target/release/dsv4-decode-oracle-check /home/ubuntu/dsv4-decode-out/full \
  /home/ubuntu/dsv4-decode-out/full/cpu-verify /home/ubuntu/dsv4-greedy-out/gpu_greedy.json
  -> banked: lane6/oracle-check-PASS.log  (exit 0)
target/release/dsv4-gpu-gate /home/ubuntu/models/dsv4-flash-nvfp4 \
  /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json 0,1
  -> banked: lane6/lane4-regression-gate-at-lane6-tip-PASS.log (15 arrays, 0 skips,
     exit 0 — table numerically identical to the lane-4 banked run-4 values)
```

Checkpoint logits rows (63 × 129,280 f32, decode + re-prefill) banked:
decode_ckpt_logits.bin sha256 b1bb64ad11af7cc5670bbdb30acb1226858e3bb8bfaf5da10b56d252
b0a597a0, reprefill_ckpt_logits.bin sha256 7f48ce4705dbaca11c5e8b4caa982bad8d76f58b0b2e
7257c2c671bb5ad6669d — rig-side ~/dsv4-lane6-bins/ (66 MB, not committed), box
/home/ubuntu/dsv4-decode-out/full/ (+ the 1.1 GB cpu_logits_all.bin, box only).

### Gate table (verdicts under the banked corrected policy)

| gate | verdict | measured | evidence |
|---|---|---|---|
| (a1) step-equivalence vs re-prefill, 63 checkpoints | **PASS 63/63** | max-abs 0.27–5.71 vs thr 12.8–17.4 (2·thr₄ pair bound); worst 5.71 at s=2052 (saturation onset); ONE argmax flip (s=47) at the in-band 0.21-margin near-tie; top-5/top-20 band violations 0 everywhere; m-floor rows 0.18–3.08 printed alongside (decode drift sits in the reference's own realization-noise band) | decode-gate-run3-PASS.log |
| (a2) step-equivalence vs the CPU ORACLE, 62 checkpoints (factor 1) | **PASS 62/62** | max-abs 0.34–5.44 vs thr₄ 6.4–10.4; worst 5.44 at s=2052 — same class as lane-4's own 4.35 vs 6.71; argmax SAME at every checkpoint except the s=47 in-band flip; band violations 0 (final checkpoint s=2208 has no CPU row — past the verified sequence end, reported not skipped-as-green) | oracle-check-PASS.log |
| (b) greedy identity vs lane-4 banked, corrected verdict | **PASS** | raw identity 18/160 with FIRST divergence at step 15 — CPU margin 0.2106 vs band 6.23 = in-band near-tie (legitimate realization flip; lane-4's own min crossed margin was 0.0061); CPU teacher-forcing over the full 2176-token decode trajectory: **2164/2176 agree, all 12 disagreements in-band** (margins 0.0103–1.42 vs bands 3.29–6.58) | oracle-check-PASS.log, cpu-verify.log |
| (c) long-context probe ≥1024 | **PASS** | 8 checkpoints at s ≥ 1024 (need ≥6); the s=2208 run crosses 17 coarse-block completions, 544 fine blocks, 16+ window wraps, and the **top-512 indexer saturation (REACHED: nb > 512 from s=2052**; at s=2052 top-20 overlap dips to 16/20 with 0 band violations — the saturation boundary shuffling near-tie blocks, as predicted) | decode-gate-run3-PASS.log |
| (d) determinism | **PASS** | 2 in-process runs: token sequences IDENTICAL and 2176-row logits streams sha256 BYTE-IDENTICAL (3bc516d925af…); PLUS the run-2 (pre-crash) sequence from a different process/build is byte-identical to run-3's (`decode_seq_run2.json` == `decode_seq_for_verify.json`) | decode-gate-run3-PASS.log |
| (e) VRAM / cache math | **PASS** | allocator bytes == design-formula bytes on BOTH devices ([26,943,488 / 28,237,824] at max_seq 2304); vram post-load 82.21/82.14 → post-alloc 82.24/82.17 → post-prefill/post-runs 82.27/82.21 GiB used (deltas = the caches + transients, ≥12.7 GiB free throughout); zero OOM, zero CUDA errors across load + 3×2176-step decodes + 63 re-prefills up to s=2208 | decode-gate-run3-PASS.log |

**Per-component classification (final, as mandated)** — Class I (bit-exact given
identical inputs, proven by the control's x@31 = 0.0 and by construction): embed
gather, every f32-island kernel (rmsnorm/headrms, dots GEMMs, rowsq, rope, hadamard,
QAT grids, pooling — decode emission reuses the prefill kernel with the 2-block pend
layout), host Sinkhorn/routing/topk, attention given the proven-identical slot order,
cache row construction. Class II (accumulation order legitimately differs): the
cuBLASLt bf16 GEMMs at m=1 vs m=s — measured per-row m-sensitivity ≤ 1.2e-7; the trunk
amplifies this to the measured decorrelation floor (0.18–3.08 on logits per single
m-step of the reference itself), hence the pair bound 2·thr₄ through the CPU oracle,
under which every checkpoint passes with ≥2.4× headroom. Class III (discrete): e2m1
flips (selection-only below saturation), router/argmax near-ties — governed by the
derived 3·√2·u·√d bands; every observed flip (1 vs re-prefill, 12 vs the oracle over
2176 steps) measured in-band; zero out-of-band discrete events in the entire run.

### Informational latency shape (single runs, NOT perf claims)

Decode: **102 ms/step at s≈200** and **112 ms/step at s≈1024** (2176 steps in 247 s
end-to-end; 119 ms/step at s≈2200) vs lane-4's ~1,100 ms/step re-prefill at s≈190 —
~11× at s≈200 and near-flat vs s where re-prefill grows ~linearly: the O(n²)→O(n)
shape landed. No CUDA-graph, no batching, host round-trips per layer — perf-lane
headroom is large and unclaimed.

### Lane 6 summary

The true decode path (incremental KV: 128-slot window rings, append-only compressed-
block stores with the coff=2 two-block pending state machine, FP4-QAT'd indexer
stores, hc-state carried, one PP host-bounce per step, batched per-expert MoE dispatch
at s=1) is landed and gated: equivalent to the lane-4 re-prefill path at the
reference's own realization-noise floor (a1), within the lane-4 derived bf16 bound of
the CPU oracle at every checkpoint (a2), greedy-consistent with the oracle at
2164/2176 with only in-band near-tie flips (b), across coarse/fine block completions,
window wraps and indexer top-512 saturation (c), byte-deterministic (d), and exactly
on its banked VRAM math (e). Support claim stays per the house law: (DeepSeek-V4-Flash,
NVFP4 artifact, clamp-only) DECODE correctness on this box is gated at the bring-up
rung — bf16 dequant-on-the-fly GEMMs, host-bounce PP boundary; NOT a serving or perf
claim.

### Open for the next lanes

1. **Native quant GEMMs** (NVFP4 MMQ / fused MXFP4): new numeric class — own
   output-sample + greedy + decode gates; this lane's thresholds do not transfer.
2. **PP-2 serving door**: fold placement + decode caches into PpNRt
   (MEMRA_PP_STAGES=2), peer-copy boundary (pp.rs:1527 mempool grants), admission/KV
   plumbing, CUDA-graph capture of the decode step, batched decode.
3. **Drafter cells**: MTP spec-decode with the lane-3 V3 shift on top of this decode
   loop; acceptance-rate cells.
4. Long-PROMPT arrival needs a chunked prefill (lane-4 prefill transients are O(s);
   the decode caches themselves fit 1M in headroom per the banked table).
5. The indexer-score kernel is one thread per (query, block) — fine at gate lengths,
   a perf-lane rewrite before long-context decode claims.

---

# Lane 7 — native quantized expert GEMMs (NVFP4 trunk / MXFP4 MTP) (2026-08-18)

Lane 7 of dsv4-flash support: replace the lane-4 dequant-to-bf16 expert GEMM rungs with
NATIVE quantized arms implementing the reference GEMM semantics (SEMANTICS.md §7.1: the
reference GPU stack runs FP8/FP4 GEMMs with dynamic pow2-ceil activation scales —
natively-quantized expert GEMMs are the REFERENCE behavior; lane 4's bf16-dequant rung
was the conservative bring-up deviation). Branch `lane/dsv4-flash-loader`, NOT merged,
NOT pushed. This section is the plan of record, banked BEFORE the arms were built.

## Reference GEMM semantics (the law, cited from the artifact's own inference/kernel.py
## + model.py — read on the box before this design)

- **Dispatch (model.py:108-120)**: FP4-dtype weight → `act_quant(x, 128, scale_fmt,
  scale_dtype)` then `fp4_gemm` (M:113-115); FP8 weight → same act_quant then `fp8_gemm`
  (M:116-118). Runtime knobs (config.json + ModelArgs M:39-42): `dtype:"fp8"`,
  `scale_fmt:"ue8m0"` ⇒ `round_scale=True`, `expert_dtype:"fp4"`, `scale_dtype:"fp8"`
  ⇒ scales stored E8M0 (exact for pow2 scales — a storage no-op numerically).
- **act_quant (kernel.py:40-125, non-inplace = the GEMM path)**: per contiguous 128
  group along K: `amax = max(|x|, 1e-4)`; `s = fast_round_scale(amax, 1/448)` =
  2^ceil(log2(amax/448)) (K:36-37, pow2 bit math); `y = clamp(x/s, ±448)` CAST TO FP8
  E4M3 (K:92-96 — REAL RNE rounding in BOTH kernel variants; the clamp-only artifact
  fork at kernel.py:88 applies only to the INPLACE KV-QAT path). Scale exact pow2.
- **fp4_gemm (kernel.py:441-536)**: `C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T`. A = e4m3
  codes + per-128 scales; B = e2m1 packed 2/byte along K + per-32 scales (K:447-451).
  Per 32-K sub-block (block_K=32=weight_group_size, K:462): FP4→FP8 cast (K:494-496 —
  EXACT: every e2m1 value is representable in e4m3), FP8×FP8 `T.gemm` into f32
  fragment (tensor cores, f32 accum), then `C_accum += C_local * scale_a * scale_b`
  (K:508-509) — scales applied per sub-block in f32, sub-block sums accumulated f32
  sequentially over K/32 iterations. `scale_a` indexed k//4 (per-128, K:502-504),
  `scale_b` per-32 (K:498-500).
- **Expert.forward (model.py:596-606)**: w1/w3 outputs `.float()`; clamps
  (up two-sided, gate one-sided) + silu in f32; routing weight multiplied BEFORE w2
  (M:604-605); `w2(x.to(bf16))` → act_quant per-128 → fp4_gemm again. So the swiglu
  intermediate IS re-quantized, per-row-per-128, with the routing weight inside the
  quantization (scale is pow2 so pow2-ratio weights commute, general weights do not).
- **MoE (model.py:629-644)**: routed accumulation f32, `y.type_as(x)` (bf16) at end;
  shared expert = FP8-weight Expert → fp8_gemm path.
- **Artifact recipes vs this law**: the artifact's own convert.py consumes experts as
  int8 e2m1 pairs + per-32 scales (convert.py:142-149, the fp4_gemm `scales_b`
  contract) — i.e. the reference checkpoint's expert recipe is MXFP4-shape. The
  artifact's TRUNK experts as shipped are the modelopt NVFP4 re-encoding (per-16 E4M3
  × F32 scale_2) — lane-1 measured law: a LOSSLESS cast (every effective scale
  e4m3×scale_2 is an exact pow2; adjacent per-16 pairs identical = 32-group ancestry).
  MTP experts are stored EXACTLY in the reference recipe (e2m1 + E8M0/32).

## Existing-kernel survey (task 1) — verdict: NO engine arm matches; write the dsv4 arms

Per the no-generic-support law, match = same weight-scale granularity + same activation
quantizer + same scale application; "runs" is not "matches". Cited arithmetic:

| engine arm | weight side | activation side | mismatch vs dsv4 law |
|---|---|---|---|
| cu/mmq_fp4.cu (W4A4 MMQ) | llama block_nvfp4: 36-B interleaved [4 UE4M3 scale bytes + 32 qs], per-16 | `quantize_mmq_nvfp4` → FP4 e2m1 codes, 2-level e8m0/ue4m3 scales per 32 | activations are FP4 not FP8; scale rule ≠ pow2-ceil(amax/448) per-128; interleaved layout ≠ artifact planar; no scale_2 |
| cu/cutlass_fp4_sm120.cu (W4A4 CUTLASS) | e2m1 K-major + swizzled ue4m3/16 SF + alpha epilogue | A operand = e2m1 activations + ue4m3/16 SF | same class mismatch: FP4 activations, per-16 float scales |
| cu/mmq_nvfp4_w4a8.cu (W4A8 int8) | FP4→int8 LUT at tile load + per-16 UE4M3 float scales | q8_1 int8, float scale amax→127 per 32 | int8 linear grid ≠ e4m3 float grid; per-32 float scale ≠ per-128 pow2-ceil |
| cu/mmq_nvfp4_f8f4.cu (+ tile in w4a8 file, MEMRA_MMQ_F8F4) | per-16 scale FOLDED into e4m3 containers at load (`cvt_e4m3(kvalue*s16)` — a weight rounding the reference never performs) | e4m3 codes + f32 amax/448 scale per **32** | act scale rule (float amax/448, not pow2-ceil) and group (32 ≠ 128); weight fold rounding |
| cu/mmq_fp8_blk.cu (FP8 128×128) | e4m3 [out,in] + f32 128×128 grid — LAYOUT matches dsv4 FP8-blk | e4m3 per-128, scale = f32 amax/448 (v2 header: "all four d4 slots carry the same per-128 amax/448") | act scale is FLOAT amax/448, dsv4 law is pow2-ceil (ue8m0) — a real re-quant difference; also FP8 linears are out of this lane's expert scope (below) |
| cu/qmatvec.cu NVFP4 MMVQ / qmatvec_gemm.cu | GGUF NVFP4 per-16 ue4m3, no scale_2 | int8 q8_1 per-32 (dp4a / int8 MMA) | activation class + scale rule |

None implement (e4m3 codes per-128 pow2-ceil activations) × (e2m1 + pow2 group scales)
with per-sub-block f32 scale application, and none read the artifact's planar
U8[out,in/2] + E4M3[out,in/16] + F32 scale_2 slabs without a repack. **Decision: write
the dsv4-native arms in cu/dsv4_gpu.cu following its conventions** (correctness rung:
deterministic fixed-order f32 accumulation, CPU-mirrorable bit-exactly; tensor-core MMA
is a later perf rung under the interleaved A/B law).

## The native arms (task 2)

New kernels (cu/dsv4_gpu.cu), switch `MEMRA_DSV4_EXPERT_ARM=native` (default = the
gated lane-4 bf16-dequant rung; the switch is read once at load and printed):

1. `memra_dsv4_act_quant_fp8` — f32 x [rows,K] → u8 e4m3 codes [rows,K] + f32 scales
   [rows,K/128]: amax (fixed-tree max, order-free), max(amax,1e-4), s = pow2_ceil
   (amax/448) (the kernel.py:36-37 bit formula, already in the TU), codes =
   `__nv_cvt_float_to_fp8(clamp(x/s,±448), SATFINITE, E4M3)` (the oracle-pinned RNE).
2. `memra_dsv4_fp4_gemm` — out[g,n] f32 from A codes+scales × as-stored expert slabs.
   kind NVFP4: per-16 groups, group scale = e4m3(sc[j])·scale_2; kind MXFP4: per-32
   groups, scale = 2^(sc[j]−127). Per output (g,n): thread t owns groups j ≡ t (mod
   128); per group: 16/32 products decoded f32 (each product EXACT in f32 — e4m3 ≤4
   sig bits × e2m1 ≤2 sig bits = ≤6), summed f32 sequentially; sub·(w_scale·a_scale)
   f32; accumulated f32 per thread in ascending j; 128-partial fixed halving tree.
   Deterministic by construction; the gate's CPU mirror replicates it BIT-EXACTLY.
   g > 65535 refused (grid-dim contract, lane-6 lesson).
3. `memra_dsv4_gather_rows_u8` — byte-row gather (codes + scale rows to expert groups;
   per-row quantization commutes with gathering exactly).

**Declared realization deviations from the reference stack (all conservative, all also
present in the CPU truth instrument so they cancel in gating):**
- Activations quantize from f32, not from a bf16-rounded copy (our pipeline keeps f32
  between ops — the lane-4 blessed realization; one fewer rounding).
- NVFP4 trunk arm applies scales per-16 as stored (the artifact recipe) instead of the
  reference per-32 grouping: identical scaled-product SET (products and pow2 scales are
  exact), different f32 summation GROUPING only — same lawful-reorder class as the
  tilelang kernel's own tensor-core reduction order, which no host implementation can
  replicate anyway. The per-16↔per-32 scale equality is the lane-1 measured pair-share
  law; gate (a)'s f64 cross-check bounds any violation.
- f32 sub-block accumulation order: ours is fixed thread-strided + tree; the reference
  tile's is the MMA fragment order. Both are f32-reorder realizations of the same exact
  products; gate (a) bounds ours against an f64-ordered reference.
- Shared experts stay on the bf16 rung (they are FP8-blk weights — see the FP8
  decision below), and the routed-expert f32 MoE accumulation + f32 hc stream stay
  (reference casts MoE output to bf16, M:644 — ours is the conservative side).

## FP8 linears decision (task 3): stay bf16 — deviation stated

The reference CLEARLY runs every FP8 linear quantized (model.py:116-118 → fp8_gemm,
K:204-254: act per-128 e4m3 + 128×128 weight grid, per-K-block scaled f32 accumulation
with a second-level accumulator). But no engine FP8 arm matches: the closest,
mmq_fp8_blk.cu, shares the weight layout yet quantizes activations with a FLOAT
amax/448 scale, not the dsv4 pow2-ceil ue8m0 law — a different quantizer (its own
header calls the activation quantizer "v2's one arithmetic change, declared not
smuggled"). Adapting it = a new activation quantizer + rewiring every attention/shared
projection — outside this lane's expert mandate. **The FP8-blk linears therefore stay
host-dequant→bf16 (bit-exact-refusal-checked in lane 4): a conservative, lawful
realization that REMOVES a reference quantization rather than approximating one.** The
deviation direction equals the CPU oracle's (which also runs those GEMMs unquantized),
so gates against the oracle do not hide it. Banked as the explicit follow-up arm.

## Numeric class + gate derivations (task 4; banked BEFORE any gate run)

Constants: u_b = 2⁻⁸ (bf16 roundoff, lane 4); **u_q = 2⁻⁴ (e4m3 roundoff: 3 mantissa
bits → relative spacing 2⁻³, RNE half-step 2⁻⁴)**. The act scale is exact pow2 and the
clamp never bites (448·pow2ceil(amax/448) ≥ amax); e4m3-subnormal elements have LARGER
relative but SMALLER absolute error than u_q·|x| — the model upper-bounds them.

Per-array class coefficient (the lane-4 random-walk + extreme-value doctrine with a
second variance term): **C(d_b, d_q) = √(d_b·u_b² + d_q·u_q²)**, thr = C·√(2 ln n)·
absmax_ref, discrete bands = 3·√2·C·|ref boundary logit| (pair comparisons ×2, lane-6
triangle rule). Hop counts (native arm): each MoE sub-block injects 2 quantized hops
(w1/w3-input quant; w2-input quant — the two independent e4m3 roundings of that
block's expert path). d_b stays the lane-4 count (attention + shared-expert + expert
bf16 hops — slightly double-counting the expert hops now carried by u_q: conservative).

| array | d_b | d_q |
|---|---|---|
| embed_out | bit-exact | — |
| layerN_out | 2(N+1) | 2(N+1) |
| layerN_attn_out | 2N+1 | 2N |
| layerN compressor/indexer/index_score | 2N | 2N |
| final_logits_last | 86 | 86 |
| mtp_logits_last | 88 | 88 |

Worked: layer0_attn_out keeps the lane-4 bound EXACTLY (no MoE upstream — a tight
canary that the attention path is untouched). layer0_out: C = √(2u_b²+2u_q²) ≈ 0.0885
(16× the bf16 class). final logits: C ≈ 0.583 → thr ≈ 0.583·4.85·35.51 ≈ 100.4.

**Honesty note, banked up front: at trunk depth the class bound is LOOSE and the id
bands are wide.** This is the physics of the class, not gate generosity: (i) the model
was QAT-trained — the f32-pipeline fixtures are the idealized realization, the
quantized GEMMs are the reference behavior; (ii) any comparison across the quantizer
boundary rides flip amplification (lane-3 measured: one rounding fork → 0.39 logits
from 1e-6 skew; lane-6: sub-bf16 noise amplifies to a 0.2–3.08 decorrelation floor).
The SHARP instruments for this class are gate (a)'s bit-exactness, the shallow arrays
(layer0/2/3 rows at 16×-tight bounds), selection/mask exactness, and determinism. Deep
id-level agreement is CHARACTERIZED (rates + margins printed) and gated only by the
derived bands. Discrete-channel budgets, re-derived for the native class: indexer_kv
upstream rel drift ≈ √(4u_b²+4u_q²) ≈ 0.125 vs e2m1 min one-step gap ≈ 0.25 ⇒ expected
one-step flip fraction ≈ 0.5 → budget 0.75 of elements, adjacency bound |diff| ≤
1.0·max(|got|,|ref|) unchanged; index_score exceeders ≤ 0.25·absmax (flip-noise on a
128-dot, ~√(0.5·128)·one-flip ≈ 0.1–0.2 of scale), budget 0.5. Selection semantics
stay EXACT gates: −inf causality mask equality; top-min(512,nb) = all completed blocks
at gate lengths.

### The truth instrument: CPU oracle extended with the expert act-quant emulation

`dsv4_forward.rs` gains the native class under the same switch: expert GEMM inputs
(xg before w1/w3; h AFTER the routing-weight multiply, before w2 — M:604-606 order)
pass through act_quant(128) with REAL FP8 rounding (the RefFp8Round grid — the GEMM
path rounds in BOTH kernel variants). Weights stay the lane-1 exact f32 decode;
products are exact; f64 accumulation = the ideal-order reference of the SAME quantized
arithmetic. Shared expert unquantized (mirrors the GPU arm's banked deviation).
Pin against truth, not siblings: gates (b)/(c)/(d) adjudicate against THIS oracle.

### Gate protocol

- **(a) kernel-level paired gate** (`dsv4-native-gemm-gate`, new binary): moe-input x
  rows captured from the REAL fixture-prompt GPU forward at layers {0, 2, 20, 22, 42}
  + the MTP block (new moe_x capture), ≤8 rows each. Samples = the lane-4 sub-gate
  set: (layers.0 e0, layers.2 e100, layers.20 e7 = lane-1 NVFP4 pin, layers.22 e31,
  layers.42 e255, mtp.0 e7 = lane-1 MXFP4 pin, mtp.0 e200) × all three projections
  (w2 input h derived from the captured x through the quantized CPU chain). Three
  comparisons each:
  1. GPU native kernel vs the CPU BIT-EXACT MIRROR (same fixed thread-strided groups +
     halving tree in f32, same decode helpers): **bit-exact required** — the kernel
     computes exactly the intended arithmetic or fails.
  2. GPU native vs the f64-ordered quantized reference: |err| ≤ n_ops·u32·Σ|p_scaled|
     per output (u32 = 2⁻²⁴; n_ops = gs−1 sequential + K/(gs·128) per-thread + 7 tree
     ≈ 24 (NVFP4) / 40 (MXFP4)) — the stated tensor-class f32-accumulation error model.
  3. native vs the bf16-dequant rung on the SAME inputs: the numeric-class shift,
     measured against the prediction |Δ| ≈ (u_q/√3)·‖x⊙w‖₂ per output (rms model;
     reported as a distribution, informational).
- **(b) output-sample gate**: `dsv4-gpu-gate` under MEMRA_DSV4_EXPERT_ARM=native vs
  the artifactvariant fixtures — thresholds/bands per the C(d_b,d_q) table (the binary
  prints the class), logits rows gated by band rule at top-1/5/20 (a top-1 flip must
  be in-band); expert-dequant sub-gate unchanged (the bf16 rung stays resident).
  PLUS the class-shift pin: CPU-quantized-oracle final-logits row (from gate (c)'s
  teacher-forcing run, position 31) vs the f32 fixture row = the clean CPU-only
  measurement of the numeric-class shift, banked.
- **(c) greedy trajectory**: `dsv4-gpu-greedy` (native env) 160 tokens from the
  fixture prompt; `dsv4-greedy-verify` (native env → quantized oracle) teacher-forces
  the GPU trajectory; every disagreement adjudicated by the native pair band
  2·3·√2·C(86,86)·|cpu top1| (verify gains the in-band adjudication for the native
  class; raw agreement + margins always printed).
- **(d) decode-path integration**: `dsv4-gpu-decode-gate` (native env) n_new = 260 —
  ≥40 consecutive early checkpoints + fine-block completions + coarse block 0 (s=127/
  128) + window wrap (129) + coarse block 1 (256) from the lane-6 checkpoint list;
  (a1) decode-vs-reprefill at native pair bounds 2·C(86,86)·√(2 ln n)·absmax + band
  rule ×2; (a2/b) `dsv4-decode-oracle-check` vs the quantized oracle at factor-1
  bounds C(86,86); (e) cache math + VRAM unchanged. Saturation (s ≥ 2052) is NOT
  reachable at this length and is SAID so (lane-6 covered it; the caches are
  arm-independent).
- **(e) determinism**: two runs byte-identical (in (c) runs×2 and (d) run×2).
- **Informational only (single runs, labeled, NOT claims)**: decode ms/step at
  s∈[190,210], native vs bf16-dequant rung, same binary same box same day — the shape
  of the speed win only; the CLAIM comes later under the interleaved A/B law.

Cert lines carry binary + invocation (incl. the env seam) + banked output path.

## Gate (a) run 1/2 — encoder zero-sign canonicalization (banked BEFORE the rerun)

Run 1 (`lane7/native-gemm-gate-run1-FAIL.log`): 13/21 projections BIT-EXACT (all
w1/w3 except layer 22, incl. both lane-1 pins, both recipes); every failure is
`FAIL(quant)` — code BYTES differ while every f64-bound check PASSES at 1e-7-class
errors and the class-shift median ratio is a flat 0.48–0.51 of the banked
(u_q/√3)·‖x⊙w‖₂ prediction (the error model is right, conservative ×2).

Run 2 with the named-mismatch instrument (`lane7/native-gemm-gate-run2-diag.log`):
every code diff is a TINY NEGATIVE input (|x/s| ≈ 4e-4..7e-4, below half the least
e4m3 subnormal 2⁻¹⁰) that RNE-rounds to ZERO: GPU `__nv_cvt_float_to_fp8` emits IEEE
**-0.0 = 0x80**; the house CPU encoder `f32_to_fp8_e4m3` **canonicalizes any
zero-magnitude result to +0 = 0x00** (nvfp4_repack.rs, documented "±0 → 0x00",
roundtrip-pinned, and consumed by the gated MEMRA_PP_FP8 loader arm — not this lane's
to change). This was previously UNEXERCISED: under the clamp-only contract the KV-QAT
sims never round to FP8, so no prior gate ever compared the two encoders' code bytes.
w2 rows expose it because swiglu's h crosses zero; layer-22 x rows carried 2 such
elements by chance.

**Decoded values are IDENTICAL (±0.0)** — no arithmetic fork; the only observable
would be the sign of an exactly-zero output. Correction (value-semantics unchanged,
codes made bit-canonical on both sides): the dsv4 act-quant kernel canonicalizes a
zero-magnitude e4m3 code to +0 (`(c & 0x7F) == 0 → c = 0`) — matching the house
canonical form; the reference itself stores torch-style signed zeros, and the choice
is invisible to the GEMM (products/sums of ±0 are value-equal). Banked as a
canonicalization deviation, not a numeric one.

## Recon-lane intel folded in (orchestrator, mid-lane; RECON.md in darklanes)

1. **CUTLASS sm_120a NVFP4 grouped GEMM is broken upstream (issue #3096)** — vLLM's
   sm12x path falls back to Marlin bf16-dequant (storage-only NVFP4). This lane's
   correctness arms are hand-written scalar-f32 kernels on the as-stored slabs and use
   NO CUTLASS — unaffected. Banked for the PERF rung: do not build the fast arm on
   CUTLASS grouped GEMM; the working precedent on this card class is hand-rolled
   block-scale mma.sync (the mmq_fp4.cu lineage) — gate exact shapes either way.
2. **vLLM's sm12x cutedsl fork lost the swiglu_limit clamp.** Gate (a) gains a
   targeted saturation cell (real captured row ×50 → w1/w3 outputs beyond ±10): clamp
   engagement counted on both sides, h vs CPU clamp math at expf-ULP bound, GEMM legs
   mirror-bit-exact — both recipes.
3. Precedent note: MXFP8-act × FP4-weight is the most-precedented Blackwell recipe
   elsewhere (TRT-LLM Gen, DeepGEMM MegaMoE). Our arm implements the dsv4 REFERENCE
   act-quant law (e4m3 per-128 pow2-ceil — kernel.py act_quant), which is what the
   artifact was evaluated with; precedent noted, law wins.
4. TRT-LLM keeps NVFP4 and MXFP4 paths distinct and keeps the clamp — matches this
   lane's two-kind design; available as a cross-check reference if ever needed.

## Lane 7 gate runs — ALL PASS (cert lines; logs in lane7/, box /home/ubuntu/dsv4-lane7-out/)

Binaries built on the box from the lane files at the final tip (sha-parity verified
rig↔box before the first build and at close), rust 1.97.1 release, MEMRA_CUDA_ARCH=120a,
CUDA 13.2. Every native invocation carries the class seam `MEMRA_DSV4_EXPERT_ARM=native`
(read identically by the GPU forward and the CPU oracle — one seam, no class mixing).

```
(a) MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-native-gemm-gate <model> <fixtures-artifactvariant.json> 0,1
      run 1 -> lane7/native-gemm-gate-run1-FAIL.log   (encoder zero-sign finding, banked above)
      run 2 -> lane7/native-gemm-gate-run2-diag.log   (named-mismatch instrument: -0.0 -> 0x80 vs 0x00)
      run 3 -> lane7/native-gemm-gate-run3-PASS.log   (PASS, exit 0)
(b) MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-gpu-gate <model> <fixtures> 0,1
            -> lane7/gpu-gate-native-run1-PASS.log    (PASS, 15 arrays, 0 skips, exit 0)
    target/release/dsv4-gpu-gate <model> <fixtures> 0,1          (bf16 regression)
            -> lane7/gpu-gate-bf16-regression-PASS.log (PASS — every measured value
               NUMERICALLY IDENTICAL to the lane-4 banked run-4 table: the refactor is
               inert on the gated rung)
(c) MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-gpu-greedy <model> <fixtures> <out> 160 2 0,1
            -> lane7/gpu-greedy-native.log
    MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-greedy-verify <model> <out>/gpu_greedy.json <out>/cpu-verify
            -> lane7/greedy-verify-native.log         (PASS, exit 0)
(d) MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-gpu-decode-gate <model> <fixtures> <native gpu_greedy.json> <out> 260 0,1
      run 1 -> lane7/decode-gate-native-run1-cFAIL.log (lane-6 long-probe criterion at a
               short probe; the banked lane-7 short-probe criterion implemented, no
               measured value changed between runs)
      run 2 -> lane7/decode-gate-native-run2-PASS.log  (PASS a1/c/d/e, exit 0)
    MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-greedy-verify <model> <out>/decode_seq_for_verify.json <out>/cpu-verify
            -> lane7/decode-verify-native.log          (252/260, 8 in-band)
    MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-decode-oracle-check <out> <out>/cpu-verify <native gpu_greedy.json>
            -> lane7/decode-oracle-check-native-PASS.log (PASS a2+b, exit 0)
(speed pair) target/release/dsv4-gpu-decode-gate ... 260 0,1  (bf16 arm, same binary/box/day)
            -> lane7/decode-gate-bf16-260-PASS.log     (PASS, exit 0)
```

### Gate table (native class; C(d_b,d_q) = √(d_b·u_b² + d_q·u_q²), u_b=2⁻⁸, u_q=2⁻⁴)

| gate | verdict | measured (vs derived bound) |
|---|---|---|
| (a) kernel paired gate | **PASS 21/21 BIT-EXACT** | GPU == CPU fixed-tree mirror bit-for-bit on every (expert, projection): both recipes, both lane-1 pins, real captured activations (m=8 rows); f64-ordered reference max err 4.5e-8..1.4e-6 vs n_ops·2⁻²⁴·Σ|p| bounds (all PASS); act-quant codes+scales byte-identical after the zero-sign canonicalization |
| (a) class-shift characterization | measured | native vs bf16-rung per-output |Δ| median ratio 0.47–0.51 of the (u_q/√3)·‖x⊙w‖₂ prediction — the banked error model verified, conservative ×2 |
| (a) swiglu_limit saturation cells | **PASS 2/2** | clamp engaged (643+1662 / 587+1700 of 2048 elements past ±10), h vs CPU clamp math ≤1.9e-6 (thr 2e-4, expf-ULP class), GEMM legs bit-exact — the vLLM-fork clamp-loss failure mode is excluded on both recipes |
| (b) output-sample, native | **PASS 15/15** | final logits max-abs **4.996** vs thr 107.6 (0.05× of the class bound); top-1 2581 exact, top-5 set EXACT, top-20 18/20 with 0 out-of-band; layer0_attn_out **1.832e-2 == lane-4 exactly** (attention path untouched — canary); layer0_out 4.259e-1 vs bf16's 5.064e-2 (the class shift, visible and bounded); indexer_kv/index_score exceeders **0** (the re-derived flip budgets went unused — the class thresholds absorb one-step flips) |
| (b) CPU-only class-shift pin | measured | quantized-oracle vs f32-fixture final logits: **max-abs 3.68, p99 2.10, median 0.53**; top-1 same, top-5 exact, top-20 19/20 — the reference's expert act-quant moves logits by the SAME order as the bf16 drift itself (4.35), ~27× inside the random-walk class bound |
| (c) greedy 160, native | **PASS** | 2 runs byte-identical (tokens + logits bins); CPU quantized-oracle teacher-forcing **157/160**, all 3 disagreements in-band at margins **0.064–0.446** — inside even the ~6-wide bf16-class band, ~450× inside the derived native band |
| (d) decode integration | **PASS** | (a1) 52/52 checkpoints vs re-prefill at pair bounds (worst 3.097 at s=44 vs thr ≈206; argmax same at ALL 52; top-5/top-20 violations 0); boundary coverage 127/128/129/256/257 + 40 consecutive early steps; (a2) all checkpoints vs the quantized oracle, worst **6.374 vs pair bound ~214** (bf16-class magnitude); (b) 252/260 with all 8 disagreements in-band (margins 0.036–0.75); cache math == allocator bytes on both devices; zero OOM/CUDA errors |
| (e) determinism | **PASS** | greedy 2× byte-identical; decode 2× tokens identical + logits streams sha-identical (ee737e83d8f2…) |

### The two headline measurements (measured, not modeled)

1. **The numeric-class shift is small.** The honest pre-run bound (random-walk with
   u_q=2⁻⁴ over 86 quantized hops → ~100 on logits) is ~27× conservative: the
   measured end-to-end shift of the reference's own expert quantization is
   max-abs 3.7 / median 0.53 on final logits (CPU-only pin), and the GPU native
   forward sits at 4.996 vs the f32 fixtures — barely above the bf16 rung's own 4.35.
   The QAT training story holds up numerically.
2. **The trajectory is semantically locked.** 157/160 and 252/260 greedy agreement vs
   the quantized oracle with every disagreement a sub-0.75-margin near-tie (median
   crossed margin ≈0.3 vs typical top-1 margins ~9) — the same in-band physics lanes
   4/6 established, now for the native class.

### Informational latency shape (single runs, same binary/box/day, NOT perf claims)

Decode at s∈[190,210], n=21 steps each: **native 91 ms/step vs bf16-dequant rung
101 ms/step** (lane-6 banked 102 for the same rung) — ~10% end-to-end; the expert
GEMMs are a fraction of a step that is dominated by per-layer host round-trips
(lane-6 note). The point of the native arm at this rung is correctness + the
~8× expert weight-traffic reduction it unlocks; the CLAIM waits for the perf lane
under the interleaved A/B law (and per the recon intel, that lane must avoid
CUTLASS grouped GEMM on sm_120a — issue #3096).

### What landed (commits on lane/dsv4-flash-loader, unmerged, unpushed-to-origin)

| commit | content |
|---|---|
| 25ac3c971c | lane-7 plan of record (survey, arm design, FP8 decision, derivations) |
| 24aa7ab0f1 | the native arms + class seam + gate binaries + class-aware thresholds |
| c47c7bc9cf | Cargo.lock sync (lane-4 sha2 dep) |
| ed2940fb4e | zero-sign canonicalization + named-mismatch diagnostics |
| eea0541cc7 | swiglu_limit saturation cell + recon-intel notes |
| 1e87f918d7 / ffada7825d / 23309b8fe5 / ea6cd60ddc | gate data banked per run |
| (this) | decode-gate short-probe criterion + final receipts |

New engine surface: `memra_dsv4_act_quant_fp8` / `dsv4_fp4_gemm` / `dsv4_gather_rows_u8`
(cu/dsv4_gpu.cu), `ExpertArm` + native moe_forward branch + moe_x capture (dsv4_gpu.rs),
oracle expert act-quant emulation + class helpers (dsv4_forward.rs), gate binary
`dsv4-native-gemm-gate`, class-aware thresholds in the four existing gate/verify bins.

### Support claim (house law)

**(DeepSeek-V4-Flash, NVFP4 artifact, clamp-only contract) natively-quantized expert
GEMMs (NVFP4 trunk + MXFP4 MTP, reference kernel.py fp4_gemm semantics) are
correctness-gated on 2× RTX PRO 6000 Blackwell 96GB** — prefill, greedy, and the
lane-6 decode path, behind `MEMRA_DSV4_EXPERT_ARM=native` with the bf16-dequant rung
as the default fallback and A/B reference (its own gates re-run green and numerically
identical). NOT a perf claim; NOT a serving claim. The FP8-blk linears remain
host-dequant→bf16 — the reference runs them quantized, no engine FP8 arm matches the
pow2-ceil act-scale law (survey above), banked as the explicit follow-up arm.

### Open for the next lanes

1. **Perf rung for the native arms**: batched/tensor-core fp4 GEMM (hand-rolled
   block-scale mma.sync lineage, NOT CUTLASS grouped — #3096), fused gather+GEMM,
   CUDA-graph decode step; interleaved A/B for any claim.
2. **FP8-blk linear arm** (attention/shared/e-h_proj): needs a pow2-ceil e4m3
   activation quantizer + per-128×128 grid GEMM matching kernel.py fp8_gemm
   (mmq_fp8_blk.cu is layout-compatible but its amax/448-float act scale is a
   different quantizer).
3. **PP-2 serving door** (PpNRt, peer-copy, batched decode), chunked prefill,
   indexer-score kernel rewrite before long-context decode claims — unchanged from
   lane 6.
4. **MTP drafter cells** on the decode loop (V3 shift confirmed lane 3; MTP native
   arm now gated here).

---

# Lane 8 — decode PERF rung (host-round-trip elimination, graphs, fused dispatch) (2026-08-18)

Lane 8 of dsv4-flash support: remove the structural decode overheads with gated
correctness and produce the program's FIRST perf numbers under the interleaved A/B law.
Branch `lane/dsv4-flash-loader`, NOT merged, NOT pushed. Baseline = the lane-7 tip
(1dc08d00f3) native arm: 91 ms/step at s≈200 ≈ 11 tok/s (informational). Public bar on
this card class (RECON.md §2): canada-quant 2× RTX PRO 6000 TP2 vLLM-fork W4A16
**47.5 tok/s bs=1** (TPOT 20.8 ms) / ~70 tok/s with the MTP sibling. Recon landmine
honored: NO CUTLASS grouped GEMM on sm_120a (issue #3096) — any fast-GEMM arm is
hand-rolled (mmq_fp4.cu block-scale mma.sync lineage or vectorized GEMV).

## Structural inventory of one decode step (read from the code, banked BEFORE profiling)

Per step at s≈200 (42 trunk layers: 21 fine, 19 coarse, 2 window-only):
- **Blocking D2H syncs ≈ 151**: hc_pre Sinkhorn mixes readback 2/layer (84), MoE router
  raw-score readback 1/layer (42), fine-indexer score readback + host topk (21),
  head-collapse mixes (1), final logits (1), PP layer-22 boundary host bounce (1+sync).
- **Pageable H2D uploads ≈ 600**: pos scalar per layer + per emission, idx list per
  layer, per-expert row/weight vectors (2 × ~6 experts × 42), Sinkhorn pre/post/comb
  re-upload 3 × 2/layer.
- **Stream-ordered allocations ≈ 2,000+** (alloc_zeros = mallocAsync + memset each),
  including one inside every `gemm()` call for the bf16 cvt scratch.
- **Kernel launches ≈ 2,500–3,000** incl. ~50/layer for the MoE per-expert loop
  (6 experts × {gather×2, GEMM×3, swiglu, act_quant, scatter}).
- Host compute per step: 84 × Sinkhorn(4×4, 20 iters), 42 × top-6-of-256 sort, 21 ×
  top-512 sort — trivial FLOPs; the cost is the round-trip serialization.

## Rung plan (correctness gate after EACH rung before stacking the next)

Seam: `MEMRA_DSV4_DECODE_PATH=device` (default `legacy` = the lane-6/7 gated path),
read once at load and printed — one binary carries both arms for interleaved A/B.
Prefill (`forward_impl`) is NOT touched by rungs 0–3: the lane-4/7 output-sample gate
must stay numerically identical (regression witness per rung).

0. **Profile** (new `dsv4-decode-bench` + nsys): bank the breakdown — kernel-time sum
   vs launch gaps vs D2H/H2D syncs vs host logic; per-kernel top table. Ranks rungs 2/4.
1. **Device-resident decode step** (host-round-trip elimination):
   - 1a. per-step workspace arena (zero per-step allocs; explicit zeroing only where a
     kernel accumulates). Arithmetic untouched → decode logits stream must be
     BYTE-IDENTICAL to the legacy path (gated).
   - 1b. device Sinkhorn kernel, single-thread, host loop order preserved verbatim.
     Device `expf` vs host libm `exp` is a REALIZATION FORK (ulp-level) — banked
     deviation, adjudicated by the lane-6/7 doctrine (CPU oracle = truth, class bounds).
   - 1c. device router (softplus→sqrt, +bias, top-6 selection value-desc/index-asc,
     weight renorm ×1.5 — one thread, host ordering) + INDIRECT fused expert dispatch:
     one launch per projection covering all 6 slots (grid.y = slot, expert id read from
     the device selection), combine in ascending-expert-id order = the legacy
     scatter_add order. This IS attack #3 (fused dispatch) at the s=1 shape.
     tid2eid/gate_bias/experts_s2/hc base+scale become device-resident at load.
   - 1d. device fine top-k: sort on (score,index)-packed orderable integer keys —
     integer comparator ⇒ BIT-EXACT same selection+order as the host sort — and
     device index-list assembly (window ring order is a pure function of pos).
   - 1e. PP layer-22 boundary: `cudaMemcpyPeerAsync` (PIX P2P) + cross-device event
     wait replaces the host bounce.
   - 1f. logits row stays dtoh (the gates consume it). Bench gains a device-argmax
     greedy mode (token fed back device-side; argmax = max with lowest-index tie —
     associative, any reduction tree identical); cross-checked against the
     full-logits mode token stream.
2. **CUDA graph capture** of the steady-state step — ONLY if the post-rung-1 profile
   shows launch-gap domination. Lane-4 history: graphs LOST ~12% on gemma eager — do
   not assume; measure and bank either verdict. If pursued: capacity-padded shapes
   (idx pads = -1, sink_attn skips them — proven inert), device step counter feeding
   rope/ring-slot/emission predicates (SGLang in-graph metadata prep), per-stage graphs
   + peer copy between.
3. Folded into 1c at s=1 (one launch per GEMM set per layer instead of per-expert).
4. **Fast fp4 GEMM** only where the post-rung-1/2 profile ranks it: vectorized
   warp-per-output GEMV or block-scale mma.sync on the as-stored slabs. NEW NUMERIC
   CLASS (accumulation order changes): fresh derived thresholds, output-sample 15/15,
   greedy teacher-forcing vs the quantized CPU oracle, decode spot-gates — the lane-7
   suite rerun; the lane-7 fixed-tree arm stays resident as the switchable truth
   realization (MEMRA_DSV4_EXPERT_ARM seam).

## Gate protocol per rung

- (i) determinism: two runs byte-identical (tokens + logits stream sha256), graphs
  included if adopted;
- (ii) `dsv4-gpu-decode-gate` n_new=260 under the rung's seams (a1 pair bounds vs
  re-prefill, a2 factor-1 vs the quantized oracle via `dsv4-greedy-verify` +
  `dsv4-decode-oracle-check`, lane-7 short-probe boundary criterion);
- (iii) byte-identity of the decode logits stream vs the PREVIOUS rung where the
  arithmetic is untouched (1a, 1d, 1e, 2, and the 1c combine order), stated per rung;
  realization forks (1b sinkhorn expf, 1c router softplus) are gated by (ii) instead —
  which comparisons apply is declared per rung below;
- (iv) prefill regression witness: `dsv4-gpu-gate` (both arms) numerically identical to
  the lane-7 banked tables while rungs don't touch prefill;
- (v) VRAM delta banked per rung; zero OOM / zero CUDA errors.

## Perf banking (the program's first REAL numbers)

Interleaved A/B ×5 alternating same-binary runs (`MEMRA_DSV4_DECODE_PATH` legacy vs
device), same box + clock window, single-stream greedy from the fixture prompt:
ms/step MEDIANS + spreads at s≈200 and s≈1024 (s≈8k if time allows), tok/s, per-rung
attribution table from the profiles. Banked as MEASURED (bench, not serving); serving
cells stay a serving-lane claim. Conditions line: box, GPU clocks as read, artifact,
seam values, binary tip.

## Rung 0 — measured baseline + step profile (nsys, steps 16–48 of a 64-step run, s≈50–80;
## baseline windows from a separate 1024-step run)

Baseline (single runs, informational): `dsv4-decode-bench`, native arm, legacy path,
tip 77ef38924a — **s≈200: 90.4 ms/step (11.1 tok/s) · s≈512: 95.5 · s≈1024: 99.5**
(min/max spreads ≤ ±2%; banked bench-baseline-native-1024.json, logits sha
691ded3a9f27…). VRAM 82.21/82.14 → 82.27/82.17 GiB used.

nsys per-step breakdown (÷32 steps; `step-profile.nsys-rep` box-side):

| bucket | per step | detail |
|---|---|---|
| GPU kernel busy (sum) | **≈ 76.5 ms** | indexer_score **22.4 ms** (21 × 1.066 ms — a 60-THREAD kernel per fine layer); fp4_gemm **15.9 ms** (774 × 20.6 µs ≈ 8× off weight-bandwidth: per-byte loads + divergent __constant__ LUT + per-element e4m3 exp2f); dots_f32 **11.5 ms** (255 inst; head-logits instance ≈ 3.0 ms); sink_attn **9.2 ms** (43 × 214 µs — 64 blocks re-gather the SAME 188 kv rows per head, uncoalesced); cuBLASLt nvjet+splitK **9.8 ms**; memsets 2.5 ms; small dsv4 kernels ≈ 5.2 ms |
| host API wall | ≈ 84.8 ms (overlaps GPU) | **153 blocking D2H drains (54.2 ms — 99.9% stream-wait, actual copy 0.08 ms)**; 5,872 kernel launches (17.4 ms CPU); 3,086 × allocAsync+memset+free (8.5 ms CPU); 874 pageable H2D (1.5 ms) |
| wall − GPU busy | ≈ 11–14 ms | launch gaps + sync/serialization slack |

**CONTRADICTION vs the lane brief (banked):** the brief's premise "decode steps are
host-round-trip dominated" is WRONG at this tip — the step is ~85% GPU-kernel-busy;
the 153 round-trips cost only the ~12 ms gap layer (their 54 ms API wall is
stream-WAIT on the underparallelized kernels, double-counted). The first-order money
is bit-exact kernel reparallelization at s=1 shapes. f64-accumulation oracle parity
is a structural tax on this card class (consumer-Blackwell f64 = 1/64 rate) — dots/
rowsq/rmsnorm floors are law-bound, not sloppiness.

**Re-ranked rung order (profile-driven, correctness gate after each):**
- **Rung A — bit-exact kernel reparallelization** (same arithmetic, same per-value
  expression trees, byte-identity gated): (i) indexer_score → block per (t,j), thread
  per head (per-head 128-dot stays one sequential f64 chain; head sum stays thread-0
  sequential in h order); (ii) sink_attn → smem-tiled kv rows (tiles loaded coalesced
  in slot order; score/denominator/output orders unchanged); (iii) fp4_gemm →
  vectorized weight/activation loads + smem LUT decode (identical products, identical
  sub-group/tree order).
- **Rung B — device-resident step** (the banked rung-1 list: arena, device sinkhorn/
  router/topk/index/argmax, fused indirect dispatch, peer boundary) — removes the
  ~12 ms gap layer + memsets + launch count, prerequisite for graphs.
- Re-profile → graphs verdict (rung 2), then cuBLASLt-GEMV / dots micro-work only
  as the profile ranks them (class-II realization forks, decode-gate adjudicated).

## Rung A — LANDED, ALL GATES PASS (commit 0a0a53a2c5)

indexer_score: block per (t,j) × thread per head (per-head 128-dot = the same single
sequential f64 chain; head sum = thread-0 sequential in h order); fp4_gemm +
fp4_gemm_sel: uint2/uint4 vector loads + per-block smem decode tables whose entries
equal dsv4_e4m3()/DSV4_E2M1 exactly (same products, same in-group order, same group
ownership mod 128, same halving tree). Bit-exact BY CONSTRUCTION and verified:

```
MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-decode-bench <model> <fixtures> \
  ~/dsv4-lane8-out/bench-rungA-native-1024.json 1024 0,1
  -> logits stream sha256 691ded3a9f27… == the baseline (77ef38924a legacy) sha EXACTLY
     over 1024 decode steps × 129,280 logits — BYTE-IDENTICAL; the lane-7 oracle
     adjudications transfer verbatim.
MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-gpu-gate <model> <fixtures> 0,1
  -> PASS 15/15, exit 0; measured-value table diff vs lane-7 banked
     gpu-gate-native-run1.log: IDENTICAL (prefill witness — the kernels are shared).
MEMRA_DSV4_EXPERT_ARM=native target/release/dsv4-gpu-decode-gate <model> <fixtures> \
  <lane7 greedy json> ~/dsv4-lane8-out/decode-gate-rungA 260 0,1
  -> PASS a1/c/d/e exit 0 (a1 52/52 class rows, determinism 2× byte-identical;
     b-raw 47/160 vs the re-prefill greedy = the lane-6 realization-flip phenomenon,
     adjudication owned by the byte-identity witness above).
```

Measured (single runs, informational): s≈200 **90.4 → 59.6 ms/step** (16.8 tok/s),
s≈512 95.5 → 62.3, s≈1024 **99.5 → 66.4** (15.1 tok/s). VRAM unchanged
(82.27/82.17 GiB post-run). −30.8 ms/step at s≈200 from two bit-exact kernel bodies.

## Rung B (mechanical half) — device-hostmath arm BYTE-IDENTICAL (commit eeeb6b9ae2)

The device-resident step behind `MEMRA_DSV4_DECODE_PATH` with Sinkhorn/router/
fine-top-k/head-gate math still on the HOST (`device-hostmath`): workspace arena
(zero per-step allocations), device window/coarse index build, device fine-top-k
(integer-key sort — bit-exact by the orderable-key argument), indirect fused expert
dispatch (one launch per projection for the 6-slot set, combine in ascending-eid
order), grouped-wo as offset GEMMs on one o-cvt (pure pointer math at s=1), peer-copy
PP boundary (cuCtxEnablePeerAccess + mempool grants + TX-stream cuMemcpyPeerAsync +
event, the pp.rs idiom), device argmax greedy mode.

```
MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DECODE_PATH=device-hostmath \
  target/release/dsv4-decode-bench <model> <fixtures> \
  ~/dsv4-lane8-out/bench-rungB-hostmath-1024.json 1024 0,1
  -> logits stream sha256 691ded3a9f27… == baseline EXACTLY (1024 steps) — the whole
     mechanical restructure is BYTE-IDENTICAL; oracle adjudications transfer.
  -> s≈200 **48.9 ms/step** (20.5 tok/s) · s≈512 51.0 · s≈1024 **55.2** (18.1 tok/s)
     — with ~148 host-math round-trips still in place. VRAM +0.04 GiB (arena).
```

## Rung B' — sink-attention three-kernel split (042d542556): BYTE-IDENTICAL

scores (block/slot, kv row smem-staged, one thread/head, same f64 dot chain) →
per-head softmax + f64 slot-order denominator (+0.0 pad adds proven bit-inert) →
output (block per (dim-chunk × head-chunk), kv tiles staged in slot order, per-(h,x)
sequential f64 chain; evals==0 skip == the legacy ix<0 skip by the ±0.0 argument).
Re-gate: hostmath 1024-step sha 691ded3a… UNCHANGED (byte-identical);
**s≈200 44.2 ms/step (22.6 tok/s) · s≈1024 47.7 (21.0 tok/s)** hostmath.

## Rung B (fork half) — device-math arm v1: gates green, perf REGRESSION found

`MEMRA_DSV4_DECODE_PATH=device` (Sinkhorn/router/top-k/head-gate as kernels — the
banked expf/log1pf realization fork; fine top-k is integer-exact):
- bench 1024: sha 7508ad50bcab… (fork, as expected), tokens greedy-stable;
  **s≈200 55.5 ms/step — SLOWER than hostmath's 44.2**: the v1 single-thread route
  kernel serializes ~1,536 sequential transcendentals per layer and the v1
  single-thread Sinkhorn ~20 dependent iterations — ~10 ms/step of dead GPU time.
  Banked as a measured contradiction of "device-side is automatically faster";
  v2 kernels (parallel transcendental precompute + row/col thread ownership,
  BIT-IDENTICAL values to v1) replace them.
- `dsv4-gpu-decode-gate` 260 @ device: **PASS a1/c/d/e** exit 0 — a1 52/52 vs the
  legacy re-prefill at native pair bounds (worst ~3.1), argmax same at ALL
  checkpoints, top-5/top-20 in-band, determinism 2× byte-identical, cache math
  exact, zero CUDA errors (decode-gate-device.log).
- a2/b: **PASS** — `dsv4-greedy-verify` (native env, quantized oracle) over the
  260-step device-arm trajectory: **255/260 agree, all 5 disagreements in-band
  near-ties** (cpu margins 0.064–0.454 vs bands ~196–216; first divergence step 14);
  `dsv4-decode-oracle-check` **PASS (a2 true | b true)**, a2 all checkpoints vs the
  oracle at factor-1 native bounds. Same physics as lane-7's 252/260. The
  Sinkhorn/router expf fork is semantically inert at the class bounds.
  (decode-verify-device.log, decode-oracle-check-device.log)

## Device-arm v2 kernels + second profile (commit e05782df03 predecessor state)

Route/Sinkhorn v2 (parallel transcendental precompute + row/col thread ownership):
1024-step sha **7508ad50… == v1 EXACTLY** — bit-identity proven, v1 gates transfer.
**s≈200 43.3 ms/step (23.1 tok/s) · s≈512 44.5 · s≈1024 46.8 (21.4 tok/s).**
Cumulative: 90.4 → 43.3 = **2.09×** over the lane-7 baseline (single-run numbers;
the A/B tables are the claims).

## Interleaved A/B ×5 — seam table at the rung-B' stack (MEASURED, bench not serving)

`~/ab_interleaved.sh seam 1024 5` — binary at 2855d68c06 (v2 kernels), ONE binary, arms
alternate A(legacy), B(device) ×5, n_new=1024 each, same box + clock window
(SM 2295–2325 MHz logged per run), fixture 32-token prompt, greedy. Median-of-medians
(per-run medians over 21-step windows; per-run min/max spreads ≤ ±2.3 ms):

| window | A legacy ms/step (5 medians) | B device ms/step (5 medians) | speedup | B tok/s |
|---|---|---|---|---|
| s≈200 | 59.56 (59.49–59.65) | **43.31** (43.30–43.35) | 1.375× | **23.1** |
| s≈512 | 62.31 (62.28–62.45) | **44.49** (44.44–44.52) | 1.400× | 22.5 |
| s≈1024 | 66.31 (66.29–66.36) | **46.77** (46.74–46.77) | 1.418× | **21.4** |

(A = the lane-6/7 decode path with the rung-A bit-exact kernels shared; the lane-7
BINARY baseline table is banked separately below. Every A run reproduced sha
691ded3a…, every B run sha 7508ad50… — the gated realizations, byte-stable across
all 10 runs.)

## Rung C — deterministic bf16 GEMV for the m=1 device path (e05782df03): gated, small win

Replaces every cuBLASLt m=1 GEMM on the DEVICE decode path (wq_a/wq_b/wkv/indexer
wq_b/weights_proj/wo_a×8/wo_b/shared w1·w2·w3) with a fixed-tree f32-accumulation
GEMV (vectorized uint4 bf16 loads; class-II f32-reorder fork; prefill + legacy path
keep cuBLASLt — output-sample witness untouched).
- bench 1024 device: sha 384ecde1bcff… (fork v3), **s≈200 42.6 · s≈512 43.8 ·
  s≈1024 46.0 ms/step** — only **−0.7 ms** vs the cuBLASLt arm. Banked
  CONTRADICTION: the ~2.3×-off-bandwidth cuBLASLt m=1 profile is the GEMV physics
  of this card class at these shapes, not a cuBLASLt deficiency; the custom kernel
  matches it (+ removes the plan-cache/mutex layer — graph-friendlier).
- gates: `dsv4-gpu-decode-gate` 260 @ device **PASS a1/c/d/e** (decode-gate-deviceC.log);
  `dsv4-greedy-verify` **252/260 agree**; `dsv4-decode-oracle-check` **PASS
  (a2 true | b true)**, 8 disagreements, 0 out-of-band (decode-verify-deviceC.log,
  decode-oracle-check-deviceC.log).

## THE HEADLINE TABLE — interleaved A/B ×5, lane-7 baseline binary vs the lane-8 stack

`~/ab_interleaved.sh lane7 1024 5` — A = the 77ef38924a binary (lane-7 kernels + the
bench harness only; ~/memra-lane7 worktree build), B = the e05782df03 binary,
`MEMRA_DSV4_EXPERT_ARM=native`, B adds `MEMRA_DSV4_DECODE_PATH=device`. Runs
alternate A,B ×5 in one window (SM 2302–2317 MHz logged per run), n_new=1024,
single-stream greedy from the fixture 32-token prompt, full-logits step contract.
**MEASURED (bench, not serving).** Per-run medians over 21-step windows:

| window | lane-7 baseline ms/step (×5) | lane-8 device ms/step (×5) | speedup | tok/s |
|---|---|---|---|---|
| s≈200 | 90.44 (90.32–90.48) | **42.60 (42.598–42.635)** | **2.12×** | 11.1 → **23.5** |
| s≈512 | 95.65 (95.52–95.71) | **43.74 (43.713–43.750)** | **2.19×** | 10.5 → 22.9 |
| s≈1024 | 99.55 (99.51–99.62) | **46.02 (46.003–46.036)** | **2.16×** | 10.0 → **21.7** |

Byte-stability across ALL runs: every A run sha 691ded3a… (== the tip legacy path —
a cross-binary witness that the rung-A kernel rewrites are bit-exact), every B run
sha 384ecde1… (the rung-C gated realization). Conditions: hyperscaler box1 (instance id redacted; full conditions banked in darklanes),
2× RTX PRO 6000 Blackwell Server 96GB PIX P2P, CUDA 13.2, MEMRA_CUDA_ARCH=120a,
nvidia 595.91.07, artifact dsv4-flash-nvfp4 (clamp-only contract), PP split at
layer 22. Bar context (RECON.md): canada-quant TP2 W4A16 47.5 tok/s bs=1 —
**NOT yet beaten** (23.5 vs 47.5); the remaining path is banked under "remains".

## Seam attribution A/B ×5 at the final tip (e05782df03, one binary)

`~/ab_interleaved.sh seam 1024 5` (rerun at the rung-C tip; the rung-B' seam table
above is the intermediate record): A = legacy path, B = device path, alternating ×5:

| window | A tip-legacy ms/step (×5) | B tip-device ms/step (×5) | path speedup |
|---|---|---|---|
| s≈200 | 59.68 (59.65–59.75) | 42.62 (42.601–42.634) | 1.40× |
| s≈512 | 62.44 (62.33–62.59) | 43.74 (43.737–43.764) | 1.43× |
| s≈1024 | 66.43 (66.38–66.52) | 46.05 (46.020–46.069) | 1.44× |

Shas byte-stable (A 691ded3a… ×5, B 384ecde1… ×5). Attribution vs the headline:
of the total 2.12× at s≈200, the shared rung-A bit-exact kernel rewrites give
90.44→59.68 (1.52×) and the device path gives 59.68→42.62 (1.40×).

nsys re-profile (device arm, steps 16–48): GPU busy ≈ 41.5 ms of the 43.3 ms wall —
**gap layer ≈ 1.8 ms ⇒ the banked rung-2 condition (launch-gap domination) is NOT
met; CUDA-graph capture not pursued at this kernel mass** (a graph buys ≤ the gap
layer). Forward-looking: host launch API ≈ 19 ms/step (2,691 launches) overlapped
under the 41.5 ms GPU time — graphs become mandatory once kernels drop below
~25 ms/step. Clock check under load: 2295–2325 MHz SM (max 2430), 140–175 W of
600 W, ~50% util per device (PP alternation) — the kernel floors are real, not
throttling. Remaining ranking (ms/step): dots_f32 11.5 (f64-island law; head row
≈ 3.0), cuBLASLt m=1 GEMMs ≈ 10.0 (~2.3× off weight bandwidth → rung C GEMV),
sink trio 6.5 (scores 2.9 + out 3.2 — further headroom exists), fp4_gemm_sel 4.8
(774→129 launches, 3.3× from vectorization), rowsq 2.65, rmsnorm 2.4, hc_sinkhorn
1.57, route 0.96, indexer_score 0.51 (was 22.4).

## Serving-shape greedy mode (device argmax) — verified + measured

`MEMRA_DSV4_BENCH_GREEDY=1` (decode_step_greedy: device argmax, one 4-byte D2H per
step instead of the 517 KB logits row): **token stream IDENTICAL to the full-logits
mode over all 1024 steps** (json tokens equality check); s≈200 42.4 · s≈512 43.6 ·
s≈1024 45.9 ms/step (informational single) — the logits-row readback costs ~0.2 ms.

## Long-context shape to s≈8k (single runs each arm, informational — NOT A/B claims)

n_new=8200 from the fixture prompt, one run per arm, same window; the device run
crosses the fine-top-k saturation boundary (nb > 512 from s=2052) on the DEVICE
bitonic top-k — zero CUDA errors, exit 0 both:

| s window | legacy ms/step | device ms/step | device tok/s |
|---|---|---|---|
| ≈200 | 59.7 | 42.6 | 23.5 |
| ≈1024 | 66.4 | 46.1 | 21.7 |
| ≈2048 | 73.4 | 49.9 | 20.0 |
| ≈4096 | 74.8 | 50.9 | 19.6 |
| ≈8192 | 77.6 | **53.2** | **18.8** |

Near-flat: +25% step time across 40× context growth (the O(n) decode shape holding
through indexer saturation). The 1M-ctx claim still needs the indexer-score
long-context rewrite (separate lane) + chunked prefill.

## Lane 8 summary — support claim + what remains

**Claim (house law):** the lane-8 decode stack — bit-exact kernel reparallelizations
(indexer_score, fp4 GEMM decode, sink-attention split), the device-resident decode
step (arena, device index/top-k/Sinkhorn/router/head-gate, fused indirect expert
dispatch, peer-copy PP boundary), and the fixed-tree bf16 GEMV — is
CORRECTNESS-GATED on the 2-card box behind `MEMRA_DSV4_DECODE_PATH=device`
(legacy path byte-stable as the fallback and A/B reference), and MEASURED at
**42.6 ms/step ≈ 23.5 tok/s single-stream at s≈200 (2.12× the lane-7 baseline;
21.7 tok/s at s≈1024)** under the interleaved A/B ×5 law. Bench numbers, NOT
serving numbers; NOT a public-bar claim (canada-quant 47.5 tok/s bs=1 stands).

Gate chain: every mechanical rung byte-identical (1024-step logits-stream sha) to
the lane-7-gated stream; the two realization forks (device transcendentals; GEMV
f32 reorder) each decode-gate green (a1 52/52 pair bounds, argmax same everywhere,
determinism 2×) + quantized-oracle adjudicated (255/260 and 252/260, every
disagreement an in-band near-tie). Prefill untouched (output-sample table
value-identical to lane-7). VRAM delta: +0.04 GiB arena. Zero OOM/CUDA errors.

**Remains (ranked by the post-C profile):**
1. dots_f32 11.5 ms — f64-island law cost; head row ≈ 3 ms ≈ 2× its f64-throughput
   floor; wkv+wgate one-launch batching is bit-exact and easy; the rest needs an
   owner call on relaxing the f64 law for the SERVING arm (the oracle keeps it).
2. sink trio 6.5 ms — scores/out still ~10× their op floors; next-shape rewrite
   (per-(slot,head) thread scores; wider out blocks) stays bit-exact.
3. fp4_gemm_sel 4.8 ms vs ~2.6 ms weight-bandwidth floor.
4. CUDA graphs — re-evaluate when kernels < ~25 ms/step (launch API ≈ 19 ms/step
   becomes binding); requires the in-graph metadata prep design banked in the plan.
5. Chunked prefill, indexer-score long-context rewrite (GVR lane), PP-2 serving
   door (PpNRt), MTP/DSpark drafter cells (RECON: +1.5× class), TP-2 evaluation
   when memra TP lands — all unchanged from lanes 6/7, all out of this lane's
   mandate.
Bar math: 42.6 ms → with items 1–4 fully landed ≈ 28–32 ms (31–36 tok/s); the
47.5-tok/s bar on this artifact class needs the drafter (× ~1.5) or TP2 on top.

# Lane 9 — decode perf iteration 2: ranked micro-kernel remains + CUDA graphs (2026-08-18)

Baseline = the lane-8 tip (ac477a9677) device stack: 42.60 / 43.74 / 46.02 ms/step at
s≈200/512/1024 (interleaved A/B ×5, sha 384ecde1…), kernel mass ≈ 41.5 ms, gap layer
≈ 1.8 ms, host launch API ≈ 19 ms overlapped (2,691 launches). Public bar unchanged:
canada-quant 47.5 tok/s bs=1 plain. **dots_f32 (11.5 ms) is BLOCKED on an owner call
(f64-island relaxation for a serving arm) — untouched in this lane.** Indexer GVR
long-context rewrite, chunked prefill, drafter: other lanes.

## Rung plan (banked BEFORE the fresh profile; correctness gate after EACH rung)

0. **Fresh profile** at the lane-9 baseline (nsys, device arm, steps 16–48): re-rank
   the remains; bank per-kernel table. The lane-8 ranking (sink trio 6.5; fp4_gemm_sel
   4.8 vs 2.6 floor; rowsq 2.65; rmsnorm 2.4; hc_sinkhorn 1.57; route 0.96) is the
   prior, not the plan — the fresh table decides.
1. **Sink-trio next-shape rewrite (BIT-EXACT)**: K1 scores → (slot-tile × head-tile)
   blocks with smem-staged q/kv chunks — each (slot, head) keeps ONE sequential f64
   chain over hd in x order (the lane-8 expression verbatim; only the thread mapping
   and staging move). K2 soft → skip ev == 0.0f entries in the thread-0 f64
   denominator (+0.0 adds are bit-inert — the banked lane-8 K3 argument, same class).
   K3 out → wider grid (smaller head-chunks), same per-(h,x) sequential slot chain.
   Gate: 1024-step logits-stream sha == 384ecde1… (byte identity), determinism ×2,
   decode-gate, prefill witness (dsv4_sink_attn_kernel untouched — prefill kernels
   shared with lane-4 stay byte-identical).
2. **fp4_gemm_sel toward its 2.6 ms weight-bandwidth floor (BIT-EXACT)**: multi-col
   blocks (amortize smem-table init + act-row loads; per-col group ownership mod
   blockDim and the per-col halving tree UNCHANGED). Same gate class as rung 1
   (byte identity — device path only; dsv4_fp4_gemm_kernel/prefill untouched).
3. **Launch-count fusions (BIT-EXACT pointer math)** as the fresh profile ranks:
   grouped-wo 8×GEMV → one grid.y=groups launch; other pure-launch merges. Byte
   identity gated per change.
4. **CUDA graphs — measure, don't assume** (gemma lost 12%; lane-8 condition not met
   at 41.5 ms mass): after rungs 1–3 re-profile; the banked flip condition is kernel
   mass < ~25 ms (launch API becomes binding). With dots_f32 blocked the projected
   mass is ~30–34 ms — if the measured gap layer stays ≤ ~2 ms the graph buys ≤ that
   and the verdict is banked from the profile without adoption; if gaps grow or the
   mass lands near the boundary, implement the in-graph metadata-prep design
   (device-pos plumbing: pos/nb/slots read from device buffers, ring/pend writes as
   indexed kernels, boundary emission predicated in-graph, capacity-padded idx with
   -1 pads proven inert) and bank the honest amortized A/B including any re-capture
   cost at block-completion boundaries.

Gates per rung: (i) determinism ×2 byte-identical (tokens + logits stream sha256);
(ii) byte identity vs the PREVIOUS rung where arithmetic is untouched (all rungs here
are declared bit-exact — any fork found in flight is a STOP + re-derivation, not a
quiet gate swap); (iii) dsv4-gpu-decode-gate 260 under the rung's seams; (iv) prefill
regression witness when a shared kernel is touched; (v) VRAM delta banked; zero
OOM/CUDA errors. Final: interleaved A/B ×5 vs the lane-8 stack (A = ac477a9677 binary,
device path — preserved box-side before any rebuild), n_new=8200 for s≈200/1024/8k
windows, plus one informational lane-7-baseline column.

## OWNER RULING mid-lane (2026-08-19, via orchestrator): dots_f32 UNBLOCKED

An **f32-accumulation serving arm for the f32-island dots** is approved — the
reference's own numeric class; **f64 stays resident as the switchable oracle-truth
arm** (new seam `MEMRA_DSV4_DOTS_ARM`, default `f64` = today's bytes; `f32` = the
serving arm, DEVICE decode path only — legacy path and prefill keep f64 unconditionally
so the A/B reference and the output-sample witness stay pinned). Owner's condition
verbatim: "if the quality stays for me its ok" — enforced as the standard fork
discipline: derived bands banked BEFORE the rerun; teacher-forcing vs the CPU
quantized oracle over the decode trajectory; EVERY disagreement must classify as an
in-band near-tie (the expf/GEMV bar: 255/260-class). Anything worse: the arm does not
ship and the failure is reported.

## Rung 0 — fresh profile at the lane-9 baseline (nsys, device arm, steps 16–48 of a
## 64-step run, s≈50–110; box-side step-profile-l9base.{nsys-rep,sqlite}, kernwin.py)

Wall 43.38 ms/step; **GPU kernel busy 41.55 ms** (dev0 19.56 + dev1 21.99, PP
sequential) → **gap layer ≈ 1.8 ms**; 2,755 launches/step costing 23.2 ms host CPU
(cudaLaunchKernel, overlapped) + 219 DtoD (1.2 ms CPU); memsets gone (arena).

| kernel | inst/step | ms/step | µs/inst |
|---|---|---|---|
| dots_f32 | 255 | 11.56 | 45.3 |
| gemv_bf16 | 687 | 8.96 | 13.0 |
| fp4_gemm_sel | 129 | 4.79 | 37.2 |
| sink_out / sink_scores / sink_soft | 43 / 43 / 43 | 3.16 / 2.92 / 0.41 | 73.5 / 67.9 / 9.6 |
| rowsq_scale | 87 | 2.65 | 30.5 |
| rmsnorm | 183.5 | 2.41 | 13.1 |
| hc_sinkhorn | 86 | 1.56 | 18.2 |
| route | 43 | 0.96 | 22.4 |
| indexer_score | 21 | 0.51 | 24.2 |
| cvt_bf16 | 322 | 0.32 | 1.0 |
| rest (rope_at, act_quant*, hc_*, swiglu, pool, combine, topk, hadamard, …) | ~800 | ~1.5 | — |

Reading: dots_f32 is ~5.8× its ~2.0 ms weight-bandwidth floor (its traffic ≈ 3.1 GB/step:
head bf16 row 1.06 GB + fine/coarse compressor f32 matrices ≈ 1.7 GB + hc fn_w + gate)
because every dot rides the 1/64-rate f64 pipe; gemv_bf16 is GEMV physics (lane-8
banked); sink scores runs 2 warps/SM with uncoalesced q; sink out puts 72 blocks on
188 SMs; rowsq/rmsnorm are single-block latency-exposed chains; route is a
single-thread 1,536-compare scan.

**Re-ranked rungs (fresh profile + the owner ruling):**
- Rung A (bit-exact): sink trio reshape — as planned.
- Rung B (bit-exact): fp4_gemm_sel 4-col blocks — as planned.
- Rung C (FORK, the unblocked item): `dsv4_dots_f32acc_kernel` — deterministic
  fixed-tree f32 accumulation in the gemv_bf16 shape (thread-strided contiguous
  8-element chunks, sequential in-chunk, 128-leaf halving tree), vectorized f32/bf16
  weight loads. Projected ~11.6 → ~2.0–2.5 ms (bandwidth floor).
- Rung D (bit-exact): rowsq/rmsnorm vectorized loads (same per-thread chain partition,
  same tree); route parallel top-k via 6 × parallel argmax rounds (max-with-lowest-
  index is associative — value-identical selection); misc launch merges as profiled.
- Rung E: graphs — with rung C landed the projected kernel mass is ~24–26 ms, AT the
  banked ~25 ms flip condition, and the 23.2 ms host submit becomes binding →
  expect graphs mandatory; implement the banked in-graph metadata-prep design and
  measure honestly (amortized re-capture cost included) either way.

## Lane-9 rung record (gate after each; every negative result banked)

**Rung A — sink trio (bit-exact), three takes.** Take 1 (K1 smem-phase staging)
REGRESSED 2.6× (67.9 → 177.7 µs/inst; 2 warps/block cannot overlap staging/compute
phases) — reverted. Take 2 (float4 q vectors, same chain) measured IDENTICAL
(67.87 µs/inst — loads are not the binding resource). Take 3 (4 slots/block, 4
interleaved f64 chains, hoisted products) REGRESSED 3.1× (211.9 µs/inst — the wide
f64 register set defeats nvcc's scheduling) — reverted. ncu (sudo; SpeedOfLight):
scores = 0.2% DRAM / 8.3% SM — pure f64 dependency latency ~346 cycles/element.
**K3 out WON: head-chunks 64 → 8 = 73.5 → 31.9 µs/inst (3.16 → 1.37 ms/step),**
byte-identical. K2 zero-skip regressed 9.6 → 12.4 µs — reverted. Scores stands at
the lane-8 shape (2.93 ms) — movement needs the f32-island ruling extended to
attention dots (owner call, flagged). Gates: every take byte-identical (1024-step
sha 384ecde1… / 1acee23f… as applicable) ×2 determinism.

**Rung B — fp4_gemm_sel: two NEGATIVE results banked.** 4-col blocks: 37.2 → 37.8
µs/inst (table-init/act-reuse amortization is NOT the cost). ALU decode (bit-equal
e4m3/e2m1 constructions, exhaustive 256+16 device probe PASS —
lane9-out/alu-decoder-probe.log): 37.8 → 45.9 µs/inst (the "random" smem LUT reads
broadcast well — activation codes cluster; ALU decode only added integer-issue
pressure) — reverted to tables. **CONTRADICTION banked: fp4_gemm_sel is NOT
weight-bandwidth-bound at these shapes** (4.87 ms vs the 2.6 ms floor derived from
25 MB/launch); the gap is issue/latency inside the in-group sequential f32 chains.
A tensor-core/dp4a-class rewrite is an iteration-3 item (new numeric class → fork
gates). Kernel stands at 4-col tables, 37.75 µs/inst, byte-identical.

**Rung C — f32-dots serving arm (the owner-unblocked fork): LANDED, ALL GATES PASS.**
`dsv4_dots_f32acc_kernel` (gemv-class fixed tree: thread-strided contiguous 8-element
chunks, sequential in-chunk, 128-leaf halving tree; float4/uint4 loads) behind
`MEMRA_DSV4_DOTS_ARM=f32`, routed ONLY on the device decode path (`dots_dev`);
legacy path + prefill pinned to f64; f32-without-device refused at load. Gates:
- default-arm regression: 1024-step sha **384ecde1… UNCHANGED** (the routing refactor
  is byte-inert when off);
- f32 arm: sha 1acee23f… ×2 (determinism), tokens greedy-stable;
- `dsv4-gpu-decode-gate` 260 @ device+f32: **PASS a1/c/d/e** exit 0, worst step-equiv
  max-abs 4.19 at s=128, cache math exact, zero CUDA errors
  (lane9-out/decode-gate-f32dots.log);
- `dsv4-greedy-verify` (CPU quantized oracle, teacher-forcing): **255/260 agree, ALL
  5 disagreements in-band near-ties** (cpu margins 0.0296–0.4777 vs bands ~200–210;
  steps 14/15/33/54/90) — EXACTLY the expf-fork class (lane-8 device fork was also
  255/260). `dsv4-decode-oracle-check`: **PASS (a2 true | b true)**, worst a2 max-abs
  6.29 vs thr ~2.2e2 (decode-verify-f32dots.log, decode-oracle-check-f32dots.log).
  **The owner's condition is met; the arm ships.**
- Measured: dots 11.56 → **2.75 ms/step** (45.3 → 10.8 µs/inst); step 40.8 → 32.0 ms
  at s≈200 in the rung-C single runs.

**Rung D — micro cluster (bit-exact): LANDED.** route v3 parallel argmax tree (host
tie rule; associative ⇒ value-identical selection): 22.4 → 11.5 µs/inst (0.96 →
0.50 ms). rowsq_scale 8-wide load batching (per-thread order unchanged): 30.5 → 15.5
µs/inst (2.65 → 1.35 ms). rmsnorm same treatment: 13.1 → 11.0 µs/inst (2.41 → 2.02
ms). gemv 2-row blocks tried and REVERTED (13.04 → 13.44 µs/inst — GEMV physics
confirmed a third time, after lane-8's cuBLASLt and GEMV arms). All byte-identity
gated (f32 sha 1acee23f ×2, default sha 384ecde1).

**Prefill witness at tip:** `dsv4-gpu-gate` native PASS 15/15, exit 0, measured-value
table IDENTICAL to the lane-8 banked log (rmsnorm/rowsq are shared with prefill and
bit-exact; everything else lane-9 touched is decode-only)
(lane9-out/gpu-gate-tip-native.log).

## Fresh profile at the lane-9 tip (nsys, device+f32, steps 16–48, s≈50–110;
## step-profile-tip.{nsys-rep,sqlite} box-side)

Wall 30.76 ms/step; **GPU kernel busy 28.93 ms** (dev0 14.35 + dev1 14.57); **gap
layer ≈ 1.83 ms**; 2,755 launches costing 16.6 ms host CPU (overlapped); 219 DtoD.

| kernel | ms/step | µs/inst | note |
|---|---|---|---|
| gemv_bf16 (687) | 8.96 | 13.0 | GEMV physics (3rd confirmation) |
| fp4_gemm_sel (129) | 4.87 | 37.8 | NOT BW-bound (contradiction banked) |
| sink_scores (43) | 2.93 | 68.2 | f64-latency-bound; needs owner f32 call |
| dots_f32acc (255) | 2.75 | 10.8 | was 11.56 f64 |
| rmsnorm (183.5) | 2.03 | 11.0 | was 2.41 |
| hc_sinkhorn (86) | 1.58 | 18.3 | untouched; fuse w/ rowsq+collapse = iter-3 item |
| sink_out (43) | 1.37 | 32.0 | was 3.16 |
| rowsq_scale (87) | 1.35 | 15.5 | was 2.65 |
| indexer_score (21) | 0.51 | 24.4 | lane-8 |
| route (43) | 0.50 | 11.6 | was 0.96 |
| rest | ~2.1 | — | soft/cvt/rope/act_quant/hc/… |

## CUDA-graphs verdict (rung E of the plan): NOT ADOPTED this iteration — measured

The banked flip condition is kernel mass < ~25 ms (host launch API becomes binding).
Measured at the lane-9 tip: **mass 28.93 ms, host submit 16.6 ms (overlapped, not
binding), gap layer 1.83 ms** — the condition is NOT met; a captured graph buys at
most the 1.83 ms gap (~6%) while costing the in-graph metadata-prep rewrite
(device-pos plumbing through build_idx/rope_at/cmp emission predicates/ring writes +
capacity-padded topk with smem-bucket re-captures). Same doctrine as lane-8's
verdict, re-evaluated at the new mass with fresh numbers. The flip WILL come: the
DSpark drafter (iteration 3) multiplies effective per-step kernel mass down by the
acceptance rate, and an owner extension of the f32 ruling to the attention/norm
islands (sink scores/out ≈ 4.3 ms + rowsq/rmsnorm/indexer residual f64 ≈ 3 ms)
takes mass to ~21–22 ms — at that point graphs are mandatory and the banked
in-graph design applies unchanged.

## Rung C fork derivation (banked BEFORE any f32-dots run)

Class: f32 accumulation replaces f64 in the island dots (inputs unchanged: x f32,
w f32 or exact-bf16). Per-dot error bound vs the f64 arm: |Δ| ≤ k·u·Σ|x·w| with
u = 2⁻²⁴ and the fixed-tree depth factor √(log₂ 128 + k/128 seq) — extreme-value
corrected as in the lane-4 doctrine: Δ ≈ u·√k·√(2 ln n)·max|x||w|. At the lane shapes
(k = 4096–16384) the relative class error is ≤ ~1e-5 — two orders below the native
pair bounds already in force for decode-gate a1 (bf16-GEMM class, u = 2⁻⁸ dominated)
and three orders below the greedy-verify margin bands (~196–216). Therefore: existing
a1 factor-1 native bounds and in-band near-tie bands carry over UNCHANGED; the fork
is expected to classify exactly like the expf/GEMV forks (255/260, 252/260 class).
What can move: router expert selection near-ties, sinkhorn mixes, head logits — all
adjudicated by teacher-forcing, which is the gate. Consumers of these dots that go
through DISCONTINUOUS functions (top-k selection, argmax) are exactly why the gate is
trajectory-level teacher-forcing, not per-array deltas.

## THE LANE-9 HEADLINE — interleaved A/B ×5, lane-8 stack vs lane-9 stack, n_new=8200

`~/ab_lane9.sh 8200 5` — A = the PRESERVED lane-8 tip binary (ac477a9677 build,
sha256 c253cec3c3358fde…, copied to ~/memra-lane8-bin BEFORE any lane-9 rebuild),
`MEMRA_DSV4_DECODE_PATH=device`; B = the lane-9 tip binary (eb31f6f7bf),
`MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_DOTS_ARM=f32`. Runs alternate A,B ×5 in one
window, single-stream greedy from the fixture 32-token prompt, full-logits step
contract, SM clocks 2302–2317 MHz logged per run. **MEASURED (bench, not serving).**
Median-of-medians over the 5 per-run window medians (per-run spreads ≤ ±0.1 ms):

| window | A lane-8 ms/step (×5) | B lane-9 ms/step (×5) | speedup | B tok/s |
|---|---|---|---|---|
| s≈200 | 42.60 (42.60–42.60) | **29.80 (29.70–29.80)** | 1.430× | **33.6** |
| s≈512 | 43.70 (43.70–43.80) | **30.40 (30.40–30.40)** | 1.438× | 32.9 |
| s≈1024 | 46.10 (46.00–46.10) | **31.90 (31.80–31.90)** | 1.445× | **31.3** |
| s≈2048 | 50.20 (50.20–50.30) | 34.30 (34.20–34.30) | 1.464× | 29.2 |
| s≈4096 | 51.20 (51.10–51.30) | 35.10 (34.90–35.10) | 1.459× | 28.5 |
| s≈8192 | 53.30 (53.20–53.40) | **37.10 (37.10–37.10)** | 1.437× | **27.0** |

Byte-stability across ALL 10 runs: every A run stream sha 007170226a47… (the lane-8
gated f64-device realization at 8200 steps), every B run sha eeecacbd3ee4… (the
lane-9 gated f32-dots realization). Near-flat long context holds: B rises +24.5%
over 40× context growth (A: +25.1%). The A arm's s≈200/1024/8192 medians reproduce
lane-8's banked table (42.60/46.02/53.2) across days — cross-day witness.
Conditions: hyperscaler box1 (instance id redacted; full conditions banked in darklanes), 2× RTX PRO 6000 Blackwell Server 96GB PIX P2P,
CUDA 13.2, MEMRA_CUDA_ARCH=120a, nvidia 595.91.07, artifact dsv4-flash-nvfp4
(clamp-only), PP split at layer 22; another lane's CPU-only fixture job (taskset
cores 24–47) ran during parts of the lane-9 rung gates but NOT during this A/B
window (verified idle); the interleave law covers residual host noise either way.

Cumulative program line at s≈200: lane-7 90.44 → lane-8 42.60 → lane-9 **29.80
ms/step** (3.03× over lane-7; 1.43× this lane) — 11.1 → 23.5 → **33.6 tok/s bs=1**.

Informational lane-7-baseline column (SINGLE run, 77ef38924a binary rebuilt in a
fresh worktree — removed after the run; NOT an A/B claim; sha d3b2a6b2… — the lane-7
legacy realization): s≈200 90.4 · s≈512 95.4 · s≈1024 99.6 · s≈2048 106.6 · s≈4096
107.6 · s≈8192 109.6 ms/step. The s≈200/1024 points reproduce lane-8's banked
baseline exactly; the 8k point is new. Lane-9 vs lane-7: 3.03× / 3.12× / 2.95× at
s≈200 / 1024 / 8192.

## Lane 9 summary — claim + bar math + remains

**Claim (house law):** the lane-9 decode stack — sink-out 8-wide chunks, fp4_gemm_sel
4-column blocks, the OWNER-GATED f32-accumulation island-dots serving arm
(MEMRA_DSV4_DOTS_ARM=f32; f64 stays the resident oracle-truth arm and the default),
route parallel-argmax, rowsq/rmsnorm load batching — is CORRECTNESS-GATED on the
2-card box (byte-identity for every mechanical rung ×2 determinism; the one fork
gated at the owner's bar: teacher-forcing 255/260 with ALL disagreements in-band
near-ties + decode-gate a1/c/d/e + oracle-check PASS) and MEASURED at **29.80
ms/step ≈ 33.6 tok/s single-stream at s≈200 (31.3 at s≈1024, 27.0 at s≈8192)** under
the interleaved A/B ×5 law vs the lane-8 stack. Bench numbers, NOT serving numbers;
NOT a public-bar claim.

**Bar math (47.5 tok/s plain = 21.05 ms/step):** at 29.80 ms we are at 71% of the
bar; 8.75 ms must still come out. Ranked path, all measured-grounded:
1. Owner call: EXTEND the f32-island ruling to the attention/norm f64 chains — sink
   scores+out (4.30 ms, both latency-bound on the f64 pipe), indexer_score (0.51),
   rowsq/rmsnorm/headrms residual f64 (~3.5) → projected −5.5 to −6.5 ms (same fork
   discipline; the sink dots are the same reference-f32 numeric class). Without this
   call those kernels are at their measured bit-exact floors (three shaped takes
   banked).
2. fp4_gemm_sel deep rewrite (dp4a/tensor-core class, NEW numeric class + fork
   gates): 4.87 → ~2.6 ms floor.
3. CUDA graphs once mass < ~25 ms (post item-1 mass ≈ 21–22): buys the then-binding
   launch layer (today: gap 1.83 ms, submit 16.6 ms overlapped).
4. hc_sinkhorn fusion with rowsq+collapse (1.58 ms of launch-latency-bound work).
Items 1–4 fully landed project ≈ 20–21 ms ≈ 47–50 tok/s — the plain bar falls at the
edge; **the DSpark drafter (iteration 3, community 1.5–1.8× class) clears it with
margin either way** (29.80/1.5 ≈ 19.9 ms effective ≈ 50 tok/s already without 1–4).

**Contradictions banked this lane:** (i) fp4_gemm_sel is NOT weight-bandwidth-bound
(two amortization rewrites flat, ALU decode −21%); (ii) sink-scores resists every
bit-exact reshape tried (smem phases −2.6×, float4 flat, 4-chain ILP −3.1×) — ncu:
pure f64 dependency latency; (iii) GEMV physics confirmed a third time (2-row blocks
−3%); (iv) the f32-dots arm is worth −8.8 ms/step, 4.3× more than every micro-rung
combined.

**VRAM:** post-run 82.27 / 82.21 GiB used — deltas vs lane-8 ≈ 0 (no new resident
allocations; the f32acc kernel reuses the arena). Zero OOM, zero CUDA errors across
all lane-9 runs (gates + benches + profiles + A/B ×10 × 8200 steps).

**Out of this lane / iteration 3:** DSpark drafter cells on this decode loop (the
headline lever), indexer long-context GVR rewrite + chunked prefill (other lanes),
PP-2 serving door, the owner call on extending the f32 ruling (item 1), fp4 GEMM
tensor-core arm, graphs at the post-item-1 mass.

---

# Lane 0731-regate — GPU re-gate on the MINTED 0731 artifact, REF contract (2026-08-19)

Publish-gate checklist item 2 (darklanes PLAN.md): re-gate the GPU stack (lanes 4-9,
tip 9b5de6f16d) on `/home/ubuntu/models/dsv4-flash-0731-nvfp4` (the lossless mint,
0731-MINT-RECEIPTS.md) under the REF contract — 0731 ships kernel.py byte-identical to
the reference law, so `ActQuantVariant::RefFp8Round` everywhere; the preview clamp-only
contract is NEVER mixed in (one variant per invocation, read from the fixture JSON —
gate construction, not policy). Fixtures: the Gate C 14-array 0731 set + the banked
160-token REF greedy trajectory (`fixtures-0731/`, Rust-verified 160/160; box copies
under /home/ubuntu/dsv4-oracle0731/fixtures-mint + greedy/). Plus one bounded extension
rung (task B) under a fresh owner authorization (PLAN.md lane-9 section, pending
ratification).

Code stage (commit de8de1c116; cherry-picks d995da97e4 + 0e00533308 from
lane/dsv4-oracle0731, cited): 0731 config parse (nextn derived from compress_ratios),
NextN-vs-DSpark discrimination at GPU load on stored structure (`mtp.0.e_proj`), the
`dsv4-gpu-tf-gate` bin, `MEMRA_DSV4_DOTS_ARM=f32x` extension arm.

## Threshold derivations (banked BEFORE any gate run — the lane-4 gate-formula protocol)

### (1) Output-sample gate, REF variant, GPU class: the Gate C fork rule re-derived

The CPU Gate C rule (`thr(ref logits) = contract_fork/3`, fork = 3.361 banked in the
fixture JSONs) was built for the CPU class, where the only REF-specific noise is
~1e-6-relative reduction skew flipping isolated e4m3 codes in the window/compressor KV
QAT (measured draw 0.979 on logits) and the bound's job is contract-MIXING detection
(fork = 3.361 fails a fork/3 = 1.120 bound by 3x).

The GPU class is different in both directions, and the rule cannot transfer verbatim:

- **The REF flip channel is larger on GPU.** Upstream bf16 drift (u_b = 2⁻⁸, rel
  σ ≈ u_b·√d) crosses e4m3 rounding midpoints (relative quantum q ≈ 2⁻³) at rate
  f ≈ u_b·√d/q per element — ~6% at layer 2, ~19% at the head — where the CPU class
  flipped isolated codes. Flip magnitude per element ≤ one e4m3 step (≤ q·|v| on
  normals; == max(|got|,|ref|) near zero, the e2m1 adjacency argument). Propagated
  with the fork's own gain, the channel's magnitude class is
  Δ_flip ≈ fork·√(12·f) ≈ 1.5·fork ≈ 5.1 on final logits (rms argument: the fork is
  an all-elements half-quantum-residual walk, rms q/√12; flips are a fraction-f
  full-quantum walk, rms q·√f).
- **The mixing-detection role does NOT transfer.** On the GPU the legitimate bf16
  drift (measured 4.35 / bound 6.71 lane-4 class; native-arm bound 107.6 lane-7
  class) is itself ≥ the fork (3.361) — a logits max-abs bound can no longer separate
  mixing from lawful drift. Mixing exclusion for the GPU gates is STRUCTURAL (the
  variant is read from the fixture JSON and pinned through one invocation; the
  loaded kernels take it as a flag) and the fixtures themselves are guarded by the
  CPU Gate C fork/3 rule. Stated plainly: the GPU output-sample max-abs row is a
  correctness bound, not a mixing detector.
- **Under the NATIVE expert arm (the shipping stack), the existing lane-7 class
  bounds subsume the REF flip channel on every banked array.** Logits: thr ≈
  C(86,86)·√(2 ln n)·absmax ≈ 107 vs flip channel ≈ 5 (5% of bound). compressor_kv
  (the QAT'd array itself): native coeff C(4,4) ≈ 0.125 → thr ≈ 0.51·absmax vs
  one-step e4m3 flips ≤ 0.125·|v| and near-zero flips of negligible absolute size —
  inside the base threshold, no budget needed. indexer_kv / index_score: the fp4 QAT
  is identical in both variants; the native budgets (0.75 / 0.5 with their bounds)
  absorb the slightly-raised flip rates. attn/layer outs: flip noise rides the same
  √d·absmax geometry at ≤ 0.5·bf16-share — inside the 0.3-0.7x measured headroom.
  THEREFORE: `dsv4-gpu-gate` runs the 0731 REF set under the native arm with the
  lane-7 thresholds UNCHANGED; the expected-values line above is the banked
  prediction, and any array outside its bound is a finding to analyze, not to widen.
- Cross-receipt: the clamp-only 0731 fixture set is also run (GPU-clamp vs
  clamp fixtures — the lane-4/7 tight class on the minted weights), mirroring Gate
  C's clamp cross-receipt. Two invocations, two JSONs, never mixed.

### (2) Greedy-160 teacher-forcing gate (the item-2 instrument)

`dsv4-gpu-tf-gate` teacher-forces the BANKED 0731 REF trajectory (Gate C bank,
cpu_greedy_ref.json sha 920901870d3a…, flattened to cpu_greedy_ref.tf.json sha
63a6e13500b3…) through the GPU decode path: per-position argmax vs banked token — 160
independent checks, no divergence cascade (the lane-6 realization-stability doctrine;
the banked trajectory is the truth instrument). In-band rule: the dsv4-greedy-verify
native-class band VERBATIM, band = 3·√2·(C(86,86)+C(0,86))·|cpu top1| — a disagreement
whose CPU margin (banked top-8) is inside the band is a legitimate realization flip
(the 255/260 expf/GEMV class); outside is a REAL bug; a pick beyond the banked top-8
that cannot be adjudicated FAILS loudly (never a silent pass). The REF flip channel
adds ≤ ~5 absolute to the GPU-vs-CPU logits distance — an order under the ~170-200
band — so the band carries over unchanged. Determinism x2 inside the gate. Runs: the
shipping stack (native + device + dots f64 default) AND MEMRA_DSV4_DOTS_ARM=f32 —
both must hold the bar. Banked prediction: agreement in the 255/260 class scaled to
160 (>= 155/160), every disagreement in-band.

### (3) Decode-path spot gates at boundaries (lane-6 a1-style pairs)

`dsv4-gpu-decode-gate` short-probe protocol (lane-7 banked): n_new = 260 covers 40+
consecutive early steps, fine-block completions, coarse block 0 (127/128), the window
wrap (129), coarse block 1 (256/257) — >= 20 steps required by the brief, 260 run.
Pair bounds unchanged (2·C(86,86) class — the a1 comparison is two GPU realizations
of the SAME arm; the REF flip channel is common-mode at matched prefixes and its
differential part is « the native pair band, argument above). Gate (b)-raw feeds the
BANKED 0731 trajectory json (raw counts informational; the corrected verdict is the
tf-gate + the CPU teacher-forcing below). CPU adjudication: `dsv4-greedy-verify`
(native class, REF variant) over the decode trajectory = the 260-class teacher-forcing
receipt. Determinism x2 inside the gate; VRAM table; zero CUDA errors mandatory.

### (4) Task B — f32x extension rung fork derivation (banked BEFORE any f32x run)

Owner authorization (PLAN.md lane-9 section, conditional on the quality bar, pending
ratification): extend the lane-9 f32 ruling to the remaining f64 dependency chains —
sink scores/soft/out (4.30 ms class), rmsnorm/headrms/rowsq residual f64 + indexer
score (~3.5 ms class). Arm: `MEMRA_DSV4_DOTS_ARM=f32x` (implies the gated f32 dots
arm; f64 default untouched; the lane-9 `f32` arm's bytes untouched; legacy path +
prefill pinned f64 unconditionally). Implementation: f32-accumulation TWINS of the
seven kernels, same launch geometry, same per-thread element order, same reduction
topology — a pure accumulator-type substitution on the identical expression DAG
(lane-9 rung-C class). NOT included (not authorized): hc_sinkhorn (iteration-3 fuse
item), compressor pool (prefill-shared emission kernel), the legacy sink/indexer
kernels.

Numeric class (the rung-C derivation, extended): replacing f64 with f32 accumulation
on a k-term chain (k = 128 sink dots/indexer dots, 4096-16384 norms, ≤ idx_tail+win
slots sink den/out) perturbs each output by Δ ≈ u32·√k·√(2 ln n)·max|term|, u32 =
2⁻²⁴ — relative class error ≤ ~1e-5, two orders under the native a1 pair bounds and
three orders under the greedy-verify margin bands. The softmax den in f32 divides
out per-head mass consistently (same class). What can move: attention near-tie slot
weights, argmax/top-k near-ties downstream — exactly why the gates are
trajectory-level teacher-forcing, not per-array deltas. Banked prediction: the fork
classifies exactly like the expf/GEMV/f32-dots forks (255/260-class, all in-band).
Gates, in order, ANY failure = revert the rung and bank the negative: (i) f64-default
regression: decode-gate logits-stream sha unchanged vs the f64 run of this lane;
(ii) f32 regression: sha unchanged vs the f32 run of this lane; (iii) f32x decode-gate
260 (a1/c/d/e) exit 0; (iv) f32x CPU teacher-forcing 260-class all in-band;
(v) f32x tf-gate 160 vs the banked trajectory all in-band; (vi) determinism x2 (inside
the gates). Informational single-run ms/step delta after (interleaved A/B belongs to
iteration 3's final table).

## Lane 0731-regate — gate runs, ALL GREEN (2026-08-19; logs + JSONs in lane-0731-regate/, box /home/ubuntu/dsv4-0731-regate-out/)

Two box binaries, both built from this branch on the box (release, MEMRA_CUDA_ARCH=120a,
CUDA 13.2, rust 1.97.1): **B1 = 3bea90c1ea** (task-A gates + the preview bench
regressions) and **B2 = ad9f0ade30** (adds ONLY the NextN-probe raw-name fix; preview
witness run 2, the f64 decode-gate rerun, task B). Cross-binary seam closed by the
rerun below. Seams on every GPU run: `MEMRA_DSV4_EXPERT_ARM=native`; decode runs add
`MEMRA_DSV4_DECODE_PATH=device` (+ the dots arm as tagged). Artifact under test:
`/home/ubuntu/models/dsv4-flash-0731-nvfp4` (the mint). One fixture variant per
invocation, from the JSON.

### Task A — publish item 2 gate table

| gate | invocation (binary) | verdict | measured | banked output |
|---|---|---|---|---|
| (1) output-sample REF on the mint, 14 arrays | `dsv4-gpu-gate <mint> fixtures-mint/dsv4_0731_fixtures_ref.json 0,1` (B1) | **PASS 14/14, exit 0** | logits max-abs 3.530 vs thr 121.9 (native class); top-1 2581 exact; top-5 one band-adjudicated swap (native rule, no OUT-OF-BAND flag); top-20 18/20, 4 set-diff ids ALL in ±53.1 band; compressor_kv 0.25 vs 2.09 (the predicted e4m3-flip class, inside base thr — no budget consumed); indexer_kv 1.0 vs 1.87 (e2m1 flip class); expert-dequant sub-gate 5/5 BIT-EXACT on the mint's NVFP4 (incl. the lane-1 pin layers.20.experts.7.w1) | gpu-gate-0731-ref-native.log |
| (1b) clamp cross-receipt on the mint | same, `_artifactvariant.json` (B1) | **PASS 14/14, exit 0** | logits 3.120 vs 125.0; top-1 exact; top-20 19/20, 2 set-diff in ±54.7 band (0731's rank-20 cluster is tighter than preview: gap 0.0476) | gpu-gate-0731-clamp-native.log |
| (1c) preview witness (shared load path touched) | `dsv4-gpu-gate <preview> dsv4-lane2-fixtures/..._artifactvariant.json 0,1` | run 1 (B1): 14/14 + mtp SKIPPED — **CAUGHT the stem-probe defect** (mtp.0.e_proj vs raw mtp.0.e_proj.weight; fix = ad9f0ade30, measured key sets in-comment). run 2 (B2): **PASS 15/15, 0 skips, table byte-IDENTICAL to the lane-9 banked gpu-gate-tip-native.log (diff = empty)**, MXFP4 sub-gate 7/7 restored | gpu-gate-preview-witness{,-run2}.log |
| (2) tf-gate 160 vs the banked REF trajectory, f64 dots (shipping default) | `dsv4-gpu-tf-gate <mint> greedy/cpu_greedy_ref.tf.json out 0,1` (B1) | **PASS 158/160, exit 0** | 2 disagreements, BOTH in-band near-ties (steps 22/134, cpu margins 0.0738/0.1579 vs bands 213.5/203.6); determinism ×2 sha 58823c81aa81…; 0 unresolved | tf-gate-0731-f64.log, tf-f64/tf_gate.json |
| (2') tf-gate, MEMRA_DSV4_DOTS_ARM=f32 | same +f32 (B1) | **PASS 158/160, exit 0** | steps 22/44, margins 0.0738/0.1458 vs bands 213.5/176.8; determinism ×2 sha b255b75a5e89… | tf-gate-0731-f32.log |
| (3) decode-gate 260 boundary short-probe, f64 | `dsv4-gpu-decode-gate <mint> ..._ref.json cpu_greedy_ref.json dec-f64 260 0,1` (B1) | **PASS a1/c/d/e, exit 0** | a1 52/52 checkpoints (33..72 consecutive, 127/128/129 wrap, 255/256/257), worst max-abs 4.984 at s=52 under the native pair bound; (d) tokens + 260-row logits stream BYTE-IDENTICAL ×2, sha 2dc470e23f26…; (e) cache alloc == formula [14373888, 14441984] both devs | decode-gate-0731-f64.log, dec-f64/decode_gate.json |
| (3') decode-gate 260, f32 | same +f32 (B1) | **PASS a1/c/d/e, exit 0** | 52/52, worst 5.243; det sha 7a07601ac419… | decode-gate-0731-f32.log |
| (3-CPU) teacher-forcing 260-class, f64 trajectory | `dsv4-greedy-verify <mint> dec-f64/decode_seq_for_verify.json …` (B1, native oracle) | **PASS, exit 0** | **256/260 agree, ALL 4 disagreements in-band** (margins 0.0057–0.2920 vs bands ~173–197) | cpu-verify-dec-f64.log, dec-f64/cpu-verify/cpu_verify.json |
| (3'-CPU) same, f32 trajectory | (B1) | **PASS, exit 0** | **258/260, both in-band** (0.0162/0.0413) | cpu-verify-dec-f32.log |
| (4) determinism / cross-binary | f64 decode-gate RERUN on B2 | **sha 2dc470e23f26… REPRODUCED byte-identically across binaries/processes**; every gate above carries its own ×2 | decode-gate-0731-f64-run2.log |
| (5) byte-inertness regression of the f32x code when OFF | `dsv4-decode-bench <preview> …_artifactvariant.json 1024 0,1` ×2 arms (B1) | **PASS** | f64 stream sha == lane-9 banked **384ecde1bcff28a3…** exact; f32 sha == **1acee23f157087a4…** exact; medians also reproduce the lane-9 table (38.55/40.70 f64, 29.73/31.88 f32 ms/step at s≈200/1024 — cross-day witness) | bench-preview-{f64,f32}-1024.{json,log} |

**VRAM (identical at every checkpoint of every run):** post-load dev0 82.24 / dev1
78.64 GiB used (loader-resident 81.51 / 77.94); post-prefill/post-runs 82.27 / 78.67;
free ≥ 12.64 GiB dev0 / 16.30 GiB dev1 throughout. No NextN block on 0731 (DSpark
skip line printed at load). **Zero OOM, zero CUDA errors across all 16 GPU
invocations** (every gate exits 0 and checks internally).

**Realization-flip consistency note (banked):** the raw first divergence vs the banked
trajectory is step 22 in every decode-path run (f64/f32/f32x), and the tf-gate
adjudicates that exact position as a 0.0738-margin in-band near-tie — the instruments
agree with each other and with the lane-6 doctrine (banked CPU margin at step 22 is
the 255/260-class coin-flip position of this trajectory).

**Threshold-derivation verdict:** every measured value sat inside the banked
predictions (REF logits flip channel predicted ≈ +5 over the clamp class: measured
3.530 REF vs 3.120 clamp GPU-side; compressor_kv flips at the predicted 0.125·|v|
class; no gate-formula correction was needed on the GPU side this lane). The one
correction of the lane was the NextN structural probe (1c) — caught by the witness,
fixed, witnessed byte-identical.

### Task B — f32x extension rung (owner-authorized, PENDING RATIFICATION; defaults untouched)

All gates in the banked order, binary B2, artifact = the mint, REF contract:

| gate | verdict | measured | banked output |
|---|---|---|---|
| (iii) decode-gate 260 @ f32x | **PASS a1/c/d/e, exit 0** | 52/52, worst 4.089 at s=52; det ×2 sha 97ec528ee29c…; cache math exact | decode-gate-0731-f32x.log, dec-f32x/decode_gate.json |
| (iv) CPU teacher-forcing 260-class @ f32x | **PASS, exit 0** | **257/260 agree, ALL 3 in-band** (margins 0.0162/0.0413/0.1294 vs bands ~183–221) — the 255/260 owner-bar class, same as the f64/f32 arms this lane | cpu-verify-dec-f32x.log, dec-f32x/cpu-verify/cpu_verify.json |
| (v) tf-gate 160 vs the banked trajectory @ f32x | **PASS 158/160, exit 0** | the SAME two in-band positions as the f64 arm (steps 22/134, margins 0.0738/0.1579); det ×2 sha ded1cc58c93a… | tf-gate-0731-f32x.log, tf-f32x/tf_gate.json |
| byte-inertness when OFF | **PASS** | table row (5) above: both existing arms' 1024-step shas reproduced exactly with the f32x code in the binary | bench-preview-*.json |

**Informational single-run ms/step on the mint (NOT an A/B claim; iteration 3 owns
the interleaved table):** `dsv4-decode-bench <mint> ref.json 1024` — f32 arm
29.79 / 30.41 / 31.89 ms/step at s≈200/512/1024 (sha eaf1a4153391…) vs **f32x
24.73 / 25.21 / 25.80** (sha aa5ea6639cae…): **−5.1 to −6.1 ms/step**, matching the
lane-9 item-1 projection (−5.5 to −6.5: sink 4.3 + norms/indexer ~3.5 at partial
latency overlap). tf-gate means: 38.6 (f64) → 29.8 (f32) → 24.8 (f32x). Per-chain
attribution stays the lane-9 tip profile (sink_scores 2.93 + sink_out 1.37 + rmsnorm
2.03 + rowsq 1.35 + headrms/indexer ~1.0 of f64-latency-bound mass); a fresh nsys
split belongs to iteration 3.

**Rung state: NOT default.** `MEMRA_DSV4_DOTS_ARM` default stays `f64`; the lane-9
`f32` arm's bytes are proven untouched; `f32x` ships only behind the seam pending the
owner's ratification (the authorization's quality condition is MET on this evidence:
every gate green, every disagreement an in-band near-tie).

### Support claim (house law)

**(DeepSeek-V4-Flash-0731, minted NVFP4 artifact `Avifenesh/DeepSeek-V4-Flash-0731-NVFP4`,
REF/RefFp8Round contract, native expert arm, device decode path) trunk forward +
decode correctness on the 2× RTX PRO 6000 Blackwell 96GB box is GATED** — output-sample
14/14 both variants, teacher-forced greedy vs the Gate C banked trajectory 158/160
(in-band-only) on both shipping arms, decode boundary probe 52/52 with 260-class CPU
teacher-forcing 256–258/260 all-in-band, deterministic ×2 (and across binaries), VRAM
exact, zero CUDA errors. The DSpark drafter GPU path is NOT part of this claim (not
wired; separate lane). NOT a serving or perf claim; the release bar (faster than every
competitor) is a separate owner call.

## Lane ws-ref-thresholds — the (REF variant × bf16 arm) compressor_kv flip policy, derived (2026-08-20; box7 2× RTX PRO 6000 Blackwell WORKSTATION)

Trigger: the it5 box7 cell banked "REF bf16 default-env FAIL, layer2_compressor_kv idx
3263, 2.500e-1 vs thr 1.304e-1, ×3 boots, both tips" with the standing read "WS card
class needs its own thresholds". Characterized per-element (gate `MEMRA_DSV4_GATE_DUMP`
instrument, six-run battery, dumps sha-compared): **the read was arm-confounded, and no
card-class threshold exists to derive.**

- The failing element is a ONE-STEP e4m3 code flip: ref 2.25 → got 2.00 — adjacent
  codes at the group's pow2 scale, |diff|/max = 0.111 ≤ 2⁻³ — exactly the "0731 REF
  set" flip channel above ("compressor_kv flips at the predicted 0.125·|v| class").
  466/4096 elements flip codes under the bf16 arm (62/512 on c160_layer3), every one
  a lawful RNE-boundary flip, only the one top-binade element exceeds the bf16 analog
  threshold. Rope-tail cols show pure analog drift; zero scale flips; zero off-grid.
- Bit-deterministic: all 14 dumped arrays byte-identical across 3 fresh boots AND the
  card-swapped placement (devices 1,0). The native-legacy and fp8-device runs dump
  byte-identical arrays to each other; embed_out/layer0_attn_out are byte-identical
  across ALL six runs (no MoE upstream ⇒ arm-independent bits).
- Cross-class: the WS native-legacy table matches box4's banked Server table row for
  row at print precision (max-abs AND max-rel, logits near-ties/gaps included), and
  the SAME element (idx 3263, 2.25→2.00) is the max-abs carrier under both arms — the
  flip channel is present on Server silicon too, hidden by the native-class threshold
  (2.090e0 ≥ 0.25). **No Server-class default-env (bf16-arm) REF-battery receipt
  exists anywhere** (grep over box4/5 banks: every 14/14 is native-arm). The 8e1bb1ed
  "Server passes / WS fails" comparison compared different threshold classes.
- Root cause: the derivation above proves the flip channel subsumed by the NATIVE
  class threshold only. The (REF variant × bf16 arm) cell was never derived: bf16
  analog thr ≈ 0.032·absmax at layer 2 cannot cover a top-binade one-step flip
  (0.125·|v| ≈ 0.25 at |v|≈2.25) on ANY silicon.

Policy landed (`dsv4_gpu_gate.rs`, keyed on (variant, arm) — the policy surface has no
card input by construction): ref+bf16 compressor_kv gets flip budget 5% of n with
adjacency bound |diff| ≤ 2⁻³·max(|got|,|ref|) (one e4m3 step among normals; near-zero
multi-code moves are absolutely tiny and never reach the analog threshold, measured).
Native arm and clamp-only variant keep ZERO budget — byte-identical to every banked
receipt — pinned by the policy-keying tests (mutation-checked: mis-keying panics with
named MIS-KEYED/LOOSENED messages).

Margin policy (budgets = measured calibration rows):

| row | layer2_compressor_kv | c160_layer3_compressor_kv |
|---|---|---|
| measured exceeders (bf16 arm, ×3 boots + swap) | 1/4096 | 0/512 |
| measured code flips (bf16 / native arm) | 466 / 1143 (24 group-scale, native only) | 62 / 99 |
| worst-case flippable-over-thr census (every element whose one-step flip exceeds the bf16 analog thr, from the REF npz) | 100/4096 = 2.44% | 16/512 = 3.12% |
| budget 5% of n | 204 | 25 |

The budget dominates the theoretical worst case (every flippable element flipping) by
≥ 1.6× and the measured worst by 200×, while still failing systematic scale-law
violations (off-grid, multi-step at magnitude, or >5% mass flips); the adjacency bound
fails any exceeder that is not one e4m3 step. Verification: full REF battery 14/14
PASS with the landed gate under default env ×3 fresh boots + card-swap, fp8 arm ×3,
native-legacy ×1, clamp-only ×1 (unchanged path), on box7; invocations + logs in
darklanes `research/deepseek-flash-20260818/box-mirror-box7/ws-ref-thresholds/`.
OWED (prediction banked): one Server-window default-env REF battery under the landed
gate — expected PASS 14/14 with ≤ a few budget-adjudicated flips.

## OWED DISCHARGED: Server-window default-env battery (2026-08-21, box6 sbox Server Edition)

Prediction held, verdict shape (a): default-env REF battery at this tip
(binary rebuilt on box6 at clean f6554a5777, sha 0cd98786c970e8be…) = PASS 14/14
x3 fresh boots + fp8-arm control, each with exactly the one banked
budget-adjudicated flip (layer2_compressor_kv idx 3263, 2.25->2.00,
|diff|/max 0.1111 <= 2^-3, exceeders 1/4096 — the calibration row verbatim).
Stronger than predicted: all flip-relevant default-env dumps BYTE-IDENTICAL to
the banked box7 WS dumps — cross-class bit-equivalence is Server<->WS
byte-for-byte. Evidence: darklanes
research/deepseek-flash-20260818/box-mirror-box7/ws-ref-thresholds/server-window/
(commits 1ce981ea + b0e04410). No open items remain on this derivation.
