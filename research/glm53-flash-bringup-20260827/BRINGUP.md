# GLM-5.3-Flash bring-up plan (lane/glm53-flash-bringup, opened 2026-08-27)

Target: NativeReference → NativeQualified → NativeTuned for
zai-org/GLM-5.3-Flash @ 04c4e9e95c5da8862dced7e5056455116f83a7e0 (FP8 e4m3, 328 GB,
MIT). Hardware: a 2× RTX PRO 6000 Blackwell Server (96 GB each) bench box with
384 GB host RAM (box identity in the private ops repo). CENSUS.md beside this file is the tensor
truth; modular_glm5_next-ref.py is the banked reference math (transformers).

## Architecture read (verified against modeling source, not the card)

- 45 decoder layers + 1 MTP: 34 KDA linear-attention, 11 DSA (MLA+indexer);
  MoE on 42+MTP layers (288 routed sigmoid noaux_tc + 1 shared), 3 dense.
- KDA: delta rule, fp32 state [64h,128,128]; per-CHANNEL log-decay
  g = −5·sigmoid(exp(A_log)·(f_b(f_a(x))+dt_bias)) (gate_lower_bound −5);
  beta = sigmoid(b_proj) per head; q/k l2norm fp32 (FLA semantics: /sqrt(Σx²+eps));
  fused q|k|v grouped short-conv (kernel 4, silu) — checkpoint stores q/k/v convs
  SEPARATELY, reference fuses at load; output = o_proj(RMSNormGated_sigmoid(core,
  gate=g_b(g_a(x)))). Chunked prefill path (chunk 64, UT transform) + recurrent
  decode path — same family as our GDN kernels, decay is per-channel not per-head.
- DSA: MLA NoPE (qk_rope_dim=0 — NO rotary anywhere in the text stack);
  q_a(1536)→q_b, kv_a(512)+mqa→expand via kv_b (BF16, absorbable); indexer =
  wq_b(q_resid), k_norm(wk(x)), pools of 4 compressed by learned softmax
  (gate_scores + APE), ReLU-scored, weights_proj head-mix, top-512 pools →
  ≤2048 token indices + always-append tail; reference applies as boolean mask
  over full KV (SDPA); a serving kernel gathers instead.
- mHC (from DeepseekV4HyperConnection — dsv4 lineage): 4 residual streams,
  per-layer attn_hc + ffn_hc, output = post⊗branch_out + combᵀ·streams; sinkhorn
  (20 iters) normalizes the mixing weights. Streams expand at model entry,
  reduce at exit.
- MTP layer: DeepSeek-shaped (eh_proj/enorm/hnorm/shared_head.norm), contains its
  own DSA attention; indexer shared with base per `index_share_for_mtp_iteration`.
- Triple EOS [154820,154827,154829]; template drives reasoning_effort
  {low,high,max}, default max; clear_thinking default false.

## Phase 0 — oracle bank (bench box, external impl OFF-serving; runs now)

- Shards downloading (328 GB, pin-verify the revision post-download).
- oracle-mint.py: 4 real prompts × {max,low} effort, greedy, 64 steps, banked
  argmax + top-8 logits + completion bytes. transformers device_map=auto with
  host offload. This bank is the truth every gate pins against
  (GATE:pin-against-truth — anchored OUTSIDE the feature under test).

## Phase 1 — NativeReference (memra)

ModelPlan + census wiring + unfused executor path for glm5_next text:
- Loader: safetensors FP8-e4m3 block-scale ingest (weight_scale_inv), BF16 KDA/
  mHC/embeddings; fuse q|k|v convs; absorb kv_b per MLA convention.
- Reference forward: KDA recurrent (fp32 state), DSA as masked SDPA (indexer in
  fp32, exactly the banked reference math), mHC stream mixing, noaux_tc router,
  dense-first-3.
- Gate R1: per-step argmax parity vs oracle bank on all 8 rows (64 steps each).
- Gate R2: top-8 logit closeness (report only; argmax is the pass bar at
  reference precision differences).
- Vision + MTP deferred out of R gates (text first; MTP loads but unused).
- [DOOR INTEGRATION SECTION — pending the code map: where glm5_next dispatches,
  HybridModel vs new door, what Dsv4Gpu shares.]

## Phase 2 — NativeQualified (serving gates)

- Tokenizer/template gate (marker + diff vs canonical; triple-EOS handling).
- Serve parity: streamed surface vs reference, spec==plain byte identity class.
- Standard-surface law: 3 wire formats + tools on first boot, real agentic-CLI
  round-trip; reasoning_effort plumbed (template kwarg, not sampler).
- Fresh-boot output-sample gate on the serving card class; capacity-keyed seams
  pinned per GATE:card-keyed-full-pins.
- Placement: 2-card PP target (328 GB FP8 > 96 GB; NVFP4 mint later may fit
  tighter). KDA state 137 MB/seq fp32 + tiny MLA KV (12 layers × 512 lora) —
  1M-ctx session memory is cheap; admission math needs new per-token costs.

## Phase 3 — NativeTuned (performance)

- Interleaved A/B ×5 on box hardware only; vendor-default sampled rows
  (temp 1.0, top_p 0.95); EVERY decode cell pins reasoning_effort explicitly
  (TRAP:reasoning-effort-unpinned-decode-cell — this model thinks by default).
- Kernel arcs, in expected-value order: KDA chunk/recurrent CUDA kernels (extend
  GDN family), DSA gather-attention kernel (mask→gather), indexer kernel, MoE
  512→288 router reuse, MTP spec route (dspark/NextN experience applies only
  after acceptance-parity gates — LAW:acceptance-parity-gate).
- Multi-turn cache twin before any serving default (LAW:multiturn-cache-twin).

## Serving/business (darklanes side, not this lane)

MIT license — clear. Pricing/placement/roster via facts.json ripple only after
NativeQualified. HF NVFP4 mint decision after reference lands
(TRAP:convert-direct-q8 applies to any GGUF mint path).

## Gate status (2026-08-27, real artifact @ 04c4e9e)

- Config: PASSED · TokenizerTemplate: PASSED · **TensorCensus: PASSED** — all 76,108
  physical tensors of the FP8 artifact bind exactly once (5 census iterations: HF MLA
  schema, hc_* entries with streams! rows, plural shared_experts, gate-nested router
  bias, hc-less Serial NextN residual, hand-censused vision tower from shard headers).
- TinyParity: PASSED (`memra model verify tiny --against glm5_next`).
- CheckpointParity: in progress — streaming f32 reference runner over the real
  checkpoint (layer-at-a-time residency) + the CLI capture bundle; the independent
  transformers oracle bank (oracle-mint.py) is generating on the bench box.

## NVFP4 mint (owner directive 2026-08-27: "mint with nvidia tools, should be the leading ones")

Tool: NVIDIA TensorRT Model Optimizer (nvidia-modelopt) — the leading NVFP4 toolchain,
and memra's nvfp4_repack consumes modelopt layout natively. Source: the BF16 twin
zai-org/GLM-5.3-Flash-BF16 @ f12e0fe1f6b2ea274c11a569582edfd99d993c5e (656 GB; FP8
main repo is itself quantized — never quantize from a quant when the vendor ships the
full-precision twin). Precision split mirrors the vendor's own FP8 exclusions + census:
quantize MoE experts + shared + dense MLPs + MLA projections; KEEP high-precision:
all KDA tensors, kv_b_proj, mHC, router gates + e_score bias, norms, embeddings,
lm_head, vision. Every mint gates argmax-vs-reference before any serving or publish
(TRAP:convert-direct-q8). Target: ~165 GB -> 2x 96 GB PP serving fit.

## Cross-precision checkpoint evidence (2026-08-27, out-of-gate report)

Native f32 streaming runner vs self-consistent bf16 transformers capture, real
checkpoint, tokens [1,2,3,4], full last-position vocab row (154,880):
**argmax MATCH (id 5), top-6 ranks identical**, ranks 7-8 swap; max_abs 5.006,
mean_abs 0.695 (bf16-weights instrument drift through 46 layers; max_rel is
tail-noise on near-zero logits). Receipts: parity-evidence/. The pack's strict
0.005 CheckpointParity stays reserved for a same-numeric-class run (GPU fp32 or
fp8-native capture); this report is wiring evidence, not the formal gate.
transformers CPU trap recorded: FP8 dequantize path hard-defaults bf16 on CPU and
leaves native-BF16 params bf16 under dtype=float32 (mixed-dtype crash) — the bf16
self-consistent instrument sidesteps it.

## NVFP4 mint COMPLETE (2026-08-28)

nvidia-modelopt 0.46.0, streaming per-tensor W4A16_NVFP4 from the BF16 twin
@ f12e0fe1. Census 38,770 tensors -> 37,338 quantized + 1,432 kept; 20 shards,
190.7 GB on disk; self-verify OK (every quantized tensor carries its
weight/weight_scale/weight_scale_2 triple). Exclusions land as designed: lm_head,
embed_tokens, every KDA projection (b_proj/f_a/f_b/g_a/g_b/q/k/v + convs), kv_b_proj,
mHC, router gates, norms, vision. Receipts: mint-receipts/.

Size note for the serving decision: 190.7 GB, not the ~165 GB sketch. Full-VRAM
residency on 2x 96 GB does not close on this split alone — expert offload (SLRU
residency) on 2 cards or a 3-card placement is the owner-facing choice, surfaced
rather than silently re-split.

Gate in flight: the same f32 streaming reference runner over the NVFP4 artifact vs
the FP8 artifact, identical tokens — argmax parity + logit drift is the mint's
admission bar (TRAP:convert-direct-q8: never serve or publish a mint that has not
been gated against the reference).

## The mint's keep list was inert against the loader (found + fixed 2026-08-28)

memra's loader law re-encodes any BF16 2-D weight of >=1M elements to Q8_0 unless
`preserves_source_dtype` matches it, and that function reads ONLY
`quantization_config.modules_to_not_convert`, matching exact-or-dotted-prefix on the
HF name after `model.language_model.` is unwrapped. Our mint wrote its keep list as
compressed-tensors `ignore` in `model.language_model.*` form: wrong key AND wrong
name form, so it was invisible. Effect on our own artifact: the large KDA
projections (q/k/v/o_proj 33.5M elements, f_b/g_b exactly 1,048,576 and so right at
the threshold) arrived Q8_0 rather than the BF16 the precision split intends. The
vendor FP8 artifact does not have this problem: it writes `modules_to_not_convert`
already.

Fixed in both places, metadata only, no weight re-mint: mint-nvfp4.py now emits both
dialects, and the minted artifact's config.json was patched in place (backup
config.json.pre-keeplist-fix on the box). Verified against the banked tensor index:
the corrected list protects 793 tensors, including 204/216 matched projections --
exactly the 34 KDA layers x 6, leaving the 12 MLA o_proj quantized by design.

Rejected alternative: teaching the engine to read `ignore`. That would flip every
compressed-tensors model's kept tensors to Float, which is the outcome the loader
law exists to prevent. Stating our own fact in the dialect the reader speaks is the
narrow fix.

## MINT GATE PASSED (2026-08-28)

Same engine (memra-reference f32 streaming runner), same tokens [1,2,3,4], full
last-position vocab row (154,880), all three artifacts run on one box.

| comparison | argmax | top-k rank-identical | max_abs | mean_abs |
|---|---|---|---|---|
| our NVFP4 vs BF16 twin (its own source) | MATCH | top-3 | 3.117 | 0.534 |
| vendor FP8 vs BF16 twin | MATCH | top-3 | 3.489 | 0.490 |
| our NVFP4 vs vendor FP8 | MATCH | top-5 | 4.184 | 0.705 |

The middle row is the calibration and the reason the first row means anything: our
4-bit mint's deviation from full precision is COMPARABLE TO the vendor's own 8-bit
quantization deviation, at half the bit width. An absolute logit delta has no
interpretation without a same-instrument reference point.

Method note: the first comparison run was NVFP4 vs the vendor FP8 artifact, which is
NOT the gate -- it measures our quant against another quant and blends two error
sources. The BF16 twin is the mint's own source, so it is the only comparison that
isolates mint error. Receipts: mint-receipts/{nvfp4,bf16,fp8-samebox}-oracle.tsv.

Not claimed by this gate: serving accuracy (4 tokens, one position), long-context
behaviour, sampled decoding, or the engine's own quant arms (the runner dequantizes
to f32; the serving path rides MMVQ/W4A8 and is gated separately).

## Indexer performance, measured (2026-08-28, 2x RTX PRO 6000 Blackwell Server, 500 W)

Release build, warmup 3 + 11 trials, `NVIDIA_TF32_OVERRIDE=0`, shipped shape
(index_topk 2048, pool 4, 32 index heads, d 128, kv_rank 512). Raw:
kpool-bench-Frankfurt.txt. Per MLA layer; a forward runs 12 of them.

DECODE (t_q=1), ms:

| ctx | build_step | build_cold | score | select | attend | total | x12 layers |
|---|---|---|---|---|---|---|---|
| 4k | 0.0106 | 0.0129 | 0.0222 | 0.0155 | 0.5469 | 0.595 | 7.1 |
| 16k | 0.0101 | 0.0178 | 0.0514 | 0.0286 | 0.5538 | 0.644 | 7.7 |
| 64k | 0.0102 | 0.0369 | 0.1733 | 0.0591 | 0.5534 | 0.796 | 9.6 |
| 256k | 0.0103 | 0.2193 | 0.6454 | 0.1818 | 0.5544 | 1.392 | 16.7 |
| 1M | 0.0087 | 0.8407 | 2.5360 | 0.6562 | 0.5534 | 3.754 | 45.1 |

THE RESULT THE INDEXER EXISTS FOR: attend is FLAT at ~0.553 ms from 4k to 1M, a
256x context increase. That is bounded work per query (the 2048-token budget)
instead of the linear growth dense attention would show. The sparse program is
doing exactly what it claims.

RESIDENCY CONFIRMED: build_step is flat at ~0.01 ms across the whole range while
build_cold (the pre-residency per-call cost) grows to 0.841 ms at 1M, a 96x saving
per call at the top end, and the gap widens with context exactly as the design
predicts.

THE NEW BOTTLENECK, stated plainly: at 1M decode, scoring is 2.536 ms and selection
0.656 ms, so the indexer now costs 3.2 ms against 0.55 ms of attention. Selection is
no longer the problem after the radix change; SCORING is. It is O(t_q * n_pools *
heads * d) and shows it worst at prefill: 1294 ms per layer at 1M with t_q=512,
which is 15.5 s across 12 layers for one chunk. Prefill scoring is the next arc, not
selection.

Not claimed: end-to-end model throughput (this measures the indexer stages on
synthetic state, not a loaded checkpoint), and nothing here is a customer-facing
number.

SUPERSEDED, PENDING RE-MEASURE (2026-08-28). Every `score` column above is the
block-per-(query, pool) kernel. That kernel is now retained only as the arithmetic
oracle (`Engine::mla_kpool_score_ref`); the shipped scorer is a register-tiled fused
GEMM+head-reduce, gated BIT-IDENTICAL to it across both tile-dispatch boundaries,
ragged tiles and three causal horizons
(`glm5_kpool_indexer_gpu::gpu_kpool_scoring_is_byte_identical_to_the_reference_kernel`).
The bench now prints BOTH — `score_ref_ms` and `score_ms` — from the same run, so the
ratio is one box's clock. Until that run lands on serving-class hardware, the `score`
columns above describe the OLD kernel and no new number belongs in this table. The
selection, attend and build columns measure unchanged kernels; the one attribute the
new scorer sets (a max-shared carveout) is per-FUNCTION, so it does not reach them.


## Indexer scoring, re-measured after the tiled rewrite (2026-08-28)

Same box, same release harness, old and new kernels timed in ONE run so the ratio is
one clock. Raw: kpool-bench-Frankfurt-tiled.txt.

PREFILL (t_q=512), score ms, per MLA layer:

| ctx | ref | tiled | speedup |
|---|---|---|---|
| 4k | 4.747 | 0.797 | 6.0x |
| 16k | 19.898 | 0.942 | 21.1x |
| 64k | 80.572 | 2.896 | 27.8x |
| 256k | 324.048 | 11.098 | 29.2x |
| 1M | 1294.533 | 41.028 | **31.6x** |

At 1M that turns one 512-token chunk from 15.5 s across 12 layers into 0.49 s.

DECODE (t_q=1), score ms — AND THE REGRESSION THE SAME RUN EXPOSED:

| ctx | pools | ref | tiled | ratio |
|---|---|---|---|---|
| 4k | 1024 | 0.0214 | 0.1399 | **0.15x (6.5x SLOWER)** |
| 16k | 4096 | 0.0523 | 0.1406 | **0.37x (2.7x SLOWER)** |
| 64k | 16384 | 0.1706 | 0.1510 | 1.13x |
| 256k | 65536 | 0.6457 | 0.4216 | 1.53x |
| 1M | 262144 | 2.5367 | 1.5598 | 1.63x |

The tile's fixed setup dominates when there is one query and few pools, which is
SHORT-CONTEXT DECODE: the most common serving shape, not a corner case. The
prediction for decode had been 10-20x faster; the measurement said 6.5x slower at 4k.
That is why the harness times both kernels rather than only the new one.

Fixed by dispatching decode on pool count at the measured crossover
(MLA_KPOOL_SMALL_TILE_MIN_POOLS 16384): below it the reference kernel runs, above it
the tile. Both are bit-identical (gate 12), so the choice is speed only and cannot
move a selection. Pending re-measure to confirm decode now tracks the better of the
two curves at every size.

## Crossover dispatch CONFIRMED (2026-08-28, same box, gate 12 re-run green there)

Raw: kpool-bench-Frankfurt-crossover.txt. Decode now takes the better of the two
kernels at every size, and prefill is unchanged.

DECODE (t_q=1) score ms: ref / dispatched
- 4k (1024 pools): 0.0219 / 0.0219 — reference path, regression eliminated
- 16k (4096): 0.0522 / 0.0514 — reference path
- 64k (16384): 0.1734 / 0.1511 — tile, 1.15x
- 256k (65536): 0.6464 / 0.4217 — tile, 1.53x
- 1M (262144): 2.5371 / 1.5604 — tile, 1.63x

PREFILL (t_q=512) unchanged: 31.6x at 1M (1294.6 -> 41.0).

Full-forward indexer+attend cost, x12 MLA layers, decode:
4k 7.1 ms/token (unchanged, as intended) · 1M 33.4 ms/token (was 45.1).

## Tokenizer: the `glm4` pre-tokenizer split landed (2026-08-28)

The first real-artifact server load refused at the tokenizer — correctly: `tokenizer.ggml.pre`
is `default` and the tokenizer.json Split regex matched none of memra's four families, so token
ids would not have been exact. The split is now ported.

It is qwen2's pattern with EXACTLY ONE ATOM changed (`\p{N}` -> `\p{N}{1,3}`, i.e. digit runs
group up to three) and a SECOND divergence the regex diff does not show: the shared qwen35
state machine folds `\p{M}` into its letter runs, and the literal GLM classes do not. So it is
its own machine (`unicode::split_glm4`), leaving qwen35/qwen2 byte-untouched.

Named `glm4` after upstream llama.cpp: this checkpoint's llama.cpp `chkhsh` is
`cdf5f353…`, identical to `zai-org/GLM-4.7-Flash`'s registered `glm4` entry, so a future GGUF
mint through the upstream converter resolves with no further change.

Gate: 509-case corpus against the checkpoint's own tokenizer through HF `tokenizers` —
**0 split mismatches, 526/526 token-id parity in both add_special modes** (the corpus includes GLM's control-token literals and real chat-template renders). Full receipt and
method in `parity-evidence/PRETOKENIZER-GATE.md`; generator `pretok-ref-glm4.py`.

## FIRST END-TO-END LOAD: it serves, and a staging stall blocks generation (2026-08-28)

The real 190.7 GB NVFP4 artifact LOADS AND SERVES through memra-server on one card:
`[server] worker ready`, `listening on`, `ctx=1048576`, `tok="glm4"`, template caps
tools=true think=true effort_levels=true. VRAM 23 GB at 2048 expert slots, 56.9 GB at
12000 slots, 89.5 GB once the cache fills. Non-expert tier landed at 14.8 GB against
the placement receipt's predicted 14.67 GB.

Getting there took two more loud refusals, both of which would have been silent
wrongness under a fallback:
- `blk.0.hc_attn_fn is absent` — the six mHC tensors had no ggml->HF name mapping,
  same class as the missing MLA names. Fixed, and the pin that should have caught it
  was widened (it built a Serial-residual plan AND filtered on `blk.0.kda`, so it was
  blind twice over).
- `unsupported tokenizer.ggml.pre 'default'` — GLM's split regex differs from qwen2 by
  exactly one atom (`\p{N}{1,3}` vs `\p{N}`, the llama-3/cl100k digit grouping).
  Implemented as the standalone `glm4` family; token ids 526/526 exact vs the
  checkpoint's own tokenizer.

GENERATION IS BLOCKED, and the diagnosis is specific but NOT yet root-caused:
- TTFT exceeds the 90 s platform ceiling on every attempt, streaming or not.
- GPU utilization 0% for 60+ s continuously; CPU 2.4%; one thread in state D.
- That thread is `memra-gpu-worke` with `wchan = folio_wait_bit_common` — blocked on
  major page faults against a memory mapping.
- Disk read during a request: ~5.7 MB/s. Random 4 KiB faults serialized on one thread
  at EBS latency is almost exactly that number.

THREE HYPOTHESES TESTED AND FALSIFIED, recorded so nobody re-runs them:
1. Disk bandwidth / cold cache. Warming the artifact changed nothing.
2. Expert-slot thrash. 2048 -> 12000 slots (9.6 -> 57 GB resident, 89.5 GB once full)
   changed nothing.
3. Cold-start cache fill amortized over requests. Three consecutive requests, VRAM
   stable at 89.5 GB, identical 90 s timeouts.
The decisive datum against all three: `cat` of the whole 178 GB artifact completes in
30 s (5.9 GB/s), i.e. it is ALREADY fully page-cache resident, while the worker still
blocks on folio faults. So the faults are not cache misses on the weights.

Remaining suspects, in order: the NVFP4 expert staging path on a safetensors source
(`find()` repacks modelopt->gguf per access via `repack_modelopt_to_gguf`, and the
per-expert sequential loop is the arm glm5 is left on because it is denied every
fused epilogue), and whatever mapping is actually faulting. `decode wave cap 1` and
`scheduler tick cap 1` in the boot log are also unexplained for this model.

Not yet measured, therefore not claimed: any tok/s number, TTFT, or throughput.

## 2026-08-28 — the stall is root-caused and fixed (two defects, two fixes)

Box: sbox bench box (2 x RTX PRO 6000 Blackwell 96 GB, 499 GB RAM, ext4 on a **ssd-vol EBS
root volume**). Artifact `~/models/glm53-nvfp4`. All timing on the box; the rig is
correctness-only.

### What was actually measured

`bpftrace` on `block:block_rq_issue` + `kprobe:filemap_fault`, one 90 s request, stock config:

```
@bytes_by_comm[memra-gpu-worke]: 743845888      # 743845888 / 181603 = 4096 B per read, EXACTLY
@ios_by_comm[memra-gpu-worke]:   181603         # ~3027 IOPS
@faults[memra-gpu-worke]:        186286         # one filemap_fault per page
```

Blocked-thread stack (`/proc/PID/task/TID/stack`), sampled during the request:

```
folio_wait_bit_common -> filemap_fault -> __do_fault -> do_read_fault -> ... -> exc_page_fault
```

`mincore` residency of what the worker actually maps (126 files in `/proc/PID/maps`):

| file set | size | resident |
|---|---|---|
| `.memra-repack/*.nvfp4` (the mmap the worker faults on) | 159.5 GiB | **13.2 %** |
| `*.safetensors` (what the earlier warm test `cat`-ed) | 177.6 GiB | 99.4 % |

The earlier "artifact is already cache-resident" datum warmed the **wrong files**. The expert
bytes the MoE forward reads live in the `.memra-repack` disk tier, not the safetensors.

Cold device characterisation (caches dropped, single stream):

| access shape | rate |
|---|---|
| 4 KiB random pread (what `MADV_RANDOM` produces) | 1835 IOPS = **7.5 MB/s** |
| 1 MiB sequential | **145 MB/s** |
| 4.72 MiB expert-stride pread | **331 MB/s** |
| 16 parallel 1 MiB readers | **69 MiB/s aggregate** (contention, not scaling) |

The volume is a ~125 MiB/s ssd-vol. `TIMEOUT_MS_MAX` is a hard 90 s platform ceiling.

### Root cause 1 — the stall

The PATH-B `.memra-repack` expert slab is mmap'd under `MADV_RANDOM`
(`MEMRA_MOE_MMAP_ADVICE` default). Each expert access is a **contiguous 4.72 MB stride**, but
the VMA policy suppresses readahead, so it is served as ~1150 serialized 4 KiB faults on the
single `memra-gpu-worker` thread. Measured 11.8 MB/s against a slab of 159.5 GiB: no request
could reach a first token inside 90 s.

Confirming arm: `MEMRA_SPILL_IO=worker` (the existing explicit positioned-read backend) raised
the same request from 756 MiB/90 s to **11 313 MiB/90 s = 126 MB/s** — the device ceiling.
Still no first token, which proves the I/O *path* was one binding constraint and the *volume*
is the other. Cold start cannot work on this storage; the expert bytes must be resident.

**Fix**: `MEMRA_MOE_SLAB_POPULATE` (default `fits`) — read each `.memra-repack` slab
sequentially at load when it leaves >=20 % of MemTotal available. Read through the file
handle, not the mapping: the VMA carries `MADV_RANDOM`, so prefaulting through it would
reproduce the 4 KiB pattern. The fits-guard preserves the >RAM sizing case the disk tier
exists for (MiniMax-M3 REAP50, 122 GB slab on a 60 GB host). Not applied to the GGUF spill
tier — those maps are whole shards, not expert slabs.

Load receipts, cold (`[slab-populate]`, 126 slabs):

```
[slab-populate] blk20-gate: 1296 MiB in 10.3s (125 MiB/s)   # x126 => 159.5 GiB, ~22 min
```

Warm restart is near-free and idempotent — the guard and the read both no-op against a
resident cache:

```
[slab-populate] blk44-down: 1296 MiB in 0.1s (8988 MiB/s)   # load ready in 40 s
```

### Root cause 2 — no first token even after the I/O fix

With experts resident, decode was reached in ~1 s and then returned `engine_error`:

```
decode_step_batch_sampled_lean_masked runs a serial residual, but this model's ModelPlan
declares ResidualTopology::HyperConnections{ streams: 4, collapse: Mean }. Refusing: that
path would compute a different model. Converted paths: forward, forward_last, prime_cache,
decode_step.
```

The batched decode chunks, batched prime core, graph capture and every speculative entry
point run a serial residual and `refuse_hyper` this topology. `decode_batch_program` returns
`Generic` for glm5, so the scheduler placed it in the batched chunks.

Suspect 3 from the brief (`decode wave cap 1; scheduler tick cap 1`, unexplained for this
geometry) is the SAME defect seen from the other side, and it is not a degenerate branch:
`chunk_cap_for` returns 1 at its first branch because
`DECODE_BATCH.trunk_capabilities(plan).batch.supported` is false —
`OperationKind::HyperConnections` is not in `decode_batch_support`'s supported list. qwen38
prints 16 because its plan is fully covered and qualifies for the exact-16 tier. So the
capability manifest ALREADY said "no batched decode arm, width 1" at boot, while the scheduler's
`eager_only_model` predicate — keyed on `decode_batch_program == Gemma` — placed the session in
the batched chunks anyway. The cap was a correct warning nobody routed on.

**Fix**: `decode_batch_unconverted(plan)` (plan-level, next to `decode_batch_program`) puts
hyper-connections plans in the server's existing `eager_only` class — the same route built for
gemma4, whose converted paths are exactly `forward / forward_last / prime_cache / decode_step`.
Boot now says:

```
[worker] zai/glm-5.3-flash: EAGER-ONLY serving (hyper-connections residual — no batched
decode arm): per-session eager decode, monolithic prefill, no graph promotion, no prime batching
```

### Before / after, on the box

Invocation (both arms), `~/cell.sh` then `~/ttft2.py`:

```
MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 CUDA_VISIBLE_DEVICES=0,1 MEMRA_COMPAT=openai \
MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18400 \
MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 ./target/release/memra-server
```

| arm | TTFT | server tok/s | disk read during request |
|---|---|---|---|
| BEFORE (stock, mmap default) | **never** (90 s deadline, streaming and not) | — | 756 MiB @ 8.4 MB/s |
| BEFORE (`MEMRA_SPILL_IO=worker`) | **never** (90 s deadline) | — | 11 313 MiB @ 126 MB/s |
| AFTER greedy, 32 tok | **1.98 s** | 9.83 | 0 MiB |
| AFTER greedy, 128 tok | **1.57 s** | 19.70 | 0 MiB |
| AFTER vendor-default sampled, 128 tok | **1.54 s** | 12.80 | 0 MiB |
| AFTER vendor-default sampled, 600 tok | **2.06 s** | 12.70 | 0 MiB |

`disk read = 0 MiB` is the point: expert bytes now come from page cache, so nothing blocks the
worker thread.

**No spec-engagement receipt is possible on this model.** Every speculative entry point
(`generate_spec`, `generate_spec_dflash`, `generate_spec_dspark`, `generate_spec_eagle`)
`refuse_hyper`s this topology, and the admission log confirms `path=plain` on every request.
The vendor-default arm is therefore verified as plain-path generation only. Spec/MTP on
hyper-connections models is unbuilt work, named below — not a silent omission.

### STILL BROKEN: the model computes garbage

Generation is fast and no longer errors, but the output is not a model:

```
mt= 1 reasoning='<'
mt= 2 reasoning='<|'
mt= 4 reasoning='<|im2'
mt= 8 reasoning='<|im2-2|>'
mt=16 reasoning='<|im2-2|>  <p>  <p>'
```

**Control — the garbage is upstream of both fixes.** Fix 2 changes the prefill route too
(batched scheduler -> monolithic prefill), so "pre-existing" needed proving, not asserting. Same
prompt, same greedy settings, stock binary (`~/memra-server.stock`) vs fixed binary; the stock
arm emits token 1 from the converted prefill before the batched decode step refuses:

```
STOCK  "Say hi." -> delta {"reasoning":" ","role":"assistant"}   (then engine_error)
FIXED  "Say hi." -> delta {"reasoning":" ","role":"assistant"}   (then " *", " and", ... )
```

Token 1 is byte-identical across the two binaries, so neither fix moved the forward path's
output. Greedy continuations are consistent and deterministic, and the very FIRST token after a
31-token prefill is already wrong — so this is the **forward/prefill numerics**, not decode drift and not
expert-cache eviction. That is a distinct bring-up defect, untouched by either fix here, and it
is the next lane: pin the glm5 forward against the Memra-native reference executor
(hyper-connections collapse, KDA, MLA/kpool indexer, NVFP4 expert macro folding, sigmoid router)
before any customer-facing claim. **GLM-5.3-Flash is not servable until that lands.**

### Open, not fixed here

- Batched decode / batched prime / graph capture / speculative arms for
  `ResidualTopology::HyperConnections`. Until built, this class is single-stream eager decode.
- The two sources of truth still disagree in general. `eager_only_model` now covers the hyper
  case by name, but the authoritative answer is already computed: `chunk_cap_for` returns 1 at
  its FIRST branch because `DECODE_BATCH.trunk_capabilities(plan).batch.supported` is false —
  `OperationKind::HyperConnections` is absent from `decode_batch_support`'s list. Keying
  `eager_only_model` on that capability instead of on named programs would generalise to every
  unsupported op, but it changes routing for any plan with any uncovered op and has no receipts
  yet; it is the follow-up, not this increment.
- `[spill-pread] falling back to mmap: worker read ring is busy` at `depth=2`
  (`buffer_bytes=4718592`): the ring is one expert deep, so a rolling window cannot form.
- Deploy smell: `.memra-repack` on a 125 MiB/s ssd-vol root volume. This box has an idle 3.5 TB
  local NVMe at `/opt/dl-image/nvme`; the slab belongs there, which turns the 22 min populate into
  roughly a minute.
- 159.5 GiB of experts against 2 x 96 GB VRAM with GPU1 idle at 3 MiB and
  `MEMRA_MOE_RESIDENT=0` forced. TP2 / multi-device expert residency would remove the disk tier
  entirely and is likely the real serving shape. Open question for the owner.

## 2026-08-28 — the garbage forward is root-caused and fixed (two NAME defects, one gate)

Method: bisect the ENGINE against `crates/memra-reference`, the unfused f32 executor, on the
SAME artifact, the SAME token ids, layer by layer. Receipts: `forward-bisect-receipts/`.

### Making the two sides comparable

Both sides already emit the same `memra-checkpoint-oracle-v1` last-position logits row:
`glm5_checkpoint_runner` (reference) and `run-safetensors` (engine, `MEMRA_ORACLE_OUT`) —
`HybridModel::forward_last`, which forks to `forward_hyper` on this plan. That gave a
**4-token repro in ~50 s per engine run** instead of the 31-token server prompt:

| | argmax | top-1 logit | cosine vs reference | mean_abs |
|---|---|---|---|---|
| reference (f32, banked `mint-receipts/nvfp4-oracle.tsv`) | **5** | 17.379 | — | — |
| engine BEFORE | **3186** | 12.359 | **0.5709** | 1.657 |
| engine AFTER | **5** | 17.219 | **0.9930** | 0.201 |

For the layer bisect, one new env-gated seam written ONCE and read by BOTH crates:
**`MEMRA_HYPER_TRACE=<path>`** (`memra_reference::hidden_trace`, FLAGS.md row in this change).
It emits the last token row at `expand` / `mixer` / `attn` / `router` / `route` / `routed` /
`ffn` / `layer` / `collapse` in one format, from `execute_hyper_layer` + `moe_mlp` on the
reference side and from `forward_hyper` + `moe_ffn_sequential_zq8` on the engine side. It
selects no arm and changes no dispatch (unlike the `MEMRA_MOE_*` traces, which move
`observation_mode`): the traced engine run reproduced argmax 3186 exactly.

### The bisection

`forward-bisect-receipts/layer-diff-before.txt`, cosine per stage:

| stage | layer | cosine | reading |
|---|---|---|---|
| `expand` | — | 1.000000 | embedding + hc stream expansion exact |
| `mixer`/`attn`/`ffn`/`layer` | 0,1,2 | ≥ 0.99976 | KDA, mHC pre/post/sinkhorn, dense MLP, NVFP4 2-D dequant all correct |
| `mixer` | 3 | **0.999988** | **MLA + DSA k-pool indexer at real `index_topk` 2048 is correct** |
| `router` | 3 | **0.9999994** | the router MATMUL was never wrong |
| `route` | 3 | **0.679** | **wrong experts selected** |
| `routed` | 3 | 0.9919 | routed-expert sum, norm 44.86 (ref) vs 41.01 (engine) |
| `ffn` | 3 | **0.878** | norm 66.46 (ref) vs **41.01 (engine) — identical to its own `routed`** |

The last row is the whole second defect in one number: `ffn - routed` is EXACTLY 0.0000 in the
engine and 33.64 in the reference. The shared expert contributed nothing at all.

FULL DEPTH (all 45 trunk layers, both arms, `layer-diff-{before,after}.txt`). The reference side
of that run reproduced the banked `mint-receipts/nvfp4-oracle.tsv` BIT-FOR-BIT (cosine 1.000000,
max_abs 0.0), so the taps changed nothing on either side. Worst cosine per stage, and the first
layer where the stage drops below 0.999:

| stage | BEFORE worst | first <0.999 | AFTER worst | first <0.999 |
|---|---|---|---|---|
| `mixer` | 0.0392 @L28 | L4 | 0.9597 @L23 | L16 |
| `attn` | 0.1617 @L30 | L4 | 0.9728 @L24 | L17 |
| `router` | 0.8927 @L44 | L5 | 0.9980 @L22 | L21 |
| `route` | 0.5282 @L42 | **L3** | 0.7451 @L19 | L7 |
| `routed` | **-0.0236** @L29 | **L3** | 0.9126 @L17 | L15 |
| `ffn` | **-0.0345** @L29 | **L3** | 0.9264 @L17 | L15 |
| `layer` | 0.1340 @L44 | **L3** | 0.9714 @L23 | L16 |

BEFORE the break is a STEP at layer 3 — the first MoE layer — and nowhere earlier; `routed` and
`ffn` go NEGATIVE by layer 29, i.e. the engine's FFN branch ends up pointing away from the
reference's. AFTER nothing breaks anywhere: every residual-stream stage stays above 0.95 and
degrades MONOTONICALLY with depth from about layer 15, which is the f32-vs-4-bit instrument drift
this comparison is supposed to show. The `route` row is the one to read carefully — it is the
flattened (expert, weight) pairs IN SELECTION ORDER, so two top-k implementations emitting the
same set in a different order lower its cosine without changing anything the model computes
(layer 7 after the fix: identical 8-expert set, two adjacent slots swapped).

Prime suspects from the brief, DISPROVEN by the same table, not by argument:
NVFP4 macro-scale folding (layers 0-2 are dense NVFP4 MLPs at cos 0.99998, and `weight_scale_2`
reads 3.78e-05 on this artifact — a dropped macro is a 2.6e4x error, not a 0.6x one); Q8_0
re-encode of the KDA projections (the boot `[loader-law] WARNING` census lists every one of the
nine KDA projection families loading as 2-D Float, and the layer 0-2 KDA mixers match the f32
reference at cos 1.000000 / max_abs 6.7e-08); the mHC stream collapse and
stream layout (`expand` bit-identical, `attn`/`layer` exact for three layers); the kpool indexer
at real topk 2048 and the `MlaKeyUp` byte-order convention (layer-3 `mixer` at cos 0.999988).

### Root cause: two missing rows in the engine's ggml -> HF name map

`hf_mapping::ggml_to_hf`'s glm5_next arm had no entry for either. Both names then fell through
to the GENERIC map, which spells them the qwen3moe way, so they resolved to tensors this
checkpoint does not contain — and both load sites treat "absent" as a legal shape:

1. **`exp_probs_b.bias`** — no row at all. `MoeWeights::load` substituted `vec![0.0; n_expert]`.
   noaux_tc selects on `sigmoid(logit) + e_score_correction_bias`, so a zero bias reduces
   selection to the raw top-k: at layer 3 the engine chose `{41,253,255}` where the reference
   chose `{10,16,24}` (5/8 overlap), on all 42 MoE layers. HF name is
   `mlp.gate.e_score_correction_bias` — nested under the ROUTER module, not the DeepSeek-V3
   `mlp.e_score_correction_bias`.
2. **`ffn_{gate,up,down}_shexp.weight`** — the generic map spells the shared expert SINGULAR
   (`mlp.shared_expert.*`); this checkpoint spells it PLURAL (`mlp.shared_experts.*`, which
   CENSUS.md had already recorded). `load_opt` answered `None` three times and the always-on
   branch was silently dropped from every MoE layer.

Why the tensor census and the reference never saw it: they load through
`TensorContract`, which declares BOTH spellings correctly. The ENGINE loads through
`hf_mapping::resolve_ggml`. Nothing pinned that the two agree. Same class as this lane's two
earlier name gaps (the six mHC parameters, the MLA family) — third occurrence.

### The fix

- `hf_mapping.rs`: the three shared-expert rows and the selection-bias row, in the arch-scoped
  glm5_next block.
- `hybrid.rs::load_ffn`: a plan that DECLARES `RouterPlan::{Sigmoid,SqrtSoftplus}{selection_bias:
  true}` may no longer fall back to a zero bias, and a plan that declares `moe.shared` may no
  longer run without one — both now refuse by name, the way the hc topology refusal already
  does. The zero row stays legal for routers that genuinely have no bias.

### The gate that would have caught it, and the property it exercises

`glm5_next_every_contract_tensor_resolves_through_the_engine_map` (`hf_mapping.rs`, CPU-only,
no checkpoint, no GPU): compile the real glm5_next plan for a miniature config that carries BOTH
mixer kinds AND a MoE layer, then require that **every** tensor the GGUF-dialect contract
declares resolves through `resolve_ggml` — the function the ENGINE's loader calls — onto a name
the HF dialect of the SAME contract declares for the SAME `TensorId`. 58 tensors, pinned exactly,
with NO per-name allowlist.

**The property every existing gate misses: NAME RESOLUTION ON THE ENGINE'S OWN PATH.** Every
fixture gate in this lane (`glm5_routed_router_gpu`, `glm5_moe_residency_gpu`,
`swiglu_preclamp_gpu`, `kda_fixture_gpu`, `mla_gpu_forward`, `glm5_kpool_indexer_gpu`) serves
its own synthetic tensors under names the test itself chose, so none of them touches the map at
all; `TensorCensus` and the reference runner exercise the CONTRACT's names, which were right.
The predecessor pin `glm5_next_kda_names_match_the_contract` was blind twice over — it filtered
to `blk.0.kda*`/`blk.0.hc_*` AND built a dense-MLP plan, so no router bias or shared expert row
existed in its contract. The new pin caught the shared-expert defect on its first run, before
any box time was spent on it.

The pin was made to go RED on each defect and then restored
(`forward-bisect-receipts/gate-goes-red.txt`) — it covers two distinct failure modes, and each
defect exercised a different one: a name with NO row (the bias), and a name that falls through
to the generic map and resolves to a plausible spelling no checkpoint carries (the shared
expert). The second mode is why the pin compares against the contract's own names for the same
`TensorId` instead of merely asserting `Some(_)`.

The onboarding runbook now carries the general form of this trap:
`docs/ONBOARDING.md` §2, "The name map is a SECOND surface, and a missing row is silent" —
what the two spellings are, why a census-green artifact proves nothing about the engine's
resolver, and the completeness-pin template every future architecture writes.

### Both refusals EXECUTED, not merely written

`forward-bisect-receipts/refusal-negative-arms.txt`. A refusal nobody has watched fire is not a
control (LAW: loud failures fail quietly — three times in one night the diagnostic written to end
a silent failure was itself silent). Each arm deleted the corresponding name-map row on the box,
rebuilt, and loaded the real 190.7 GB artifact:

- ARM A (no `exp_probs_b.bias` row): refused at layer 3 naming the tensor, the plan's
  `Sigmoid { normalize_selected: true, scaling_factor: 2.5, selection_bias: true }`, and the
  name map as one of the two possible causes.
- ARM B (no `ffn_*_shexp.weight` rows): refused at layer 3 naming all three tensors.
- CONTROL (rows restored, same binary path, same artifact): argmax 5, logit 17.2195.

Before this change both arms LOADED and SERVED.

### Before / after on the box (bench box, warm restart, `~/cell.sh FIXED 0`)

```
BEFORE  greedy "Say hi." x24  ->  "  * 1. 1. 1. 2. 2. 2. 2. "
AFTER   greedy "Say hi." x24  ->  "The user wants me to say hi. This is a simple greeting
                                   request. I should respond in a friendly, natural way"
AFTER   sampled "What is 17 times 24? Answer with just the number." (vendor defaults, no
        sampling params) -> reasoning: "17 x 20 = 340 / 17 x 4 = 68 / 340 + 68 = 408 ...
        The answer is 408."   content begins "408"
```

Gates, all green in both forms, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`:
glm5_kpool_indexer_gpu 4+9 · glm5_moe_residency_gpu 1+2 · glm5_routed_router_gpu 5+3 ·
swiglu_preclamp_gpu 3+7 · hyper_connections_gpu 1+6 · mla_gpu_forward 5 · kda_fixture_gpu 3 ·
kda_quant_operand_gpu 4 · `memra-gguf --lib` 170 (was 169, +1 = the new pin) ·
`memra-reference --lib` 22 · `memra-tokenizer` 55+3 · `cargo check --workspace --tests` ·
`cargo fmt --all`.

### THE NEXT DEFECT, visible for the first time: generation does not stop

(Diagnosed here; FIXED in the section that follows.)

With a correct forward, the next defect is legible. `generation_config.json` declares THREE eos
ids — 154820 `<|endoftext|>`, 154827 `<|user|>`, 154829 `<|observation|>` — and the boot log says
`eos=154820`. Only one is wired. A 400-token sampled request answered correctly and then kept
going, emitting `<|user|>` and a whole synthetic multi-turn conversation, `finish_reason=length`.
The triple-EOS handling was READ in this document's architecture section on day one and never
implemented.

The decisive probe is the GREEDY control, banked in
`forward-bisect-receipts/eos-not-wired.txt`: `"Name the capital of France. One word."` answers
`The capital of France is Paris.` and then emits `<|user|>` — a DECLARED eos — and the server
keeps generating. Everything before that token is correct; everything after it is post-stop text.

It is NOT a template defect, checked rather than assumed: `chat_template.jinja` contains zero
`<|im_start|>`/`<|im_end|>` occurrences and renders turns with `<|user|>`/`<|assistant|>`. The
ChatML markers in the sampled output appear only AFTER the model emits `<|user|>` — i.e. after
the token that should have stopped it — so they are the model drifting past an unhonored stop,
not a prompt rendered in the wrong dialect.

Also unchanged from the previous section: no batched decode / graph capture / speculative arm
exists for `ResidualTopology::HyperConnections`, so this class is single-stream eager decode and
no spec-engagement receipt is possible (`path=plain` on every request).

### FIXED: multi-EOS stop handling (2026-08-28)

**Which file the engine reads, and where the ids died.** `Tokenizer::from_hf_dir`
(`crates/memra-tokenizer/src/lib.rs`) reads `tokenizer.json` + `tokenizer_config.json` +
`generation_config.json`. It never reads `config.json` for eos, so this artifact's
`text_config.eos_token_id = [154820, 154827, 154829]` (banked as `glm-config.json`) is inert —
the file that matters is `generation_config.json`, and it WAS being read. The bug was one line
inside it: the array arm was

    json::Value::Arr(a) => a.first().and_then(|x| x.as_u64()),

i.e. the vendor's whole declared stop set was truncated to its first entry, and ids 2 and 3
were dropped on the floor at load. Everything downstream of the tokenizer was already a SET and
already correct: `GenParams::eos` is a `Vec<u32>`, `worker::run` unions `tok.eog_ids()` into it,
both decode loops (`advance_sample_emit`, `advance_token_emit`) and every spec emit path test
`params.eos.contains(&id)`, the stop token's text is deliberately emitted as `""` (never
streamed as content), and `stop_reason_to_finish` maps `Eos -> "stop"`. Nothing in that chain
needed to change.

**The change** (`crates/memra-tokenizer/src/lib.rs`, plus one log line in `worker.rs`):
`Tokenizer` gains `eos_ids: Vec<u32>` — every id the checkpoint declares, `eos_id` first — and
`eog_ids()` starts from that instead of `vec![self.eos_id]`. The SCALAR `eos_id` keeps its exact
old selection (tokenizer_config `eos_token`, else the FIRST generation_config id), so the boot
log, embed pooling, `TokRxInfo`/`SpecGrammar` and every `eos_id()` caller are untouched; the set
rides alongside it. `from_gguf` stores `vec![eos_id]` — GGUF metadata declares one id, so no GGUF
family moves. `dsv4_serve.rs` builds its own stop set from `eos_id()` and is likewise unmoved.
The boot log now prints the set it actually stops on: `eos=154820, stop=[154820, 154827, 154829]`
(it previously printed only the scalar, so it agreed with the bug and could not disagree with it).

**Why `<|user|>` / `<|observation|>` are legitimate stops** — checked, not assumed. Both are
`special: true` added tokens in the artifact's own `tokenizer.json` (ids 154827 / 154829,
verified on the box), and `chat_template.jinja` uses them as TURN BOUNDARIES: `<|user|>` is the
user-turn prefix (line 139), `<|observation|>` the tool-result prefix (line 167), `<|assistant|>`
opens the generation the server asked for (line 256). An assistant emitting either has left its
own turn.

**Census — what else moves.** Reading the whole array changes the stop set of every
`from_hf_dir` family whose vendor declares more than one id. Measured over the artifacts on the
rig, 2026-08-27:

| family (HF dir) | `generation_config.eos_token_id` | delta to the stop set |
|---|---|---|
| qwen36 / qwen38 27B | `[248046, 248044]` | +248044 `<|endoftext|>` (248046 `<|im_end|>` was already in via the name backstop) |
| qwen35 9B modelopt | `248044` (scalar), tc `<|im_end|>` | +248044 `<|endoftext|>` |
| qwen3 1.7B synth | `[151645, 151643]` | +151643 `<|endoftext|>` |
| gemma4 26B NVFP4 | `[1, 106, 50]` | +50 `<|tool_response>` (1 `<eos>` is the scalar, 106 `<turn|>` was already in) |
| hy3 | `120025` (scalar) | none |
| m3 (`MiniMaxAI/MiniMax-M3`) | `200020` (scalar) | none |
| step35 (`stepfun-ai/Step-3.7-Flash-NVFP4`) | `[1, 2, 128007]` | +1 `<｜end▁of▁sentence｜>`, +2 `<｜▁pad▁｜>` (128007 `<|im_end|>` is the scalar) |
| every GGUF family | n/a | none, by construction |
| dsv4 | n/a | none (`dsv4_serve.rs` builds its set from `eos_id()`) |

qwen3x / gemma4 / hy3 measured against the artifacts on the rig; m3 and step35 against the
vendor repos' own `generation_config.json` on the Hub (no serving box touched). Every delta is a
`special: true` control token the vendor itself names as a stop; none can fall out of ordinary
BPE over text. The one row worth an eyebrow is step35's id 2, which is that checkpoint's PAD
token — the vendor lists it as an eos anyway; it is still a control token, it is never produced
by a healthy decode, and stopping on a generated pad is the right answer if one ever appears. Shipped general rather than gated on GLM, because dropping
declared ids is the bug in every family that has them — and pinned, so the delta is a decision
with a test behind it.

**Pins** (`crates/memra-tokenizer/src/lib.rs`, `hf_tests`):
- `hf_dir_glm53_flash_carries_all_three_declared_eos_ids` — the REAL banked sidecars
  (`generation_config.json` sha256 `230c3060…`, `tokenizer_config.json` sha256 `98b12715…`,
  both now banked in this lane dir) over a fixture whose `added_tokens` carry the artifact's real
  ids. Asserts `eos_id() == 154820` (unchanged) and SET EQUALITY
  `eog_ids() == [154820, 154827, 154829]` — this vocab has no `<|im_end|>`/`<turn|>`, so the name
  backstop must contribute nothing.
- `hf_dir_single_eos_family_stop_set_unchanged` — the regression pin: both single-id shapes
  (tokenizer_config-only, and hy3's scalar generation_config) give a one-id set.
- `hf_dir_multi_eos_array_extends_the_stop_set_beyond_the_first_id` — the qwen shape, holding the
  census delta above.
- `hf_dir_generation_config_eos_fallback_and_jinja` updated deliberately (id 14 is no longer
  dropped), with the reason in the test.
All three fail on the pre-fix `a.first()` behaviour (executed, not assumed).

**On-box before/after**, `"Name the capital of France. One word."`, `max_tokens=400`, raw JSON in
`eos-receipts/`:

```
BEFORE  greedy (temperature 0)  finish_reason=length  completion_tokens=400
        reasoning: "The capital of France is Paris.<|user|>\n<|im_start|>user\nName the capital
        of France. One word.<|im_end|>\n<|im_start|>assistant\nThe capital of France is
        Paris.<|user|>…"  (the same hallucinated turn ~12x until the budget ran out)
BEFORE  sampled (vendor defaults, NO sampling params)  finish_reason=length  completion_tokens=400
        content: 'Paris<|user|><|user|>The user is asking me to find a word that means the same
        as "happy" - they're asking for a synonym…'

AFTER   greedy (temperature 0)  finish_reason=stop  completion_tokens=8
        reasoning: 'The capital of France is Paris.'   content: ''
        no <|user|> and no <|observation|> anywhere in the response
AFTER   sampled (vendor defaults, NO sampling params)  finish_reason=stop  completion_tokens=228
        content: 'The capital of France is Paris.'
        no <|user|> and no <|observation|> anywhere in the response
```

Boot line on the after-binary: `[worker]   loaded "zai/glm-5.3-flash": 46 layers, eos=154820,
stop=[154820, 154827, 154829]`, `/proc/<pid>/exe` NOT `(deleted)` (the receipt is from the
rebuilt binary, not a survivor of the restart).

**Observed, not fixed:** in the GREEDY arm the model stops at a declared eos while still inside
`<think>`, so it never emits `</think>` and the whole answer lands in `reasoning` with `content`
empty. The product shape — vendor-default sampled, no sampling params — closes the think block
and puts the answer in `content`, so this does not block; greedy is the instrument, not the
product. The sampled arm's reasoning text still wanders (it produced an unrelated ChatML-looking
few-shot transcript as literal text before answering); that is decode quality on a bring-up mint,
not a stop-handling defect — no control token leaked and the request terminated on a declared eos.

Gates, all green in both forms, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`:
glm5_kpool_indexer_gpu 4+9 · glm5_moe_residency_gpu 1+2 · glm5_routed_router_gpu 5+3 ·
swiglu_preclamp_gpu 3+7 · hyper_connections_gpu 1+6 · mla_gpu_forward 5 · kda_fixture_gpu 3 ·
kda_quant_operand_gpu 4 · `memra-gguf --lib` 170 · `memra-reference --lib` 22 ·
`memra-tokenizer` 58+3 (was 55+3: +3 pins, 1 updated) · `cargo check --workspace --tests` ·
`cargo fmt --all --check` clean. No new `MEMRA_*` env read, so no `docs/FLAGS.md` row.

## Standard-surface audit and fix (2026-08-28)

The launch blocker is the house standard-surface law: every served model exposes the IDENTICAL
full API — three wire formats plus tools — and "no tools branch" is a launch blocker.

### Audit BEFORE any change (live receipts in `surface-receipts/before-*`)

Every surface answered with a 200. All of them answered on the WRONG prompt.

| surface / feature | status before | why |
|---|---|---|
| `/v1/chat/completions` non-stream | worked, ChatML-rendered | template markers `<think>` + `add_generation_prompt` matched the qwen detector; the glm5 dialect had no arm |
| `/v1/chat/completions` streaming | worked, ChatML-rendered | same render; SSE reasoning/content deltas fine |
| `/v1/completions` (raw) | worked | no template involved |
| `/v1/messages` (Anthropic) | worked, ChatML-rendered | translation surface over the same core |
| `/v1/responses` (OpenAI) | worked, ChatML-rendered | ditto |
| tools: definition | rendered as the QWEN `<tools>` + `<function=…>` instruction block | `tools_json` (whole tool object) instead of the template's unwrapped `function` object with `defer_loading`/`strict` dropped |
| tools: call emission | parsed only because the model OBEYED the qwen instruction it was handed | the native `<tool_call>NAME<arg_key>…` wire has no parser branch; it would have surfaced VERBATIM as content |
| tools: result round-trip | rendered as a qwen `<|im_start|>user\n<tool_response>` turn | the native `<|observation|>` block, its run-grouping and its id-keyed re-ordering did not exist |
| `reasoning_effort` | ACCEPTED AND IGNORED | reached `Request::reasoning_effort` (`effort_levels` fires — the template really does contain `reasoning_effort is defined`) and died in the qwen arm. Proof: `/v1/responses` with `reasoning.effort:"low"` hit the prefix cache at `cached_tokens: 288` of 288 against the identical no-effort request — zero rendered bytes differed |
| `reasoning_effort: none/minimal` | 400, correct | the template opens `<think>` unconditionally and has no `enable_thinking`; kept |
| `response_format` | 400, correct — but `/v1/models` advertised `structured_output: true` | claim the server itself refuses |
| `tool_choice: required` | 400, documented contract-wide | needs constrained decoding |

The frame it rendered — `<|im_start|>` / `<|im_end|>` — is not in this checkpoint's special
vocabulary at all (`extra_special_tokens` is `[gMASK] <sop> <|system|> <|user|> <|assistant|>
<|observation|>` …), so it tokenized as ordinary text. Fluent and invisible: the
GGUF-template-mint failure mode.

### Fix

- `memra-tokenizer/src/chat.rs`: `template_is_glm5` (`[gMASK]<sop>` AND `<|observation|>` —
  unique across every committed template), `apply_glm5_template`, `glm5_effort_level`,
  `glm5_tool_json`, `glm5_can_sort`; dispatched from BOTH entry points ahead of the qwen arm,
  and named in `template_has_tools_branch`. `dsv4_json` renamed `py_json` — it is
  `json.dumps(ensure_ascii=False)`, which both dialects' `tojson` needs.
- `ModelCaps::glm5`, probed template-keyed in `worker::run` and stated false in `dsv4_serve`;
  `instruct_type` "glm"; boot log prints `glm5=`.
- `canonical_effort_for`'s `dsv4_max` generalized to `max_tier` (`c.dsv4 || c.glm5`): GLM is the
  second template with a real rung above `high`, and the clamp was silently eating it.
- `toolcall.rs`: `ToolStreamParser::glm5` — the `<arg_key>`/`<arg_value>` body grammar, no
  `</think>` separator swallow, and the template's single `\n` before the first call stripped as
  wire syntax (left in, a tool-only turn answers `content: "\n"` instead of `content: null`).
- `worker::plain_chat_render_path`: the fast path maps turns to `(role, content)` and DROPS
  `reasoning`, so on a dialect that replays it the two render paths would disagree — the shape
  that also stops a parked session from matching its own stream. Turns carrying `reasoning` no
  longer take it. Byte-neutral for every other family by construction.
- `/v1/models` `structured_output` (contract-v2 row) AND the OpenRouter catalog's
  `json_mode`/`structured_outputs` now false/absent wherever the server refuses
  `response_format` (`dsv4`, or `qwen_think && !think_switch`). Fleet ripple, owner-visible:
  step35 carries the same shape and its rows flip too — it 400s that request as well, so the
  rows are now the truth, but darklanes surfaces may read these fields.
- Fixture oracle: `gen_surface_fixtures.py` + `surface-fixtures/` (22 cases), rendered from the
  checkpoint's own `chat_template.jinja` under the transformers jinja environment.

### Gates

`memra-server` 461 (435 at lane start -> 447 after the origin/main merge 626b827954, which
added 14 and removed 2 independently of this work -> 461 with 14 new glm5 pins) ·
`memra-tokenizer` 58+3 (held) · `memra-gguf --lib` 179 = 178 pass + 1 ignored (170 before the
same merge) · `memra-reference --lib` 22 (held) · `cargo check --workspace --tests` clean ·
`cargo fmt` clean. No new `MEMRA_*` env read, so no `docs/FLAGS.md` row.

`tools/check-public-boundary.py`: 0 new violations from this work. It DOES report one new
sev4 (`rented_ipv4`) in `docs/FLAGS.md` — the decode-perf lane's in-flight `MEMRA_ST_PINNED`
receipt row, which spells the bench box's public IP. Left untouched (another lane's uncommitted
work in a shared checkout) and raised instead: memra is the public repo, so it has to come out
before that row is pushed.

One repair not from this work but red on this branch: the origin/main merge (626b827954)
brought PR #57's tokenizer sparsity guard, which refused the multi-EOS lane's `write_glm53_fixture`
— a 20-entry synthetic vocab declaring id 154829. The guard is right (a dense `id_to_token` Vec
allocated from a few hundred bytes of JSON); the FIXTURE was sparse. It now fills ids 17..154820,
giving it the artifact's real density.

### Outstanding

The live agentic round-trip IS banked (`surface-receipts/roundtrip-*`, 2026-08-28), taken in the
window after the decode-perf lane's chained cells finished — no timing captured, no restart, GPU
idle and no request in flight. Fixture-pinned NATIVE prompt bytes to `/v1/completions`: turn 1
answered `</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>
</tool_call>` (`finish_reason: "stop"`, 12 tokens) — the GLM wire, not the qwen wire the ChatML
fallback used to instruct it into; turn 2, with the result in an `<|observation|>` block, read it
and answered "21°C and sunny" (`stop`, `cached_tokens: 176`). Both emissions are pinned verbatim
through the shipped parser by `glm5_live_emissions_parse_into_the_serve_surface`.

Still owed: the post-deploy battery on the FIXED binary — boot `glm5=true` caps line, full
chat/`/v1/messages`/`/v1/responses` tool cycles through the server's own rendering, and a
vendor-default sampled probe with a spec-engagement receipt. The round-trip proves the dialect
against the model, not the server-side wiring on the deployed build.

## The PP door is OPEN for the mHC trunk (2026-08-28, lane/glm53-pp)

ROADMAP step 3's blocker is lifted. All three hyper-connection trunk walks
(`forward_hyper`, `prime_cache_hyper`, `decode_step_hyper` in `hybrid_forward.rs`) refused
`MEMRA_PP_STAGES` outright — "the sharded stage handoff is unwired for this residual
topology" — and that refusal was the one thing between GLM-5.3-Flash and expert residency
across BOTH cards. 171.2 GB of routed experts against 2x96 GB: card 0 alone holds at most
~57-66 GB of them, so single-card residency is arithmetically impossible and the second card
is the only route.

Each walk now has a ppN twin built on SHARED layer-range helpers and SHARED trunk exits, so
the split walk and the unsplit walk run the same code and differ only in where the stages
run. Exactly one thing differs from the generic ppN arm: the boundary payload is the mHC
stream state, `[streams, n_embd]` for decode and `[t, streams, n_embd]` for prime, not the
serial trunk's `[n_embd]` / `[t, n_embd]`. `pp.rs` BoundarySlot buffers are lazily sized from
the caller's `n`, so no slot-sizing change was needed. Routing: decode.rs routes `hyper` before
the generic ppN door, so the hc walks own their own door rather than the order being flipped;
each carries the same `rewrite_allowed(RewriteSurface::Pipeline)` check the generic door does.

Roadmap item 3 (per-stage KDA recurrent + MLA/kpool KV placement) needed NO new contract:
`pp::new_cache` already picks the owning stage's `KvDev` per layer for `Recurrent`,
`LatentKvCache` and `KvCache` alike, and the kpool `index_pool_keys` plane is lazily allocated
through the engine the mixer is called with, which under these walks is the stage's engine.
The gate asserts the fence actually SEPARATES those state classes across stages, so it is a
tested property rather than an argued one.

**Gate**: `glm5-hyper-ppn-gate` (new binary), fixture-driven, bit-identical logits vs the
unsplit hc walk on three arms — decode-serial, prime-twin, prefill-twin. 10 knob arms x 3
comparison arms, ALL PASS. Four mutations (dropped TX; off-by-one layer range in the decode,
prime and prefill walks) each turn it red, and the per-walk mutations leave the other arms
green, which is how the arms are shown independent. Receipts, green and red logs, driver
script and full scope: `ppn-hyper-gate/RECEIPTS.md`.

**Truth chain**: split-vs-unsplit is arm-equality, not truth. It closes only by composition
with `tests/hyper_connections_gpu.rs`, which anchors the UNSPLIT hc walk to `memra_reference`
(6/6 PASS on the same tree). Cite both halves or neither.

**The door is now proven as a PLACEMENT too (2026-08-28, two-card box).** Six cross-device
arms x three comparison arms, 18/18 BIT-IDENTICAL: `0,1` N=2, `0,1` with `SHARD=0`, `0,1,0,1`
N=4, reversed `1,0`, `0,1` with `SPLITS=1`, and a longer P=16/N=24 arm. Peer transport, the
sharded weight load and cross-device per-stage cache placement all carry the mHC stream state
exactly. Mutation M1 re-run cross-device turns the gate red on both `0,1` and `0,1,0,1`, so it
still binds in the new topology rather than passing there by construction. Receipts and the
full story: `ppn-hyper-gate/XDEV-FINDINGS.md`.

One arm cannot run and it is not a bug: `MEMRA_PP_HOST_BOUNCE=1` deliberately revokes peer
access before serving, which makes the gate's door-OFF UNSPLIT reference impossible over
sharded cross-device weights. Phase markers put the failure in the reference walk, before any
split code. Closing it needs a bank/compare harness (`--dump-logits` / `--against`), which
would be a stronger gate than today's sibling-arm comparison; named in XDEV-FINDINGS.md, not
built here.

**SCOPE — still NOT a throughput result.** This gate runs a synthetic 4-layer fixture and
reads no clock, by construction. Step 3's 26.7 ms/token and 37.4 tok/s remain PROJECTIONS:
turning them into measurements needs the real 190.7 GB artifact, residency actually achieved
on both cards, real prompts with `reasoning_effort` pinned, interleaved A/B x5, vendor-default
sampled rows, and METHOD.txt's staging-subtracted decomposition. That cell is now unblocked —
the artifact is resident on the two-card box.

Still refused, deliberately: the deferred-readback (pipelined) arm.
`decode_step_h_ppn_deferred` calls `refuse_hyper`, and this lane did not change that.

Still in the way of a clean DEFAULT, and not taken by this lane: `hybrid.rs:216` computes
`exact_expert_bytes` only for `match src.gguf()`, so a safetensors model has no whole-model
expert census. That is why `MEMRA_MOE_RESIDENT=0` is forced in the serving recipe and why
`MEMRA_ST_PINNED` cannot yet become capacity-keyed by default.

## STEP 3 MEASURED: full expert residency on both cards (2026-08-28, lane/glm53-pp)

The ppN door opened, so the cell could finally run on the real 190.7 GB artifact. It works, on
the **f32 trunk**, without `MEMRA_BF16_MMV`:

| arm | p5 greedy | p5 sampled | p7 greedy |
|---|---|---|---|
| 1-card, `ST_PINNED`, 12000 slots | 20.41 tok/s (48.99 ms) | 24.36 (41.06) | 18.92 (52.85) |
| **2-card PP, full residency** | **29.95 (33.39)** | **27.08 (36.93)** | **29.74 (33.62)** |

1.47x on p5 greedy, 3.51x against the 8.5 tok/s configuration that was being served. Reproduced
across two boots to 0.29%. Every output sha is IDENTICAL between the one-card and two-card arms
(greedy and seeded-sampled, two prompts) — byte identity across the placement on the real model,
not just the fixture.

**Staging goes to zero and the intercept holds.** The attribution's staging-free constant for
the f32 trunk is X = 33.05 ms/token; resident, the machine measures 33.39 ms. The +0.34 ms
residual matches card 1 sitting at 99.2% residency.

**Two corrections to the roadmap, both load-bearing.**
1. Step 3's `26.7 ms / 37.4 tok/s` carries the BF16 roofline — it stacks step 2 underneath. On
   the f32 trunk the prediction is `33.05 ms / 30.26 tok/s`, and the measurement is 29.95, −1.0%.
   The projection was right; it was being quoted against the wrong trunk arm.
2. "only with the BF16 trunk" was a double-count: it sized the trunk per card from the halved
   BF16 figure while treating the f32 trunk as if all 23.6 GB landed on each card. Under PP the
   trunk shards too — measured 11,987 / 13,013 MiB per card — so f32 + full residency is
   ~93.7/94.0 GiB of 95.6 GiB. It fits with ~4 GiB to spare.

What actually had to be raised was `MEMRA_MOE_HARD_VRAM_FRAC`, 0.80 -> 0.95 (the slot count was
never the binding clamp). OOM-gated per FLAGS.md and receipted; a machine-specific pin at this
context/session shape, not a new default.

Receipts, decomposition, the residency arithmetic, and a measurement gap that had to be worked
around (`moe_cache_stats` is not wired through PP stage engines, so two-card staging counters
read 0.0 whether or not staging happened): `residency-cell/RESIDENCY-CELL.md`.
