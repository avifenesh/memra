# DeepSeek-V4.1-Flash onboarding, step 1: config, pack and census (2026-11-09)

Verdict: the new `deepseek_v41` pack parses the 1017 config, compiles a typed plan for it and
binds every one of the 96,085 header tensors of the pinned checkpoint. Config, TokenizerTemplate
and TensorCensus pass at inspect. The pack's support stays none: no executor runs CSA2,
single-pass mHC or Engram, so TinyParity, CheckpointParity, RewriteParity and Serve are pending.
No GPU was used.

(Superseded on the support state by step 2 at the end of this file: the reference executor runs
the plan, TinyParity passes and the pack is `NativeReference`. The step-1 numbers below stand.)

Step 2b (same day, `encoding/README.md`): the `encoding_dsv41` chat dialect renders and parses
byte for byte as the vendor's `encoding/encoding.py` across 172 render (80 through the OpenAI
surface), 18 refusal and 15 parse fixtures (166, 67 and 17 before step 2c), with the artifact
tokenizer's ids for every render. The pack's renderer now resolves, and the TokenizerTemplate
gate exercises it on the artifact (`inspect/tokenizer-contract.tsv`).

Step 2c (same day, at the end of this file): fixes from the independent review of PR #1128.

## Artifact

`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`: 48 safetensors
shards, 510,286,023,000 tensor bytes, 96,085 tensors. Headers were read by HTTP range requests;
no weight was downloaded. `inspect/artifact.lock`:

- `config_sha256=8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` (the
  `config.json` bytes; the same file is the pack fixture `fixtures/config.json`)
- `census_sha256=42e86e668ad65b03c252064d34ce424b7a575eb2927222bbb2aa1574028057c1`
  (`inspect/tensor-census.tsv.gz` unpacks to the hashed text)
- `plan_sha256=c4d9ad3bfdeac3d8152963b4c8449080b4e8a6024e2fdf7c1f67799764a31b8c`
- `binding=passed`, `tokenizer=passed`, `dialect=encoding_dsv41`

Code: commit `8e86db30fd` on `dsv41/onboarding-20261109` (base `06b6a8136a`).

## Commands

```bash
memra model inspect deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --against deepseek_v41 --out inspect
# family=deepseek_v41 tensors=48496 artifacts=inspect
memra model verify config deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --against deepseek_v41 --out verify-config
# family=deepseek_v41 verified=Config
```

The census has 48,496 rows because each quantized row carries its `.scale` plane (47,589 of
them). Rows by storage class:

| rows | bytes | storage |
|---:|---:|---|
| 47,232 | 295,997,276,160 | MXFP4 (I8-packed e2m1, e8m0 scale per 32): routed experts, trunk and DSpark |
| 2 | 202,758,032,400 | FP8 e4m3, e8m0 scale per 32 of a row: the two Engram tables |
| 355 | 7,408,601,600 | FP8 e4m3, e8m0 scale per 32x32 block: dense projections |
| 520 | 3,952,883,712 | BF16: embedding, head, norms, router, compressor, index-key projection, tower |
| 387 | 169,229,128 | F32: hyper-connection parameters, sinks, router biases |

## What the plan says

`inspect/model-plan.txt` is the compiled plan. Per layer:

| layers | CSA2 mode | compressed KV source | top-k source | candidates |
|---|---|---|---|---|
| 0, 1 | Window (ratio 0, `rope_theta` 10000, no YaRN) | - | - | - |
| 2, 8, 14 | Full, ratio 2, softmax-gated compressor | own | own | none |
| 20 | Full, ratio 1, projection compressor | own | own | produces 2048 blocks of 8 |
| 24, 28, 32, 36 | Reindex, ratio 1 | 20 | own, restricted to 20's pool | restricted |
| other 3..39 | Reuse | group's Full layer | most recent index layer | - |
| DSpark 40..42 | Window | - | - | - |

Every layer is a 384-expert top-6 (DSpark: 128 top-3) sqrtsoftplus MoE with one shared expert
and pre-clamped SwiGLU (limit 10), inside single-pass mHC (4 streams, Sinkhorn 20). Engram sits
on layers 1 and 14. The plan carries the checkpoint numerics: 32x32 FP8 dense blocks, per-32 FP8
activation quantization on every quantized linear, FP4 index query/key (32, ue8m0), FP4
compressed KV (16, e4m3), FP8 window KV and attention output (32, ue8m0).

New plan operations, all registry rows with no implemented surface:
`Fp8Block32Linear`, `Csa2Attention`, `Csa2Compressor`, `Csa2Indexer`, `Csa2CandidatePool`,
`Csa2KvReuse`, `Csa2IndexReuse`, `Csa2State`, `SinglePassHyperConnections`, `EngramLookup`.
`inspect/execution-rewrites.tsv` lists them as blockers on every surface.

## DeepSeek-V4 is unchanged

The 0731 artifact (`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`)
inspected `--against deepseek_v4_dspark` with the base binary and with the step-1 commit writes
byte-identical `artifact.lock`, `model-plan.txt`, `tensor-census.tsv`, `execution-rewrites.tsv`,
`gates.txt` and `tokenizer-contract.tsv`. Only the refusal text moved: the base refuses the
output head (`head.weight` was not a recognized spelling), this commit names the next refusal,
the upstream MXFP4 trunk experts against the pack's NVFP4 house-mint contract. (From step 2b on,
`tokenizer-contract.tsv` gains the renderer lines of the TokenizerTemplate check; see the step 2
inspect section.)

## Files

- `census_classes.py`: compresses the header census into the pack fixture
  `crates/memra-gguf/src/model_packs/deepseek_v41/fixtures/census-classes.tsv` and refuses to
  write unless the fixture expands back to every name, dtype and shape.
- `inspect/`: the inspect outputs listed above, `normalized-config.txt` included.
  `tokenizer-contract.tsv` is step 2b's: the same contract plus the renderer probe.
- `encoding/`: step 2b, the chat dialect (oracle files, fixture generator, fixtures, receipts).

# Step 2: the plan runs in the reference executor; tiny gate (2026-11-09)

Verdict: the `deepseek_v41` plan runs in memra's reference executor, prefill and decode, every
one of the ten V4.1 operations, DSpark drafting included. `memra model verify tiny --against
deepseek_v41` passes against an oracle the vendor `model.py` produced on the fixture's own
checkpoint bytes (with torch stand-ins for what it calls outside torch, named under Oracle), and
memra's run equals that oracle byte for byte: 6,549 rows, max_abs 0, plus two variants (step 2c)
at 9,084 and 17,860 rows. The pack is `NativeReference` <!-- support: deepseek_v41-reference -->
(record `deepseek_v41-reference`); checkpoint, rewrite and serve gates are pending. The pack,
the reference and the oracle runs used no GPU; one check did: torch's CUDA `topk` tie rule, on an
RTX PRO 6000 Blackwell Max-Q (Oracle below).

## What runs

- `crates/memra-gguf/src/dsv41_forward.rs`: the V4.1 primitives, a sibling of `dsv4_forward`
  (which is untouched). bf16 rounding at every point the vendor stores a bf16 tensor; FP8
  quantize-dequantize per 32 with a power-of-two scale on every quantized-linear input, the
  window KV and the attention output; FP4 per 32 (e8m0) on the index query and key; FP4 per 16
  (e4m3 scale) on the compressed KV; the sparse-attention kernel's online softmax over blocks of
  64 selected entries (bf16 probabilities into PV, f32 denominator, sink once at the end);
  single-pass mHC mixes and Sinkhorn; RoPE with YaRN on compressed layers; sqrtsoftplus routing
  and clamped SwiGLU; the Engram hash and gate; checkpoint FP8/FP4 dequantization. The Engram
  hash multipliers come from a bit-exact port of numpy's `SeedSequence` + `PCG64` + Lemire
  bounded draw, pinned against the multipliers the vendor code derives for the 1017 config.
- `crates/memra-reference/src/deepseek_v41.rs`: the plan-derived checkpoint census (a test pins
  it to the pack's config-derived census for the 1017 config, tower and `bias_vl` excepted), the
  dequantizing loader, the tiny fixture, and the session: per-layer window rings, Full / Reindex
  / Reuse, the ratio-2 partial-group carry, the indexer and candidate pool, DSpark ring seeding
  and drafting.
- `memra_reference::execute_tokens(plan, weights, tokens, decode_from)`: one prefill chunk, then
  teacher-forced decode steps, a DSpark draft after each step but the last. `verify tiny` runs
  every family through it (`decode_from` is `None` for every other family, the old `execute`).

## Tiny fixture

`crates/memra-gguf/src/model_packs/deepseek_v41/fixtures/tiny-config.json`: hidden 64, two query
heads of 64, rope 16, 4 mHC streams, Sinkhorn 20, window 64, index top-6 over 2 heads of 32,
candidate pool 3 blocks of 4, Engram at layers 1 and 3 (2 heads, n-grams up to 3, tables of 88 and
152 rows, compressed vocabulary 40), routed experts top-2 of 6 (DSpark blocks top-1 of 4, which
the vendor does not renormalize), swiglu limit 1.5, YaRN factor 4 over 32 positions.

| layer | CSA2 mode | ratio | notes |
|---|---|---|---|
| 0 | Window | 0 | |
| 1 | Full | 2 | Engram |
| 2 | Reuse (1, 1) | 2 | |
| 3 | Full | 2 | Engram; second ratio-2 group |
| 4 | Reuse (3, 3) | 2 | |
| 5 | Full | 1 | produces the candidate pool |
| 6 | Reuse (5, 5) | 1 | |
| 7 | Reindex (5) | 1 | restricted to the pool; DSpark target |
| 8 | Reuse (5, 7) | 1 | DSpark target |
| 9 | Reindex (5) | 1 | restricted; DSpark target |
| 10, 11 | DSpark | 0 | |

Tokens: 81, prefilled 73 then 8 decode steps, 7 drafts of 3. The prefill crosses the window by 9;
ratio-2 layers see up to 36 compressed entries against top-6, ratio-1 layers up to 73 against a
12-entry pool; the odd prefill leaves a ratio-2 group for the first decode step to complete;
window 64 plus 6 picks puts every query across the kernel's 64-entry block boundary. Weights are
seeded per tensor name and stored the way the checkpoint stores them (FP8 32x32 and row-32 with
e8m0 scales, FP4 packed with e8m0 scales, BF16, F32); the reference reads them back through the
same dequantization a checkpoint goes through. `tiny/hashes.tsv` holds the checkpoint image
hashes.

## Oracle

`tiny/make_tiny_oracle.py` runs the vendor `inference/model.py` and `engram.py` at the pinned
revision (sha256-checked, no line of them edited) on CPU over `OUT/tiny-checkpoint/` (which
`verify tiny` writes: `model.safetensors`, the vendor `model-args.json` the plan corresponds to,
and `fixture.json` with the tokens, the split and the compressed token map). The vendor code
does not run alone: everything it calls outside torch, and three things inside it, are
memra-written torch stand-ins:

- the tilelang kernels `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn` and
  `hc_split_sinkhorn` (`oracle/kernel.py`, which shadows the vendor's `kernel` module; the
  sparse-attention stand-in walks the 64-entry blocks the kernel walks);
- torch's GEMMs, reductions and transcendental functions, evaluated in f64 and rounded once
  (`oracle/exact.py`);
- `Tensor.topk`, replaced by the CUDA tie rule;
- the Engram fp8 gather (`ParallelEngramEmbedding.forward`, replaced by a uint8-view gather with
  the vendor's dequantization, since CPU `F.embedding` has no fp8 kernel);
- the logits of every position but the last, taken as the final norm output (a forward hook)
  times the head, the product the vendor head computes for the last one;
- a stand-in tokenizer the vendor `build_compressed_token_map` turns into the fixture's map.

`oracle/test_kernel.py` checks the kernel stand-ins against themselves only (the blocked
attention against its direct form). The independent review of PR #1128 read them side by side
against the tilelang source and found no semantic difference (act_quant floor and power-of-two
ceiling, both FP4 scale paths, GEMM scale grids and nibble order, sparse_attn sink and masking,
Sinkhorn eps and iteration order); the stand-ins are still memra's code, not the vendor's.

Two free choices of the vendor program had to be pinned, and both are pinned the same way on
both sides:

- Primitive rounding. The first run, with torch CPU's own GEMMs and transcendental functions,
  matched memra exactly at positions 0..11 and then drifted: max_abs 0.423 on the main logits,
  0.173 on the drafts, 2 argmax flips. A one-ulp f32 difference flips a bf16 rounding, and the
  flip changes a top-k selection. `oracle/exact.py` evaluates every GEMM, einsum, sum and mean
  in f64 and rounds once to f32 (then to the op's output dtype), and every transcendental op in
  f64 rounded once; `dsv41_forward` evaluates its primitives the same way. With that, every row
  and every layer's residual stream at every position match to the bit.
- Ties. The index scores are bf16, so equal scores at a selection boundary are common: 110 of
  the 1,338 top-k rows of this run had one. torch's CUDA `topk` (the vendor runtime) takes ties
  in index order, so the lowest index wins; torch's CPU `topk` decides by libstdc++
  `nth_element`. The oracle's `topk` stand-in and memra both use the CUDA rule.
  `oracle/topk_ties.py` checks it at the vendor indexer's shapes, bf16 scores with eight distinct
  values, widths 64 to 262,144, k up to 512: on an RTX PRO 6000 Blackwell Max-Q (torch
  2.14.1+cu132) the lowest index wins in every row of every case
  (`oracle/receipts/topk_ties.cuda.json`); on CPU it does not below width 131,072
  (`oracle/receipts/topk_ties.cpu.json`). Run: `python3 oracle/topk_ties.py cuda` (or `cpu`).

Environment: Python 3.14.4, torch 2.15.0+cu132 on CPU, numpy 2.5.4, sympy 1.14.0, tokenizers
0.23.3. `oracle/test_kernel.py` prints `KERNEL-SELFTEST-OK` (it now also checks the multi-block
attention form).

```bash
memra model verify tiny --against deepseek_v41 --out OUT
uv run --with torch,numpy,sympy,tokenizers,pillow python3 -I \
  research/modelplan-onboarding-dsv41-20261109/tiny/make_tiny_oracle.py \
  OUT/tiny-checkpoint VENDOR/inference crates/memra-cli/tests/fixtures/tiny-oracle/deepseek_v41.tsv
# {"tokens": 81, "prefill": 73, "drafts": 7, ..., "topk_rows": 1338, "topk_rows_with_boundary_tie": 110,
#  "sha256": "8131074dfa4abae13d585307b5bd4366866e5dcfbb78707775670d521e4a52fe"}
memra model verify tiny --against deepseek_v41 --out OUT
# family=deepseek_v41 verified=Tiny
```

`tiny/tiny-gate.tsv` is the gate's own receipt (pinned oracle sha256 `8131074d...`), and
`tiny/generator-run.json` is the generator's output line.

Tolerance: the pack's `checkpoint_parity` is atol 1e-5, rtol 1e-5, argmax required. The tiny run
measures 0. The bound only absorbs a last-ulp libm difference between machines; anything that
moves a bf16, FP8 or FP4 rounding or a selection moves later rows far past it.

Unit tests beside it: the plan census equals the pack census; a prefill of every token and a
prefill-then-decode split agree bit for bit at splits at or past the window (64, 70, 73 and 80 at
window 64; 128, 130, 137 and 144 at the variant's window 128), and differ below it (step 2c);
confusion guards (zeroing an Engram table moves its layer and not the one before; lifting layer
7's candidate restriction moves layer 7 and not layer 6; DSpark weights move the drafts and never
the trunk); the PCG64 port reproduces the vendor multipliers; the prime layout reproduces the
1017 table rows.

## DeepSeek-V4 and V4.1 inspect after step 2

Re-inspecting the 0731 artifact `--against deepseek_v4_dspark` at this commit writes the same
`artifact.lock`, `model-plan.txt`, `tensor-census.tsv`, `execution-rewrites.tsv`, `gates.txt`
and refusal text as step 1. Its `tokenizer-contract.tsv` gains the five renderer lines of the
TokenizerTemplate check (step 2b; `encoding/README.md`, `encoding/inspect-dsv4-0731/`), the
step-1 lines unchanged. Re-inspecting the 1017 artifact writes the same plan, census, rewrites,
gates and normalized config; the lock gains `support=NativeReference` and the three
`checkpoint_*` tolerance lines, and the tokenizer contract the renderer lines.

## Checkpoint tape, for the checkpoint gate

`memra model verify checkpoint` reads `memra-checkpoint-oracle-v1`: `format`, `engine`,
`numeric_class`, `tokens` (comma-separated), `vocab`, then `logit\t<index>\t<f32 bits hex>` for
every vocabulary index, the full logits row after the prompt's last position. It runs the native
runner as `<runner> <checkpoint_dir> 1 2 3 4` with `MEMRA_ORACLE_OUT` and `MEMRA_FULL_PREC=1`,
and requires an equal `numeric_class`, equal tokens and vocab, every logit within
`atol + rtol * |reference|`, and the argmax when the pack asks for it. Only one case, on tokens
1,2,3,4, so the greedy ids, top-32 rows, later steps and hidden states of an oracle case have no
consumer.

- `oracle/oracle_to_checkpoint_tape.py CASE.pt OUT.tsv` converts one oracle case
  (`prompt_ids`, `full_logits[0]`) into that tape, numeric class
  `deepseek-v41-bf16-correctly-rounded` by default, and flags a case whose prompt is not
  1,2,3,4.
- `oracle/run_oracle.py` and `oracle/make_cases.py` are the lane's checkpoint oracle with
  `exact.py` and the CUDA `topk` rule installed (the manifest records both) and a `tape-1234`
  case with `prompt_ids` [1, 2, 3, 4].
- No V4.1 native checkpoint runner exists yet; it has to stream the 510 GB checkpoint (the
  reference holds dequantized f32) and write the same numeric class.

# Step 2c: review fixes (2026-11-09)

The independent review of PR #1128 (at `b5d18a0954`) found eight items. Each is fixed here; the
support state does not move.

## Tiny variants: splits below the window and a two-block window

The main fixture splits past the window (73 of 64), so it never decodes while the window ring
fills, never drafts from a short DSpark ring and never runs a window that spans two of the
sparse-attention kernel's 64-entry blocks. Below the window the vendor's prefill and decode are
different numeric programs for the same row: `get_window_topk_idxs` (`model.py`) lays a prefill
row out `min(seqlen, window)` wide with its empty slots last, a decode row as the ring oldest
first with its empty slots first, and `sparse_attn` rounds probabilities to bf16 per 64-entry
block, so where a row's valid entries fall in those blocks changes its numbers. The prefill
width also moves where the compressed entries start, so even rows a shorter prefill computes
differ.

`verify tiny` now runs two variants beside the main fixture, each pinned to an oracle the vendor
`model.py` produced on the variant's own checkpoint image (`tiny-checkpoint-<variant>/`), through
the same `make_tiny_oracle.py`, exact arithmetic and tie rule:

| variant | config | tokens | split | decode steps (ring filling) | drafts | rows | oracle sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| main | tiny, window 64 | 81 | 73 | 8 (0) | 7 | 6,549 | `8131074d` |
| `split60` | tiny, window 64 | 81 | 60 | 21 (4) | 20 | 9,084 | `c3bb258e` |
| `window128` | tiny at window 128 | 145 | 100 | 45 (28) | 44 | 17,860 | `8eff857f` |

memra matches all three to the bit; `tiny/tiny-gate.tsv` records the three pinned digests,
`tiny/hashes.tsv` every checkpoint file and oracle, `tiny/generator-run-<variant>.json` the
generator's output line. The window-128 config
(`crates/memra-gguf/src/model_packs/deepseek_v41/fixtures/tiny-config-window128.json`) is the tiny
config with `sliding_window` 128 and `max_position_embeddings` 512 and nothing else (a pack test
pins that). The reviewer's scratch run of the same two splits gave the same two digests.

```bash
memra model verify tiny --against deepseek_v41 --out OUT
for v in split60 window128; do
  uv run --with torch==2.15.0 --with numpy==2.5.4 --with sympy==1.14.0 --with tokenizers==0.23.3 \
    --with pillow python3 -I research/modelplan-onboarding-dsv41-20261109/tiny/make_tiny_oracle.py \
    OUT/tiny-checkpoint-$v VENDOR/inference crates/memra-cli/tests/fixtures/tiny-oracle/deepseek_v41-$v.tsv
done
```

The unit test that compared a full prefill with a prefill-then-decode split now claims only what
holds: at or past the window the two agree to the bit (splits 64, 70, 73, 80 at window 64; 128,
130, 137, 144 at window 128). A second test pins that below it they do not: split 60 at window 64
moves every row from row 0, splits 70 and 100 at window 128 every row from row 64.

Constraint for the serve route, from this: below the window, a V4.1 request split differently
(prime versus decode, other chunk lengths) runs a different numeric program for those rows.
Under "one numeric program per request" a serve route must fix the split it uses below the window
or prove the rows it emits equal, and no serve gate may cite the at-or-past-window test for a
split below it.

## The other fixes

- The reference reads the plan's numerics. Each activation, cache and table step
  (`linear_activation`, `window_kv_quant`, `output_quant`, `latent_quant`, `query_quant`,
  `key_quant`) is resolved from the plan as a `dsv41_forward::QuantStep` and applied as named; a
  form with no vendor kernel (FP8 with an e4m3 scale), a block that does not divide its row, a
  storage the loader does not decode (`dense_weight_block`, `expert_weight`, `table_quant`), a
  program switch it does not implement (`routed_weight_before_down`,
  `output_lora_a_dequantized_bf16`), a state plan that is not its attention's (`partial_group`)
  or a gated shared expert refuses by name before a tensor is read.
  `vendor_model_args` derives `dtype`, `expert_dtype` and `n_shared_experts` from the plan and
  refuses any step the vendor's module constants do not fix, so no oracle is exported for another
  program. Test: `the_plan_numerics_are_read_or_refused_by_name`.
- Tool namespaces on the V4.1 serve path (`encoding/README.md`): a request tool's tool-level
  namespace is folded into the function object the renderer reads, schemas key by `ns::name`,
  and a history call in the vendor parser's own shape (`namespace` beside `function`) re-renders
  as the vendor does. 13 namespaced cases are now on the wire (gate 3), vendor bytes. DSv4 0731
  reads neither field, as before (pinned).
- Gate 4 parses through the parser `build_chat_request` arms, and a test pins that V4.1 caps arm
  the V4.1 parser and DSv4 caps the DSv4 one. The TokenizerTemplate gate also requires every
  control string the probe render spells, and every special token its ids hold, to be a listed
  marker; the gate-2 test applies the same check to all 172 vendor renders. The task tokens are
  one table for the renderer and the marker list.
- `dspark_target_layer_ids` must be strictly ascending (the vendor concatenates the taps in layer
  order). `dsv41_forward::e4m3` decodes 0x7f and 0xff as NaN, as torch `float8_e4m3fn` does, so a
  NaN weight or Engram byte reaches the loader's non-finite refusal instead of loading as zero.
- Wording: the stand-ins are named wherever a record said the vendor `model.py` ran unchanged;
  the CUDA `topk` tie check and its receipts are in `oracle/`; the 0731 tokenizer contract is
  called out as gaining the renderer lines; the three `test_encoding.py` cases with no fixture
  are listed as divergences (`encoding/README.md`).
- `docs/models/deepseek-v41-flash.md` is the model card; `docs/MODELS.md` links it.
- Red arms on the serve path (`step2c-red-arms.txt`, each reverting one fix in a scratch copy):
  swapping the request builder's V4.1 and DSv4 parser arms fails gate 4, the namespaced agent
  test and the caps test; dropping the tool-level namespace fold fails gate 3 and the
  same-named-tools test; dropping the history call namespace fails gate 3 and the namespaced
  agent test.

Not changed: adding `ModelConfig::dsv41` moves `config_sha256` (the `Debug` text of the
normalized config) for every GGUF and Hy3-repack source on its next inspect, as earlier family
fields did (documented in `docs/ONBOARDING.md`); HF-source locks do not move.

Re-inspect at step 2c (`step2c-reinspect.tsv`, sha256 of every receipt against the step-2b
inspect): the 1017 artifact writes the same plan (`plan_sha256` `c4d9ad3b`), census, rewrites,
gates, placements, normalized config and tokenizer contract (the probe passes the new
control-string and special-token checks), and the lock differs from step 2b's only by the step-2
lines; the 0731 artifact `--against deepseek_v4_dspark` writes every receipt byte for byte as
step 2b (lock `000ce525`, the same as step 1).

# Step 3 (GPU): single-card kernels and checkpoint layers (2026-11-09)

Single-card kernels for every V4.1 operation and real-checkpoint layers on one card, then (step 4) the whole trunk on one card as the native checkpoint runner, CheckpointParity on a pack-declared probe, and long-context gates to 257k tokens: `gpu/README.md`.

# Step 5 (GPU, N ranks): the TP/EP partition, gated at N=2 (2026-11-09)

The V4.1 layer split across N ranks (heads, output groups, `wo_b` rows, routed experts, head rows;
gathers only, so the N-rank program writes the one-card bytes): byte-identical to the one-card
`block_forward` on six real layers at prefill and decode on two cards, the whole trunk on two cards
gives the one-card logits bit for bit up to 128,745 tokens, the indexer splits its query rows at
prefill (measured against a head split), and the list of what the 4-card stage adds: `tp/README.md`.

# Step 6 (GPU): the decode loop, gated against the vendor's generations (2026-11-09)

A prefill, then one token per call with every sequence's window rings, compressor groups,
compressed KV, index keys and token history carried as the vendor carries them (`dsv41_gpu::decode`,
`dsv41-run --force / --gen`), on one rank or N. Teacher-forced with the vendor's greedy tokens it sits
inside the vendor program's own decode floor on 711 steps of 17 tapes (16 to 128,745 tokens of
context); greedy generation, the identity gates (layer-by-layer replay against token-by-token
generation, N=2 against N=1) and what serving still needs: `decode/README.md`.

# Step 7 (GPU): the serving route on one card (2026-11-09)

memra-server's deepseek-v4.1 route over the decode loop, every layer's own weights on the card and the
routed experts in pinned host memory behind a device LRU cache that writes the streamed runner's bytes;
CheckpointParity again on that residency, `memra model verify serve` passing, served greedy answers
equal to the native runner's, batched equal to solo, the decode rate, and the support record that makes
the pack NativeQualified on this topology: `serve/README.md`.
