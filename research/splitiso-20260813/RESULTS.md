# Split-boundary divergence isolation — results

Date: 2026-08-13
Verdict: **BOUNDARY-IDENTIFIED**

The bounded prefix restore is byte-correct. The divergence is not an alignment, tiling, page, or
split-position boundary. It is the numerical-program boundary that the worker names
`eager_mono && carried`: a genuinely cold Gemma request executes one monolithic
`gemma4_prime`, while a partial-prefix hit must execute every suffix token through T=1
`decode_step`. The prime attends over transient pre-quantized/bf16 Q/K/V operands even though it
also appends quantized K/V rows; decode attends through the quantized cache. Those transient prime
operands and that arithmetic program are not part of the prefix-cache object.

In the restored-versus-full-cold comparison, the first captured persistent model-state witness is
`kv.layer.1.first_suffix_k_sha256` (layer 1, the first suffix token) at every detailed split.
Layer 0's quantized first-suffix K and V rows still match. The first uncaptured numeric difference
is therefore inside layer 0's prime-versus-decode attention block (Q, pre-quantized K/V, or the
attention result); it propagates into layer 1 K/V. Boundary-logit vectors differ at every detailed
split, including positions whose 60-token greedy output happens to match.
Output failures occur exactly where that always-present cold-prime versus restored-decode numeric
perturbation crosses a greedy argmax during the completion. The failure positions are therefore
content-sensitive islands, not a geometric threshold.

No eligibility-gate, suffix-prime, runtime-default, or generated-board change was made. Partial
restore should remain default OFF.

## Frozen cell and provenance

- Branch: `lane/cx-splitiso`, based on `v0.81.3` (`7cf5fd842`), with the committed lcprestore
  harness/receipt and its default-off follow-up imported unchanged before lane instrumentation.
- Rig: box1 physical GPU 1, NVIDIA RTX PRO 6000 Blackwell Server Edition,
  `GPU-2b4cf166-fd33-f161-8536-ca04bc72280c`.
- Every accepted box1 device cell acquired `/tmp/memra-gpu.lock` and `/tmp/memra-gpu-1.lock` non-blocking,
  required `compute_apps=none`, pinned `CUDA_VISIBLE_DEVICES=1`, and stopped its server before
  parsing. Cleanup receipts contain no compute application.
- Server binary: SHA-256
  `fc94f06645cebcc483dc32b4dfc7a3f65050ca7c9d6f74a88d847fc121e8de95`, native sm_120a build
  from `4fcd4bd1b`. There is no engine/server source diff between that build source and the later
  map-harness source `ebee2a2c0`.
- Model: `gemma-4-12b-it-qat-q4_0.gguf`, SHA-256
  `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
- Workload: 4,860 prompt tokens, 60-token generation cap, SHA-256
  `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34`.
- Mechanism was explicitly armed with `MEMRA_PREFIX_PARTIAL_RESTORE=1` and
  `MEMRA_PREFIX_SPLIT_TRACE=1`; detailed cells additionally used the opt-in detail trace.
- These are exactness cells only. No score, throughput, or latency claim is made.

## Original lcprestore receipt reproduced

The first box1 cell reproduced the committed four-point output hashes exactly. Source-entry state
and immediate restored state also matched at all four splits.

| Split | Restored output SHA-256 | Genuinely cold SHA-256 | Output |
|---:|---|---|:---:|
| 64 | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | PASS |
| 512 | `bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525` | `719a43f41b407364130580b2f12a8c09e78da460dc25ada2f1781dd436780079` | **FAIL** |
| 2048 | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | `223618bfd84e4f30bb454fb7383f139753011e918926af620cf047dda7c136c2` | **FAIL** |
| 4374 | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | PASS |

Evidence: `raw/box1-smoke-original-four/`.

## Boundary maps

### Frozen lcprestore constructor

The required 69-point dense sweep (every 64 tokens from 64 through 4,352, plus 4,374) is **51
PASS / 18 FAIL**. All 69 source→restored state receipts match. The full per-position output-hash
table is in `DENSE-MAP.md` / `DENSE-MAP.json`.

| Dense failure positions |
|---|
| 448, 512, 576, 640, 1024, 1280, 1408, 1600, 1664, 1728, 1792, 2048, 2752, 2816, 2944, 3904, 4096, 4224 |

Targeting every sampled transition plus the named boundaries expanded this to **116 cells, 80
PASS / 36 FAIL**, again with 116/116 source→restored matches and no infrastructure failure. The
complete table is `FROZEN-TARGETED-MAP.md` / `.json`.

This map is a reproduction table, not a valid split-only causal sweep. The frozen constructor says:

```python
prefix = fixed_prompt_ids(split, 370)
prompt_a = prefix + fixed_prompt_ids(total - split, 407)
prompt_b = prefix + fixed_prompt_ids(total - split, 444)
```

Changing `split` therefore changes request B's token sequence as well as the boundary. The 47 new
targeted positions have 47 distinct request-B hashes; the seven older dense segment summaries
predate prompt hashing. Adjacent PASS/FAIL flips in this table cannot identify geometry while their
target content also changes.

### Fixed-target causal control

Map-only control construction held request B byte-identical at every position, with canonical token
hash `21ef4227fcb0993c341e03c4df6bf01b27f6012021c881fc4a8f451364495397`, and changed only request A
after the requested exact LCP. Gate mode and the frozen default constructor were left unchanged.

The controlled map is **106 cells, 92 PASS / 14 FAIL**, with 106/106 source→restored state matches,
one target hash, no missing named-boundary cells, and no infrastructure failure. The full pass/fail
and output-hash table is `FIXED-TARGET-MAP.md` / `.json`.

| Controlled failure islands at one-token resolution |
|---|
| 64; 384; 1472; 1727–1728; 1791; 1856; 1919; 2304–2305; 3136–3137; 3775–3776 |

The fixed input reverses the original four-point story: split 64 FAILS, while 512, 2048, and 4374
all PASS. Detailed logits still differ at all four. This is direct evidence that final output
equality is an argmax-basin observation, not state equality.

## Correlation against named boundaries

The controlled table has no exact discriminator among the requested geometry fields
(`exact_discriminators: []`). Counts below are PASS/FAIL.

| Candidate | Source contract | Controlled result | Finding |
|---|---|---:|---|
| Prefix eligibility | `const PREFIX_CACHE_MIN_TOKENS: usize = 64;` | eligible 92/14 | All cells are eligible; failures recur far above 64. Split 64 F -> 65 P is not an eligibility transition because both are eligible. |
| Worker prefill chunk | `PREFILL_TICK_T = 1024`, `SOLO_PREFILL_TICK_T = 8192`, but Gemma takes `if eager_mono { q }` | monolithic T=4860 92/14 | The cold target's actual chunk is invariant; restored suffixes do not enter chunk prime. |
| Cached/chunk FA tile | generic quantized-view path: `const BLOCK_Q: usize = 64; const BK: usize = 32;` | not reached | `eager_mono && carried` excludes Gemma continuation prime, so this tile cannot select a split outcome. |
| Cold Gemma FA tiles | SWA `BLOCK_Q=64/BK=32` (paired arm 32/32); global single-pass `SP_M=16/BKS=32` (fallback 32/32) | invariant T=4860 92/14 | Split does not select the cold tile because the same full prompt is primed each time. |
| Global first-suffix arm | `kvl.len >= fa512_min_tkv()`, default 512 | scalar-sp16 12/2; rows-sp32 80/12 | Both arms contain both outcomes. Exact split bracket 510/511/512 is P/P/P; the post-append switch occurs from split 510 to 511. |
| SWA first-suffix arm | `kvl.len > win`, `win=1024` | kvmod-sp16 23/2; rows_w-sp64 69/12 | Both arms contain both outcomes. Splits 1023/1024/1025 are P/P/P. |
| Generic 188-SM ladder | `if t_kv <= 2048 { 16 } else ... { 64 }` | rung 16: 47/8; rung 64: 45/6 | Both rungs contain both outcomes; splits 2047/2048/2049 are P/P/P. The ladder is not live there because Gemma rows/rows_w have already taken over. |
| Decode batch width/row | detailed worker receipt | eager B1, width/row null: 92/14 | No width or row transition exists in this single-session lane. |
| KV plane stride/page | live rows: global 512 B/token, SWA 2048 B/token; allocation is `rows * token_bytes + 8` | global offset 0: 60/8, 512: 16/2, 3584: 14/4; SWA offset 0: 62/8, 2048: 30/6 | Every populated offset class contains PASS; the main offsets contain both outcomes. |
| Alignment residues | split modulo 64/32/16 | residue 0: 60/8; 1: 16/2; 63: 14/4 | PASS and FAIL occur at the same residues. |

The opening alignment/tiling hypothesis is therefore falsified. The named boundary is instead the
worker's program fork in `crates/memra-server/src/worker.rs`:

```rust
let eager_mono = eager_only_model(lm);
let carried = s.cache.as_ref().is_some_and(|c| c.pos > 0);
// ...
&& !(eager_mono && carried)
// ...
let mut take = if eager_mono { q } else { q.min(budget) };
```

The adjacent comment states the contract directly: fresh Gemma prompts prime whole, while carried
suffixes “ride the tokenwise `decode_step` path.” The engine independently rejects continuation
prime when `cache.pos != 0` with:

```text
gemma4 prime v0 is fresh-prompt only (no continuation/chunked prime) — prime the full prompt in one call or decode tokenwise
```

## First differing field

The detailed restored-vs-cold comparison was run at splits 64, 512, 2048, and 4374. The same field
matrix occurred at all four positions.

| Captured state at the full 4,860-token boundary | Restored partial hit | Genuinely cold | Result |
|---|---|---|:---:|
| Source prefix state vs immediate restore | independently hashed | independently hashed | equal at every split |
| Retained-prefix logical K/V | per-layer hashes | per-layer hashes | equal |
| Layer 0 first-suffix K/V | per-row hashes | per-row hashes | equal |
| **Layer 1 first-suffix K** | `kv.layer.1.first_suffix_k_sha256` | same field | **first logical model-state difference** |
| Whole suffix K/V aggregates | 48 layers | 48 layers | different |
| `kv.layer.0.len_d` device mirror | `[4860]` | `[0]` immediately after prime | different, reported separately |
| Canonical conv/SSM and spare SSM | absent (transformer checkpoint) | absent | equal |
| Cache position / next RoPE position | 4860 / 4860 | 4860 / 4860 | equal |
| Sampler | seed 3407, RNG `9e3779b97f4a8964`, prompt history hash | same within each split | equal |
| Per-Engine scoped flags | `capture_keep_on=false`, `verify_exact=false` | same | equal |
| Logits producer | `decode-step-prefill` | `prime-cache` | **different program** |
| Decode batch width / row | `null` / `null` | `null` / `null` | equal |
| Boundary logit vector | 262,144-value f32 hash | 262,144-value f32 hash | different |
| Selected first generated token | token id | token id | equal |

`len_d` is the earliest separately captured representation difference, but it is not promoted as
the cause. Cold `gemma4_prime` has already produced its boundary logits without consuming
`len_d`; eager rows decode explicitly writes the host logical length into `len_d` before it reads
the counter. The first persistent logical model-state difference is the layer 1 suffix row above.

A second same-position probe compared the retained source slice (computed inside the 4,860-token
prime) with a cold prime ending exactly at the split. Its first difference is
`kv.layer.0.k_sha256` at all four detailed splits. That shows even causal prefix rows can depend on
the prefill execution shape; it is mechanism evidence, not the output discriminator, because it
also occurs at both original PASS positions.

Compact field evidence: `FIELD-COMPARISON.json`. Full detailed reductions and tee'd server logs:
`raw/box1-smoke-original-four/` and `raw/fixed-pilot/`.

## Mechanism

The code and field receipts form one consistent chain:

1. Fresh Gemma runs `gemma4_prime` over all 4,860 tokens. The function is explicitly fresh-only.
2. `gemma4_attn_prime` appends post-RoPE K and normalized V to the quantized cache, but then computes
   prompt attention from the transient Q/K/V operands passed directly to `fa_prefill*` / SDPA.
3. A partial hit restores the quantized prefix rows exactly. Because its cache position is nonzero,
   the worker routes every suffix token through `decode_step`.
4. `gemma4_decode_attn` appends the current quantized K/V row and attends by viewing the quantized
   `kvl.k` / `kvl.v` planes through `fa_decode_rows`, `fa_decode_rows_w`, or `fa_decode_kvmod`.
5. At the first suffix token, the quantized layer 0 K/V rows match. That does not establish equality
   of transient Q/K/V before row quantization. The first uncaptured difference is bounded to the
   layer 0 prime-versus-decode attention block; layer 1 K/V are the first persisted witness, and
   boundary logits subsequently differ.
6. The perturbation is present at PASS and FAIL positions. A completion hash changes only when a
   later greedy top-1 choice changes, which explains narrow, non-monotonic, content-sensitive
   islands without invoking an alignment defect.

Step 5's layer-0 block boundary is an inference from the equal quantized layer-0 row, differing
layer-1 row, and the two source call paths. The lane did not copy transient Q/K/V or attention
output to host, so it does not claim which one is the first floating-point value to differ.

## Relation to cx-eosclass

This result does not close the Q27 11-token EOS class. cx-eosclass isolated that failure to an
eager-B1 → generic-batched width transition; its terminal EOS id 248046 was rank 1 and its
one-program B=1/B>=2 default passed the controlled sweep. Here, every detailed sample has batch
width/row `null` and the program fork is monolithic prime versus tokenwise decode. The two lanes
share the general risk of switching floating-point programs, but restore is not the Q27
discriminant and no EOS work was duplicated here.

## Fix direction — unshipped

There is no small safe diff to propose. Exact resumption requires a canonical Gemma numerical
contract on both sides of the cache boundary, for example:

- make cold prime attend through a canonical quantized-cache view whose per-token results are
  proven bit-identical to resumed decode; or
- persist enough higher-precision prime attention state to reproduce the transient program.

The first is an engine/kernel project: merely selecting an existing chunked FA kernel does not
guarantee the same reduction order as T=1 decode. The second materially changes cache size and
format. Routing every cold prompt tokenwise would be a useful slow oracle but an unacceptable
serving default. None of these was implemented in this isolation lane.

## Gate ledger

| Gate | Result |
|---|---|
| Original lcprestore 64/512/2048/4374 output receipt | reproduced byte-for-byte |
| Frozen dense map | COMPLETE: 69 cells, 51 PASS / 18 FAIL |
| Frozen targeted map | COMPLETE: 116 cells, 80 PASS / 36 FAIL |
| Fixed-target controlled map | COMPLETE: 106 cells, 92 PASS / 14 FAIL, one prompt hash |
| Source → restored state | 69/69, 116/116, and 106/106 verified; no mismatch |
| Detailed field reducer | COMPLETE at 64/512/2048/4374; no reducer failure |
| Infrastructure failures / fatal CUDA markers | none in accepted cells |
| Python byte-compilation, ShellCheck, `git diff --check` | PASS |
| `DOCS_RS=1 TMPDIR=/home/avifenesh/tmp-lanes cargo check -p memra-server` | PASS; logs in `raw/build-checkpoint1/` and `raw/build-checkpoint2/` |
| `kernel-check`, `run-gen`, `run-spec` | NOT RUN — no runtime fix or shipping candidate exists in this isolation lane |
| Scored/timed cells | NOT RUN |
| Eligibility/suffix-prime/default change | NOT MADE |
| Merge/tag/push/perf-board edit | NOT DONE |

One dense seed request at split 448 ended normally after two generated tokens with HTTP 200,
SSE completion, and `finish_reason=stop`. The frozen scored helper required exactly 60 tokens and
therefore rejected that seed row. Map mode retained the raw receipt and narrowly classified normal
EOS as model output; the independent source→restore verifier passed the cell. No conclusion treats
that normal EOS as a runtime failure.

## Evidence index

- `DENSE-MAP.md` / `.json`: required 69-point frozen dense table and correlations.
- `FROZEN-TARGETED-MAP.md` / `.json`: 116-point frozen reproduction table.
- `FIXED-TARGET-MAP.md` / `.json`: 106-point single-target causal table and correlations.
- `FIELD-COMPARISON.json`: compact restored/cold field matrix.
- `raw/box1-build/`: build source, full build log, and server hash.
- `raw/box1-smoke-original-four/`: detailed byte-for-byte reproduction and state/logit receipts.
- `raw/dense-seg*/`, `raw/frozen-targeted-seg*/`: frozen-request raw cells.
- `raw/fixed-dense-seg*/`, `raw/fixed-targeted-seg*/`, `raw/fixed-named-global510/`: controlled raw
  cells.
- `raw/*-reduce.log`: tee'd reductions; every reducer parsed logs only after raw capture.

No live serve host was touched.
