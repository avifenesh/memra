# Models and hardware

What runs, on which path, on which card. The README links to one concise
[card per model](models/); this file remains the complete support matrix and reasoning.

The generated support table at the bottom is written by `tools/update-perf-board.py` from
`research/tune-data/current-board.json` — edit the board, not this file.

---

Support is specific to a model, quantization, and drafter combination — never to a format. A
checkpoint loading is not the same thing as a model being supported: every family has its own
tensor census, quantization arithmetic and topology, and it counts as supported here only once
it has passed its own exactness gates. The generated table below is numbers-free; per-model
measurements stay in [docs/PERFORMANCE.md](PERFORMANCE.md).

### Direction: safetensors first, GGUF still supported

For a new upstream model the official safetensors checkpoint — with its config,
tokenizer/template, quantization metadata and auxiliary tensor files — is the preferred semantic
source. memra repacks those tensors once into measured rig-native layouts; safetensors is not the
internal compute format. GGUF remains a fully supported, self-contained portable import and
distribution format.

**safetensors is the tuned path from here on.** New work goes there — NVFP4 and FP8 trunks,
with load-time head trimming instead of a separate pre-trimmed draft file.

**GGUF stays supported.** Many models are served through it today and those paths are not going
anywhere. What is *not* offered is a promise that a format keeps getting new work: a GGUF that
lands after this line was written may or may not be picked up, and that is a decision each
time, not a policy. The reverse holds too — a safetensors checkpoint can load without being on
the tuned path.

Where each family actually sits right now:

Supported on both paths means both, and where the tuning is happening is a separate
question from what is supported:

| Family | Supported on | Tuning now |
|---|---|---|
| Qwen3.8-27B | **both**, both tuned — safetensors NVFP4 and NVFP4+Q5_K GGUF | done. Performance defaults stay artifact-specific |
| Gemma-4 31B | **GGUF** (QAT Q4_0), supported and tuned | NVFP4 safetensors |
| Step-3.7-Flash 196B-A11B | **GGUF** (IQ4_XS + Q8_0 MTP head, two-card PP-2) since v0.73.1 | FP8 — not Q8 |
| Gemma, rest of the family | GGUF | NVFP4 safetensors rework in progress |
| DeepSeek-V4-Flash | **safetensors** checkpoint dir through its own two-card door — **experimental engine support**, functional and gated, **not serving-grade** (see its section below) | performance ([#32](https://github.com/avifenesh/memra/issues/32)) |
| GLM-5.3-Flash | **safetensors** (FP8 e4m3, MIT) on the hand-written `glm5_next` path — **NativeReference**; multi-card TP serving, vision, and MTP spec gated (see [its card](models/glm53-flash.md)) | — |
| Qwen3.8-Flash-Next | **bring-up only** — hand-written `qwen4_exp` gate path on the minted NVFP4 artifact, real-checkpoint eager gate green; ModelPlan loader not wired, no serving surface (see [its card](models/qwen38-flash-next.md)) | — |
| Hy3 | **safetensors** — canonical BF16 plan **NativeReference**; the exact all-expert ModelOpt W4A16 artifact is **NativeQualified** on four-card Blackwell receipts (see [its card](models/hy3.md)) | — |
| everything else in the table | GGUF | — |

Nothing above is a roadmap. It is where the code is, and it changes by decision. A row moving
from one path to another does not retire the old one: when a model is supported on both, both
keep working.

### In progress

Tensor parallel, P2P and 3-stage pipeline parallel are being built now and are close, which is
exactly why they are named here as unfinished rather than listed as features. When each one has
its gates it moves into the table.

### Which models should be next?

This is an engine anyone can run, and the support list is a series of decisions rather than a
plan — so the most useful thing a reader can send is which model they want served, and on what
card. Open an issue or a discussion. Requests with a concrete checkpoint and a reason carry more
weight than a wishlist, and they are read.

- **RTX PRO 6000 Blackwell (`sm_120a`) — co-primary target.** Workstation and Server Edition, 96 GB.
  Carries verification, final tuning, and serving, single-card and as PP-2 pairs.
- **RTX 5090 / 50-series (`sm_120a`) — co-primary target.** Same architecture, its own measured
  settings, and its own defaults where they differ. Local 5090 performance is never traded away to
  simplify a remote default; a perf claim needs numbers from both cards before it sets a global one,
  and one-card evidence sets a one-card default at most.
- **Hopper `sm_90a`** — separately compile-gated H100 source-build lane with its evidence
  ledger in [ARCHITECTURE-H100.md](../ARCHITECTURE-H100.md). Secondary: it does not change the
  naked `sm_120a` build or its defaults. Its standing battery was retired with the Hopper CI
  lane (2026-09-02); run the gates directly on Hopper hardware.
- **Ada `sm_89`** — portable source-build lane, not a tuned performance target.
- **B200 `sm_100a`** — runtime-qualified source backend with automatic source-build detection;
  no prebuilt is published. Pinned Qwen3.5-9B NVFP4 is `NativeQualified` on default W4A8.
  Opt-in W4A4 is correct but 0.521x raw W4A8 prefill. Pinned Qwen3.8-27B block-FP8 is
  `NativeReference`; its explicit FP8-MMQ twin is correct but 0.173x the fallback, so it is not
  tuned or default. See [the B200 card](rigs/b200.md).
  Other architectures are not a tuned support promise.

<!-- PERF-MODELS:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
| Model | Class | Quant | Drafter | Supported since |
|---|---|---|---|---|
| Qwen3.5-9B | dense | NVFP4 (5090), Q8_0 (H100) | MTP + own-gen trimmed draft | v0.1.0 |
| Qwen3.8-27B | dense hybrid (GDN + gated attention) | both paths tuned: safetensors NVFP4 · NVFP4+Q5_K GGUF; performance receipts are not interchangeable | DFlash2 block-diffusion drafter (MEMRA_DSPARK_SPEC=1, q4 default + FR-Spec trim — HF Avifenesh/Qwen3.8-27B-DFlash2-memra; the qualified serving route since v0.113.0, beats the MTP arm on every rung of the vendor sampled shape) · MTP + own-gen FR-Spec masked ranks (safetensors: MEMRA_FRSPEC_TRIM; GGUF: pre-trimmed head — HF Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF) | v0.82.2 (MTP) / v0.113.0 (DFlash2 serving default) |
| Qwen3.6-27B | dense | NVFP4, Q4_K_M MTP-baked | MTP + own-gen trimmed draft | v0.1.0 |
| Qwen3.6-35B-A3B | MoE | IQ4_XS | MTP + own-gen trimmed draft | v0.1.0 |
| Gemma-4 26B-A4B | MoE | QAT Q4_0 | Gemma assistant draft — served since 2026-08-17 via the gspec attach (MEMRA_DRAFT, shipping depth K=5); table spec numbers remain gemma-gate CLI measurements | v0.23.0 |
| Gemma-4 31B | dense | QAT Q4_0 GGUF (supported + tuned) · NVFP4 safetensors tuning in progress | Gemma assistant draft — served since 2026-08-17 via the gspec attach (MEMRA_DRAFT, shipping depth K=5); table spec numbers remain gemma-gate CLI measurements | v0.35.0 |
| Gemma-4 E4B | dense | QAT Q4_0 | Gemma assistant draft — served since 2026-08-17 via the gspec attach (MEMRA_DRAFT, shipping depth K=5); table spec numbers remain gemma-gate CLI measurements | v0.35.0 |
| Gemma-4 12B | dense | QAT Q4_0 | Gemma assistant draft — served since 2026-08-17 via the gspec attach (MEMRA_DRAFT, shipping depth K=5); table spec numbers remain gemma-gate CLI measurements | v0.40.0 |
| Ornith-1.0-9B | dense | Q8_0 | own-gen donor-block draft | v0.63.0 |
| Ornith-1.0-35B | MoE | Q4_K_M | own-gen donor-block draft | v0.64.0 |
| Ornith-1.5-35B-A3B | MoE hybrid (GDN + gated attention) | NVFP4+Q5_K GGUF (own-published: HF Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF, first NVFP4 of this model) · image input via the checkpoint ViT (MEMRA_VISION_DIR, parity-gated) | continued-trained MTP head + FR-Spec self-trim (MEMRA_FRSPEC_TRIM; v0.105 serves cached-long at K=5 with a trimmed head, v0.108 replays the verify trunk as a captured graph by default — +19.7% on a current-generation host, and much less host-CPU-sensitive than before) — artifacts on HF | v0.97.0 |
| Qwen-AgentWorld-35B-A3B | MoE | UD-IQ4_XS (avoid UD-Q4_K_M — its Q5_K expert mix sits outside fast-path coverage) | own-gen drafter | v0.66.0 |
| Step-3.7-Flash 196B-A11B | MoE | IQ4_XS + Q8_0 MTP head (two-card PP-2), supported · FP8 tuning in progress · image input via the checkpoint perception_encoder ViT (MEMRA_STEP_VISION_DIR, parity- and serve-gated, default off) | MTP (single-card); plain batched decode on PP-2 | v0.73.1 |
<!-- PERF-MODELS:END -->

Step-3.7-Flash serves on a two-card PP-2 pair (`MEMRA_PP_STAGES=2`); its explicit
configuration and qualification boundaries are recorded under
[bring-up notes](PERFORMANCE.md#bring-up-notes). On PP-2 it decodes in a single
numeric class at every batch width — served bytes do not depend on load history.

Since lane/step37-vision-20260830: **image input** on the same endpoint (OpenAI
`image_url` content parts, base64 data URIs; no video for this family). The
checkpoint's own perception_encoder tower (47 blocks, unquantized BF16 inside the
NVFP4 artifact) runs in-engine behind `MEMRA_STEP_VISION_DIR` (default OFF; see
docs/FLAGS.md), with the vendor's exact tiling law (728 main view + 504 crops,
crops-first token layout, 169/81 tokens per view). Gated before any serving path:
per-token cosine parity min-cos 1.000000 on the projected rows vs BOTH an
independent NumPy reference and the checkpoint's own torch code, on both grids;
end-to-end can't-hallucinate probes through the real endpoint (single, tiled,
multi-image, mid-conversation) with exact prompt-token accounting (171 per plain
image, 670 tiled); text requests through the vision-enabled binary stay
byte-identical to the seam-off boot and keep MTP spec engaged, while vision
sessions admit at K=0 (plain path) by the family-agnostic admission law. Vision
tokens bill as ordinary prompt tokens. Receipts:
`research/step37-vision-20260830/`.

Since lane/step37-postthink-grammar: **structured output** (`response_format`
`json_object` / `json_schema`) on the same endpoint, POST-THINK. This family's
template force-opens a think channel with no `enable_thinking` switch, so the old
behavior was a named 400; constrained requests now serve two-phase: the think
phase runs unconstrained exactly as the model was trained (every end-of-generation
id banned, so the response cannot end inside think: the receipted
EOS-inside-think empty-content quirk is closed for constrained requests), and the
llguidance grammar clamps every token from the tokenizer's atomic `</think>` close
token (id 128799) on. `reasoning` carries the think text; `content` is the
grammar-conformant JSON. Constrained sessions on this family take plain decode
(MTP spec disengages, `[spec-k]` K=0 admit receipt); `MEMRA_POSTTHINK_CEILING`
(default off) is the forced-close guard.

The failure face is FAIL-CLOSED, never a success a client can mistake: if
generation ends inside the reasoning channel before the think-close token (the
model reasons at length; measured natural closes on real agentic prompts run
p50 2119 / p90 3554 think tokens, so small `max_tokens` values get here), the
request returns a named 400 `invalid_request_error` (param `max_tokens`, or
`stop` when a stop sequence cut the reasoning) instead of a 200 with empty
content; a stream that already delivered reasoning deltas ends with the same
error object. `finish_reason: length` still occurs only AFTER the grammar
engaged (truncated JSON with non-empty content, the same face every constrained
model has). Raise `max_tokens` (stream for large budgets) or set the ceiling.
Receipts: `research/step37-postthink-grammar-20260830/`.

Since lane/step37-draft-graph-serving-20260830: the speculative DRAFT chain is
CUDA-graph captured on the qualified serving shape itself — the 3-head step-modulo
prefix-replay chain captures per-head single-row graphs (greedy AND sampled), and
truncation-filtered sampling (the vendor default temp 0.5 / top_p 0.9) runs its
filter in-graph, so vendor-default requests launch a captured chain. Gated by
greedy byte identity K=1..8, per-K acceptance identity, seeded sampled twins
(byte-identical streams graph-vs-eager), spec-on == spec-off serving bytes, and a
vision no-interaction cell; doors `MEMRA_MTP_CHAIN_GRAPH` /
`MEMRA_SPEC_GRAPH_FILTERED` / `MEMRA_STEP35_DRAFT_DCW` all default ON with `=0`
rollbacks (docs/FLAGS.md). Receipts:
`research/step37-draft-graph-serving-20260830/`.

The Gemma-4 drafter is a separate-assistant format, not a NextN/MTP head — the generic NextN
loader refuses it (`draft n_embd != model n_embd`). Since 2026-08-17, `memra-server` serves it
through its own attach path (`gspec`): `MEMRA_DRAFT=<assistant.gguf>` with `MEMRA_GEMMA4_SPEC`
unset arms served speculative decode at the shipping depth K=5 (`MEMRA_GEMMA4_SPEC=0` is the
plain kill switch — see docs/FLAGS.md). The speculative numbers in the table above remain
`gemma-gate` CLI measurements, not serving ones.

## Capture models — embeddings and rerank (v0.116.0)

These two checkpoints are served through the **capture surfaces**
(`POST /v1/embeddings`, `POST /v1/rerank`), not through generation. They are
prefill-only: the route reads the final prompt position of a causal LM and no decode
step runs, so they carry no drafter, no spec path and no perf-board row — the
serving contract is in [Serving](SERVING.md#embeddings-and-rerank--the-capture-surfaces-laneembed-serve-2026-08-26).

| Model | Class | Quant | Surface | Supported since |
|---|---|---|---|---|
| Qwen3-Embedding-8B | dense | Q8_0 GGUF (official `Qwen/Qwen3-Embedding-8B-GGUF` mint) | `/v1/embeddings` — last-token post-final-norm hidden state, L2-normalized, 4096-dim, MRL `dimensions` truncation | v0.116.0 |
| Qwen3-Reranker-8B | dense | Q8_0 GGUF (community `mradermacher/Qwen3-Reranker-8B-GGUF` f16→`llama-quantize` mint; direct convert-to-q8 mints of this model are broken) | `/v1/rerank` — P("yes") over the {"yes","no"} logit pair at the final position, under the Qwen3-Reranker prompt | v0.116.0 |

Both were qualified against a vendor `transformers` fp16 reference on pinned real-corpus
probes before serving. The embedding gate is a cosine floor of **0.995** (the measured
floor with margin), both prefill paths exercised — prime at or above the prime floor,
`decode_step_h` below it. The rerank gate is **strict full-order parity plus
|Δscore| ≤ 0.12** against the reference logit-score path (fp16 dir: 32/32 positions
exact, max |Δ| 0.0996; the shipped Q8_0 mint: 30/32 exact with one bottom-two tail swap
at reference margin 0.052, under the documented Q8 tail rule of strict top-half parity
and tail flips < 0.06 — rerank is a `top_n` product).

Two mint traps this bring-up paid for, both artifact-side rather than engine-side:
`convert_hf_to_gguf --outtype q8_0` **direct** from the reranker's safetensors produces a
broken rerank head — order scrambled across every gate query, reproduced byte-identically
from two independent llama.cpp checkouts — while the f16→`llama-quantize` pipeline of the
same weights gates PASS. And f16 GGUF is not a usable discriminator here: the engine
refuses it at load with `embed_gather: unsupported dtype F16`.

A capture model loads PLAIN beside a spec'd chat model — `MEMRA_MODELS` takes the extra
`alias=path` entry, and a model with no trained NextN block no longer fatals the worker
when `MEMRA_FRSPEC_TRIM` is set globally. A three-model single process (a spec'd chat
model plus both capture models) is a proven shape; `/health` lists all three. On such a
process, pin `MEMRA_Q8RP=0`: the capacity-keyed decode mirrors auto-arm on the Q8
embedder and cost 8.5 GB for a decode path a capture model never takes.

Both capture models are **subordinate by design**: serve them on batch-class keys so they
ride the harvest lane and shed under interactive load — the SLO admission that protects
decode p99 is the isolation mechanism, and a shed answers a retryable
`429 rate_limit_exceeded`, not an error.


---

## Ornith-1.5-35B-A3B, in detail

memra runs **Ornith-1.5-35B-A3B at its full native 262,144-token context** on one
RTX PRO 6000 Blackwell — self-published NVFP4 GGUF (the first NVFP4 of this model,
published ~34.5 h after the checkpoint dropped), continued-trained MTP head, and a
self-trimmed FR-Spec draft lm_head, all on HF
([Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF](https://huggingface.co/Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF)).

Measured on one rented RTX PRO 6000 Blackwell WS (vendor sampling T=0.6/top-p 0.95/
top-k 20, memra v0.105.0; N and windows in `research/orndecode-20260822/`). The
single-stream rows are host-CPU-dependent — see the note below the table:

| Metric | Measured |
|---|---|
| Single-stream cached-long | **319 tok/s** on a current-generation host with the v0.108 verify-graph default (4+4 interleaved boots, per-arm spread under 1%); 266 with that default killed; 326–354 measured on the serving box's own host class |
| Single-stream fresh short | 354 tok/s (Zen 5 host) · 235–262 (Zen 3 host) |
| 8-turn agentic session (14.7k doc, verbatim history) | **11.6–12.9 s** total |
| Shared-prefix aggregate | c8 ~915 tok/s · c16 ~700 tok/s |
| Warm-turn TTFT | 25–40 ms (full prefix restore) |
| Spec exactness | run-spec K=1..8 self-consistency PASS; verify arbitrates every drafted token |

**This model's decode used to depend on the host CPU as much as on the card, and since
v0.108 it depends on it much less.** Its architecture forces the *sampled* speculative path,
and that path was host-LAUNCH-bound: with `MEMRA_SPEC_PHASE=1` the round read verify-issue
44–58% and **verify-wait 0.0%** — the GPU was never what the host waited for. So a 1.6×
faster host moved end-to-end decode ~1.6× (195–230 → 326–354) with the GPU untouched.

v0.108 captures the verify trunk as a CUDA graph and replays it, which collapses that phase
(55–62 → 8–10 ms per burst) and hands the round to the device: **+19.7% on a
current-generation host** (266 → 319 tok/s, 4+4 interleaved boots), byte-identical output
(fixed-seed sampled hashes match across arms *and* across host generations). The remaining
host sensitivity is real but much smaller, and the ON arm lands at ~320 on both host
generations tested — that is a device ceiling rather than a host one. Still worth pairing the
card with a current-generation host, and a cold host still reads low until its clocks ramp.
Receipts: `research/orndecode-20260822/VGRAPH.md`.

Serving recipe: `MEMRA_FRSPEC_TRIM=<ranks.gguf> MEMRA_PRIME_CHUNK=0 memra-server
--model <nvfp4-q5k-mtp.gguf>` — the trim drops the draft lm_head 248,320 → 32,768 rows
(~221 → ~29 µs/draft step; c1 short +10%, c8 +2.9%, ABBA N=6/shape), and v0.105's
automatic depth table serves cached long prompts at K=5 when a trimmed head is loaded.
**Image input** ships since v0.102: the checkpoint's qwen3_5 ViT runs in-engine via
`MEMRA_VISION_DIR` (tower width derived from the shard), gated by the per-token cosine
parity oracle (min-cos 0.99983 worst probe) before serving.

## Qwen3.8-27B, in detail

memra runs **Qwen3.8-27B at its full native 262,144-token context** on RTX PRO 6000 Blackwell,
supported the day after the checkpoint's release and gated by the full exactness battery before
it was called supported at all.

**Serving route since v0.113.0 (2026-08-25): the DFlash2 drafter** — vendor-default
sampled shape, RTX PRO 6000, x3 interleaved medians: c=1 127 / c=2 128 / c=4 87 agg
tok/s vs the MTP head's 117/120/85; single-stream wall prose ~131-146, code ~208-239,
digit-heavy ~287-339 tok/s (acceptance rises with output predictability). Recipe in
[COOKBOOK.md](COOKBOOK.md); receipts in the FLAGS `MEMRA_DSPARK_SPEC` row.

**Current engine number: 250 tok/s** on one RTX PRO 6000. That is memra: DFlash2,
512-token digits, wall clock with TTFT included. The digit-heavy ~287-339 tok/s
band above is the decode window on the same route (TTFT excluded); 250 is that
window after TTFT. Do not quote the 140 tok/s p50 row below as current; it is
the 2026-08-15 MTP-era battery.

Historical MTP-era battery, kept as measured (NVFP4+Q5_K artifact, real agentic prompts,
3-rep medians, zero sheds or errors across every cell — 2026-08-15):

| Metric | Measured |
|---|---|
| TTFT p50, cold | **0.156 s** (c=1) — ≤0.32 s through c=4 |
| TTFT, cached conversation turn | **0.130 s** on a 5.7k-token context (full prefix restore) |
| Decode p50, single stream | RTX PRO 6000: **140 tok/s** (rep medians 138–141) · RTX 5090 Laptop: 75 tok/s (range 71–80) — GGUF trunk with its masked GGUF draft; plain decode 75 and 44 |
| Sampled-config throughput | top-p/top-k/min-p requests sample on-device — sampled aggregate **equals greedy** (240–245 tok/s at c=16–32) |
| Aggregate completion | **238–245 tok/s** at c=16, flat to c=32, zero sheds across capacity mixes |
| Sustained soak | 576/576 requests, 0 errors, 0 sheds, −0.27% drift |
| Spec ON/OFF exactness | 8/8 byte-identical; verify gate: zero differing logits at T=1..4, K=1/3/8 |

Since v0.86: **image and video input** on the same endpoint (OpenAI `image_url` /
`video_url` content parts, base64 data URIs; videos as animated GIF, decoded
in-process) — the checkpoint's native ViT tower runs in-engine, gated by a per-token
cosine parity oracle against the HF reference before serving (images min-cos 0.9997,
video 0.99999); vision tokens bill as ordinary prompt tokens.

Both paths are supported, but the decode figures above are the GGUF path's numbers. They were
misattributed to safetensors in the 2026-08-16 documentation-direction change: the surviving
138.40/141.18/139.85 cells were produced by a script that hardcodes the GGUF trunk and draft.
The last valid direct format cell from that window measured safetensors 105.44 versus GGUF 146.79
tok/s. Do not transfer either format's performance default without a new same-binary comparison.

The trim costs nothing in correctness: verification runs on the target's full 248,320-token
vocabulary, so a trim moves draft acceptance and never output. Heads are trimmed
248,320 → 32,768.

Published in [Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF), three ranks flavours, chosen by
workload. On a **safetensors** trunk the `.txt` ranks drive the trim at load time
(`MEMRA_FRSPEC_TRIM=<ranks.txt>`) and no separate draft file is needed; the pre-trimmed `.gguf`
head is the GGUF path (`MEMRA_MTP_DRAFT=<head.gguf>`):

| Flavour | Ranks (safetensors) | Pre-trimmed head (GGUF) | Corpus |
|---|---|---|---|
| **agentic** — serving default | [`q38-ranks-sxc32768.gguf.txt`](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-sxc32768.gguf.txt) | [`mtp-…frspec-sxc32768.gguf`](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf) | 163k own-generated tokens over real agentic sessions |
| **prose** | [`q38-ranks-prose-32768.txt`](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-prose-32768.txt) | [`mtp-…frspec-prose32768.gguf`](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-prose32768.gguf) | 154k own-generated tokens, essay/story/letter prompts, ~15% non-English |
| **mixed** | [`q38-ranks-mixed-32768.txt`](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-mixed-32768.txt) | [`mtp-…frspec-mixed32768.gguf`](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-mixed32768.gguf) | 50/50 normalised blend of both streams, same rank law |

Short generic probes put the three within noise of each other; the differences live in
domain tail tokens, which is why the flavour is a choice and not a default anyone should
inherit blindly. Ranks are 100% model-generated — external text is used as prompts only.

Two honest architecture notes, published because they were measured:
on this GDN-hybrid class the prefix cache serves extension shapes
(conversation continuation, session affinity) rather than fan-out shapes, because the
recurrent state exists only at an entry's end boundary.

Step-3.7-Flash already serves on the GGUF path; its current tuning is on **FP8**, not Q8. That work and the Qwen3.6 family continue; numbers land
in [docs/PERFORMANCE.md](PERFORMANCE.md) as they are measured; a default only ships for a
card class it was measured on.

Two results from that tuning are worth stating up front, because both are counter-intuitive and both
were measured on one RTX PRO 6000 Blackwell Workstation serving a 35B MoE at a 4,860-token prompt
shape with a shared prefix:

- **Prefix-cache depth dominates.** With the cache holding the full shared prefix, concurrency 16 ran
  at 8.50 req/s and a 1.87 s median. With a shallower entry covering the same prefix class, the same
  build on the same card ran 2.72 req/s at 5.84 s — a 3.1x swing in both throughput and latency from
  cache depth alone.
- **Speculative decoding is not free on cache-carried shapes.** It is numerically exact, and offline
  it looks like a 1.5-1.7x win, but on this serving shape it cost 4x because a v1 speculative session
  gave up the cross-request prefix cache. That mechanism is closed as of v0.93.0: a spec session's
  boundary capture publishes its prefix entry with the draft plane (entry layout v2), and a greedy
  whole-entry hit re-arms a warm spec session instead of downgrading — the sold-shape gate holds
  1.00x req/s at a 0.987 hit rate with spec engaged on hits
  ([research/spec-cache-20260818/](../research/spec-cache-20260818/)). `MEMRA_SERVE_SPEC=0` remains
  the rollback posture. See [docs/FLAGS.md](FLAGS.md).

Raw per-run receipts for both are under [research/](../research/), and the second is the kind of result
this project publishes either way: the arm expected to win lost.


## GLM-5.3-Flash serves its OWN chat dialect, not ChatML (lane/glm53-flash-bringup-20260827)

`glm5_next` ships a chat template that shares three markers with the qwen ChatML class —
`<think>`, `add_generation_prompt`, `<tools>` — and NOTHING else. Its frame is
`[gMASK]<sop>` + `<|system|>` / `<|user|>` / `<|assistant|>` / `<|observation|>`; its tool
calls are `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`; its tool
results are `<tool_response>` blocks inside one `<|observation|>` turn; its `<think>` tail is
unconditional (no `enable_thinking` anywhere) and it renders an always-present
`<|system|>Reasoning Effort: {Low|High|Max}` line whose default rung is **Max**.

Because every qwen marker check matched it, the renderer used to fall through to the ChatML
arm and serve `<|im_start|>` turns — tokens this checkpoint does not carry as specials at all,
so the whole frame tokenized as ordinary text. It answered anyway (GLM follows the qwen
tool-format instruction it is handed in-context), which is exactly the template-mint failure
mode: fluent, and invisible without a byte oracle. `chat::template_is_glm5`
(`[gMASK]<sop>` AND `<|observation|>`) now keys the renderer dispatch, the tools-branch probe,
`ModelCaps::glm5`, the streaming tool/reasoning parser and the effort ladder from one law.

Byte parity against the checkpoint's own `chat_template.jinja` is the acceptance bar, pinned
by 20 generated fixtures (darklanes
`research/glm53-flash-bringup-20260827/{gen_surface_fixtures.py,surface-fixtures/}`) replayed
through the REAL request pipeline on all three wire formats. Two honesty consequences: an
explicit reasoning-off request is a named 400 (the template cannot close its tail), and the
`/v1/models` row no longer claims `structured_output` on any model whose think tail opens
unconditionally without an `enable_thinking` switch — the server already refused
`response_format` there by name, so the claim was one the server itself would not keep.


## deepseek-v4 checkpoint dirs serve through their own door (lane/dsv4-flash-revival-20260822)

A dsv4 checkpoint dir (`config.json model_type deepseek_v4`) loads at worker boot onto the
engine's dedicated 2-card PP stack (`Dsv4Gpu`) — never `HybridModel` — and serves the full
chat surface through the worker's `Cmd::Generate -> Event` contract on a dedicated thread
(the FIFO is the admission queue; bs=1 queueing is measured, not hidden). Routes:
{greedy|sampled} x {spec|plain} keyed on temperature + DSpark drafter residency; the
serving numeric contract keys on the artifact's own encoding revision (0731 =>
`RefFp8Round`, preview => `ClampOnly`, undetectable REFUSES at boot). The real artifacts
ship their chat dialect as CODE (`encoding_dsv4.py`) with no `chat_template` string —
dispatch and caps key on the detected `Dsv4Encoding` census, and the DSML tool protocol
counts as a tools branch. Penalties serve over the true per-state window (row-incremental
on the spec verify; cross-pinned against `memra-sampling`); penalized greedy demotes to
the plain path. Deliberate v1 limits, refused by name: `min_p`, `response_format`; no
prefix cache (`n_cached` honestly 0, model row says so). The rung-3 serve cell (S1-S8:
boot/caps, stream identity, spec==plain byte-identity at the surface, penalties+named
400s, seeded determinism, DSML->OpenAI tools round-trip, c=16 concurrency, cancel
mid-stream) is the real-surface gate; receipts in darklanes
`research/deepseek-flash-20260818/revival-20260822/`.

**Status: experimental engine support — functional and gated, not a serving-grade path.**
Both halves of that sentence are meant. The support is real: the full chat surface serves,
the S1–S8 cell is green, spec==plain byte-identity holds at the surface, and the refusals
are named. And it is not serving-grade: the engine decodes bs=1 behind the FIFO (no
batching — that IS the aggregate ceiling), it requires a two-card PP-2 stack, and the
measured numbers sit far below the per-card bar the served roster sets. Current numbers,
protocol-labeled (2× RTX PRO 6000 Blackwell 96 GB, PP-2; receipts above): single-stream
drafted **75.7 tok/s** (greedy instrument, DSpark drafter, verify window `slot@0.5`, c=1,
128-token agentic prompts); single-stream plain **54.3 tok/s** (greedy instrument); c=16
aggregate **26.7 tok/s** through the honest bs=1 FIFO (S7 cell — queueing measured, not
hidden, which is why aggregate lands *below* plain single-stream). Compare the served
flagship at ~259 tok/s single / 238–245 aggregate on one card. The tuning path to
serving-grade — batched decode, PP-2 placement, round-cost work, prefix cache — is
tracked in [#4](https://github.com/avifenesh/memra/issues/4); until those rows exist,
do not read this section as a serving recommendation.
