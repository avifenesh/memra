# MODELS history

Records moved out of [docs/MODELS.md](../MODELS.md) and the model and hardware cards
([docs/models/](../models/README.md), [docs/rigs/](../rigs/README.md)) on 2026-11-12, verbatim
except for the removal of private deployment detail. The current contract is in the main doc
and the cards; this file is not. A note in italics under a heading is added on 2026-11-12 and
says what in the record is stale.

## What a support claim meant (untitled in the source)

From `docs/MODELS.md`, lines 13-17 at d1ed663256.

Support is specific to a model, quantization, and drafter combination — never to a format. A
checkpoint loading is not the same thing as a model being supported: every family has its own
tensor census, quantization arithmetic and topology, and it counts as supported here only once
it has passed its own exactness gates. The generated table below is numbers-free; per-model
measurements stay in [docs/PERFORMANCE.md](../PERFORMANCE.md).

## Direction: safetensors first, GGUF still supported

From `docs/MODELS.md`, lines 21-34 at d1ed663256.

For a new upstream model the official safetensors checkpoint — with its config,
tokenizer/template, quantization metadata and auxiliary tensor files — is the preferred semantic
source. memra repacks those tensors once into measured rig-native layouts; safetensors is not the
internal compute format. GGUF remains a fully supported, self-contained portable import and
distribution format.

**safetensors remains the preferred path for admitted native work.** This covers NVFP4 and FP8 trunks,
with load-time head trimming instead of a separate pre-trimmed draft file.

**GGUF stays supported.** Many models are served through it today and those paths are not going
anywhere. What is *not* offered is a promise that a format keeps getting new work: a GGUF that
lands after this line was written may or may not be picked up, and that is a decision each
time, not a policy. The reverse holds too — a safetensors checkpoint can load without being on
the tuned path.

## Where each family actually sits right now (untitled in the source)

From `docs/MODELS.md`, lines 36-56 at d1ed663256.

Where each family actually sits right now:

Supported on both paths means both, and where the tuning is happening is a separate
question from what is supported. The last recorded tuning directions are retained below:

| Family | Supported on | Recorded tuning direction |
|---|---|---|
| Qwen3.8-27B | **both**, both tuned — safetensors NVFP4 and NVFP4+Q5_K GGUF | done. Performance defaults stay artifact-specific |
| Gemma-4 31B | **GGUF** (QAT Q4_0), supported and tuned | NVFP4 safetensors |
| Step-3.7-Flash 196B-A11B | **GGUF** (IQ4_XS + Q8_0 MTP head, two-card PP-2) since v0.73.1 | FP8 — not Q8 |
| Gemma, rest of the family | GGUF | NVFP4 safetensors rework in progress |
| DeepSeek-V4-Flash | **safetensors** checkpoint dir through its own two-card door — **experimental engine support**, functional and gated, **not serving-grade** (see its section below) | performance ([#4](https://github.com/avifenesh/memra/issues/4)) |
| GLM-5.3-Flash | **safetensors** (FP8 e4m3, MIT) on the hand-written `glm5_next` path. The `glm5_next` pack is **NativeReference**; multi-card TP serving, vision, and MTP spec are gated on the hand-written path and do not qualify the pack (see [its card](../models/glm53-flash.md)) | - <!-- support: glm5_next-reference --> |
| Qwen3.8-Flash-Next | **bring-up only** — hand-written `qwen4_exp` gate path on the minted NVFP4 artifact, real-checkpoint eager gate green; ModelPlan loader not wired, no serving surface (see [its card](../models/qwen38-flash-next.md)) | — |
| DeepSeek-V4.1-Flash | **reference executor only**: the `deepseek_v41` pack is **NativeReference**. It binds all 96,085 header tensors of the 1017 checkpoint, its CSA2 / single-pass mHC / Engram / DSpark plan runs in the reference executor, the tiny gate matches the vendor model.py (run with torch stand-ins for its tilelang kernels) to the bit, and its `encoding_dsv41` chat dialect renders and parses byte for byte as the vendor's `encoding.py`. Checkpoint parity and serving are pending; nothing serves it (see [its card](../models/deepseek-v41-flash.md)) | - <!-- support: deepseek_v41-reference --> |
| Hy3 | **safetensors**: canonical BF16 plan and the exact all-expert ModelOpt W4A16 profile (`hy3_nvfp4`) are both **NativeReference**. The four-card Blackwell qualification receipts are not tracked, so checkpoint parity and serve are pending (see [its card](../models/hy3.md)) | - <!-- support: hy3-bf16-reference,hy3_nvfp4-modelopt-w4a16 --> |
| everything else in the table | GGUF | — |

Nothing above is a roadmap. It is where the code is, and it changes by decision. A row moving
from one path to another does not retire the old one: when a model is supported on both, both
keep working.

## Unfinished native work

From `docs/MODELS.md`, lines 60-72 at d1ed663256.

*Stale against code: `/v1/audio/*` exists and fails closed; DeepSeek-V4-Flash TP/EP, GLM-5.3-Flash TP-2/TP-4 and Hy3 EP-4 run. The current state is in [docs/MODELS.md](../MODELS.md).*

Whisper large-v3 has a [native speech plan and metadata pack skeleton](../../ASR-MODALITY-PLAN.md).
The source census is bound, and native CPU mel and encoder stages have passed a 2-second
HF oracle gate in their documented numeric classes. The complete speech program has no native
support state. Decoder policy and FastConformer/RNNT remain unfinished; neither path is
admitted for serving.

The speech capability as a program, which families are in scope, what each support state means
for a speech model, the repeatable onboarding path for a new one, the gate set and the
performance thesis with its kill criteria, is [docs/SPEECH.md](../SPEECH.md). There is no GPU
execution path and no audio endpoint yet, so no speech model has a speed number of any kind.

Tensor parallel, P2P and 3-stage pipeline parallel were recorded here as unfinished. This record
does not schedule their completion; any support change still requires its applicable gates.

## Which models should be next?

From `docs/MODELS.md`, lines 76-97 at d1ed663256.

*The hardware list now lives in [docs/rigs/](../rigs/README.md). It omitted the GB10 `sm_121a` target, and B200 is best effort since 2026-11-04.*

There is no next-model roadmap during the pause. An issue may propose a concrete research
question with a checkpoint, target card, baseline, and reason to use Memra. Such a proposal does
not create a support commitment; admitted work follows the project scope above.

- **RTX PRO 6000 Blackwell (`sm_120a`) — co-primary target.** Workstation and Server Edition, 96 GB.
  Carries verification, final tuning, and serving, single-card and as PP-2 pairs.
- **RTX 5090 / 50-series (`sm_120a`) — co-primary target.** Same architecture, its own measured
  settings, and its own defaults where they differ. Local 5090 performance is never traded away to
  simplify a remote default; a perf claim needs numbers from both cards before it sets a global one,
  and one-card evidence sets a one-card default at most.
- **Hopper `sm_90a`** — separately compile-gated H100 source-build lane with its evidence
  ledger in [ARCHITECTURE-H100.md](../../ARCHITECTURE-H100.md). Secondary: it does not change the
  naked `sm_120a` build or its defaults. Its standing battery was retired with the Hopper CI
  lane (2026-09-02); run the gates directly on Hopper hardware.
- **Ada `sm_89`** — portable source-build lane, not a tuned performance target.
- **B200 `sm_100a`** — runtime-qualified source backend with automatic source-build detection;
  no prebuilt is published. Pinned Qwen3.5-9B NVFP4 is `NativeReference` on default W4A8: <!-- support: qwen35-9b-nvfp4-b200-w4a8 -->
  kernel, self-consistency and sampled serving gates pass, checkpoint parity is still pending.
  Opt-in W4A4 is correct but 0.521x raw W4A8 prefill. Pinned Qwen3.8-27B block-FP8 is
  `NativeReference`; its explicit FP8-MMQ twin is correct but 0.173x the fallback, so it is not <!-- support: qwen38-27b-fp8-b200 -->
  tuned or default. See [the B200 card](../rigs/b200.md).
  Other architectures are not a tuned support promise.

## Step-3.7-Flash and Gemma-4 notes under the generated board (untitled in the source)

From `docs/MODELS.md`, lines 117-182 at d1ed663256.

*The vision, structured-output and draft-graph lanes were gated on the `stepfun-ai/Step-3.7-Flash-NVFP4` safetensors artifact, not the IQ4_XS GGUF PP-2 route; see the [Step card](../models/step37-flash.md).*

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
`gemma-gate` CLI measurements, not serving ones. E4B is the exception to the served attach: the
worker skips it, so its assistant drafter runs through `gemma-gate` only, and `memra-server`
serving of E4B is not qualified ([E4B card](../models/gemma4-e4b.md)).

## Capture models — embeddings and rerank (v0.116.0)

From `docs/MODELS.md`, lines 186-223 at d1ed663256.

*No tracked receipt under `research/` holds the gate figures below.*

These two checkpoints are served through the **capture surfaces**
(`POST /v1/embeddings`, `POST /v1/rerank`), not through generation. They are
prefill-only: the route reads the final prompt position of a causal LM and no decode
step runs, so they carry no drafter, no spec path and no perf-board row — the
serving contract is in [Serving](../SERVING.md#embeddings-and-rerank--the-capture-surfaces-laneembed-serve-2026-08-26).

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

## Ornith-1.5-35B-A3B, in detail

From `docs/MODELS.md`, lines 230-270 at d1ed663256.

memra runs **Ornith-1.5-35B-A3B at its full native 262,144-token context** on one
RTX PRO 6000 Blackwell — self-published NVFP4 GGUF, continued-trained MTP head, and a
self-trimmed FR-Spec draft lm_head, all on HF
([tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF](https://huggingface.co/tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF)).

Measured on one RTX PRO 6000 Blackwell WS (vendor sampling T=0.6/top-p 0.95/
top-k 20, memra v0.105.0; N and windows in `research/orndecode-20260822/`). The
single-stream rows are host-CPU-dependent — see the note below the table:

| Metric | Measured |
|---|---|
| Single-stream cached-long | **319 tok/s** on a current-generation host with the v0.108 verify-graph default (4+4 interleaved boots, per-arm spread under 1%); 266 with that default killed |
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

From `docs/MODELS.md`, lines 274-337 at d1ed663256.

*No tracked receipt backs the 250 tok/s figure or the 2026-08-15 MTP-era battery (576/576 soak and the rows around it); both arrived with the 2026-11 migration sync. The DFlash2 sampled rows cite a lane directory that is not in this repository.*

memra runs **Qwen3.8-27B at its full native 262,144-token context** on RTX PRO 6000 Blackwell,
gated by the full exactness battery before it was called supported at all.

**Serving route since v0.113.0 (2026-08-25): the DFlash2 drafter** — vendor-default
sampled shape, RTX PRO 6000, x3 interleaved medians: c=1 127 / c=2 128 / c=4 87 agg
tok/s vs the MTP head's 117/120/85; single-stream wall prose ~131-146, code ~208-239,
digit-heavy ~287-339 tok/s (acceptance rises with output predictability). Recipe in
[COOKBOOK.md](../COOKBOOK.md); receipts in the FLAGS `MEMRA_DSPARK_SPEC` row.

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

Published in [tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF), three ranks flavours, chosen by
workload. On a **safetensors** trunk the `.txt` ranks drive the trim at load time
(`MEMRA_FRSPEC_TRIM=<ranks.txt>`) and no separate draft file is needed; the pre-trimmed `.gguf`
head is the GGUF path (`MEMRA_MTP_DRAFT=<head.gguf>`):

| Flavour | Ranks (safetensors) | Pre-trimmed head (GGUF) | Corpus |
|---|---|---|---|
| **agentic** — serving default | [`q38-ranks-sxc32768.gguf.txt`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-sxc32768.gguf.txt) | [`mtp-…frspec-sxc32768.gguf`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf) | 163k own-generated tokens over real agentic sessions |
| **prose** | [`q38-ranks-prose-32768.txt`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-prose-32768.txt) | [`mtp-…frspec-prose32768.gguf`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-prose32768.gguf) | 154k own-generated tokens, essay/story/letter prompts, ~15% non-English |
| **mixed** | [`q38-ranks-mixed-32768.txt`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-mixed-32768.txt) | [`mtp-…frspec-mixed32768.gguf`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-mixed32768.gguf) | 50/50 normalised blend of both streams, same rank law |

Short generic probes put the three within noise of each other; the differences live in
domain tail tokens, which is why the flavour is a choice and not a default anyone should
inherit blindly. Ranks are 100% model-generated — external text is used as prompts only.

Two honest architecture notes, published because they were measured:
on this GDN-hybrid class the prefix cache serves extension shapes
(conversation continuation, session affinity) rather than fan-out shapes, because the
recurrent state exists only at an entry's end boundary.

## Tuning results under the Qwen3.8-27B section (untitled in the source)

From `docs/MODELS.md`, lines 339-361 at d1ed663256.

*Stale: the last e4m3 Step lane is `research/step-e4m3-x8-20260814/`; later Step lanes target the NVFP4 artifact.*

Step-3.7-Flash already serves on the GGUF path; its current tuning is on **FP8**, not Q8. That work and the Qwen3.6 family continue; numbers land
in [docs/PERFORMANCE.md](../PERFORMANCE.md) as they are measured; a default only ships for a
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
  whole-entry hit re-arms a warm spec session instead of downgrading — the gate holds
  1.00x req/s at a 0.987 hit rate with spec engaged on hits
  ([research/spec-cache-20260818/](../../research/spec-cache-20260818/)). `MEMRA_SERVE_SPEC=0` remains
  the rollback posture. See [docs/FLAGS.md](../FLAGS.md).

Raw per-run receipts for both are under [research/](../../research/), and the second is the kind of result
this project publishes either way: the arm expected to win lost.

## GLM-5.3-Flash serves its OWN chat dialect, not ChatML (lane/glm53-flash-bringup-20260827)

From `docs/MODELS.md`, lines 366-389 at d1ed663256.

*The fixtures and their generator are tracked in this repository under [research/glm53-flash-bringup-20260827/](../../research/glm53-flash-bringup-20260827/).*

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
by 20 generated fixtures
(`research/glm53-flash-bringup-20260827/{gen_surface_fixtures.py,surface-fixtures/}`) replayed
through the REAL request pipeline on all three wire formats. Two honesty consequences: an
explicit reasoning-off request is a named 400 (the template cannot close its tail), and the
`/v1/models` row no longer claims `structured_output` on any model whose think tail opens
unconditionally without an `enable_thinking` switch — the server already refused
`response_format` there by name, so the claim was one the server itself would not keep.

## deepseek-v4 checkpoint dirs serve through their own door (lane/dsv4-flash-revival-20260822)

From `docs/MODELS.md`, lines 394-426 at d1ed663256.

*Superseded. Since memra #710 (2026-09-25) the two-card load is TP/EP by default, 16 lanes coalesce into B-row steps, parked-prefix reuse is opt-in through `MEMRA_DSV4_KV_HOST_MB`, and a preview-encoding checkpoint refuses at load. Current state: the [DeepSeek-V4-Flash card](../models/deepseek-v4-flash.md).*

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
mid-stream) is the real-surface gate.

**Status: experimental engine support — functional and gated, not a serving-grade path.**
Both halves of that sentence are meant. The support is real: the full chat surface serves,
the S1–S8 cell is green, spec==plain byte-identity holds at the surface, and the refusals
are named. And it is not serving-grade: the engine decodes bs=1 behind the FIFO (no
batching — that IS the aggregate ceiling), it requires a two-card PP-2 stack, and the
measured numbers sit far below the per-card bar. Current numbers,
protocol-labeled (2× RTX PRO 6000 Blackwell 96 GB, PP-2; receipts above): single-stream
drafted **75.7 tok/s** (greedy instrument, DSpark drafter, verify window `slot@0.5`, c=1,
128-token agentic prompts); single-stream plain **54.3 tok/s** (greedy instrument); c=16
aggregate **26.7 tok/s** through the honest bs=1 FIFO (S7 cell — queueing measured, not
hidden, which is why aggregate lands *below* plain single-stream). The tuning path to
serving-grade — batched decode, PP-2 placement, round-cost work, prefix cache — is
tracked in [#4](https://github.com/avifenesh/memra/issues/4); until those rows exist,
do not read this section as a serving recommendation.

## A pack declares WHERE its chat dialect comes from (lane/dsv4-template-contract-20260910)

From `docs/MODELS.md`, lines 430-469 at d1ed663256.

`TemplateContract` on a `ModelPack` has two arms and no third:

- `ArtifactRequired`: the artifact carries the dialect, as a `tokenizer_config.json`
  `chat_template` or a `chat_template.jinja`. Absent means the onboarding gate FAILS.
- `EngineRenderer("<name>")`: the family's dialect is CODE in this engine, because the vendor
  ships it as code and no artifact in the family carries a template. deepseek-v4 is the case this
  exists for. Upstream `deepseek-ai/DeepSeek-V4-Flash-0731` at revision `7872f01b…` lists 77
  entries with zero `.jinja` files and an 801-byte `tokenizer_config.json` with no
  `chat_template`; the dialect is `encoding/encoding_dsv4.py`. `ArtifactRequired` there refuses a
  correct artifact forever.

**The renderer arm is not a fallback and must never become one.** Four properties keep it
honest, and a new family adopting it inherits all four:

1. it is NAMED, and the pack test rejects a nameless renderer, because a nameless renderer is
   indistinguishable from a fallback;
2. it is TWO-SIDED. A renderer family also REFUSES an artifact that carries a template of its
   own, as a franken artifact whose two dialects would disagree silently. Neither arm can go
   vacuous when a family's distribution changes;
3. `artifact.lock` records the gate's own judgment through the same function the gate calls, and
   writes `dialect=<renderer>` with `template_sha256=engine-renderer` rather than a hash of
   nothing.

4. it RESOLVES. The gate looks the name up in memra-tokenizer's `chat::EngineDialect` table,
   loads the artifact's own tokenizer, and requires the artifact's config.json census to select an
   encoding that renderer covers, every control string the renderer emits to encode as one id,
   and a fixed probe conversation to render. `tokenizer-contract.tsv` records the renderer, the
   encoding and the probe's digests; `artifact.lock` keeps its bytes. Before this a pack could
   name a renderer that did not exist and pass on the name.

The renderer carries its own byte oracle elsewhere in the tree. For `encoding_dsv4` that is the
vendor's four `encoding/tests` pairs, gated in `memra-server` BOTH through the sentinel template
and, since this lane, through the shape the serve path actually uses (`chat_template = None` plus
the config-census `Dsv4Encoding`), with a red arm that drops the dialect and asserts the render
comes back as ChatML. For `encoding_dsv41` (DeepSeek-V4.1-Flash, its own dialect: spaced DSML
tags, a numeric reasoning budget, mid-conversation system turns, image parts, tool namespaces) it
is fixtures generated by running the vendor's `encoding/encoding.py` over every branch, its five
`encoding/tests` pairs and every `test_encoding.py` case but three documented divergences
included, plus the token ids of every render
(`research/modelplan-onboarding-dsv41-20261109/encoding/`).

## Qwen3.8 27B card (the table before 2026-11-12)

From `docs/models/qwen38-27b.md`, lines 3-16 at d1ed663256.

*DFlash2 is opt-in (`MEMRA_DSPARK_SPEC=1` plus `MEMRA_DSPARK_DRAFT`), not a 5090 engine default. The +3.87% figure is the q4 drafter-precision flip (`MEMRA_DFLASH_PREC`), and neither it nor the 250 tok/s figure has a tracked receipt.*

| | Recommended use |
|---|---|
| **Status** | Supported and tuned on both safetensors and GGUF paths |
| **5090 default** | DFlash2 on the NVFP4+Q5_K GGUF trunk with q4 weights and the agentic 32K FR-Spec rank map; the masked MTP head remains the rollback path |
| **Best starting path** | The qualified 5090 DFlash2 q4 + agentic-rank route; safetensors NVFP4 MTP remains the native-format fallback |
| **Portable path** | NVFP4+Q5_K GGUF with the embedded or attached MTP head |
| **Fastest spec route** | DFlash2 drafter on the GGUF trunk (`MEMRA_DSPARK_SPEC`, q4 drafter default + FR-Spec trim; sampled sessions stack in the LOW band since v0.113.0) — beats the MTP head on every rung of the vendor-default sampled shape on RTX PRO 6000 (c1 127/117, c2 128/120, c4 87/85 agg tok/s); +3.87% local 5090 E2E over full-head DFlash2 |
| **Hardware** | RTX 5090 for single-card use; RTX PRO 6000 Blackwell for larger context and serving headroom |
| **On-box speed** | **250 tok/s** wall on one RTX PRO 6000 (DFlash2, 512-token digits, TTFT included). Same route's digit-heavy decode-window band is ~287-339 tok/s. 140 tok/s is the 2026-08-15 MTP-era p50, not current memra. |
| **Use this when** | You want the main general-purpose and agentic path exercised by Memra |

Start with the [Qwen3.8 cookbook](../COOKBOOK.md#qwen38-27b). Choose the agentic, prose, or mixed
ranks file only after reading the [model detail](#qwen38-27b-in-detail). Conditions and
receipts are in [performance](../PERFORMANCE.md).

## Step 3.7 Flash 196B-A11B card (before 2026-11-12)

From `docs/models/step37-flash.md`, lines 3-28 at d1ed663256.

*On PP-2 the K policy picks K=0, so the MTP head drafts only on one card. The three surfaces were gated on `stepfun-ai/Step-3.7-Flash-NVFP4@4275532ffd9a9496ff36b7a2dc4a9db1048da438`.*

| | Recommended use |
|---|---|
| **Status** | Served on the hand-written GGUF pipeline path with its own receipts: PP-2 serving, image input, structured output, and graphed draft chains. The `step35` model pack declares no support state (`support: None`). FP8 tuning is separate work |
| **Best starting path** | IQ4_XS trunk with the Q8_0 MTP head on two-card PP-2 |
| **Hardware** | 2× RTX PRO 6000 Blackwell |
| **Use this when** | The model does not fit one card and you want the qualified pipeline-parallel path |

Use the [Step cookbook](../COOKBOOK.md#step-37-flash-196b-a11b). This is PP-2, not tensor
parallelism. Topology and admission behavior are in [serving](../SERVING.md#pipeline-parallel-pp-2-serving).

Since the qualified pipeline landed, three more surfaces serve on the same endpoint:

- **Image input** (lane/step37-vision-20260830): the checkpoint's own perception_encoder
  tower runs in-engine behind `MEMRA_STEP_VISION_DIR` (default OFF; [flags](../FLAGS.md)),
  with the vendor's exact tiling law. Per-token parity-gated against two independent
  references before any serving path; text requests stay byte-identical to the seam-off
  boot and keep MTP spec engaged. Receipts: `research/step37-vision-20260830/`.
- **Structured output** (lane/step37-postthink-grammar): `response_format`
  `json_object`/`json_schema` served post-think — the think phase runs unconstrained as
  trained, and the grammar clamps every token from the tokenizer's atomic think-close
  token (id 128799). Fail-closed named 400s if generation ends inside the reasoning
  channel. Receipts: `research/step37-postthink-grammar-20260830/`.
- **Draft-graph serving** (lane/step37-draft-graph-serving-20260830): the 3-head
  step-modulo draft chain is CUDA-graph captured on the serving shape, greedy and
  filtered-sampling arms byte-identical to eager. Receipts:
  `research/step37-draft-graph-serving-20260830/`.

## DeepSeek V4 Flash card: banner

From `docs/models/deepseek-v4-flash.md`, lines 3-13 at d1ed663256.

> **Serving bring-up reopened (owner goal, 2026-09-23).** The 2026-09-12 pause is lifted for the
> bring-up and tuning lane ([issue #4](https://github.com/avifenesh/memra/issues/4)). Support is
> still experimental: the numbers below are engine measurements with their conditions, not a
> serving-grade claim.

| | Recommended use |
|---|---|
| **Status** | Experimental engine support; functional and gated, not serving-grade |
| **Path** | Exact-source sharded Safetensors checkpoint through the dedicated two-card implementation; no GGUF conversion |
| **Hardware** | 2x RTX PRO 6000 Blackwell (the bring-up and tuning rig) |
| **Use this when** | You are evaluating or contributing to the experimental path, not deploying it as a supported server |

## DeepSeek V4 Flash card: lever history, 2026-09-23 to 2026-09-28

From `docs/models/deepseek-v4-flash.md`, lines 41-157 at d1ed663256.

Since 2026-09-26 the TP/EP decode chain runs with programmatic dependent launch, a
vocab-parallel head, one kernel for the compressor snapshots and push joins
(`research/dsv4f-bringup-20260923/levers-20260926/`). The same bits, and on the Server Edition
pair, greedy aggregate tok/s:

| cell | before (N=5) | levers on (N=5) |
|---|---|---|
| c1 | 67.53 | 77.19 (decode 80.95) |
| c2 | 93.74 | 104.42 |
| c4 | 123.39 | 130.44 to 135.10 |

A DSpark round's compressor rollback and its verify row placement are single launches since
2026-09-26 (`research/dsv4f-bringup-20260923/dspark-round/`): DSpark greedy c1 89.16 to 93.17
tok/s and sampled 75.84 to 78.65 on a second SE pair (N=3), and the chunked prefill's TTFT 18% to
21% lower. Same bits.

Since 2026-09-27 a TP/EP step of 1 to 16 rows runs the routed experts as the fused pair over the
rank's experts: two launches per layer, where the grouped chain took about fourteen
(`research/dsv4f-bringup-20260923/levers-20260927/`). Same bits. On a second SE pair, greedy
aggregate tok/s, main against fused, N=3:

| cell | main | fused |
|---|---|---|
| c1 | 77.36 | 84.49 (decode 88.86), +9.2% |
| c2 | 103.75 | 116.45, +12.2% |
| c4 | 130.48 | 146.44, +12.2% |
| DSpark c1 | 88.99 | 95.63, +7.5% |

Since 2026-09-27 the prefill FP8 tile runs 8 token rows by 4 outputs at 128 registers with an
exact `cvt` E4M3 decode (`research/dsv4f-bringup-20260923/prefill-tile/`): greedy TTFT p50 at an
8k prompt 17.69 to 14.36 s (-18.9%), at 32k 73.93 to 60.97 s (-17.5%), SE pair, N=3. Same bits.

Since 2026-09-27 the HC finish runs its Sinkhorn projection on a fifth warp, beside the collapse
and the RMSNorm (`research/dsv4f-bringup-20260923/hc-finish/`). Greedy c1 goes from 89.17 to
91.01 tok/s (+2.0%, decode 96.1) on the SE pair, N=2. Same bits.

Since 2026-09-27 the shared expert runs on one TP/EP rank per layer, the one with fewer of the
step's routed slots, and its rows ride the expert join
(`research/dsv4f-bringup-20260923/levers-20260927/`). Greedy c1 goes from 91.18 to 94.66 tok/s
(+3.7%, decode 99.1) on the second SE pair, N=2. Same bits.

By the close of 2026-09-27, with the TP/EP joined MoE tail, the attention small-kernel fusions and
the router-built x mirror also landed, greedy c1 is 97.10 / 97.26 tok/s (decode 102.1, TPOT 9.8
ms) and sampled c1 97.96 / 97.97 on the SE pair. That is +9.0% over the same morning's main
(`research/dsv4f-bringup-20260923/ceiling/CEILING.md`, "Close"). The same file measures the
step's floors: 6.7 GB read per rank per step, and a 1.45 ms launch floor.

Since 2026-09-28 a multi-request step runs each per-request stage as one launch for every row,
and captured B-row steps take up to 16 rows (#906). Dense-fast M-row launches also take 16 rows
(#909). Against main of 2026-09-27, one boot per row, 16 serving lanes:

| cell | before | after #906 (second SE pair) | after #909 too (first SE pair) |
|---|---|---|---|
| greedy c4 | 160 tok/s | 206 .. 209 | 209 .. 215 |
| greedy c8 | 199 | 286 .. 288 | 289 .. 298 |
| greedy c16 | 200 (TPOT 73 ms) | 266 .. 267 (TPOT 53 ms) | 336 .. 340 (TPOT 42 ms) |
| greedy c24 | 199 | 268 .. 269 | 312 .. 315 |

A 16-row captured step goes from 68.3 to 37.4 ms. Every served text is the same as main's
(`research/dsv4f-bringup-20260923/levers-20260927/`).

Concurrency: the plain TP/EP route serves sixteen lanes whose steps share one captured B-row
graph step (memra #710, #667; four until 2026-09-27). Aggregate on the Workstation pair, at four
lanes:

| | c2 | c4 |
|---|---|---|
| TP/EP, four lanes | 102.0 tok/s, TTFT 0.24 s | 132.8, TTFT 0.42 s |
| PP-2, two pipelined lanes | 120.9, TTFT 0.30 s | 120.6, TTFT 4.5 s |

At sixteen lanes, with the coalescer that keeps them in one batch, on the SE pair
(`research/dsv4f-bringup-20260923/lanes16/`): greedy c8 190 .. 192 tok/s, c16 196 .. 198 (TTFT
p50 1.5 s), c24 193 .. 195 with every request served. Four lanes give 153 .. 160 at each of
these, and refuse a sixth of the c24 requests. c4 is the same at both widths.

Known cost of TP/EP:
- **Context.** The C4 compressed stores are split by position across the two ranks (#710,
  `research/dsv4f-bringup-20260923/kv-split/`): 8.5 KB per token per rank at 1M against 13.8 KB
  replicated. A served session reaches 1M plain and 500k with DSpark, up from 800k and 300k. The
  cost is 0.4% to 2.0% decode and about 1% TTFT on the SE pair.

Receipts: `research/dsv4f-bringup-20260923/tpep-default/RESULTS.md`, `tp-rows/RESULTS.md`,
`dspark-ep/RESULTS.md`, `ceiling/CEILING.md`.

#### Before the flip: PP-2, 2026-09-23

Medians of four boots per arm on the Workstation pair: plain 63.34 greedy / 57.38 sampled (TTFT
192 ms), DSpark 76.07 / 62.47 (TTFT 240 ms).

What the rows carry, each step measured against the tree before it with the same text on every
prompt:

| step | plain greedy | DSpark greedy |
|---|---|---|
| one-token MoE stream visitor (#664) | +29.2% over the sktail tail | |
| multi-row stream visitor on the verify rounds (#669) | | +26.8% |
| deferred MoE checks (#670): one fault readback per transaction instead of 129 synchronizing reads per step | +6.8% | +2.7% |
| small-kernel diet (#339): one fused HC finish and one fused Q norm/pack per one-token step | +4.3% (measured before #670) | flat |
| latency kernels: register-resident rmsnorm, one-warp router, one-warp expert prefix, grouped `wo_a` on the dense-fast program | +13.3% (55.92 -> 63.34) | +3.9% (73.23 -> 76.07) |

The diet and grouped `wo_a` change one-token programs only; the other latency kernels run on
verify rows too, where a round pays its layers once per 3.56 committed tokens. That is why
DSpark gains less than plain from these steps.

- The first pair measured for this bring-up had the hardware power brake latched (722 MHz
  effective SM clock behind a reported 2865 MHz) and produced 15.86 / 22.07 tok/s. Those rates
  are withdrawn; the correctness verdicts from that pair stand.

- Served DSpark greedy equals served plain greedy on every tested prompt since #660. Before that
  fix the two differed on 8/8 prompts.
- Per-token weight traffic is 11.44 GB. At the measured 1.54 TB/s practical read bandwidth
  that bounds c1 at 135 tok/s for PP-2 and 270 tok/s for TP-2 (`ceiling/CEILING.md`).

Receipts: `research/dsv4f-bringup-20260923/` (`REBASELINE.md`, `m1-stream-664/RESULTS.md`,
`mrow-stream/RESULTS.md`, `moe-defer-670/RESULTS.md`, `small-diet/RESULTS.md`, `latency/RESULTS.md`, `PRIOR-ART.md`, `power-brake/POWER-BRAKE.md`, `spec-identity-660/RESULTS.md`; `BASELINE.md` is the withdrawn braked-pair run).
Deeper history: the [DeepSeek section](#deepseek-v4-checkpoint-dirs-serve-through-their-own-door-lanedsv4-flash-revival-20260822).
Do not infer production support from a successful load or prompt.

## Hy3 card: artifact hashes

From `docs/models/hy3.md`, lines 43-54 at d1ed663256.

The complete tensor payload is 180,826,481,152 bytes (168.408 GiB) across 99 shards. The full MTP
block is 2,295,901,696 bytes (2.138 GiB); removing it is not a supported profile.

The sealed 108-file artifact manifest is
`d63f8c3da9ab144d42dbfc1136d05294e26d5b7b7a40b114bf4b301359f4092c`.
Its generated metadata hashes are:

- `config.json`: `3cb16aa29d0046ffddd2f8a4866e4c7511e4018c6fced8dd913d1a788d787af9`;
- `hf_quant_config.json`: `38e5689cd6847427cc28c26c3cd3ca30568822bf311f479f11d21cf8ab632d2e`;
- `model.safetensors.index.json`:
  `0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd`;
- logical tensor census: `566db2975edac5cd1a86061ec6943988ef695cc8ae8c6cda050ad0d354ae2600`.

## Hy3 card: required RTX gate and the 2026-08-30 outcome

From `docs/models/hy3.md`, lines 98-141 at d1ed663256.

*The outcome below is reported; its oracle and serve receipts are not tracked. The `MEMRA_BF16_MMV` line is stale: since 2026-11-04 `MEMRA_PARALLEL=auto` turns BF16 residency on for a routed W4A16 artifact placed expert-parallel.*

### Required RTX gate

Use only the exact RTX PRO 6000 Blackwell topology being qualified. Load through the structural
entrypoint:

```bash
MEMRA_PARALLEL=auto \
MEMRA_PARALLEL_DEVICES=0,1,2,3 \
memra-server
```

For this exact artifact, the bound tensor census must select whole-expert EP-4, report 79 routed
trunk layers, retain the MTP expert bank on the root, and preserve the configured per-card reserve.
No HY3 layer list is an input. Then require:

1. immutable artifact and binary manifests;
2. finite native forward and same-artifact external-oracle comparison;
3. plain-vs-MTP greedy identity with nonzero acceptance at K=1..8;
4. explicit eager, batched, PP, MTP-verify, cache, and rewrite engagement receipts;
5. vendor-default sampled serving, tools/reasoning, concurrency, context/admission, stress,
   cache-on multi-turn, and rollback gates.

Qualification outcome on four RTX PRO 6000 Blackwell Server Edition cards:

- automatic placement selected the generic W4A16 whole-expert backend from `ModelPlan` plus the
  artifact census; the checkpoint estimates were about 60.05 GB on the root and 40.26 GB on each
  peer before the 6 GiB reserve;
- same-artifact vLLM pipeline oracle: argmax equal, top-20 overlap 20/20, cosine
  `0.9995285974311506`, RMSE `0.05849238475772628`, mean absolute error
  `0.046138540558894096`, maximum absolute error `0.30402064323425293`; the pack records the
  predeclared elementwise checkpoint bound as absolute error `<=1.0` with equal argmax;
- greedy MTP K=1..8: target-identical at every K, with nonzero acceptance at every K;
- vendor-default sampled Memra serving (temperature 0.9, top-p 1.0), native tools,
  `reasoning_effort` high/none, concurrent sessions, cache reuse, client-abort rollback, and
  explicit MTP engagement all pass;
- PP-4 wavefront: default remains off. The first arm improved four-client sampled throughput but
  failed full-batch serial-vs-wave identity because wave decomposition narrowed the BF16 reduction
  width. `MEMRA_BF16_MMV=1` restores one per-row BF16 program and passes the B=4/B=8 identity
  matrix with live overlap; it remains an explicit numeric-class door pending its own sampled
  quality/performance and default decision. Without it the runtime now fails closed.

No HY3 DFlash2 checkpoint or Memra HY3 DFlash2 consumer existed at the 2026-08-30 live check.
HY3 DFlash and DFly releases are different draft architectures, so embedded MTP remains the current
speculative path. Recheck this volatile decision before promotion.

## Gemma 4 E4B card: qualification record (2026-11-04)

From `docs/models/gemma4-e4b.md`, lines 39-68 at d1ed663256.

### Qualification record

RTX 5090 Laptop (driver 615.71.09), 2026-11-04, sm_120a builds. The engine cells (load, run-gen,
calibrated argmax, speculative, verify, graph) were re-run on the fix-stack commit `fbdd631927`;
the pipeline-refusal and serve cells come from the lane's first run at `09348940b8`, whose binary
hashes are in `serve-09348940b8/binaries.sha256`. Raw logs for every cell are in
[`research/modelplan-onboarding-e4b-20261104/`](../../research/modelplan-onboarding-e4b-20261104/RESULTS.md),
which also backs the TokenizerTemplate and TensorCensus gates in `docs/support-records.toml`.

Artifact `gemma-4-E4B_q4_0-it.gguf` (5,154,941,280 bytes, from
`google/gemma-4-E4B-it-qat-q4_0-gguf`), drafter `gemma-4-E4B-it-assistant.Q8_0.gguf`
(98,653,280 bytes). Each cell is a single run.

| Cell | Command | Result |
|---|---|---|
| Load | `memra model inspect <gguf> --against gemma4_e4b`, then every run below | 666 tensors bound by the pack contract; Config, TokenizerTemplate and TensorCensus pass; plan_sha256 `5b09a55734a8b55d61d8eceb3d16742fdb1347d0bf462e5bb743326c44589f7a` |
| Tiny plan | `memra model verify tiny --against gemma4_e4b` | pinned oracle matches; an independent NumPy forward (window 2, weighted PLE norms) agrees to max abs 1.8e-7, and its red arms (no PLE tail, no KV sharing, no window, unit PLE norms) all flip argmax |
| run-gen argmax | `MEMRA_NGEN=20 run-gen` on the pinned Gemma chat ids | prefill/decode MATCH, batched prime MATCH, 20 tokens identical to the `e4b` fast-gate golden |
| Calibrated argmax | `tools/argmax-margin-gate.sh <gguf>` (board-2048, window 12, max flips 1) | PASS, flips 0, bad 0, prime MATCH |
| Speculative K=1..8 | `MEMRA_SPEC=<K> MEMRA_DRAFT=<assistant> MEMRA_NGEN=64 gemma-gate` | stream agreement 64/64 at every K; accepted/drafted 26/38 (0.684) at K=1, 33/81 (0.407) at K=8 |
| Batched verify | `MEMRA_VERIFY_GATE=7 gemma-gate` on the 1,736-id depth prompt | PASS |
| Graph door | `MEMRA_E4B_GRAPH_GATE=256 gemma-gate` | 256/256 tokens identical to eager |
| Pipeline refusal | `MEMRA_PP_STAGES=2 run-gen <gguf>` | load refused before any upload; blockers name PerLayerInputEmbedding, PerLayerInputGate and SharedKvState |
| Serve, plain surface | `tools/serve-smoke.sh <gguf>` | 7/7: model list, chat, stream, completions, greedy determinism, 3 concurrent chats, long generation |
| Serve, gemma4 chat arm | `GEMMA_MODEL=<gguf> tools/serve-smoke.sh <gguf>` | FAIL: thinking-off content is `The user is asking ... exactly that.<channel|>ok`; a `reasoning_effort: low` chat returns no reasoning |
| Serve, cache metering | serve-smoke's cache-meter gate | not applicable: the eager-only class takes no prefix snapshot, so 0 prefix-cache hits where the gate expects 4 |
| Serve, identity | one greedy chat solo, then the same chat 3x concurrent (96 tokens) | all 3 identical to solo |

`run-spec` needs an embedded MTP head, so the Gemma assistant drafter is gated with
`gemma-gate` stream agreement, the same check the 31B runs in `tools/local-ci.sh`.

## GLM-5.3-Flash card: engine surface and scope lines

From `docs/models/glm53-flash.md`, lines 14-27 at d1ed663256.

*Stale: `glm_dsa` is not loader-only; the pack declares NativeReference on a tiny fixture (record `glm_dsa-reference`). <!-- support: glm_dsa-reference -->*

- Engine surface: the `glm5_next` model plan, a hand-written multi-card TP path
  (`crates/memra-engine/src/glm5_tp.rs`), MTP spec (`glm_spec.rs`), and vision
  (`vision_glm5.rs`). The template's always-rendered `Reasoning Effort: {Low|High|Max}`
  line and unconditional `<think>` tail are probed and documented in
  [serving](../SERVING.md). The effort ladder is exactly those three rungs with `max`
  the template default; a client `medium` maps to `high` (issue #75), `max`/`xhigh`/
  `ultra` reach `max` by name, and `none`/`minimal` are a named 400 (no off switch).
- Bring-up receipts: `research/glm53-flash-bringup-20260827/` (tensor census, 262k
  two-card, batched decode gate); vision and multi-card batteries under
  `research/glm5-vision-20260830/` and `research/glm53-vision-ppn-20260901/`.

Checkpoint-, quantization-, and hardware-specific: none of this transfers to GLM-5/5.2
(`glm_dsa`, loader-only) or to any other family by analogy.

## DGX Spark (GB10) card: support per machine (2026-11-02 text)

From `docs/rigs/dgx-spark.md`, lines 15-21 at d1ed663256.

*Qwen3.6-35B-A3B was also measured on GB10 on 2026-11-10 ([device penalty decision](../decisions/GB10-DEVICE-PENALTY-DEFAULT.md)).*

Model support is per machine. No model has a support state on this machine until it is measured
here: module load, `kernel-check`, `run-gen` argmax and `run-spec` K=1..8 on the GB10 itself,
then the model's own gates. `docs/support-records.toml` has no GB10 record.

- **Qwen3.8-27B:** measured below; no support record yet (`docs/support-records.toml`).
- **Qwen3.8-Flash-Next:** not planned for this machine.
- Every other model: not planned until Qwen3.8-27B has a result here.

## DGX Spark (GB10) card: measured, 2026-11-02 to 2026-11-07

From `docs/rigs/dgx-spark.md`, lines 45-126 at d1ed663256.

*Superseded on prefill: the F8F4 v3 mainloop is the GB10 default since 2026-11-08 ([decision](../decisions/GB10-F8F4-GEMM-V3.md)), and the DFlash2 c1 conversion fix landed 2026-11-06 (f045e69cee) without a re-measure of these cells.*

### Measured (2026-11-02, one GB10, `121a` built on the device from `b014549cb9`)

Correctness: kernel-check ALL GREEN (92 cells; 22 skipped for models not on the device, every
NVFP4 GEMM/MMVQ cell among them, so the 27B's NVFP4 path is covered end to end below);
`run-gen` argmax MATCH 4/4 on the GGUF and 4/4 on NVIDIA's NVFP4 safetensors; `run-spec`
K=1..8 self-consistency PASS ([receipt](../../research/gb10-tuning-20261102/correctness-gb10-b014549cb9.log)).

Agentic bench (48 turns from 24 coding-agent sessions, thinking on, T=1, 768 max new tokens,
262,144 context, 32 sessions), aggregate tok/s, one run per cell:

| arm | c1 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| GGUF + masked MTP draft (default K table) | 21.43 | 39.36 | 59.36 | 66.77 | 66.45 |
| GGUF + DFlash2 drafter, `MEMRA_MTP_SKIP=1` | 12.02 | 39.77 | 59.70 | 66.61 | 68.02 |
| NVIDIA NVFP4 safetensors + DFlash2, `MEMRA_MTP_SKIP=1` | | 33.50 | 50.90 | 54.90 | 56.54 |

From c4 up the spec gate serves batched plain decode, so the GGUF arms converge. The DFlash2
arm serves plain at c1 too, because its multi-turn hits land on plain prefix entries (fix in memra#1031). `MEMRA_SPEC_K_CACHED_LONG_TRIM=2` is flat at every concurrency (21.29 at c1). No cell returned an
HTTP error, and no `[shed]` line was logged. Cold prefill: 11.3 s at 8,192 tokens, 49.3 s at
32,768.

Long context: the 264-task quality panel (one 131,550-token session, then 164 HumanEval and 100
GSM8K tasks on it, 8 concurrent) scored 158/164 and 98/100 on the GGUF arm; MemAvailable stayed
at or above 17.0 GiB (the bench's own 3 GiB MemAvailable watchdog never fired). A non-streaming
request with a 150,413-token prompt answered HTTP 200 in 317 s under the derived deadline (GGUF
DFlash2 boot with the MTP head loaded; 5 output tokens, so it tests the prefill allowance only).

Prefill: the W4A8 NVFP4 MMQ is 61% of an 8,788-token prefill (776.6 tok/s monolithic). GB10 now
takes the F8F4 route (`MEMRA_MMQ_F8F4`, e4m3 activations) by default: 15.4% faster in the
monolithic `run-gen` prefill, 43.5 s against 49.3 s of served cold prefill at 32,768 tokens, one
greedy acceptance run per arm on three prompts at 0.702 against 0.672, c32 68.21 against 66.45
tok/s, and the quality panel at 158/164 and 97/100 (W4A8: 158/164 and 98/100). `MEMRA_MMQ_F8F4=0`
restores W4A8 ([research/gb10-tuning-20261102](../../research/gb10-tuning-20261102/)).

Long-context prefill (2026-11-06, [research/gb10-prefill-20261106](../../research/gb10-prefill-20261106/)):
served cold TTFT of one fresh prompt with the exact levers of that lane (two-head prime attention
by default on GB10, no dead f16 operand pass, no slab copies of the dense FFN's GEMM outputs,
silu(gate)*up fused into the down projection's e4m3 quantize, each shared activation quantized
once per GEMM group, and 8,192-row prime chunks, which were then rejected for peer latency and
did not land; they were worth about 0.6% at 131k), same tokens on every prompt: 44.26 -> 36.16 s
at 32,784 tokens (-18.3%), 239.0 -> 181.1 s at 131,051 (-24.2%) and 637.5 -> 457.2 s at 259,951
(-28.3%), two fresh boots per arm in both orders in each of the lane's two campaigns on one unit.
MemAvailable stayed above 65.2 GiB at 259,951 tokens.

F8F4 GEMM v2 (2026-11-06, same lane, `MEMRA_MMQ_F8F4_V2`): F8F4 GEMMs of at least 2,048 rows fold
the weight once per call into a pooled e4m3 image and run a pipelined 16-warp 256 x 128 tile,
bit-identical to the v1 tile (about 111 TFLOPS in the tile against 79 for v1; the epilogue's
ceiling at 16 warps per SM is 132). Served cold TTFT with the default 4,096-row prime chunks, two
boots per arm in both orders on each of the two units: 36.25 -> 32.54 s at 32,784 tokens (-10.2%),
182.6 -> 168.1 s at 131,051 (-7.9%) and 461.3 -> 433.1 s at 259,951 (-6.1%) on unit 1, unit 2
within 0.5 s, same answers, MemAvailable floor unchanged (66.2 GiB or more at 259,951). With
8,192-row chunks v2 saves more (-10.7% at 131,051), since it builds each weight image once per
chunk. At 131,051 tokens attention (50.8 s) is now the largest part, then the GEMMs and the Gated
DeltaNet kernels (about 20 s).

Prime attention ring (2026-11-07, [research/gb10-prefill-attn-20261107](../../research/gb10-prefill-attn-20261107/),
`MEMRA_PRIME_KV_H2P`): h2's row program on a three-stage K/V ring handed over by mbarriers, the
second query head one QK behind the first, `ldmatrix.shared` on precomputed offsets; bit-identical
to h2. Device probe at depth 131,072: 85 against 68 TFLOPS (bf16 `mma.sync` ceiling about 102).
Served cold TTFT against main `6f6d816757`, B H H B B H on unit 1, three fresh boots per arm:
32.57 -> 32.06 s at 32,784 tokens (-1.6%), 167.9 -> 158.2 s at 131,051 (-5.8%) and 432.7 ->
393.9 s at 259,951 (-9.0%), same answers in every boot, MemAvailable floor 66.6 GiB.

GDN traffic, fused norms and gate/up (2026-11-07, [research/gb10-prefill-next-20261107](../../research/gb10-prefill-next-20261107/),
[decision](../decisions/GB10-PRIME-GDN-GLUE.md)): the GDN compact q/k layout at 3:1 (it had been
limited to 2:1, so this model's 16 key / 48 value heads ran broadcast), heads-fastest K2/K3/K5, a
fused K4+K5 state and output pass (`MEMRA_GDN_K45`), one add + RMSNorm kernel per residual site
writing the next GEMM's F8F4 v2 / q8_1 activation (`MEMRA_PRIME_NORM_FUSE`), a tiled beta/alpha
dp4a matvec (`MEMRA_DP4A_TILED`) and gate and up in one F8F4 v2 launch writing the down GEMM's silu
activation (`MEMRA_FFN_GATEUP_FUSE`); all bit-identical. Served cold TTFT against main `6f5e03e74a`,
B L L B B L on unit 2: 31.89 -> 25.38 s at 32,784 tokens (-20.4%), 157.71 -> 131.69 s at 131,051
(-16.5%) and 391.24 -> 340.89 s at 259,951 (-12.9%), same answers in every boot, MemAvailable floor
66.2 GiB; unit 1 within 0.6%.

Second pass (2026-11-07, same receipts, [decision](../decisions/GB10-PRIME-GLUE-2.md)): the residual add
in the F8F4 v2 GEMM epilogue (`MEMRA_PRIME_RESID_EPI`), K2 at C = 32 in half the shared memory
(`MEMRA_GDN_K2_32`), the GDN conv with the q/k L2 norms (`MEMRA_GDN_CONV_L2`) and the attention q path in
one kernel (`MEMRA_PRIME_Q_FUSE`); all bit-identical. Served cold TTFT against the first pass, one boot
per arm on unit 2: -3.95% / -3.26% / -3.39% at 32,784 / 131,051 / 259,951 tokens.
Both passes against main `46172c1bac`, B L L B on unit 2: 31.94 -> 24.39 s at 32,784 tokens (-23.6%),
157.42 -> 127.35 s at 131,051 (-19.1%) and 392.88 -> 332.47 s at 259,951 (-15.4%), same answers,
MemAvailable floor 66.2 GiB in the lane boots (66.1 in every boot).

## DGX Spark card: the pre-unified-memory defaults (the "was" column)

*Moved 2026-11-12 from the defaults table of docs/rigs/dgx-spark.md. The values before the
unified-memory keying ([research/gb10-defaults-20261101/](../../research/gb10-defaults-20261101/)):*

| default | was |
|---|---|
| free memory for sizing | `cuMemGetInfo` free |
| `MEMRA_PREFIX_CACHE_MB` | 2 entries: 15.9 GB for Qwen3.8-27B at 262,144 tokens |
| `MEMRA_STREAM_TTFT_MS_MAX` | 90,000 ms |
| `MEMRA_SSE_PREFILL_COMMIT_MS` | off |
| non-streaming `timeout_ms` default/max | 90,000 ms |
