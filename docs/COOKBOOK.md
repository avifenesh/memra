# Cookbook — copy-paste serving configs, per model, per card

Every block below is a configuration that has actually run — on the named card, with the
named artifacts — not a template. Blocks come from three places: the qualification cells in
[docs/PERFORMANCE.md](PERFORMANCE.md), the published model cards on Hugging Face, and serving
configurations that have carried real traffic. Numbers live in
[PERFORMANCE.md](PERFORMANCE.md); this file is the commands.

Two things to know before pasting:

- **The attach is a log line, not the absence of an error.** A wrong draft path or a
  misspelled flag does not fail — the trunk's embedded full head drafts instead and
  everything still works, just slower. After boot, look for the line that proves the
  artifact you chose is the one that loaded (each block names it).
- **`hf:` specs download on first use.** `MEMRA_MODELS` and draft flags accept
  `hf:owner/repo:file-substring`, so most blocks below need no manual download step.

Flags are documented in [FLAGS.md](FLAGS.md); what is and is not supported per
(model, quantization, drafter) is [MODELS.md](MODELS.md). If your card or model is not
here, that is a statement — a block only appears once the configuration has receipts.

---

## Qwen3.8-27B

Dense hybrid (GDN + gated attention). Both paths are tuned and supported; choose from receipts for
the exact artifact and card rather than transferring a format-level performance claim.
Artifacts: [Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF)
(trunk, pre-trimmed masked MTP head, ranks `.txt`).

### RTX 5090 / 24 GB class — DFlash2 q4 + masked head (default)

```bash
MEMRA_COMPAT=openai \
MEMRA_MODELS="q38=hf:Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp" \
MEMRA_DSPARK_SPEC=1 \
MEMRA_DSPARK_DRAFT=hf:Avifenesh/Qwen3.8-27B-DFlash2-memra \
MEMRA_FRSPEC_TRIM=hf:Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF:q38-ranks-sxc32768.gguf \
MEMRA_DFLASH_PREC=q4 \
MEMRA_CTX=8192 \
MEMRA_MAX_SESSIONS=8 \
MEMRA_PREFIX_CACHE_MB=0 \
memra-server
```

This is the measured local default: the 5090 ABBA receipt is 83.85 -> 87.10 E2E tok/s
(+3.87%) on the held-out agentic pack. Verification stays full-vocab, so the rank mask moves
proposal cost and acceptance, never emitted tokens. The 24 GB envelope is intentionally bounded
to `MEMRA_CTX=8192` and eight sessions; set a larger context only after sizing it on the actual
workload. Boot proof: the resolved q4 precision, the DFlash route path, and
`[dspark] q38: DFlash2 draft head TRIMMED to 32768 rows`.

The masked MTP head remains the rollback path: unset `MEMRA_DSPARK_SPEC` and use the prior
`+hf:Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF:frspec-sxc32768` attachment.

### RTX PRO 6000 Blackwell 96 GB — long-context serving

The shape that serves real traffic: full 262K context, 32 sessions, 16 GB prefix cache.

```bash
MEMRA_COMPAT=openai \
MEMRA_MODELS="q38=/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
MEMRA_MTP_DRAFT=/models/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf \
MEMRA_CTX=262144 \
MEMRA_MAX_SESSIONS=32 \
MEMRA_PREFIX_CACHE_MB=16384 \
memra-server
```

On a 24 GB card, set `MEMRA_CTX` to the workload instead — 262144 reserves KV for clients
that never send it; [SERVING.md](SERVING.md#admission) covers the ladder and the
`MEMRA_CTX` fallback trade.

### RTX PRO 6000 Blackwell — DFlash2 drafter (the measured-fastest spec route)

The [DFlash2 block-diffusion drafter](https://huggingface.co/Avifenesh/Qwen3.8-27B-DFlash2-memra)
replaces the MTP arm for this model (arming it disables MTP spec — two spec programs never
coexist). Defaults do the tuning: the drafter quantizes to q4_0 at load
(`MEMRA_DFLASH_PREC=q4`) and the round consumes the FR-Spec vocab trim when armed. Output is
byte-identical to plain decode by construction — the verifier arbitrates every committed token.

```bash
MEMRA_COMPAT=openai \
MEMRA_MODELS="q38=/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
MEMRA_DSPARK_SPEC=1 \
MEMRA_DSPARK_DRAFT=hf:Avifenesh/Qwen3.8-27B-DFlash2-memra \
MEMRA_FRSPEC_TRIM=q38-ranks-sxc32768.gguf.txt \
MEMRA_CTX=262144 \
MEMRA_MAX_SESSIONS=32 \
MEMRA_PREFIX_CACHE_MB=16384 \
memra-server
```

Boot receipts: `[dspark] q38: DFlash2 draft head TRIMMED to 32768 rows` and the route line.
This is the configuration serving both production Qwen3.8 origins since v0.113.0. Measured
on this card at the vendor-default SAMPLED shape — the shape real traffic has (memra >=
v0.113.0, sampled sessions stack in the spec-gate LOW band; x3 interleaved, medians):
c=1 127/117, c=2 128/120, c=4 87/85 agg tok/s vs the MTP head — the DFlash2 route wins
every rung. Single-stream wall rates on the same card: prose ~131-146, code ~208-239,
digit-heavy ~287-339 tok/s (drafter acceptance rises with output predictability). The
greedy instrument numbers (chat 142.9 / agentic 157.2 vs MTP 126.9 / 148.6) remain the
byte-exactness receipts, not the serving verdict. Under load the spec-gate sheds to plain
batching at c>=4 (aggregate parity).

Multi-turn conversations reuse the parked session by default (`MEMRA_REUSE_POOL`, no extra
flag): turn N+1 primes only the new suffix onto the parked trunk cache + draft KV instead of
re-prefilling the whole conversation. Measured on this card, 8-turn conversation: turn-over-turn
TTFT −16% → −83% vs cold re-prime, growing with depth. Resume receipt in the log:
`[worker] dspark-reuse: N committed tokens resumed`. EOS and `max_tokens` both clamp the
committed state to the public stream before park. A request whose context cap outgrew the parked
allocation still serves cold, named in the log; a max-token-terminated session requires a
non-empty next-turn suffix to resume rather than re-emitting from its terminal boundary.

### NVFP4 safetensors trunk + ranks trim

No GGUF anywhere: the compressed-tensors NVFP4 checkpoint
[`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
(`16b6615af3548b88e2d8e382457bc705b00479cf`) loads natively, its own
`model_mtp.safetensors` head drafts out of the box, and the ranks `.txt` self-trims the head at
load (byte-level row gather from the trunk's own `output.weight`, zero requant; memra ≥ v0.84).
Do not substitute the official FP8 checkpoint: its BF16 output head is a different loader class
and is not this measured NVFP4 configuration.

```bash
MEMRA_FRSPEC_TRIM=q38-ranks-sxc32768.gguf.txt \
MEMRA_MODELS="q38=/models/Qwen3.8-27B-NVFP4" \
memra-server
```

`MEMRA_FULL_PREC=1` disables the trim by design — the exactness ceiling wants the natural
full head.

---

## Qwen3.5-9B

Dense. NVFP4 GGUF on `sm_120a`, Q8_0 on the H100 lane. MTP + own-gen trimmed draft.

### RTX 5090 — NVFP4

```bash
MEMRA_COMPAT=openai \
MEMRA_MODELS="q9=/models/Qwen3.5-9B-NVFP4.gguf" \
memra-server
```

The tuned path is the default — no flags needed for speed. `MEMRA_SPEC_K` remains the
operator pin, including `0` for plain decode.

---

## Ornith-1.5-35B-A3B

MoE. NVFP4 GGUF with a trained MTP head; masked own-gen ranks trim adopted as the serving
default (see the board entry). NVFP4 is not an upstream llama.cpp tensor type — this file
runs on memra. Artifacts:
[Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF](https://huggingface.co/Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF).

### RTX PRO 6000 Blackwell — trunk + masked head

```bash
MEMRA_COMPAT=openai \
MEMRA_MODELS="ornith=hf:Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF:Q5K-mtp+hf:Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF:frspec-owngen32768" \
memra-server
```

Without the `+draft` suffix the trunk's embedded head drafts — still correct, the mask
only moves cost. Boot log proof: `[worker] ornith: regime draft attached (…owngen32768.gguf)`.

Ranks-only alternative (self-trim at load, no separate draft file — the adopted serving
default, board entry 412c45b0):

```bash
MEMRA_FRSPEC_TRIM=ornith15-ranks-owngen-32768.txt \
MEMRA_MODELS="ornith=hf:Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF:Q5K-mtp" \
memra-server
```

---

## Gemma-4 31B

Dense, vision-capable. The configuration below is the shape that served real traffic on an
RTX PRO 6000: Q6_K trunk, the official NVFP4 MTP drafter with ranks trim, and the Gemma
vision seam.

### RTX PRO 6000 Blackwell — serving with vision + drafter

```bash
MEMRA_COMPAT=openai \
MEMRA_MODELS="gemma31=/models/gemma-4-31B-it-Q6_K.gguf" \
MEMRA_DRAFT=/models/gemma-4-31B-it-official-NVFP4-MTP.gguf \
MEMRA_GEMMA_DRAFT_RANKS=/models/gemma31b-ranks-32768.gguf.txt \
MEMRA_GEMMA_TRIM_ADAPT=512 \
MEMRA_GEMMA_VISION=1 \
MEMRA_GEMMA_MMPROJ=/models/gemma-4-31B-it-mmproj.gguf \
MEMRA_CTX=262144 \
MEMRA_MAX_SESSIONS=16 \
MEMRA_PREFIX_CACHE_MB=16384 \
memra-server
```

The Gemma vision seam is `MEMRA_GEMMA_VISION=1` + `MEMRA_GEMMA_MMPROJ` — do **not** set
`MEMRA_VISION_DIR` here; that is the Qwen tower seam, and one vision path per worker.

---

## Step-3.7-Flash 196B-A11B

MoE, two-card PP-2. Receipts: `research/pp2-batch-20260806/`,
`research/pp2-spec-20260806/`, `research/pp2-hardening-20260806/` — rig 2× RTX PRO 6000
Blackwell Server Edition 96 GB. The split adds **zero deviation**: every f32 logit of every
step bit-compared, 0 differing bits across all seven gate configs.

### 2× RTX PRO 6000 Blackwell — PP-2

```bash
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
MEMRA_COMPAT=openai \
MEMRA_MODELS="step=/models/Step-3.7-Flash-IQ4_XS.gguf" \
memra-server
```

The request-conditioned K policy selects `K=0` on this sharded shape by itself — no
`MEMRA_SERVE_SPEC=0` needed. Boot log proof:
`[pp] cross-device transport: stage0=dev0 stage1=dev1` — a config that silently did not
split is the failure mode that banner exists to rule out.

---

## Everything else

The full support table — every (model, quantization, drafter) combination and the card
class it is qualified on — is [MODELS.md](MODELS.md). The audited flag catalog is
[FLAGS.md](FLAGS.md). Serving operations (admission, caching, auth, SLO) are
[SERVING.md](SERVING.md). If you run a configuration worth a block here — a different
card, a different quant — the
[hardware report template](../.github/ISSUE_TEMPLATE/hardware-validation.md) is how it
gets in with receipts.
