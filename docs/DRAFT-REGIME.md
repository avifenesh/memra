# The draft regime

How to build and attach a speculative draft for a model that drafts with an MTP/NextN head: the
three rules a draft must follow, the two-command build, the attach forms and the checklist for a
new model. For anyone adding speculative decoding to a GGUF model.

Scope: MTP/NextN heads, embedded in the GGUF or shipped as a standalone draft file. Other drafter
classes have their own attach and gates: the Gemma assistant drafter (`MEMRA_DRAFT`, with the ranks
trim below), the DFlash2/DSpark block drafters (`MEMRA_DSPARK_SPEC` plus `MEMRA_DSPARK_DRAFT`, and
`MEMRA_DSV4_DRAFTER` on DeepSeek-V4), the GLM-5.3 MTP block (`MEMRA_GLM5_MTP`, `MEMRA_GLM5_SPEC`)
and Hy3's eager draft chain. The drafter table is in
[ARCHITECTURE.md](../ARCHITECTURE.md#8-speculative-decoding), flag contracts in
[FLAGS.md](FLAGS.md), and each model card names its drafter. Verification always runs the target's
full vocabulary, so a draft or a trim moves speed and acceptance, never output.

## The three laws

Each was measured, and each was violated at a cost before it was written down (evidence at the end).

1. **Per-model, every time.** Rank files and draft heads are artifacts of the exact serving model's
   vocabulary and output distribution. Derive ranks from the model's own generations for every
   model and every requant; external text is prompts only, never the counted distribution. Derive
   with the chat template on when you serve chat (`--raw` is for pure continuation), cover every
   prompt class you serve, and generate at least 4x topN tokens. A trim wins any cell whose emitted
   distribution it covers and loses any cell it does not.
2. **Byte-verbatim extraction.** The draft block comes out of the serving GGUF's own bytes
   (`tools/extract_mtp_draft.py`); an extracted draft matches the embedded head's acceptance. Never
   re-convert the MTP block from the HF checkpoint: converter-produced drafts collapse in acceptance.
3. **Quantize after trimming, judge by end-to-end tok/s.** Head to NVFP4, block to Q4_K_M. The
   verdict metric for any draft or trim decision is end-to-end tok/s under the board protocol;
   acceptance explains why, it never decides.

## Build a draft file

Requirements: a release build of `frspec-owngen`
(`cargo build --release -p memra-engine --bin frspec-owngen`), a GPU that holds the model, and a
llama.cpp checkout whose `llama-quantize` has the NVFP4 type. `tools/make-trimmed-draft.sh` reads
`MEMRA_GGUFPY` (llama.cpp's `gguf-py` directory), `MEMRA_QUANTIZE` (the `llama-quantize` binary)
and `MEMRA_CONVERT_PY` (a Python with numpy, default `python3`). The first two default to one
developer machine's layout, so set them.

```bash
# 1. ranks from the model's own generations: the built-in mixed prompt pack, or your own
#    prompts (.txt/.md file, directory, .jsonl) or a Hugging Face dataset as hfds:owner/name
./target/release/frspec-owngen model.gguf ranks.gguf 32768

# 2. extract + trim + quantize -> the draft file (block Q4_K_M, head NVFP4)
MEMRA_GGUFPY=/path/to/llama.cpp/gguf-py \
MEMRA_QUANTIZE=/path/to/llama.cpp/build/bin/llama-quantize \
  tools/make-trimmed-draft.sh model.gguf ranks.gguf.txt draft.gguf 32768 [imatrix.gguf]
```

Step 1 writes `ranks.gguf` (the `d2t` table) and `ranks.gguf.txt` (one id per line, rank order),
and warns when the corpus is under 4x topN tokens. `--preset code|chat|agentic|mixed`, `--ngen N`
and `--temp T` (default 0, greedy) shape the corpus.

Check before trusting: `frspec-owngen model.gguf out.gguf --validate` reloads with the trim applied
and A/Bs speculative decoding (K=3) against the untrimmed embedded head. It prints `GOOD` when the
trim wins end to end, `WASH` within 2% (keep it only if the VRAM saving matters) and `BAD` below
that (widen topN or regenerate the ranks). A model without an embedded MTP head skips the check.

On a shared GPU, run the corpus in bounded chunks: `--corpus-out corpus-ids.txt --limit 64`
generates 64 prompts, appends their ids and exits; rerunning the same command resumes, and the
final chunk writes the ranks. Greedy decoding makes the chunked corpus identical to a single run.

## Attach

| Form | What it does |
| --- | --- |
| nothing | an embedded MTP/NextN head drafts with its full vocabulary head |
| `MEMRA_FRSPEC_TRIM=<ranks.txt\|ranks.gguf\|hf:...>` | trims the embedded head at load to the ranked rows; no draft file needed |
| `MEMRA_MODELS="name=/model.gguf+/draft.gguf"` | attaches a draft file to one model (both parts accept `hf:` specs) |
| `MEMRA_MTP_DRAFT=<draft.gguf>` | attaches a draft file process-wide (`run-spec`, single-model servers) |
| `MEMRA_GEMMA_DRAFT_RANKS=<ranks.txt>` | trims a Gemma assistant drafter at load (Gemma variant below) |

The server refuses to start when a `+draft` path is missing, is not a file or fails to load, and
quotes the cause, rather than serving plain decode under a config that asked for speculative
decoding. Boot proofs: `[frspec-trim] self-trimmed head: 32768 rows of` for a load-time trim,
`[worker] <name>: regime draft attached (` for a draft file. Admission (`MEMRA_SPEC_GATE`) keeps
drafting off on a cross-device PP-2 placement, where plain batched decode measured faster; see
[FLAGS.md](FLAGS.md) and the [Step-3.7-Flash card](models/step37-flash.md).

## Targets that ship no NextN head (donor-block variant)

Some published GGUFs of supported architectures strip the NextN/MTP block entirely
(`nextn_predict_layers` absent, one fewer block: Ornith-1.0-9B and 35B, KAT-Coder-V2.5). Build the
draft from a same-backbone donor GGUF that carries the block, trimmed with the target's own ranks:
`make-trimmed-draft.sh <donor.gguf> <target-ranks.txt> <draft.gguf>`.

- Law 2 still holds (the bytes are the donor's) and law 1 is not relaxed: the ranks always come
  from the target's own generations.
- The donor must match the target's trunk interface. The loader asserts `n_embd` and head_dim,
  and `n_head` and `n_head_kv` for a natural NextN clone; the vocabulary must be identical for the
  `d2t` map.
- At load the draft's token embedding comes from the serving model's `token_embd`, so only the
  donor's NextN block and trimmed head ride along. Donor and target drift costs acceptance, never
  correctness. Gate it like any draft: `run-spec` K=1..8 self-consistency, acceptance above zero,
  adoption on end-to-end tok/s.
- Known donor pairs: Qwen3.5-9B-NVFP4-MTP for Ornith-1.0-9B; Qwen3.6-35B-A3B-UD-IQ4_XS (`blk.40`)
  for Ornith-1.0-35B and KAT-Coder-V2.5-Dev
  ([research/ornith-drafters-20260801/](../research/ornith-drafters-20260801/)).

## Targets that ship their NextN head as a separate file (Step-3.7-Flash)

StepFun publishes the three chained NextN/MTP blocks of Step-3.7-Flash in their own GGUF
(`Step3.7-flash-mtp-Q8_0.gguf`), so the trunk reads `nextn_predict_layers=0` and loads with no
head. Nothing needs to be built; attach the published head:

```bash
MEMRA_MODELS="step=/models/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf+/models/Step3.7-flash-mtp-Q8_0.gguf" \
  ./target/release/memra-server
```

The loader resolves the per-layer draft geometry (64 against 96 q-heads, the SWA(512) window, the
head-wise `attn_gate` width) from the drafter file's own arrays, so no geometry flag is involved.
A trunk whose pack declares an external MTP drafter and boots without one logs one WARN naming
plain decode and the attach string.

## Gemma variant

Gemma assistant drafters are standalone byte-verbatim GGUFs already (law 2 by provenance), attached
with `MEMRA_DRAFT`. The trim applies at load: `MEMRA_GEMMA_DRAFT_RANKS=<ranks.txt>`, the `.txt`
that `frspec-owngen` writes (it skips `--validate` here; Gemma drafters validate through
`gemma-gate`). Laws 1 and 3 apply unchanged.

With ranks set, the adaptive trim is on by default (`MEMRA_GEMMA_TRIM_ADAPT`, 512 spare head slots,
`0` for the static trim). Tokens the trim cannot propose arrive as verify corrections or with the
prompt; the loop writes their head rows into the spare slots and persists the ids to
`<ranks>.learned`, which the next load pre-fills, so a distribution's escapes cost one miss per
serve lifetime, not per request.

## Prebuilt drafts

Each model card names its drafter in its **Drafters** row; most cards do not say where the file
is published, so this section does. The drafts and ranks in
[huggingface.co/tiyuvta/memra-bench](https://huggingface.co/tiyuvta/memra-bench) (`drafts/`, with
per-file provenance) cover Qwen3.5-9B, Qwen3.6-27B and Qwen3.6-35B-A3B, plus own-generated ranks
for the Gemma 4 26B, 31B and E4B QAT trunks. Newer models publish theirs in the model's own
repository (for example `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF` and
`tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF`). Build your own for any other model, requant or
finetune: a finetune's distribution moved, so its draft must too (law 1).

## Checklist for a new model

- [ ] own-generated ranks from the published artifact (`frspec-owngen`)
- [ ] a draft file from `make-trimmed-draft.sh` (byte-verbatim block, NVFP4 head, Q4_K_M block),
      or a load-time trim with `MEMRA_FRSPEC_TRIM` when the GGUF carries its head
- [ ] `run-spec` K=1..8 self-consistency and acceptance above zero
- [ ] end-to-end A/B against no draft and against any prior draft under the board protocol
      (interleaved, N of 2 or more, power pinned, window validated); adopt only on an end-to-end win
- [ ] draft and ranks published with provenance (source model, rank corpus, commands)
- [ ] the model card names the drafter and links the published draft file

## Evidence

The measurements behind the laws, with their dates. Raw rows are in
`research/tune-data/rig5090.jsonl` unless a lane is named; the previous text of this page is in
[history/DRAFT-REGIME.md](history/DRAFT-REGIME.md).

- One draft file per model replaced the earlier mix of external drafts, separate trim files and
  env knobs: on the 27B board model, p1 +3.4%, p2 +2.5% and p3 +0.9% over the previous board
  configuration (2026-07-18). The script's recipe measured 101 tok/s at 85.2% acceptance (K=3) on
  the unsloth 27B artifact against 99.6 tok/s at 83.3% for the draft it replaced
  (`tools/make-trimmed-draft.sh` header, 2026-07-17).
- Law 1: foreign ranks cost 12 acceptance points on an identical tokenizer, and wiki-heavy ranks
  lose about 12 points on code prompts. A raw-derived rank set left a Gemma 31B chat cell with 10.9%
  structurally unproposable tokens, every one a guaranteed rejection, at -15 acceptance points
  (2026-07-19). An oracle control that injected the exact escapees into the trim set flipped a -17%
  cell to +2% at identical acceptance, and a class-by-K sweep showed the sign flips only with class
  coverage, never with K. A 2.5k-token corpus cost 6.5 acceptance points.
- Law 2: an extracted draft matched the embedded head at acceptance parity. Converter-produced
  drafts collapsed to about 35-39% acceptance with no tensor-level difference found; the route is
  deprecated and the cause is still open.
- Law 3: an NVFP4 head measured zero acceptance cost against q5_K at about a quarter of the bytes,
  and a Q4_K_M block measured faster and higher acceptance than Q8_0 (cheaper rounds waste fewer
  drafts).
- Gemma: the adaptive trim closed the one cell the static trim lost (31B chat -17% to +2.5% against
  untrimmed), and trim was at or above untrimmed on every measured cell (26B and 31B, N=2
  interleaved, 2026-07-19). Verdicts: 26B trim adopted, 31B trim adopted on both cells (depth +5.2%,
  chat +2.5% warm), E4B stays untrimmed (a small head, the trim buys nothing).
  Bring-up records: [research/gemma4-bringup/](../research/gemma4-bringup/).
