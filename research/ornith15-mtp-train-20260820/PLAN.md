# Ornith-1.5-35B-A3B — MTP head continued training (lane opened 2026-08-20)

Owner diagnosis (RECEIPTS.md §Head diagnosis, ../ornith15-st-nvfp4-20260819/): base
acceptance ~50% long-gen / ~25% serve-level is the ceiling, not the mask — the vendor
1-layer MTP head is undertrained for the RL'd trunk. Corpus-flavor ranks were refuted as a
lever. This lane retrains `mtp.*` on the trunk's own generations; trunk, embeddings and
lm_head stay frozen checkpoint bytes. Directive: run in parallel with the ST end-to-end
track, card never sleeps.

## Route (decided 2026-08-20)

Offline teacher-forced fine-tune of the checkpoint's own `mtp.*` weights (785 tensors,
~0.8B params: 1 full-attn decoder layer + 256-expert unfused MoE + fc/norm glue).
transformers 5.8.1 instantiates **zero** mtp modules (verified on meta device), so the
head is a custom torch module built from `Qwen3_5MoeDecoderLayer` + glue, loading the
`mtp.*` names directly. MTP needs 1 hidden tap × 2048 dim ≈ 4 KiB/token — offline capture
is viable (~15 GB for the full corpus), unlike the DFlash 5-tap blowup.

## Semantics (pinned to the memra serve program — the head must match what serves it)

| piece | source of truth |
|---|---|
| row j input | `concat(enorm(emb(t_j)), hnorm(h_{j-1}))` → `fc` (= eh_proj, in 2·n_embd) — spec.rs `mtp_kv_fill` ops A/1-5, pairing = token + PREDECESSOR hidden (`fill_prev`) |
| h_{j-1} | trunk hidden PRE final norm (pre-forward hook on `language_model.norm`) |
| mixer | one full-attention qwen3_5_moe decoder layer, causal over the FULL committed history (memra keeps a persistent draft-KV row per committed token) |
| chain depth (owner correction 2026-08-20) | serve drafts K tokens/round by CHAINING: depth-1 seeds trunk h, depth d>=2 seeds the head's OWN pre-mtp.norm output (op-10 h_nextn) with the drafted token; depth-1-only training misses the serve distribution at depth>=2. Trainer v2 unrolls D=3 depths (hqmtp chain-rollout precedent), teacher-forced tokens, gradient through carrier AND chain K/V, serve-exact attention: (p,d) attends depth-1 rows q<=p-d+1 + chain band (p-d+k,k) |
| RoPE | row position = j+1 (`rope(token@p) = p+1` chain convention; global shift is relative-safe, matched anyway) |
| output | `mtp.norm` → shared lm_head; label = t_{j+1} |
| loss mask | response-region labels only (j+1 ≥ prompt_len); prompt rows stay as attention context |

## Pipeline (all on box3 — owner granted the whole box 2026-08-20)

1. **Prompt pack** — `build-prompt-pack.py` frozen mix, `--limit 4000` (3800 train /
   200 heldout, stratified chat/code/if/math × think/nothink, seed 20260811). The
   `--agentic-dir` hook expects verified-sft jsonl so it admitted 0; the 44 real agentic
   .txt prompts ride as an extra generation shard instead (~1% of mass either way).
   `mtp-train/prompts/` + summary sha.
2. **Own-gen corpus** — `gen_corpus.py` against memra-server (the published NVFP4-MTP
   artifact, spec-off, c8, GPU0) at vendor serving sampling T=0.6/top_p 0.95/top_k 20,
   per-row seeds; think rows 1024 max, nothink 512 via `reasoning_effort:"none"`.
   → `mtp-train/corpus.jsonl` (~3M gen tokens).
3. **Hidden capture** — `capture_hiddens.py` on GPU1, BF16 trunk, trails the corpus live
   (--follow). Re-renders the served stream (think re-wrap, finish=length stays
   mid-stream, stop gets `<|im_end|>`), right-padded batches (pad-safe by causality),
   stores h bf16 (fp16 would overflow outlier channels) → `mtp-train/hiddens/shard-*.pt`.
4. **Train** — `train_mtp.py`: AdamW fp32 params/bf16 autocast, lr 1e-4→1e-5 cosine,
   3 epochs, chunked CE over the 248,320-row shared lm_head, heldout loss + top-1 proxy
   at step 0 (vendor baseline) and each epoch; exports per-epoch `mtp-trained-epochN
   .safetensors` re-split to checkpoint tensor names.
5. **Remint + A/B (the gate)** — patch mtp.* into the BF16 source, remint the GGUF head
   (embedded + masked draft v3 in the hqmtp order), then serve-level spec-on/off A/B vs
   the vendor head on the same probes as `gates/ab-draft.log`. Publish an update ONLY on
   a measured win; the offline top-1 proxy never ships as an acceptance claim.

## Risks / open

- Response re-tokenization drift vs the exact served token stream (boundary tokens at the
  think re-wrap): self-consistent for training (h computed on the same re-tokenized
  stream), noted as a known approximation.
- MoE router fine-tune without an aux balance loss can skew expert load — low risk at
  ~600 low-lr steps; watched via heldout loss, and the serve-level A/B is the gate.
- Think rows cut at 1024 (finish=length) train the think-phase distribution heavily —
  that is where the head measured weakest (9–28% short-probe), so this is intentional.
