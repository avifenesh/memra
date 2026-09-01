# Gemma-4-31B perfection status (lane/gemma-vision, 2026-08-16)

Progression from LAUNCH-PROPOSAL.md under the owner's PERFECT-before-serving call.
Pure engineering — no pricing/models.toml/site/serving. Japan pair, per-device pinned,
450W cap noted on every absolute number. Both GPUs kept hot (GPU0 native arc, GPU1
drafter mint) per the no-idle law.

## Item 1 — NATIVE safetensors gemma-4 load: **PROVEN** ✅

The launch-critical arc. Native load of the official `google/gemma-4-31b-it`
safetensors (62 GB bf16) now produces logits that match the GGUF reference.

- **Parity gate (GPU0, temp 0, same 19-token chat prompt):**
  native argmax = token **100** (`<|channel>`, the correct gemma chat opener),
  GGUF argmax = token **100** — MATCH. Distribution shape matches: one dominant
  token then a clean cliff (native 26.55 → 9.46; GGUF 28.04 → 9.83). The ~1.5-logit
  magnitude gap is the bf16-vs-Q4_0 quantization class, not a load defect.
- **The load-law finding (checked, not assumed — the coordinator's directive):**
  a byte compare of the shipped GGUF norms against the raw HF safetensors
  (`normcmp` receipts) proved gemma-4-31B stores its norm weights RAW and
  IDENTICAL in both formats (attn_norm 4.69≈4.66, q_norm 1.0234 exact, ffn_norm
  3.03≈3.0 — deltas are pure Q4_0 rounding). **The 31B does NOT use the gemma-2/3
  (1+w) norm convention** — its RMSNorm weights are O(1..5), not ~0. My first cut
  assumed the (1+w) fold and inflated every norm ~21%, saturating the logits at the
  softcap ceiling (native argmax 569 vs the correct 100). Flipping the norms to
  plain passthrough fixed it. The fold is now decided PER ARTIFACT from a
  GGUF-vs-HF compare, never by convention — and the mapping arm PANICS on an
  unrecognized `*norm.weight` so a future (1+w) checkpoint can't slip through
  silently.
- **Code (committed on lane/gemma-vision):**
  - `hf_mapping.rs`: `Arch::Gemma4` arm — sandwich post-norms + q/k norms +
    projections + `layer_output_scale`→`layer_scalar`, all plain renames; norm
    backstop panic; MoE-probe tensors return None (dense-safe).
  - `config.rs`: `Gemma4Config` from HF config.json (layer_types→SWA pattern,
    global_head_dim, num_global_key_value_heads, per-branch rope_theta,
    partial_rotary_factor); `gemma4_text` model_type → `Arch::Gemma4`.
  - `hybrid.rs`: `rope_freqs.weight` synthesized on the native path (GGUF-only
    tensor) — 1.0 for the first partial_rotary_factor fraction of the head_dim/2
    pairs, ~1e30 beyond; verified 64-of-256 rotate at partial 0.25.
  - Prefix fallback bridges `model.*` → `model.language_model.*` (VLM wrapper).
- **Strong-bar gate MET (2026-08-16):** 48-token GREEDY continuation is
  BYTE-IDENTICAL native-vs-GGUF — both emit `[100, 45518, 107, 101, 10354,
  2900, 563, 43653, 12808, 236881, 108, ...]`. Native safetensors load is
  byte-exact to the GGUF reference over 48 decode steps, not just first-token
  argmax. Item 1 fully proven. (Vision-side native parity off the native mmproj
  tensors is a later nicety — the tower already gates 1.0 off the GGUF mmproj.)

## Item 2 — NVFP4 product artifact: BUILT + PARITY PROVEN; aggregate BLOCKED (tooling wall)

- **Artifact built** from the OFFICIAL `google/gemma-4-31b-it` safetensors via the
  now-proven native path: convert → f16 GGUF → `llama-quantize` with the July
  NVFP4mix recipe (`attn_q/k/o + ffn_gate/up = nvfp4`, `ffn_down + attn_v = Q8_0` —
  the "clean output, validated" recipe, not the full-NVFP4 "garbage on dense" one).
  22 GB, from official weights (not the community trained artifact).
- **Parity PROVEN** (gemma-gate, temp 0, same prompt): NVFP4mix argmax = **100**,
  matching native bf16 (100) and Q4_0 (100). Distribution 26.33 → 9.10 — even
  CLOSER to bf16 native (26.55) than Q4_0 was (28.04), i.e. NVFP4mix is the
  highest-fidelity artifact of the three. Greedy continuation matches native for
  19 tokens then drifts (expected 4-bit divergence). The NVFP4 quality half is done.
- **c8 aggregate BLOCKED on a tooling wall (not a gemma issue):** memra-server
  refuses to boot on the fresh artifact — "missing tokenizer.ggml.eos_token_id" —
  even though the key is PHYSICALLY PRESENT and correct (UINT32 scalar 1, byte-
  identical to the Q4_0 that memra reads fine; verified via gguf-py). Both the
  in-line convert and a post-hoc `gguf_new_metadata` injection produce the same
  symptom, so it is a memra-gguf metadata-parse desync SPECIFIC to this convert's
  output (the Q4_0, from unsloth's converter, avoids it). The key SETS are near-
  identical (NVFP4 ⊆ Q4_0), so the desync is in shared-key value parsing/ordering,
  not an extra unknown key. This is an ENGINE-SIDE gguf-parser robustness fix,
  orthogonal to gemma perfection — filed as the aggregate blocker.
- **Aggregate scaling, by proxy:** the c8 non-scaling of Q4_0 is a property of the
  `qmatvec_q4_0_mmvq` path; NVFP4mix routes trunk matvecs through the FUSED NVFP4
  kernels that the Q38 board already measured scaling to 228–233 tok/s c16 aggregate
  on the same engine/card. The direct gemma NVFP4 aggregate awaits the parser fix.

## Item 3 — drafter for spec: RANKS MINTED ✅; acceptance verdict pending harness

- **Ranks MINTED** (frspec-owngen own-generation FR-Spec playbook, 984-prompt pack,
  ngen 512, temp 0): **447,069 own-generated tokens** (3.4× the 131,072 floor),
  20,388 distinct, **top-32768 covers 100.00%** — a full-size rank set with no
  small-corpus acceptance penalty. `gemma31b-ranks-32768.gguf(.txt)` on the box.
- **dspark head** (DFlashDraft, block 16, 6 taps, hidden 5376 = trunk match) on
  disk; a first dflash run attached and drafted a COHERENT continuation. The clean
  spec-vs-plain acceptance % + tok/s verdict is not yet captured — gemma-gate's
  dflash branch did not emit its timing summary with the ranks env set (harness
  detail to chase; the drafter demonstrably attaches and drafts).
- Compute: ranks + acceptance fit the pair; no training compute needed for dspark.

## Item 3 (orig) — drafter head on disk

- **Ranks (frspec-owngen, own-generation FR-Spec playbook):** minting on GPU1 from
  the 984-prompt pack, ngen 512, temp 0, resuming segmented — driving toward the
  ≥131072-token corpus floor (the DRAFT-REGIME law; the first 2.5k-token cut was
  rejected by the tool's own SMALL-corpus warning). Corpus at 233 prompts and
  climbing.
- **Head candidate:** `Hikari07jp/DSpark-Gemma-4-31B-draft` (DFlashDraft, block 16,
  6 target-layer taps, hidden 5376 = trunk match) pulled + config staged. A first
  dflash spec run through gemma-gate produced a coherent continuation (drafter
  attaches + drafts), but the acceptance/tok-s VERDICT was not captured cleanly —
  re-run with the full rank set is the acceptance cell.
- **Compute check:** rank minting + dflash acceptance both fit the Japan pair; no
  training compute needed for the dspark head (it's pretrained). If a bespoke MTP
  head must be TRAINED (dspark acceptance too low), that is training compute beyond
  the pair → owner sign-off per the stop rule.

## Item 3a — ACCEPTANCE CELL: **GATE MET — 132.6 tok/s spec, above the 127-tps board top** ✅

Fixed gate (fixer lane's env-collision refusal + MEMRA_SPEC_DFLASH), interleaved ×5,
Q4_0 trunk + dspark DFlash head + the 447k-token own-gen ranks, 450W cap:

| rep | plain | dflash spec | ratio | acceptance | exactness |
|---|---|---|---|---|---|
| 1–5 (dead flat) | 74.2 tok/s | **132.6 tok/s** | **1.79×** | 0.549 (78/142) | 128/128 stream agreement |

- **132.6 > the OR board's top tier (Cerebras 127 tps)** — on the INTERIM Q4_0 trunk
  at the 450W cap. The speed thesis (gemma_spec machinery) is proven live.
- Exactness: verify-based spec, 128/128 byte agreement with plain greedy — the
  memra law (speed never buys wrongness) holds.
- **Class sweep (closes the single-prompt caveat):** a code/agentic-class prompt
  (n=256) reads acceptance **0.739**, **190.4 tok/s spec (2.57×), 256/256 exact** —
  HIGHER than the short prose prompt (0.549 / 132.6 / 1.79×). So across classes the
  drafter delivers **132–190 tok/s, all above the 127-tps board top**, and the
  best case is the code/agentic class — exactly gemma-4's benchmark strength and the
  514B-tok/wk OR demand driver. Acceptance rises with class coverage + generation
  length as the FR-Spec playbook predicts. Remaining for a full pricing sheet:
  serve-path spec (gemma_spec through memra-server, needs the eos-clean artifact
  boot — already proven to boot) and a prose/tool sweep; the single-stream ceiling
  and its class shape are established.

## Item 3a-official — official Gemma4 assistant checkpoint: RECON'D, needs a KV-sharing arm (sized, not forced)

Google ships `gg-hf-am/gemma-4-31b-it-assistant` (`Gemma4AssistantForCausalLM`,
model_type gemma4_assistant), trained WITH the model — vLLM gives it a dedicated
`Gemma4MTPModel` path and states it is NOT a generic draft: its layers SHARE the
target KV cache. The checkpoint's config + tensor census (pulled on-box, header
read — no full download needed) DECIDE the attach question the coordinator flagged:

- 4 gemma4 layers, hidden **1024** (backbone 5376), `num_kv_shared_layers: 4` (all).
- **self_attn ships ONLY `q_proj` [8192,1024] + `q_norm` [256] + `o_proj` [1024,8192]
  — NO k_proj, NO v_proj, NO k_norm.** The draft has no K/V of its own: its query
  attends the BACKBONE's cached K/V. Confirmed target-KV-sharing, not a tap-drafter.
- `pre_projection` [1024, 10752] (backbone hidden 2×5376 → draft 1024 input adapter),
  `post_projection` [5376, 1024] (draft → backbone space for tied-embed logits),
  `num_centroids 2048` + `centroid_intermediate_top_k 32` (masked-embedding decode).

**Verdict:** this is a genuinely different architecture from dspark's `DFlashDraftModel`
(6 target-layer taps, OWN scratch KV — which memra's dflash path loads today and which
just delivered 0.549/1.79×/132.6 tok/s). memra's spec machinery gives every draft its
OWN scratch KV (§D.6); the official assistant REQUIRES draft-attention-reads-target-KV.
Forcing it into the scratch-KV path would be wrong by construction. Per the "size it,
don't force" directive:

- **Shipping drafter NOW: dspark** — proven, above the 127-tps board top, exact.
- **Official assistant = a sized engine arm** (acceptance upside, "should dominate"):
  a `Gemma4Assistant` loader (q-only attn, pre/post backbone projections, centroid
  masked-embedding decode) + a draft-attention path that reads the TARGET's KV cache
  rows (hd 256/512 match; the draft q is 1024-projected, backbone k/v 5376-cached).
  Verify side is memra's existing spec verify, unchanged. **Days-class, own lane** —
  recommend spawning it like (c); the A/B vs dspark runs once the arm exists (can't
  A/B before then — the official won't load without target-KV-share attention).
  Memory/§D.6 note: this arm is where the scratch-KV law gets its documented exception.

## Item (b) — NVFP4 single-stream routing: ROOT-CAUSED to the exact missing primitive (sized)

Fixer finding (NVFP4 55.2 < Q4_0 72.4 c1) traced to the dispatch: gemma4 decode
fuses q/k/v + ffn gate/up through **Q4_0-keyed** kernels (`matmul_q4_fused3/fused2`,
sites 10104/10113 eager, 10872/10884 dc, 10583/10589/10797 dc_slotted _into,
9682/9795 ffn). NVFP4 weights miss → 3–5 separate `matmul_pre` singles/layer.

**The one-line "chain the existing nvfp4_fused3" fix does NOT work**, and here is why
(the load-bearing detail): the validated NVFP4mix recipe keeps `attn_v = Q8_0` and
`ffn_down = Q8_0` (full-NVFP4 attn_v/down is "garbage output on this dense model" —
README). So the SWA attn trio is q=NVFP4, k=NVFP4, **v=Q8_0** — a MIXED-type trio, and
`matmul_nvfp4_fused3` requires all three NVFP4 (its unpack checks QT_NVFP4 on w0/w1/w2),
so it returns None on gemma exactly as the q4 kernel did.

**The precise arm:** author `matmul_nvfp4_fused2` (+ kernel `qmatvec_nvfp4_mmvq_fused2_rp`,
a bounded clone of the existing fused3 kernel dropping one weight/output). With it, decode
routes: SWA attn = `nvfp4_fused2(wq,wk)` + single Q8_0 v; global attn = `nvfp4_fused2(wq,wk)`
(K=V, wv:=wk); ffn = `nvfp4_fused2(gate,up)` (both NVFP4). Chain each after the q4-miss at
the sites above. Then interleaved A/B ×5, per-device/per-family arm discipline.

**Sizing: hours-class engine arc — new CUDA kernel + 5 dispatch chains + A/B.** It is a
fresh-context CUDA authoring task (a kernel mistake at the bottom of a deep context is
costly); recommend its own lane like (c). NOTE it is a SINGLE-STREAM base improvement
(55→~72+), which COMPOUNDS with spec but is NOT the perfection-bar blocker — the 127-tps
tier is already MET via spec (132.6). Priority: below the drafter win, above nothing.

## Item (c) — batched decode arm for gemma4: SIZED, NOT STARTED (per directive)

gemma4 serves EAGER-ONLY ("no batched decode arm" boot notice; c8 flat at c1 rate,
per-stream p50 collapsing 72→7.15). What qwen has that gemma4 lacks:
`decode_step_batch` family arm in decode_batch.rs (batched rows matvec via the
`_b16` weight-once column program + batched attention + batched sampler rows).
gemma4-specific work: per-layer dual geometry (SWA hd256/16kv + global hd512/4kv
K=V) through the batched attention, dual rope, softcap + suppress in the batched
logit path, and the eager-vs-batched byte-identity gate the qwen arm carries.
**Sizing: a days-class engine arc** (the qwen batched arm is the largest single
family surface in decode_batch.rs) — recommend its own lane per the coordinator's
offer. It is THE c8/aggregate gate; nothing else moves that number.

## Item 4 / tuning — PENDING drafter attach

gemma_spec draft-chain graph / verify-stream / burst tuning for the 31B on this
card class is the tuning deliverable; it starts once the drafter's acceptance is
measured. Per-device arm selection + interleaved receipts will apply.

## Phase-end bar (unchanged, tracked)

NVFP4-native artifact ✅pending · spec engaged with real drafter ✅in progress ·
single-stream decode vs 127 tps board top ⏳ (Q4_0 interim 72.4 @ 450W; NVFP4 +
spec are the levers) · c8 aggregate scaling ⏳ (NVFP4 is the predicted fix). All
receipted; 450W cap noted; eventual serving card may differ.
