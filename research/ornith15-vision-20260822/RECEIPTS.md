# Ornith-1.5-35B-A3B vision enablement — receipts (lane/ornith15-vision-20260822)

Owner directive 2026-08-22: "so its hour class plumbing, its the same code" — wire the
existing qwen3_5 vision tower path for Ornith-1.5 rather than treating it as a new program.

## What the model carries

The official checkpoint (`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`, arch
`Qwen3_5MoeForConditionalGeneration`) ships the qwen3_5 ViT: depth 27, hidden 1152,
heads 16, patch 16, spatial merge 2, temporal patch 2, learned 48x48 pos grid —
dimensionally identical to the Qwen3.8-27B tower memra has served since v0.86, except
`out_hidden_size` (the merger's output width): **2048** here vs 5120 on q38, because it
must match the trunk's n_embd. `deepstack_visual_indexes: []` — no deepstack, same
program shape as q38. 333 `model.visual.*` BF16 tensors extracted from the official ST
shards into `outside.safetensors` (893 MB) — the format `MEMRA_VISION_DIR` already loads.

## The one code change

`vision.rs` hardcoded `V_OUT = 5120` (q38's trunk width) into tower load, video-group
sizing, and the worker's admission width check — so ornith image requests were refused
with "vision tower output width does not match this model". The width is a property of
the shard, not of the engine: the fix derives merger fc2 `out_features` from the tensor,
exposes `VisionTower::out_width()`, and admission compares the serving trunk's `n_embd`
against the loaded tower instance. Images with no tower loaded now 400 explicitly.
Commit: `feat(vision): tower output width derived from the merger shard, not a constant`.

q38 no-regression (local 5090, lane build): tower loads with derived `out_width 5120`,
forward emits [154, 5120] on the 448x336 probe image (grid 22x28), fwd 268.4 ms.
Full local battery on the lane: `tools/local-ci.sh --perf-quick` rc=0 — accept-gate
1 pass 0 fail, perf stage 0 fail 0 warn, spec-on-cache-hit gate ALL GREEN.

## Parity gate (box9: 1x RTX PRO 6000 Blackwell WS 600W, driver 595.58)

HF reference: `Qwen3_5MoeVisionModel` instantiated from the checkpoint's
`vision_config`, CPU f32, weights loaded strict-complete (missing=[] unexpected=[])
from the extracted shard; the model's own `AutoImageProcessor`; **`pooler_output` is
the merged embedding** (`last_hidden_state` is pre-merger patch states — first dump
read [560, 1152] and was wrong; correct dump [140, 2048] for the 448x336 probe).
Gate: `vision-gate <img> --ref <ref.bin>` — PASS iff per-token min cosine > 0.999.

RESULT: **PASS on all three probe shapes** (threshold min_cos > 0.999):

| probe | grid (merged tokens) | mean_cos | min_cos |
|---|---|---|---|
| 448x336 shapes | 20x28 (140) | 0.999998 | 0.999945 |
| 640x480 photo-class gradient/texture | 30x40 (300) | 0.999994 | 0.999827 |
| 224x672 tall-narrow (pos-interp asymmetry) | 42x14 (147) | 1.000000 | 0.999995 |

The first gate attempt FAILED on token count (154 vs HF 140) and found a real
preprocessing defect: HF smart_resize uses Python round() = round-half-EVEN, Rust's
f64::round is half-away-from-zero — 336/32 = 10.5 diverged (352 vs HF 320). Latent for
q38 too (any exact-.5 side). Fixed in `vision_pre.rs` (image + video paths; the gemma
path is its own sealed program, untouched). q38 tower re-checked locally after the fix:
grid 20x28, out_width 5120, forward intact.

## E2E serve

`memra-server` + `MEMRA_VISION_DIR` on the NVFP4-Q5K-mtp GGUF, OpenAI `image_url`
base64 request: **WORKS** — model correctly described the red rectangle, green circle,
and yellow diagonal (176 prompt tokens incl. 140 image pads, full reply in 1.25 s).

## Merge battery (lane tip 6f359f804, box9)

kernel-check ALL GREEN (83 cells, 22 skipped); run-spec K=1..8 SELF-CONSISTENCY PASS;
run-gen prefill/decode argmax flip at the last position — **adjudicated near-tie, not a
defect**: bit-identical failure signature reproduced at pre-lane main 3d044ad3d (same
argmax pair 369/25, same maxdiff 1.553e0, same 0.0256 margin), and
`tools/argmax-margin-gate.sh` PASS (1 flip, margin 0.0378 < config spread 0.3479,
0 bad). Local 5090: full `local-ci --perf-quick` green on the lane (accept-gate 1 pass,
perf 0 fail 0 warn).

## Spec-off controls (owner pivot 2026-08-22: decode campaign baselines)

Same box, same artifact, `MEMRA_SERVE_SPEC=0` vs the spec-on recipe:

| cell | spec-on recipe | spec-off plain | vLLM |
|---|---|---|---|
| c1 short decode (N=3) | 118.3-122.2 tok/s | **187.1-188.3 tok/s** | 277.4-277.7 |
| c1 longdoc decode (N=3) | 189.1-228.7 tok/s | 210.0-270.4 tok/s | 275.3-289.4 |
| c16 shared-prefix agg (N=2) | 677.8/718.7 | 691.4/697.2 | 1181/1190 |

Findings: (1) the pmin spec recipe is NET-NEGATIVE at c1 short (-37%) — acceptance
~0.43 doesn't pay for verify at this shape; (2) **the c16 wall is NOT the spec/cache
forfeit** (canonflip's 4x q27 pattern does not apply here — spec-off c16 is the same
~700): memra tops ~700 agg from c8 up while vLLM reaches ~1190 — a real batch-scaling
wall; (3) open anomaly: true-cold 14.7k prefill measured 0.49 s on the spec-armed boot
vs 1.37 s on the spec-off boot (twice, then a 0.076 s cache hit) — the plain prime path
looks ~3x slower than the spec-armed one on identical input; needs its own lane.

## Cache-in-action numbers (same box, vendor sampling T=0.6/top_p=0.95/top_k=20, seeded)

memra serving recipe (`MEMRA_CTX=32768 MEMRA_PREFIX_CACHE_MB=30000 MEMRA_SPEC_ADAPT=1
MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1`) vs vLLM 0.27.1 (`--enable-prefix-caching`,
official NVFP4 ST). Raw: `raw/` (jsonl per rep + probe transcripts).

All cells 2026-08-22, box9. N stated per cell; spread is real and reported, not averaged away.

**Where memra wins (the cache-in-action cells):**

| cell | memra | vLLM | ratio |
|---|---|---|---|
| 8-turn agentic session, total wall | 12.89 / 14.33 / 24.90 s (N=3, median 14.33) | 21.86 / 23.77 s (N=2) | ~1.6x |
| shared-prefix c8 agg | 913.7 / 680.6 / 596.4 tok/s (N=3, median 680.6) | 733.0 / 372.9 tok/s (N=2) | ~1.2–1.8x |
| TTFT, warm repeat of a 14.7k prompt (2 prior hits) | 0.039 / 0.025 s | 0.065 / 0.063 s | ~2x |

**Where vLLM wins (single-stream and high concurrency):**

| cell | memra | vLLM | ratio |
|---|---|---|---|
| c1 TRUE-cold TTFT, 14.7k unseen prefix (N=3) | 0.514 / 0.495 / 0.485 s | 0.322 / 0.127 / 0.126 s | ~3.9x steady |
| c1 decode, longdoc (N=3) | 228.7 / 202.2 / 189.1 tok/s | 289.4 / 275.3 / 275.6 tok/s | 1.35x |
| c1 decode, short prompt (N=3) | 122.2 / 118.3 / 118.9 tok/s | 277.7 / 277.5 / 277.4 tok/s | 2.3x |
| c16 shared-prefix agg (N=2) | 677.8 / 718.7 tok/s | 1181.0 / 1190.5 tok/s | 1.65x |
| TTFT under c16 load | 0.062 s | 0.058 s | par |

Notes on the labels: the ttftc16 probe's original "cold" 0.095 s was a CACHE-HIT number
(the session cell had already cached the same doc in that server lifetime) — true-cold is
the c1 probe's mutated-prefix arm above. The two warm-TTFT probes disagree by shape:
after TWO prior hits memra 0.025-0.039 vs vLLM 0.063-0.065; after ONE prior hit memra
0.066 vs vLLM 0.035-0.037. Both are reported; neither is averaged into the other.

**The engine-side reading (owner call 2026-08-22, mid-cell):** q38-27B DENSE serves at
259 tok/s c1 on this same card class (v0.101). Ornith is 3B-ACTIVE — it should decode
FASTER than a 27B dense, and vLLM's 277 tok/s shows the model has the headroom. memra's
118-229 is an engine gap, consistent with the receipted decode anatomy (trunk mmvq ~3x
off the BW floor, fa_decode ~8x off) plus the ornith MTP head's ~0.43 acceptance. The
campaign's next walls, in payoff order on this model: c1 decode path, c16 scaling
(memra's aggregate does not grow from c8 to c16; vLLM's tripled), true-cold prefill
TTFT (0.13 s is reachable — vLLM does it with chunked prefill on the same silicon).

Raw: rep1 recovered from `/tmp/cachecell-box9.log` (its jsonl was truncated by the
failed rep2 run — `: > "$OUT"` fired before the boots were checked); rep2
`cachecell2-rep2.jsonl`; rep3 `cachecell2-rep3.jsonl`; probes `ttftc16.txt`,
`c1probe.txt`. First rep2 vLLM boot died at flashinfer JIT `gemm_sm120` ninja exit 137
(the 29-way nvcc OOM-kill; redo ran under `taskset -c 0-5`). vLLM c8/session spread:
the 733/21.86 pair came from a manual warm pass, the 372.9/23.77 pair from the scripted
cell in the same window — both retained.

## Box discipline notes (repeated-cost lessons)

- Teardown by `nvidia-smi --query-compute-apps` PID + verify-empty between arms.
  A `setsid` group-kill missed once; the orphan `VLLM::EngineCore` held 92.5 GB and
  OOM-refused both boots of the next rep — that rep was a total loss.
- Any vLLM boot that may trigger a flashinfer JIT rebuild runs `taskset -c 0-5`
  (container RAM cap; unbounded ninja spawns 29 nvcc and gets OOM-killed, exit 137).
- The flashinfer JIT cache on a CUDA-13.0 driver must be built by nvcc 13.0 exactly;
  13.2-built ops segfault at CUDA-graph replay.
