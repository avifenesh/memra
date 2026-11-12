# Ornith-1.5-35B-A3B

Ornith's 35B-A3B mixture-of-experts on the Qwen3.5 MoE stack (Gated DeltaNet plus gated
attention, routed experts), served from an NVFP4+Q5_K GGUF with a continued-trained MTP head at
its full 262,144-token context. It is one of the two checkpoints the tag battery runs on every
release, and the one the battery requires.

| | |
|---|---|
| **Serving route** | NVFP4+Q5_K MTP GGUF on one card, sampled speculative decode. Tag battery on every release: calibrated argmax (eager and served-batch) and greedy `run-spec` K=1..8; last run v0.141.1, 2026-11-11, one RTX PRO 6000 Blackwell Max-Q. Standing gate: yes, the `own` row of `tools/release-roster.tsv` (a missing file refuses the release) and the daily regression |
| **Pack** | `qwen35_moe`, NativeReference through record `qwen35_moe-reference`, which names a Qwen3.6-35B-A3B artifact; this artifact has no pack record of its own <!-- support: qwen35_moe-reference --> |
| **Artifact** | `tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF@8de0800e9820d147ce6a735afa68592bd8451f6e`, file `Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf`, 20,188,038,400 bytes, sha256 `72ff9600aa2b0de77a5b27041a84448c2ce88c7b2055529fc23b3cd5bf518fd3` (the file the v0.141.1 battery hashed). Older receipts pin `@d3769481…`, which the repository's history (it starts on 2026-11-02) no longer holds; the file hash is the same. Base checkpoint `ornith-ai/Ornith-1.5-35B-A3B` |
| **Drafters** | the embedded continued-trained MTP head, trimmed at load by the own-generated ranks (`ornith15-ranks-owngen-32768.gguf`, the default recipe since `412c45b0ca`, 2026-08-22); the standalone masked head `mtp-…frspec-owngen32768.gguf` as the alternative; both in the same repo |
| **Hardware measured** | RTX PRO 6000 Blackwell (96 GB) |
| **Input** | text and image (since v0.102) |

## Run it

Commands are in the [cookbook](../COOKBOOK.md#ornith-15-35b-a3b). Boot proof for the ranks
trim: `[frspec-trim] self-trimmed head: 32768 rows of`; for the standalone masked head instead:
`[worker] ornith: regime draft attached (`. Without either the embedded full head drafts, which is
still exact and only costs more per draft step.

- This architecture forces the sampled speculative path. The verify trunk replays as a captured
  CUDA graph by default on the Gated DeltaNet plus routed-MoE family since 2026-08-23
  (`MEMRA_SPEC_VERIFY_GRAPH=0` is the kill switch).
- The ranks trim drops the draft head from 248,320 to 32,768 rows (about 221 to 29 µs per draft
  step), and the automatic depth table serves cached long prompts at K=5 when a trimmed head is
  loaded.
- Image input runs the checkpoint's qwen3_5 ViT in-engine through `MEMRA_VISION_DIR`, gated by the
  per-token cosine oracle (min-cos 0.99983 on the worst probe) before serving.

## Measured

One RTX PRO 6000 Blackwell Workstation Edition, vendor sampling (T=0.6, top-p 0.95, top-k 20).

| Metric | Value | Conditions |
|---|---|---|
| Single-stream cached-long | 319 tok/s (266 with the verify graph off) | v0.108, a Zen 5 desktop host, 4+4 boots in both orders, every boot within 1% of its arm; fixed-seed sampled hashes identical across arms and hosts. [Receipt](../../research/orndecode-20260822/VGRAPH.md) |
| Single-stream fresh short | 354 tok/s on a Zen 5 host, 235-262 on a Zen 3 host | v0.105.0 |
| 8-turn agentic session (14.7k document, verbatim history) | 11.6-12.9 s total | v0.105.0 |
| Shared-prefix aggregate | about 915 tok/s at c8, about 700 at c16 | v0.105.0 |
| Warm-turn TTFT | 25-40 ms (full prefix restore) | v0.105.0 |
| Ranks trim | c1 short +10%, shared-prefix c8 +2.9%, long-document and session shapes flat | 2026-08-22, ABBA, fresh boots, N=6 per shape ([receipt](../../research/orndecode-20260822/RESULTS.md), commit `412c45b0ca`) |

The v0.105.0 rows: [`research/orndecode-20260822/`](../../research/orndecode-20260822/RESULTS.md).
Single-stream decode on this model depended on the host CPU before the verify graph (the round
was host-launch bound, verify-wait 0.0%); with the graph on, both host generations land at about
320 tok/s.

## Limits

- The pack record for `qwen35_moe` names a different checkpoint. No CheckpointParity or Serve gate
  exists for this artifact; the tag battery is its gate.
- A cold host reads low until its clocks ramp. Pair the card with a current-generation host CPU.
- Pair-owner MoE ordering was measured and rejected as a runtime arm
  ([decision](../decisions/ORNITH-PAIR-OWNER.md)).

## Deeper sources

- Release battery records: [`research/release-qualification/`](../../research/release-qualification/).
- Lanes: [`research/orndecode-20260822/`](../../research/orndecode-20260822/),
  [`research/ornith15-vision-20260822/`](../../research/ornith15-vision-20260822/),
  [`research/ornith15-mtp-train-20260820/`](../../research/ornith15-mtp-train-20260820/).
- Pack: `crates/memra-gguf/src/model_packs/qwen35_moe/`.
- The earlier in-detail section: [history](../history/MODELS.md#ornith-15-35b-a3b-in-detail).
