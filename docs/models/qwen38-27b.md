# Qwen3.8-27B

Qwen's 27B dense hybrid (Gated DeltaNet plus gated attention) at its full 262,144-token context,
from an NVFP4+Q5_K GGUF or an NVFP4 safetensors checkpoint. It is one of the two checkpoints the
tag battery runs on every release.

| | |
|---|---|
| **Serving route** | NVFP4+Q5_K MTP GGUF on one card. Tag battery on every release: calibrated argmax (eager and served-batch) and greedy `run-spec` K=1..8; last run v0.141.1, 2026-11-11, one RTX PRO 6000 Blackwell Max-Q. Standing gate: yes, the `vendor` row of `tools/release-roster.tsv` and the daily regression |
| **Pack** | `qwen35`. NativeReference only for the B200 block-FP8 artifact (record `qwen38-27b-fp8-b200`); the NVFP4 GGUF and NVFP4 safetensors routes have no pack record <!-- support: qwen38-27b-fp8-b200 --> |
| **Artifact** | `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF@6c39753daccd7473d4e60735ec8ace1d30afdd10`, file `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, 15,705,922,304 bytes, sha256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a` (the file the v0.141.1 battery hashed). Older receipts pin `@c303a1b0…`, which the repository's history (it starts on 2026-11-02) no longer holds; the file hash is the same. The NVFP4 safetensors route uses `unsloth/Qwen3.8-27B-NVFP4@16b6615af3548b88e2d8e382457bc705b00479cf` ([cookbook](../COOKBOOK.md#rtx-pro-6000-blackwell-nvfp4-safetensors-checkpoint)). Base checkpoint `Qwen/Qwen3.8-27B` |
| **Drafters** | the embedded MTP head; masked FR-Spec heads and ranks files in the same repo (below); the DFlash2 drafter [`tiyuvta/Qwen3.8-27B-DFlash2-memra`](https://huggingface.co/tiyuvta/Qwen3.8-27B-DFlash2-memra), opt-in |
| **Hardware measured** | RTX 5090 Laptop (24 GB), RTX PRO 6000 Blackwell (96 GB), DGX Spark (GB10); B200 for the block-FP8 record |
| **Input** | text, image and video (since v0.86) |

## Run it

Commands are in the [cookbook](../COOKBOOK.md#qwen38-27b): the 24 GB envelope, 262k serving on a
96 GB card, the DFlash2 drafter and the NVFP4 safetensors trunk.

- A naked server runs the MTP route. DFlash2 arms only with `MEMRA_DSPARK_SPEC=1` plus
  `MEMRA_DSPARK_DRAFT`, and arming it turns MTP spec off for that model. Boot proof:
  `[dspark] q38: DFlash2 draft head TRIMMED to 32768 rows`.
- Image and video input load the checkpoint's own ViT tower through `MEMRA_VISION_DIR`. The tower
  is gated by a per-token cosine oracle against the HF reference (images min-cos 0.9997 on three
  probes, [decision](../decisions/VISION-LANE.md)). One vision family per process.
- Verification always runs on the full 248,320-token vocabulary, so a ranks trim moves draft cost
  and acceptance, never output.

FR-Spec ranks, three flavours, all model-generated (external text is used as prompts only). On a
safetensors trunk the `.txt` ranks trim the head at load (`MEMRA_FRSPEC_TRIM=<ranks.txt>`); on the
GGUF route the pre-trimmed `.gguf` head attaches (`MEMRA_MTP_DRAFT=<head.gguf>`). Heads go from
248,320 to 32,768 rows.

| Flavour | Ranks (safetensors) | Pre-trimmed head (GGUF) | Corpus |
|---|---|---|---|
| agentic | [`q38-ranks-sxc32768.gguf.txt`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-sxc32768.gguf.txt) | [`mtp-…frspec-sxc32768.gguf`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf) | 163k own-generated tokens over real agentic sessions |
| prose | [`q38-ranks-prose-32768.txt`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-prose-32768.txt) | [`mtp-…frspec-prose32768.gguf`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-prose32768.gguf) | 154k own-generated tokens, essay, story and letter prompts, about 15% non-English |
| mixed | [`q38-ranks-mixed-32768.txt`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/q38-ranks-mixed-32768.txt) | [`mtp-…frspec-mixed32768.gguf`](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/main/mtp-Qwen3.8-27B-NVFP4-frspec-mixed32768.gguf) | 50/50 blend of both streams, same rank law |

Short generic probes put the three within noise of each other; the differences live in domain
tail tokens. Pick the flavour that matches the workload.

## Measured

- **RTX 5090 Laptop, cold single-stream prompts** (2026-11-09, main `1273a16f07`, GGUF with the
  untrimmed MTP head, context 8,192, six boots per arm): T=1 decode medians 63.2 tok/s at 1.2k
  prompt tokens, 63.3 at 3.7k and 60.4 at 5.8k on the K=2 cold-long rung that is now the device
  default; TTFT p50 1.18 s at 1.2k and 3.6 s at 3.7k.
  [Receipt](../../research/spec-k-cold-long-20261109/README.md),
  [decision](../decisions/SPEC-K-COLD-LONG-K2.md).
- **DGX Spark (GB10), one cold request at a time** (2026-11-10, main `fcd66f8963` plus the
  verify-rows walk that landed after it, MTP route at automatic K, single runs):

  | prompt tokens | TTFT | decode after the first token |
  |---:|---:|---:|
  | 1,036 | 1.13 s | 31.05 tok/s |
  | 131,051 | 117.4 s | 17.00 tok/s |
  | 259,951 | 313.9 s | 12.65 tok/s |

  [Receipt](../../research/gb10-decode-20261110/README.md). More GB10 cells are on the
  [DGX Spark card](../rigs/dgx-spark.md).
- The DFlash2 sampled rows on RTX PRO 6000 (c1 127 against 117 tok/s for the MTP head, from
  v0.113.0) cite a lane directory that is not in this repository, so they are kept only in the
  [history file](../history/MODELS.md#qwen38-27b-in-detail).

## Limits

- Performance receipts are per artifact. A GGUF number does not transfer to the safetensors route,
  or back, without a same-binary comparison.
- On this GDN-hybrid class the prefix cache serves extension shapes (conversation continuation,
  session affinity), not fan-out shapes: the recurrent state exists only at an entry's end
  boundary.
- DFlash2 refuses to boot together with `MEMRA_SPEC_DFLASH`, `MEMRA_GEMMA4_SPEC` or more than one
  pipeline stage. No standing gate covers the DFlash2 route; the tag battery runs the MTP head.
- B200: the block-FP8 artifact `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
  passed only Config and TinyParity; see the [B200 card](../rigs/b200.md).

## Deeper sources

- Release battery records: [`research/release-qualification/`](../../research/release-qualification/)
  (`current.json` names the latest).
- Device-keyed spec defaults: [SPEC-K-COLD-LONG-K2](../decisions/SPEC-K-COLD-LONG-K2.md),
  [SPEC-K-COLD-SHORT-5090](../decisions/SPEC-K-COLD-SHORT-5090.md),
  [C8-SERVING-24GB](../decisions/C8-SERVING-24GB.md).
- DFlash2 drafter build: [DRAFT-REGIME.md](../DRAFT-REGIME.md); flags `MEMRA_DSPARK_SPEC`,
  `MEMRA_DFLASH_PREC`, `MEMRA_FRSPEC_TRIM` in [FLAGS.md](../FLAGS.md).
- Pack: `crates/memra-gguf/src/model_packs/qwen35/`. Record: `qwen38-27b-fp8-b200` in
  `docs/support-records.toml`.
- Older detail (the MTP-era battery, the August DFlash2 rows, the earlier card):
  [history](../history/MODELS.md#qwen38-27b-in-detail).
