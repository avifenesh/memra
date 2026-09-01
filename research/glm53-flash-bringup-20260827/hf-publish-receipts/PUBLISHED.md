# PUBLISHED: Avifenesh/GLM-5.3-Flash-NVFP4 (2026-08-28)

https://huggingface.co/Avifenesh/GLM-5.3-Flash-NVFP4 (public, model)

Uploaded from the upload box, never from the rig, per the standing rule from the
unthrottled-upload incident that flooded 25 GB of swap and stalled the machine.

## Published set: 28 files, hash-verified end to end

| check | how | result |
|---|---|---|
| Local bytes unchanged after the mid-lane reboot | `sha256sum -c` of the banked 30-entry manifest on the box | 30/30 OK, exit 0 |
| 20 shards + index + tokenizer.json as published | Hub LFS sha256 vs the manifest | 22/22 MATCH |
| The 6 non-LFS small files as published | re-downloaded from `resolve/main` and hashed | 6/6 MATCH |
| Nothing extra published | repo listing vs the expected set | 30 entries = 28 published + our README + `.gitattributes` |

Not published, by explicit allowlist rather than by absence: the vendor's own `README.md`
that ships in that directory, `config.json.pre-keeplist-fix`, and `.memra-repack/`.
The allowlist matters here rather than being belt-and-braces: the repack cache
reappeared inside the artifact directory mid-lane when another lane booted a server
against the same files, and it will keep reappearing.

Repo total as published: 190,750,154,058 B across 20 shards plus metadata.

## Metadata, verified live after publish

`Library: memra` · `Task: text-generation` · `Architecture: glm5_next` ·
`base_model:quantized:zai-org/GLM-5.3-Flash` (the parent's quantizations panel, which
is the discovery surface this publish exists for) · `arxiv:2602.15763` minted
automatically from the body link · `language: en, zh` · `license: mit` · plus
`nvfp4` `fp4` `4-bit` `modelopt` `w4a16` `quantized` `moe` `blackwell` `conversational`
`tool-calling`.

One correction made after the first publish: the Hub auto-tagged the repo **`8-bit`**.
That detector is reading the FP8-e4m3 scale plane, not the weights, which are 4-bit
e2m1. The auto-tag cannot be removed, so `4-bit` was added explicitly so the correct
filter also matches. Worth carrying forward: any NVFP4 mint will get mis-auto-tagged
this way.

## Collection

`Avifenesh/memra-nvfp4-serving-artifacts-6a91a984b0d2e29a645e1f82` holds all four
cards (this one, Qwen3.8-27B-NVFP4-MTP-GGUF, Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF,
Qwen3.8-27B-DSpark-Agentic). This closes the "collections" item the q38 publish left
explicitly owed, and it is the only prior-card retrofit that turned out to be backed:
`library_name` was already resolving to memra on both, and neither base model carries
an arXiv report or a declared language to inherit.

## Credential handling

The token was never written to disk on the box. It lives in `~/.config`-class storage
on the rig only, and reached the box piped over ssh stdin into an exported shell
variable. It is not in `upload.py` or `verify.py` (both read it from the environment),
and never appeared on a command line where `ps` could read it. No upload process
survives, so no `/proc/<pid>/environ` copy remains.

## What the card deliberately does not claim

- **That we serve this model.** We do not serve GLM-5.3-Flash. No endpoint, no model
  id, no pricing, and no "try it on our API" block, even though both prior cards open
  with exactly that block. memra and tiyuvta appear as provenance and identity only,
  and the tiyuvta link is `https://inference.tiyuvta.ai`.
- **Any performance product claim.** 20.3 tok/s appears once, as a dated bench
  measurement, with the card, the power limit, the card count, the slot count, greedy,
  single stream, and the 63 tok/s resident-traffic roofline stated above it, plus an
  explicit line that it is a bring-up figure and not a property of the artifact.
- **Anything the gate did not cover.** Long-context behaviour, sampled decoding
  quality, the engine's fused 4-bit kernels, and the vision tower are each named as
  outside the gate rather than left for a reader to assume.
- **No box IP, hostname, key path, credential, or internal lane name.** Seven metadata
  files were scanned before publish and were clean.
- **No usable context claim.** The checkpoint's `max_position_embeddings: 1048576` is
  stated as the vendor's architecture figure and explicitly not as something this mint
  delivers. The card instead publishes where the wall actually is, with the arithmetic
  a reader can re-derive from `config.json` in the same repo: `index_kpool` is 4, so a
  monolithic prefill over N tokens allocates an N-squared-bytes f32 score plane per MLA
  layer per call, giving 67 MB at 8k, 2.5 GB at 50k, 68.7 GB at 262k and 1.10 TB at 1M.
  The wall is therefore below 262,144, not at the headline. Verified independently from
  the config rather than accepted on report. Same rule applied to the Show HN draft,
  which now bans a context number outright and states the wall in the first comment.
