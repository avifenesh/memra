# HF publish checklist: GLM-5.3-Flash NVFP4 mint

Lane: `lane/glm53-hf-publish` (branched from `lane/glm53-flash-bringup`).
Owner law this file exists to honour (2026-08-18, learned on the Qwen3.8 publish):
**research current Hugging Face publishing practice BEFORE uploading, apply it AT
publish time, and bank the checklist before a single byte moves.** The Qwen3.8 card
had its tags, naming and metadata corrected after the fact and still pulled real
traffic; doing it right up front is strictly better.

Discoverability metadata is the deliverable here, not decoration. Owner's reason, in
their words: "we should publish our mint, thats part of the way we get traffic to
memra and tiyuvta."

Research date: 2026-08-28. Sources are named per row so a later reader can re-check
rather than trust this file.

---

## 0. HARD GATE, do not upload until this is green

`BRINGUP.md` states the law: "Every mint gates argmax-vs-reference before any serving
or publish." The corpus carries `TRAP:convert-direct-q8`: never serve or publish a
mint that has not been gated against the reference, from the case where a direct q8_0
mint silently scrambled rerank ranking.

The bar is **byte coverage, not family coverage**. A gate receipt that names a family,
a directory, or a model revision does not cover the bytes going to the Hub.

State of the banked receipts, checked 2026-08-28:

| receipt | what it identifies | binds our bytes? |
|---|---|---|
| `mint-receipts/nvfp4-oracle.tsv` | `format` / `engine` / `numeric_class` / `tokens` / `vocab` only | NO, no path and no hash |
| `mint-receipts/bf16-oracle.tsv` | same header shape | NO |
| `mint-receipts/fp8-oracle-samebox.tsv` | same header shape | NO |
| `inspect-receipts/artifact.lock` | the **vendor FP8 source**, 62 shards, rev `04c4e9e…` | NO, a different artifact |

So the banked receipts do not, on their own, meet the bar. Chain-of-custody reasoning
(rsync completed, sizes match, mtimes precede the gate commit) is exactly the
family-level trust the stop clause rejects, and it is not accepted here.

**Resolution taken in this lane:** re-run the gate on the exact upload bytes instead of
annotating the old receipt.

- [x] Bench box and upload box compared read-only, per-file: all 33 entries identical
      in size and mtime; artifact total 190,750,196,900 B on both. The bench box also
      holds `.memra-repack` (171.2 GB, regenerable) which is NOT in the upload set.
- [x] `sha256sum` over every file of the upload artifact on the upload box; manifest
      banked as `hf-publish-receipts/SHA256SUMS.txt`.
- [x] `glm5_checkpoint_runner` built on the upload box from the lane head, `--self-test`
      PASS (streamed trunk matches `execute()` bit-for-bit).
- [x] Gate re-run: the same memra-reference f32 streaming runner over
      `~/models/glm53-nvfp4` on the upload box, same tokens `[1,2,3,4]`, full
      last-position vocab row. Compared to the banked `nvfp4-oracle.tsv` and to the
      BF16-twin oracle. Banked as `hf-publish-receipts/nvfp4-oracle-rerun.tsv` and
      `hf-publish-receipts/GATE-COVERAGE.md`.
- [x] Nothing is edited in the artifact after the re-run. If any file has to change,
      the re-run is repeated; the receipt covers a hash set, not a directory name.

If the re-run had failed to reproduce, the answer was STOP and publish nothing.

Supporting arithmetic banked with the receipt: `model.safetensors.index.json` carries
113,446 weight-map entries = 1,432 kept tensors + 3 x 37,338 quantized tensors
(weight / weight_scale / weight_scale_2), matching `mint-log-summary.txt` exactly.

---

## 1. Repo naming

- **Repo id: `Avifenesh/GLM-5.3-Flash-NVFP4`.**
- House pattern from the two prior cards is `<base>-<format>-<extras>`:
  `Qwen3.8-27B-NVFP4-MTP-GGUF`, `Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF`.
- No `-GGUF`: this artifact is safetensors, not GGUF. Naming it GGUF would be a false
  format claim and would mislead every llama.cpp user who found it.
- No `-MTP`: the checkpoint's NextN block is present but is not executed and not gated
  here. A name that advertises a capability the card cannot back is an overclaim.
- Ecosystem convention check (searched Hub, sorted by downloads): community modelopt
  mints name the tool in the repo id (`…-NVFP4-ModelOpt`, `…-ModelOpt-W4A16-NVFP4`).
  We keep the tool in the `tags` and the card body instead of the id, matching our own
  two prior cards; consistency across our three cards is worth more than matching a
  convention that is not itself consistent.

## 2. YAML frontmatter

Field-by-field, each with the reason it is set that way.

```yaml
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
library_name: memra
language: [en, zh]
tags: [...]
```

| field | value | why |
|---|---|---|
| `license` | `mit` | Upstream `zai-org/GLM-5.3-Flash` is MIT, so redistribution of a derivative is clear. Same value as the Ornith card. |
| `base_model` | `zai-org/GLM-5.3-Flash` | This is the field that puts the mint on the PARENT model's quantizations panel, which is the single highest-value discovery surface we get: the parent has ~1,437 likes and the traffic we want walks in from there. Live check on our two prior cards confirms the mechanism: both carry a `base_model:quantized:<parent>` tag. **Single id, not a list**, even though the true quantization source is the BF16 twin `zai-org/GLM-5.3-Flash-BF16` (which exists on the Hub, 40 likes). HF docs state a list of two or more base models is how a MERGE is declared, so a two-entry list invites the wrong relation on the highest-value field. The twin is named with its revision in the card body instead, which is where a reader checking provenance looks. |
| `base_model_relation` | `quantized` | HF infers the relation but documents that it can be set explicitly. Setting it removes the guess. Both prior cards set it; keep the house habit. |
| `quantized_by` | `Avifenesh` | Attribution field both prior cards carry. |
| `pipeline_tag` | `text-generation` | Matches upstream. Drives the task filter. |
| `library_name` | `memra` | **Deliberate, and NOT `transformers`.** HF docs: for repos created after August 2024 the library is not inferred from `config.json`, so an explicit value is wanted, and `library_name` drives the auto-generated usage snippet. Our artifact is `quant_method: modelopt` on a `glm5_next` architecture, so a `transformers` value would render a "Use in Transformers" button that cannot load it, which is a false capability claim on a public card. Two candidate honest values: `Model Optimizer` (what NVIDIA's own official NVFP4 repos such as `nvidia/DeepSeek-R1-0528-FP4` use) and `memra` (what actually runs it). **`memra` wins**, on evidence: querying our two live cards shows the Hub already resolves them to **Library: memra** from the tag alone, so this value is house-consistent, it is true, and it keeps our engine in the library facet rather than handing that slot to a third party's tool. The owner's stated reason for publishing at all is traffic to memra and tiyuvta. `modelopt` stays in `tags`, where the modelopt audience looks. |
| `language` | `en`, `zh` | Upstream declares both and both parent repos carry the `en` / `zh` facets. Neither of our prior cards sets `language`, so this is a small discovery surface we have been leaving on the table. |
| `tags` | see below | Filterable facets. |

Tags, chosen as things a buyer would actually filter on:

`nvfp4`, `fp4`, `modelopt`, `w4a16`, `quantized`, `memra`, `glm5_next`, `moe`,
`blackwell`, `conversational`, `tool-calling`.

- `nvfp4` / `memra` / `blackwell` / `conversational` / `moe` are carried by both prior
  cards; keeping them makes our three cards cross-discoverable as a set.
- `modelopt` / `fp4` / `w4a16` are the facets the modelopt-mint audience searches on
  (confirmed against the top-downloaded modelopt NVFP4 repos on the Hub).
- `glm5_next` is the architecture tag from `config.json`.
- No `gguf` tag. Not a GGUF.

## 3. arXiv linkage

Upstream's technical report is arXiv **2602.15763**. HF docs: "If the model card
includes a link to a Paper page (either on HF or an Arxiv abstract/PDF), the Hub will
extract the arXiv ID and include it in the model tags with the format `arxiv:<PAPER
ID>`", which then lists the repo under that paper and lets readers filter for other
models citing it.

- [x] Card body links `https://arxiv.org/abs/2602.15763` so the `arxiv:2602.15763` tag
      is minted automatically. This is free discovery neither prior card took.

## 4. Card body: what it MAY and MUST NOT say

We are a serving vendor. An overclaim on this card is a product claim.

MAY:

- That this is an NVFP4 mint of GLM-5.3-Flash and how it was produced: NVIDIA
  TensorRT Model Optimizer 0.46.0, streaming per-tensor `W4A16_NVFP4` from the BF16
  twin `zai-org/GLM-5.3-Flash-BF16 @ f12e0fe1…`, 4.5 bits/element (4-bit e2m1 with an
  FP8-e4m3 per-16 scale plane). Quantizing from the vendor's full-precision twin
  rather than from their FP8 release is a real methodological point: never quantize
  from a quant when the full-precision twin ships.
- The precision split, tensor by group, and the exact gate the mint passed, with the
  calibration row that makes the number interpretable.
- That it runs on memra, and how to run it.
- The engineering story, which is the traffic hook and is all checkable against the
  public lane doc: bring-up found and fixed a series of silent correctness defects,
  every one caught by memra's unfused f32 reference executor disagreeing with the
  engine, never by the engine looking unhealthy. And the chat-template finding: the
  model's template is NOT ChatML, and serving it through a ChatML lookalike returns
  200 with fluent text while running off-distribution. Caught with a byte oracle,
  fixed to the native wire (`<tool_call>NAME<arg_key>…`, `<|observation|>` tool
  results, `[gMASK]<sop>` framing).

MUST NOT:

- Claim we serve this model to customers. **We do not serve GLM-5.3-Flash.** No
  "available on our API", no "served by", no "try it on tiyuvta", no pricing, no
  endpoint, no model id to call.
- State any performance number as a product claim. The decode number is 20.3 tok/s
  single-card, and it is a bring-up measurement with a road to more, not a product
  claim. It appears only with its named hardware and full conditions, or not at all.
- Include any box IP, hostname, key path, credential, internal lane name, or build
  fingerprint. The card is public. `python3 tools/check-public-boundary.py` must
  report 0 new before the push.
- Use em dashes (house rule).

## 5. Brand placement in the header (owner instruction, 2026-08-28)

Owner's words: the header should include memra and tiyuvta, "not as serving until we
actually serve, but in general."

- **memra** is named as the engine the mint was produced and validated with, and as
  the thing whose f32 reference executor caught the defects. That is a true,
  checkable statement about tooling.
- **tiyuvta** is named as the lab behind it, with its link, as attribution and
  identity.
- The link is **https://inference.tiyuvta.ai** (the inference product, verified live),
  NOT `https://tiyuvta.ai` (lab/marketing site). Owner correction, 2026-08-28.
- Forbidden shape: anything reading as an offer. "built by tiyuvta
  (https://inference.tiyuvta.ai)" is fine. "run this on tiyuvta" is not, until we
  actually serve it.
- Same rule inside the YAML: brands may appear in provenance fields, never in a way
  that implies a hosted offering.
- Both prior cards open with a "Run this exact model through an API" block. **This
  card must NOT have one.** That block is correct there and wrong here; the difference
  is that we serve those two and do not serve this one. The moment we serve it, the
  card gets the block.

## 6. What ships in the repo, and what must not

Ships:

- 20 `model-*-of-00020.safetensors` shards (178 GB of the 190.7 GB total)
- `model.safetensors.index.json`
- `config.json` (the keep-list-fixed one, emitting both `modules_to_not_convert` and
  compressed-tensors `ignore`)
- `generation_config.json`, `hf_quant_config.json`
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
  `processor_config.json`
- our authored `README.md`

Must NOT ship:

- `.memra-repack/`: the 160 GiB regenerable expert-slab cache the engine builds on
  load. It was absent from the upload box when this set was first enumerated and it is
  NOT absent any more: it appeared inside the artifact directory after another lane
  booted a server against the same files. Excluded by explicit pattern, because
  "it is not there" stopped being true once and will stop being true again.
- `config.json.pre-keeplist-fix`: an internal backup, not part of the artifact.
- The vendor `README.md` currently sitting in the model directory. It is zai's card,
  with zai's benchmark tables and citation. Publishing it under our repo id would
  republish someone else's card as ours. **A blind folder upload does exactly this**,
  which is why the upload is explicit-path, not directory-sweep.

Pre-upload scrub, mandatory:

- [x] Grep every small text file being uploaded for user paths, box hostnames, IPs, and
      producer strings. Run BEFORE the gate finished, so a forced edit would not have
      wasted the run. All seven clean; the only producer string is modelopt 0.46.0.
      Nothing scrubbed, so no re-hash and no second gate run.

## 7. Upload mechanics

Researched 2026-08-28 against the current `huggingface_hub` upload guide. **Two
corrections to older practice, both live:**

1. `hf upload-large-folder` and `upload_large_folder()` are **deprecated** and slated
   for removal. `hf upload` / `upload_folder()` is now the go-to for large folders:
   it streams in several commits, splits automatically to stay under server limits,
   and resumes if interrupted by re-running the same call.
2. **`HF_HUB_ENABLE_HF_TRANSFER=1` no longer does anything.** The docs state
   `hf_transfer` was removed in favour of `hf_xet`, and the flag is superseded by
   **`HF_XET_HIGH_PERFORMANCE=1`**. This contradicts standing lane guidance and is
   worth carrying forward: do not install `hf_transfer`, set the Xet flag instead.

Procedure:

- [x] `pip install -U huggingface_hub` on the upload box (pulls `hf_xet`).
- [x] Token passed through the ssh environment, read from stdin into a shell variable.
      It is the owner's personal credential on a shared box: it must not be written to
      a file there, must not be embedded in a script, and must not appear in a command
      line where `ps` can read it.
- [x] Upload from **18.132.250.253**, never from the local rig. A past unthrottled
      upload from the rig flooded 25 GB of swap and stalled the machine.
- [x] `HF_XET_HIGH_PERFORMANCE=1`. The docs warn it will use all available bandwidth
      and CPU cores. That is right for this transfer and wrong on the rig. Note the
      upload box is NOT dedicated to this lane: another lane runs work there
      concurrently, so the upload stays niced and the flag's appetite is a thing to
      watch rather than assume.
- [x] `HF_XET_CACHE` on local disk. The box root is a 1.5 TB EBS volume; free space
      needs re-checking at upload time, because the repack cache and a 250 GB swap file
      added during this lane both landed on it.
- [x] Small files first, weights second. Precedent: the q38 publish committed the
      small files as their own commit before the weights, so the repo is readable and
      correctly-tagged even while the shards stream.
- [x] Weights upload runs detached with a log, so a dropped ssh session does not kill
      a multi-hour transfer.
- [x] Verify after: file count, per-file size, and the repo's own sha256 against the
      banked manifest.

## 8. Launch mechanics

- [x] Publicity draft filed in **darklanes** `spec/gtm/` (business content lives in
      darklanes, never in the public memra repo). Show HN style, title plus body,
      receipt-backed. HN is the priority channel. r/LocalLLaMA is explicitly not our
      channel: self-hosters download, they do not buy.
- [ ] The owner posts (OPEN: draft ready, not posted). Agents never publish to HN, Reddit or social directly.
- [x] Buyer framing: companies shipping LLM products, and operators of personal
      autonomous agents. Not rig hobbyists.
- [x] Collections: add the artifact cards to one Hub collection so each card
      surfaces the other two. This was an explicit "owed" item left open by the q38
      publish and is still open; it costs one action and it is the cheapest
      cross-discovery we have.

## 9. Retrofit to the two prior cards

Deltas found by diffing our two live cards against current practice. Both are cheap
metadata edits; neither rewrites body copy.

Checked live rather than assumed, 2026-08-28:

| card | delta | verdict |
|---|---|---|
| both | no `library_name` field | **NOT a delta.** The Hub already resolves both to `Library: memra` from the `memra` tag. Leave them alone. |
| both | no `language` field | **NOT actionable.** Neither `Qwen/Qwen3.8-27B` nor `ornith-ai/Ornith-1.5-35B-A3B` declares `language` upstream. Adding one to a derivative would be inventing a factual claim about the model to win a filter facet. Declined. (Our new card DOES set it, because `zai-org/GLM-5.3-Flash` declares `en` / `zh` upstream.) |
| both | no `arxiv:` tag | **NOT actionable.** Checked live: neither base model carries an `arxiv:` tag, so there is no report to cite. Do not invent a citation to mint a tag. |
| both | not in any collection | **real, and the only actionable one.** It is also the "owed" item the q38 publish left open. One collection holding all three cards makes each one a door to the other two |

Both already carry `base_model` + `base_model_relation: quantized` + `quantized_by` +
`pipeline_tag`, and both show the `base_model:quantized:` tag live, so the
parent-panel surface is already working on them.

The retrofit is additive metadata only: do not touch their serving blocks, which are
correct for those two because we do serve them, and are wrong for this one because we
do not.

## 10. Order of operations

1. Bank this checklist. Commit it. **Nothing is uploaded before this file exists.**
2. Gate re-run on the upload bytes plus sha256 manifest. If it does not reproduce,
   stop and report.
3. Author the card locally, boundary-check it.
4. Create the repo, push small files, then weights.
5. Verify the published repo against the manifest.
6. Retrofit the two prior cards.
7. File the publicity draft in darklanes and surface it. The owner posts.
8. Clean up: token footprint, box scratch, local scratch.


---

## OUTCOME (2026-08-28)

Gate re-run on the exact upload bytes: **PASS, bit-identical** to the banked
`mint-receipts/nvfp4-oracle.tsv`. Receipts in `hf-publish-receipts/`.

Published: **https://huggingface.co/Avifenesh/GLM-5.3-Flash-NVFP4**, 28 files,
every published byte hash-verified (22 via Hub LFS sha256, 6 by re-download).
Collection `Avifenesh/memra-nvfp4-serving-artifacts-6a91a984b0d2e29a645e1f82` holds
all four cards. Publicity draft filed at
`darklanes:spec/gtm/HN-SHOW-GLM53-NVFP4-20260828.md`, ready, owner posts.

One finding for the next NVFP4 publish: **the Hub auto-tags an NVFP4 repo `8-bit`.**
The detector reads the FP8-e4m3 scale plane, not the 4-bit e2m1 weights. The auto-tag
cannot be removed, so add `4-bit` to `tags` explicitly at publish time rather than
discovering it afterwards.
