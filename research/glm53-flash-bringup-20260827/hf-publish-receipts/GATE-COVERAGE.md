# Gate coverage for the published bytes

Purpose: make the argmax-vs-reference gate cover the EXACT bytes uploaded to
`Avifenesh/GLM-5.3-Flash-NVFP4`, not a directory name or a model family.

Law: `BRINGUP.md`, "Every mint gates argmax-vs-reference before any serving or publish."
Corpus: `TRAP:convert-direct-q8`, never serve or publish a mint that has not been gated
against the reference, from the case where a direct q8_0 mint silently scrambled rerank
ranking.

## Why a re-run rather than an annotation

The banked receipts do not bind bytes. Checked file by file:

- `mint-receipts/{nvfp4,bf16,fp8-samebox}-oracle.tsv` carry only `format`, `engine`,
  `numeric_class`, `tokens`, `vocab`. No path, no hash, no artifact identity.
- `inspect-receipts/artifact.lock` binds the **vendor FP8 source**: 62 shards, revision
  `04c4e9e95c5da8862dced7e5056455116f83a7e0`. Our mint is 20 shards. Different artifact.

Chain-of-custody reasoning (rsync finished, sizes match, mtimes precede the gate commit)
is available and is NOT accepted as coverage. That is family-level trust wearing a
timestamp. The gate is re-run instead.

## ACCEPTANCE RULE, written before the result was seen

Pre-registered deliberately: deciding the bar after seeing the number is how motivated
reasoning gets into a publish decision.

- **PASS (primary).** The re-run TSV is bit-identical to the banked
  `mint-receipts/nvfp4-oracle.tsv`. This is the expected outcome: `BRINGUP.md` records
  that a later reference run reproduced that banked file BIT-FOR-BIT (cosine 1.000000,
  max_abs 6.7e-08), so the reference path is stable across the commits in question.
- **PASS (fallback).** If the reference code drifted between the gate commit
  `5894542b83` and the commit this binary was built from, bit identity may not hold. The
  fallback bar is the gate's own bar, re-derived on these bytes: **argmax MATCH and
  top-3 rank identity against the banked `bf16-oracle.tsv`** (the mint's own source),
  with max_abs and mean_abs no worse than the banked NVFP4-vs-BF16 row (3.117 / 0.534)
  by more than the vendor's own FP8-vs-BF16 calibration row (3.489 / 0.490).
- **STOP.** Anything else. Publish nothing, report the divergence.

## What was run

Everything below happened on the upload box, which is the machine the bytes are uploaded
from, so no copy sits between the gated bytes and the Hub.

| step | detail |
|---|---|
| binary | `glm5_checkpoint_runner` (`crates/memra-reference`), built release from the lane head on the upload box |
| self-test | `--self-test` PASS: streamed trunk matches `execute()` bit-for-bit |
| invocation | `MEMRA_ORACLE_OUT=<tsv> glm5_checkpoint_runner ~/models/glm53-nvfp4 1 2 3 4` |
| output | full last-position logits row, 154,880 entries, `memra-checkpoint-oracle-v1` |

## What the manifest covers, precisely

`SHA256SUMS.txt` covers all 30 files in the artifact directory. **28 of them are
uploaded.** Two are excluded by design and are NOT part of the published repo:

- `README.md` as it sits in that directory is the **vendor's** card (zai's benchmark
  tables and citation). Our authored card replaces it. The card is documentation, not
  gated bytes, so iterating on it never invalidates this receipt.
- `config.json.pre-keeplist-fix` is an internal backup of the pre-fix config.

The 28 published files are frozen from the moment of the re-run. If any of them has to
change, the re-run is repeated. The card may change freely.

## Corroboration, independent of the oracle

- `model.safetensors.index.json` declares 113,446 weight-map entries. Census arithmetic:
  1,432 kept tensors + 3 x 37,338 quantized tensors (`weight`, `weight_scale`,
  `weight_scale_2`) = 113,446. Matches `mint-receipts/mint-log-summary.txt` exactly.
- `total_size` in the index is 190,701,634,272 B; the directory measures
  190,750,196,900 B including the non-tensor files.
- Two metadata hashes in the manifest match values pinned independently elsewhere in the
  lane: `chat_template.jinja` `34d5ee66…` equals `template_sha256` in
  `inspect-receipts/artifact.lock`, and `generation_config.json` `230c3060…` equals the
  hash `BRINGUP.md` records for it. The tokenizer and template shipped here are the
  pinned ones.
- Bench box and upload box compared read-only, per file: identical sizes and mtimes on
  all 30 entries at the time the upload set was enumerated. The bench box additionally
  held `.memra-repack` (regenerable expert slab), which was absent on the upload box at
  that point and is not any more; see the incident section below.

## Box incident during the re-run, recorded because it changes the upload set

The first re-run attempt was killed at layer 21 of 46 when the upload box rebooted
(kernel 6.8.0-1061 to 6.8.0-1063). Unattended-upgrades is NOT configured to reboot
automatically on that box, so the reboot came from outside this lane; the box is shared.
Log kept as `oracle-rerun-attempt1-killed-by-reboot.log`. The run was restarted from
scratch, not resumed, because a partial oracle is not a receipt.

Two things changed on the box across that window and both matter:

1. The artifact's 30 files survived intact (file count and per-file sizes unchanged,
   the volume is EBS). `sha256sum -c` against the banked manifest is re-run immediately
   before upload, so the manifest binds post-reboot bytes rather than pre-reboot ones.
2. **A `.memra-repack/` directory appeared inside the artifact directory**, 160 GiB
   across 126 files, timestamped after the reboot. That is the engine's regenerable
   expert-slab cache, built when a server loads this artifact. It was not present when
   the upload set was first enumerated. It must never be published, and a folder upload
   recurses into it, so the upload excludes it by explicit pattern rather than by it
   happening to be absent.

## Privacy scrub of the published metadata, run BEFORE the gate finished

Deliberately ordered that way: if the scrub had forced an edit to any gated file, the
running gate would have been covering bytes that could not ship, and the re-run would
have had to start again. Better to know 40 minutes early.

All seven small text files scanned for `/home/<user>` paths, `/root/`, box hostnames
(`ip-172-*`), IPv4 literals, ssh user@host strings, and this lane's own directory names:

`config.json`, `generation_config.json`, `hf_quant_config.json`, `processor_config.json`,
`tokenizer_config.json`, `chat_template.jinja`, `model.safetensors.index.json`.

**All clean.** The only producer string is `{"name": "modelopt", "version": "0.46.0"}`,
which is public tool identity and is a fact worth publishing rather than hiding. The
index metadata is `{"total_size": 190701634272}` and nothing else. No file needed
scrubbing, so no re-hash and no second gate run were triggered.

## RESULT

Filled in from the re-run. See `RESULT.md` beside this file.
