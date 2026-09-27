# Step HF text-schema correction

This is commit B of the isolated Step correction. It depends on the centered-norm declaration
in A (`944516e44b4ccec3f39926a377d8934dec759a5f`) and the existing #537 Step compiler. Its
production code and pack-local tests do not depend on #541's bound-source foundation, runtime
adapter, or universal activation. Ancestry contains that work, but the patch does not require it.

The Step pack owns HF names and shapes for its stacked routed projections, router/bias and shared
experts. It declares q/k norm `+1`, uses `self_attn.g_proj.weight`, and binds the existing private
MTP output/norm names under `transformer.shared_head`. Norm and projection ownership retain the
plan's `PreferPrivateThenModel` policy. Step's MTP layers remain the inherited dense plan. GGUF
storage retains identity transforms for already-folded weights. All requirements still pass the
compiler's duplicate-ID check after family normalization. Tokenizer JSON is an accepted source
for this HF dialect; pack support state remains unchanged.

Pack-local CPU tests compare 66 roles in a four-block fixture against the existing native mapper:
3 globals, two dense blocks with 12 tensors each, two MoE blocks with 17 each, and 5 MTP glue/head
roles. They also bind the complete tiny text census and reject missing, swapped-shape, misowned,
and non-floating router tensors. Replacing only the q/k fold with identity makes the mapper
comparison fail at the intended norm. Raw green/red logs are retained. Clippy for GGUF/CLI all
targets with warnings denied passes. These are schema/source checks, not native qualification.

## Current pinned metadata boundary

The current index and config were independently hashed against the existing lock for
`stepfun-ai/Step-3.7-Flash-FP8@b3d7916fccac844cca050d7520f2aaa513f9a84f`:

- Config SHA-256: `4cff5ed015bff319274d0967aeca5a4753ad718c3ca4cdcf8216fcd93ba44037`.
- Index SHA-256: `aac0a86e46d5bc212ace03d824d893d17ecef09cf3621fde0f79c3847e03ae1e`.
- The index declares 1,471 entries: **804 text entries and 667 vision/projector entries**.
- This compiler declares all 804 indexed text names, with zero missing declarations and zero
  unclaimed text names. Private norm/output names for MTP layers 45, 46 and 47 are included.

`raw/pinned-index-names.json` binds these counts to the exact metadata hashes and probe binary.
`raw/declared-names.tsv` records semantic IDs. The 667 unrepresented names are preserved verbatim
in `raw/unrepresented-vision-projector-names.txt`; none were dropped from or rewritten in an
artifact. `names.rs` reproduces the declaration list from a local copy of the locked config.

This is **index-name evidence only**. Shard headers, actual shapes/dtypes/auxiliaries, payloads and
full-artifact completeness are not proved by an index. The perception-encoder vision program is
not represented by this Step text plan. A complete census must therefore refuse those surfaces
until an explicit supported/unused-surface contract is reviewed; this patch neither pretends they
are text tensors nor qualifies vision execution. Universal #541 activation must preserve working
text loading through a reviewed scope contract, not ignore unknown tensors or alter checkpoints.
The acceptance gap remains tracked under #541; this is not a separate model-support promotion.

Fresh integration source review and plan/binary-bound native receipts are required. Old Step
qualification records remain historical. No GPU execution or merge is part of this correction.
