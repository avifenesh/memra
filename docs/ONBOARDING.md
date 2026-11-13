# Model onboarding

Onboarding takes a new model family, or a new artifact of a known family, from a pinned checkpoint
to a support record. This procedure is for maintainers and agents. The owner rules behind it are in
[AGENTS.md, Model onboarding](../AGENTS.md#model-onboarding-compile-a-native-plan-not-a-new-engine-path-owner-2026-08-22).

Memra compiles each model into a native `ModelPlan`, binds the checkpoint's tensors to that plan,
and validates every tuned program against the reference executor. A new model is a model pack, not
a new engine path. The preferred semantic source is the official safetensors checkpoint with its
config, tokenizer, template, quantization metadata and every indexed or auxiliary tensor file.
GGUF stays a supported import format and passes the same gates. Neither format may redefine the
checkpoint's compute program, and a missing, duplicate, substituted or unsupported surface fails
closed.

Related documents: [Testing](TESTING.md) (gate catalog), [Release qualification](RELEASE-QUALIFICATION.md)
(per-cell push evidence, and [what a tag record covers](RELEASE-QUALIFICATION.md#what-a-record-covers)),
[Releasing](RELEASING.md) (the release procedure), [Flags](FLAGS.md),
[Serving](SERVING.md) (drafter attach syntax), [Draft regime](DRAFT-REGIME.md),
[Tokenizer program](TOKENIZER-PROGRAM.md), [Execution surfaces](EXECUTION-SURFACES.md) and
[`support-records.toml`](support-records.toml). A worked same-family example (Qwen 3.8 27B FP8
safetensors, 2026-08) and the procedure this page replaced are in
[history/ONBOARDING.md](history/ONBOARDING.md).

## The pieces

| Path | What it owns |
|---|---|
| `crates/memra-gguf/src/model_plan.rs` | `ModelPlan`, the typed program: norms, RoPE, attention and state mixers, gates, MLPs, residual topology, logits transforms, MTP blocks, multimodal operations. No kernel choice. |
| `crates/memra-gguf/src/tensor_contract.rs` | Checkpoint names to semantic `TensorId`s with shape, owner, transform and quant layout, per dialect (GGUF or HF). |
| `crates/memra-gguf/src/model_packs/<family>/` | The `ModelPack` (fields below). Statically linked Rust; no plugin ABI. |
| `crates/memra-gguf/src/config.rs` | Parses an artifact into `ModelConfig` (`Arch`, sizes, per-layer arrays). `ArchGeometryTable` holds per-layer geometry for Qwen35, Qwen35Moe, Qwen4Exp, Step35 and MiMoV2; other architectures keep their own config paths. |
| `crates/memra-gguf/src/checkpoint_binding.rs` | `bind_source`: the contract against the checkpoint's metadata census, before any upload. `audit_consumption`: bound tensors the loader never read. |
| `crates/memra-gguf/src/execution_manifest.rs` | Which tuned rewrites a plan is eligible for, per `RewriteSurface` (`CarriedPrime`, `DecodeEager`, `DecodeBatch`, `DecodeGraph`, `MtpSpec`, `Glm5Spec`, `Pipeline`), and the receipts that admit them. Per-operation support comes from `op_registry.rs` ([Execution surfaces](EXECUTION-SURFACES.md)). |
| `crates/memra-reference/` | The unfused CPU executor and semantic baseline. An unsupported operation is a named error, never a fallback. |
| `crates/memra-cli/` | The `memra model` commands: `inspect`, `scaffold`, `verify`, `qualify`. |
| `docs/support-records.toml` | One record per supported capability, with the receipts behind each passed gate. |

`ModelPack` fields: `family`, `aliases`, `matches_config`, `config_layout`, `output_head`,
`tensor_consumption`, `tokenizer_sources`, `template`, `support`, `gates`, `checkpoint_parity`,
`plan_builder`, `tensor_schema` and `tiny_plan`. Load-time selection walks `PACKS` in order and
takes the first pack whose `matches_config` accepts the config (`llama_dense` is last).
`ONBOARDING_PROFILES` are artifact-storage profiles (`hy3_nvfp4`, `mimo_v2_mint`, `mimo_v2_source`)
that `--against` can name but that never auto-select. `--against` takes a family name or any alias.

## Support states and their gates

A pack's `support` field is its ModelPlan state, not a claim about older handwritten paths.
`tools/check-support-states.py` (CI gates job) requires it to equal the highest state its own
records in `docs/support-records.toml` prove, and every passed gate in a record must cite a
tracked `gates.txt` receipt for the same family (or `ci:verify-tiny`, the hosted tiny test, for
Config and TinyParity). The full rules are at the top of that file.

- `NativeReference` needs Config and TinyParity. <!-- support: none; not NativeReference -->
- `NativeQualified` adds TokenizerTemplate, TensorCensus, CheckpointParity and Serve on a pinned, <!-- support: none; not NativeQualified -->
  non-tiny artifact, with a memra commit, topology and route in the record.
- `NativeTuned` adds RewriteParity. <!-- support: none; not NativeTuned -->

A pack with `support: None` inspects but has no native execution, and `verify tiny` refuses it.
A tiny fixture never promotes a pack. Every pack declares all seven gates.

| Gate | Written by | Receipt in the bundle directory |
|---|---|---|
| Config | `inspect`, `verify tiny` (`verify config` checks it and writes nothing) | `gates.txt` |
| TokenizerTemplate | `inspect` | `tokenizer-contract.tsv` (`tokenizer-error.txt` on failure) |
| TensorCensus | `inspect` | `tensor-census.tsv`, `binding=` in `artifact.lock` (`contract-error.txt` on failure) |
| TinyParity | `verify tiny` | `tiny-gate.tsv`, `reference-oracle.tsv` |
| CheckpointParity | `verify checkpoint` | `checkpoint-parity.tsv` |
| RewriteParity | `verify rewrite` | `rewrite-receipts.tsv`, `rewrite-receipts/<id>.tsv` |
| Serve | `verify serve` | `serve-gate.tsv`, `serve.log`, `serve-response.json` |

Each stage that writes a directory rewrites its `gates.txt` from the lock's verdicts and every
receipt there that still binds to the lock and the family.

## Ordered checklist

Run from source with `cargo run -p memra-cli --bin memra -- model ...`, or a built `memra`. Any
error prints `memra: <error>` and exits 2. `<bundle>` is one directory reused by inspect,
checkpoint, rewrite and serve; each stage reads what the previous one wrote there.

| Phase | Command or work | Exit condition |
|---|---|---|
| 0 | Branch, receipt namespace, pinned source | `hf-id@<40-char sha>` or a local artifact plus its byte manifest recorded |
| 1 | `memra model inspect <source> --against <family> --out <bundle>` | Config, TokenizerTemplate and TensorCensus pass, or every difference is classified |
| 2 | `memra model scaffold <family> --out <dir>`, then the Rust pack | `memra model verify config <source> --against <family>` passes |
| 3 | Typed operation plus its reference implementation, only where the plan needs one | `memra model verify tiny --against <family> --out <dir>` matches the pinned oracle |
| 4 | Tokenizer, template, tools, thinking | token-id parity and rendered-byte goldens pass |
| 5 | Drafter attach | `run-spec` K=1..8 self-consistency and non-zero acceptance |
| 6 | `tools/generate-arch-gates.py` | chunk, tick and B>1 scaffolds reviewed and merged by hand |
| 7 | Rewrite receipts, then `memra model verify rewrite <receipt> --against <family> --out <bundle>` | every enabled rewrite has a bound receipt |
| 8 | `verify checkpoint`, `verify serve`, the device battery | all green with raw logs |
| 9 | Pack `support`, record, `memra model qualify`, model card, INDEX row | `python3 tools/check-support-states.py` passes |

## 0. Freeze the lane and artifact

Work on a dedicated branch or worktree. Put receipts under a dated
`research/modelplan-onboarding-<topic>-<YYYYMMDD>/` directory (for example
[`research/modelplan-onboarding-dsv41-20261109/`](../research/modelplan-onboarding-dsv41-20261109/)).
Record the branch, the source commit, the worktree status and the GPU state at entry.

The source is either a pinned Hugging Face revision, `owner/repo@<40-character commit sha>`, or a
local path: a GGUF file (any shard of a split GGUF), an HF safetensors directory (`config.json`
plus `model.safetensors[.index.json]`), or a Hy3 repack directory with `manifest.json`. A pinned remote
source is inspected from its config, tokenizer files and safetensors headers over HTTP range reads;
no weights download. Execution (`verify checkpoint`, `verify serve`) needs the artifact on disk.
For a local artifact, hash every file the source reads: the config, tokenizer files, index and
every indexed shard, or every GGUF shard.

Do not edit model metadata to make a parser accept it. If the artifact changes, start a new
manifest and keep the old receipt.

## 1. Inspect before implementing

```bash
cargo run -p memra-cli --bin memra -- model inspect <source> --against <family> --out <bundle>
# family=<family> tensors=<n> artifacts=<bundle>
```

`inspect` writes, in `<bundle>`:

- `artifact.lock`: source, revision, family, the pack's support state and checkpoint tolerance,
  `config_sha256`, `census_sha256`, `plan_sha256`, `rewrite_manifest_sha256`, and the `binding=`
  and `tokenizer=` verdicts;
- `normalized-config.txt`: the `ModelConfig` the pack compiled from;
- `tensor-census.tsv`: every tensor with semantic and physical name, dtype, shape, storage and
  bytes;
- `model-plan.txt`: the compiled plan;
- `execution-rewrites.tsv`: each tuned rewrite, its surface, implementation, plan hash and
  eligibility;
- `tokenizer-contract.tsv`: tokenizer hash, dialect and template hash (or the engine renderer);
- `placement-checkpoint-<N>.tsv` for 2 to 4 pipeline stages, when the tensor contract binds;
- `gates.txt`.

The lock is bound to the code as well as to the bytes. `plan_sha256` hashes the compiled plan's
text. For an HF source `config_sha256` hashes the `config.json` bytes; for a GGUF or repack source it
hashes `normalized-config.txt`, so a `ModelConfig` field change moves the lock when the same artifact
is inspected again. A regenerated lock needs its rewrite receipts regenerated with it.

`gguf-inspect <model.gguf> --all` (or `--config`) prints a GGUF's raw header and metadata when the
normalized view is not enough.

Read the outputs as four inventories:

1. Config: architecture id, layer count, per-layer arrays, head dimensions, attention classes,
   RoPE parameters, window, MoE and router fields, normalization, clamp values, MTP depth.
2. Tensors: naming patterns, shapes, dtypes and layouts, shard ownership, optional gates, position
   factors, expert banks, output heads, companion draft blocks.
3. Tokenizer: model, `tokenizer.ggml.pre` or pre-tokenizer, normalizer, vocabulary and special ids,
   BOS and EOS behavior, exact encoding on a frozen parity corpus.
4. Template: raw template hash, roles, tools branch, generation prompt, whitespace, reasoning
   boundaries, `enable_thinking` and any string-valued reasoning control.

Classify every difference:

| Class | Action |
|---|---|
| Mechanically derived | Parse it once into `ModelConfig` (and the geometry table where the family uses one). |
| Existing semantic contract | Reuse it only after model-backed parity proves identical behavior. |
| New semantic contract | Add a typed operation, a narrow loader or renderer path, and confusion-guard tests (phase 3). |
| Unsupported or ambiguous | Stop. The plan refuses and the dispatch door stays closed. |

Tensor shape is evidence, not semantics. Do not borrow a shape-compatible output head, gate or
draft block without name and topology confirmation.

## 2. Write the pack and its plan

`memra model scaffold <family> --out <dir>` writes a worksheet into an empty directory:
`pack.toml`, `aliases.txt`, `config-normalization.txt`, `tensor-schema.tsv`,
`tokenizer-template.txt` and a `gates.txt` with all seven gates pending. No code reads these
files. Fill them in, then write the pack as Rust in `crates/memra-gguf/src/model_packs/<family>/`
and register it in `PACKS` (or `ONBOARDING_PROFILES` for a storage profile of an existing plan).

- If the math exists, the pack is small: aliases, `matches_config`, config normalization, the
  tensor schema and a `plan_builder` that composes existing operations. A sibling with existing
  math is one pack, not another forward implementation.
- A new architecture string needs its `Arch` variant and parsing in `config.rs`
  (`ModelConfig::from_hf`, `from_gguf`). Families that use `ArchGeometryTable` declare every
  trunk and MTP layer there; tuned arms read `layer_geometry()` or `full_attention_geometry_at()`
  instead of per-architecture match arms. A companion drafter resolves its rows from its own
  config; never fabricate an out-of-range row or borrow a trunk row because a shape looks close.
- Geometry tests: every layer resolves the expected class, class counts and transitions match the
  metadata, MTP rows resolve deliberately, windowed and full rows differ where the artifact says
  they differ, the gate kind matches tensor ownership, and missing or malformed arrays fail
  instead of falling back to projection-wide scalars.
- Runtime policy reads the plan or its manifest, never an architecture-name allowlist.

`memra model verify config <source> --against <family>` compiles the plan from the config alone.

### The name map is a SECOND surface

The tensor contract (`tensor_contract.rs`) and the engine's ggml-to-HF name map
(`hf_mapping::{ggml_to_hf, resolve_ggml}`) are two spellings of the same checkpoint. Both loaders
(`Model::load_dense_from_source`, `HybridModel::load_from_source`) call
`memra_gguf::checkpoint_binding::bind_source` before any upload. A missing, unexpected, duplicate,
ambiguous, wrong-shape or wrong-quant tensor refuses the load with
`checkpoint refused before upload (pack <family>, <dialect>, N census tensors): <error>`. What
`model inspect` refuses, the server refuses, with the same text.

- Output-head ownership is the pack's `output_head`. An absent `output.weight` selects the token
  embedding only under `TiedHeadAllowed`, and never when `config.json` says
  `tie_word_embeddings: false`. A `SeparateHead` family refuses.
- The hybrid loader reads through a `RecordingSource`, and `audit_consumption` names every bound
  tensor it never read. The pack's `tensor_consumption` makes that list a report (`Report`) or a
  refusal (`Refuse`). `Refuse` needs a receipt on a real artifact
  ([`research/loader-census-20260922/`](../research/loader-census-20260922/) for qwen35).
- Write the map-completeness pin for a new architecture: compile the real plan and require every
  tensor the GGUF-dialect contract declares to resolve through `resolve_ggml` onto a name the HF
  dialect of the same contract declares for the same `TensorId`. Pin the count; no per-name
  allowlist. `glm5_next_every_contract_tensor_resolves_through_the_engine_map` in `hf_mapping.rs`
  is the template, and it runs on CPU.
- Where the plan declares a tensor (a selection-bias router, an always-on shared expert, residual
  topology parameters), the loader refuses by name instead of substituting a default. Add the
  refusal in the same change as the semantics and run both arms before merging.

The bind catches a checkpoint that does not match the contract, the pin catches a map that does
not match it, and the consumption audit catches a loader that matches neither. The history of this
boundary (memra#541) is in [history/ONBOARDING.md](history/ONBOARDING.md).

## 3. Add new semantics behind closed doors

When the plan reports an operation memra does not have, add the typed operation to
`model_plan.rs` and its implementation to `memra-reference` first. Tuned backends then get an
explicit rewrite (phase 7); until then the operation runs only in the reference executor, and the
generated [Execution surfaces](EXECUTION-SURFACES.md) table shows it with no tuned program.

Add only what the inspection marked new: parser aliases and typed metadata, tensor ownership and
required-tensor checks, attention, FFN or router math, quantization layouts, tokenizer or chat
renderer, drafter topology, target kernels. Keep graph, device-counter, pointer-table, resident,
grouped, fused and batched doors closed until they consume the same declared geometry and have an
arm-specific exactness and liveness gate. A refusal that names the architecture and the missing
contract beats a plausible wrong answer.

Kernel tests need confusion guards, not only happy paths. Exercise the nearest wrong
interpretation: fused versus separate gates, full versus partial rotary, windowed versus full
attention, block-local versus file-level heads, uniform versus mixed quant layouts, pruned versus
retained ids.

`memra model verify tiny --against <family> --out <dir>` builds the pack's `tiny_plan`, runs a
deterministic fixture twice in the reference executor (it must be bit-identical), writes
`tiny-fixture.txt`, any `tiny-checkpoint/` image and `reference-oracle.tsv`, and compares the run
against a reviewed oracle pinned at `crates/memra-cli/tests/fixtures/tiny-oracle/<family>.tsv`
(registered in `pinned_tiny_oracle` in `crates/memra-cli/src/lib.rs`) within the pack's
`checkpoint_parity` tolerance. A family without a pinned oracle cannot pass. A divergence is
reviewed and the oracle re-pinned on purpose, never regenerated silently. Hosted CI runs `verify
tiny` for every supported pack (`ci:verify-tiny` in the records).

A tiny oracle from the vendor's code pins the program only if both sides evaluate their
primitives the same way once the source stores activations in bf16 and selects by top-k. One f32
ulp of accumulation difference flips a bf16 rounding, the flip changes a selection, and later
layers move by O(1). The `deepseek_v41` tiny gate evaluates GEMMs, reductions and transcendental
functions in f64 and rounds once on both the vendor and the reference side, takes top-k ties in
index order (the CUDA `topk` rule), and matches the vendor `model.py` (torch stand-ins for its
tilelang kernels) to the bit
([`research/modelplan-onboarding-dsv41-20261109/`](../research/modelplan-onboarding-dsv41-20261109/)).
A fixture also pins only the splits it runs. Where prefill and decode lay a row out differently, a
split there is another numeric program, so the gate pins one variant per layout (the
`deepseek_v41` `split60` and `window128` oracles).

### The checkpoint-parity probe and numeric classes

`memra model verify checkpoint` runs the reference oracle and the native runner on one prompt
and compares the full logits row after it: every logit within the pack's `atol + rtol *
abs(reference)`, the argmax equal when the pack requires it. The prompt is tokens 1,2,3,4 unless
the pack declares `checkpoint_probe`. Declare one only when the default probe cannot separate a
wrong program from a right one on the family, and record the measurement that showed it next to
the declaration. A declared probe is part of the record: its sha256 is written to the
`artifact.lock` and to `checkpoint-parity.tsv`, so a different probe is a different receipt.
Packs that declare nothing keep the default probe and byte-identical locks and receipts.

DeepSeek-V4.1 declares its `capital` chat turn (16 tokens). On 1,2,3,4 (`<eos> <pad> ! "`) the
vendor's own CPU and GPU runs of the checkpoint program disagree by 13.3 logits (KL 0.49, top-5
overlap 3/5, hidden states 0.45 apart by layer 38): a band wide enough for the vendor's two
backends admits nearly any program. The receipts are in
`research/modelplan-onboarding-dsv41-20261109/gpu/`.

The oracle and the native tape each name a `numeric_class`, and the gate refuses a mismatch. A
family whose checkpoint program runs under more than one class (a correctly rounded CPU reference
next to a hardware-primitive GPU route) gives each class its own band in
`checkpoint_parity_classes`; the oracle's class picks it, and an unnamed class falls back to
`checkpoint_parity`. Set a class band from the source program's own spread between two backends on
the probe, never from the native runner's result.

## 4. Prove tokenizer, template, tools and thinking

The pack declares where the tokenizer comes from (`tokenizer_sources`: `TokenizerJson`,
`GgufMetadata`) and who owns the chat dialect (`template`): `ArtifactRequired` fails when the
artifact carries no template; `EngineRenderer(<name>)` names an engine-owned renderer and refuses an
artifact that also carries a template. `memra-tokenizer` accepts the pre-tokenizers `qwen35`,
`qwen2`, `deepseek-v3`, `gemma4` and `glm4` and refuses any other at load. Explicit HF-to-GGUF
tokenizer programs are in [Tokenizer program](TOKENIZER-PROGRAM.md).

Tokenizer and template are separate gates:

- compare token ids against the authoritative tokenizer on a frozen corpus that covers Unicode,
  whitespace, punctuation, code and special-token boundaries (`tok-parity`);
- hash and archive the source template;
- render golden conversations for system, user, assistant and tool roles, generation prompts and
  assistant continuations;
- check tool calls and reasoning stripping on the live server;
- map `reasoning_effort` and `reasoning` through the per-architecture behavior table.

Template marker detection is only triage. A template with Qwen markers can still differ in roles,
whitespace, tool semantics or thinking control. The gate is the rendered bytes and the served
response.

## 5. Attach drafters explicitly

The served form is `MEMRA_MODELS="alias=/path/trunk.gguf+/path/draft.gguf"`; a missing or
non-file draft is a boot refusal ([Serving](SERVING.md)). Direct `run-spec` work may use
`MEMRA_MTP_DRAFT`. How each model's draft is built (ranks, trim, quantization) is in
[Draft regime](DRAFT-REGIME.md). A safetensors checkpoint carries an embedded MTP head when any
tensor named `mtp.*` is in it; `tools/serve-st-gate.sh` applies the same test.

Before allocating the full model, check architecture and vocabulary compatibility, embedding
width, attention geometry, token ids, MTP depth, block sequence and state topology, block-local
output-head ownership, and any architecture-specific transforms. Then require both:

1. naked `run-spec` K=1..8 self-consistency; and
2. non-zero acceptance and end-to-end evidence on the served model, drafter and prompt regime.

Self-consistency proves verification is correct. It does not prove that a shape-compatible head is
useful: 0% acceptance can still be self-consistent.

## 6. Generate standing gate scaffolds

`tools/generate-arch-gates.py` renders the repeated mechanics from the architecture name, a default
artifact path and an explicit JSON spec:

```bash
python3 tools/generate-arch-gates.py --list-ports       # bound ports, and the free ones in the band
python3 tools/generate-arch-gates.py "<architecture>" <artifact> \
  --spec <receipt-dir>/gate-spec.json \
  --out-dir tools/generated-arch-gates/<slug>
python3 -m unittest tools.test_generate_arch_gates
tools/test_gate_template_integrity.sh                   # runs a generated gate against a stub server, CPU only
shellcheck tools/generated-arch-gates/<slug>/*.sh
```

`--out-dir` defaults to `tools/generated-arch-gates/<slug>`; `--force` replaces generator-owned
files. The output holds `<slug>-chunk-invariance-gate.sh`, `<slug>-tick-invariance-gate.sh` and
`<slug>-b2-geometry-gate.sh`, `fast-gate-models.tsv` (naked and canary rows for
`tools/fast-gate/models.tsv`), `fast-gate-map.tsv` (rows for `tools/fast-gate/map.tsv`) and the
normalized spec with output hashes (`gate-spec.json`, `schema_version` 2). The generator never
edits the canonical registries: review the fragments and merge them by hand.

What the generated gates already do (the rule is in [Testing](TESTING.md), its history in
[history/TESTING.md](history/TESTING.md)):

- the B>1 gate's port comes from the reserved band 18300-18399, checked against a computed census
  of the ports `tools/` binds and against sibling generated specs;
- that port has its own override, `MEMRA_<SLUG>_B2GEO_PORT`;
- the B>1 gate guards the port at run time through `tools/port-guard.sh` and refuses to run
  without it;
- in every generated gate, a skip (missing artifact, `nvidia-smi`, GPUs or declared drafter) exits
  77 and is appended to `$MEMRA_SKIP_CENSUS` when wired; `MEMRA_ARCH_GATE_ALLOW_SKIP=1` accounts
  for it and the run says it proves nothing;
- a drafter requested by the environment and absent is a hard FAIL;
- an empty completion (`{"reasoning": null, "content": null}`) is refused;
- the canary must match its declared `canary_expect_regex`, and the number of assertions that ran
  must equal the count the spec implies (`1 + sum(concurrency) + 2`).

The spec carries what the generator cannot derive. The prompt must cross the real window or class
boundary. Chunk and tick sets must straddle the suspected segmentation boundary. The canary seam
must change the runtime world and be shown to bite, and `canary_expect_regex` names the verdict
line that seam is guaranteed to break. PP topology must match the only valid placement, the
concurrency widths must execute B>1, and the liveness regexes must prove the intended arm ran.
`port` is the one non-scientific field; pick a free value from `--list-ports`. The field reference
is the [appendix](#appendix-gate-spec-fields).

Copy the current base probe lists from `tools/fast-gate/map.tsv` before installing generated rows.
Do not make a canary pass by changing only its expected label,
and do not treat SKIP from a missing artifact or GPU as evidence.

## 7. Qualify rewrites and enable execution arms

A tuned program is a rewrite of the plan, admitted only by a receipt. `execution-rewrites.tsv`
lists each rewrite with its surface, implementation, plan hash and eligibility. The gate binaries
write receipts (`memra-rewrite-parity-v1`) bound to `MEMRA_ARTIFACT_LOCK=<bundle>/artifact.lock`,
to the file `MEMRA_REWRITE_RECEIPT=<file>` names ([Flags](FLAGS.md));
`modelplan_reference_gate` takes the receipt path as its second argument
(`<config.json> <receipt.tsv>`):

| Surface | Receipt writer |
|---|---|
| `DecodeEager` | `modelplan_reference_gate` |
| `CarriedPrime` | `prime-batch-gate` |
| `DecodeBatch`, `Pipeline` | `decode-batch-gate` |
| `DecodeGraph` | `graph-decode-gate` |
| `MtpSpec` | `run-spec` |

No gate binary writes a `Glm5Spec` receipt today.

`memra model verify rewrite <receipt> --against <family> --out <bundle>` checks that the bundle's
lock names the family, then checks the receipt's rewrite id (eligible in
`execution-rewrites.tsv`), surface, implementation, plan hash and artifact-lock hash against the
bundle, requires `status=passed` and `first_violation=none`, and copies it to
`<bundle>/rewrite-receipts/<id>.tsv` with an index row in `rewrite-receipts.tsv`. A receipt is
valid only for its artifact lock, serialized plan, stream and numeric class, and binary.

`MEMRA_REWRITE_BUNDLE=<bundle>` installs the bundle in `memra-server`. An invalid bundle fails the
model load. Optional batch, graph or prime surfaces without a receipt use the receipt-backed native
eager rewrite; required speculative or pipeline surfaces fail closed.

Promote arms in this order, keeping a reference path and requiring output identity plus liveness
evidence at each step:

1. parser and loader;
2. eager prefill and tokenwise decode (`DecodeEager`);
3. chunked and per-tick prefill (`CarriedPrime`);
4. MTP verify (`MtpSpec`, or `Glm5Spec` for the glm5_next loop);
5. PP placement (`Pipeline`);
6. batched prefill and decode (`DecodeBatch`);
7. device-counter, graph and session paths (`DecodeGraph`);
8. fused or grouped kernels.

A faster arm's throughput never waives a quality or routing mismatch.

## 8. Run the qualification ladder and the device battery

Checkpoint parity compares the native runner's last-position logits for the token ids 1, 2, 3, 4
with a Hugging Face transformers fp32 oracle:

```bash
memra model verify checkpoint <local-hf-dir> --against <family> --out <bundle>
# first run: inspects again, writes <bundle>/capture-hf-oracle.py and oracle-request.tsv, and stops
python3 <bundle>/capture-hf-oracle.py --out <bundle>/hf-oracle.tsv   # needs torch and transformers
memra model verify checkpoint <local-hf-dir> --against <family> --out <bundle> \
  --oracle <bundle>/hf-oracle.tsv --native-runner <binary>
```

The pack must declare `checkpoint_parity` (`max_abs`, `max_rel`, `require_argmax`). The runner is
`run-safetensors` or a `memra-reference` checkpoint runner (`glm5_checkpoint_runner`,
`llama_dense_checkpoint_runner`, `mimo_v2_checkpoint_runner`); the CLI calls it on the directory
with `MEMRA_FULL_PREC=1` and `MEMRA_ORACLE_OUT=<bundle>/native-oracle.tsv`.
Without `--native-runner`, the CLI reads `MEMRA_NATIVE_CHECKPOINT_RUNNER`.

```bash
memra model verify serve <local-artifact> --against <family> --out <bundle> \
  --native-runner target/release/memra-server
```

`verify serve` needs the bundle's lock (same source, `binding=passed`, `tokenizer=passed`) and a
passed checkpoint receipt bound to that lock. It boots the runner on a free loopback port with
`MEMRA_MODELS=verify=<source>`, `MEMRA_REWRITE_BUNDLE=<bundle>` and an API key, waits for
`/readyz` (`MEMRA_SERVE_VERIFY_TIMEOUT_S`, default 180), and requires a one-token
`/v1/completions` answer. Without `--native-runner`, the CLI reads `MEMRA_NATIVE_SERVE_RUNNER`.

Then run the device battery on the target hardware, under the box's GPU lock
([AGENTS.md, lock names](../AGENTS.md#lock-names-are-a-correctness-surface)), with raw logs teed
first and parsed second:

| Gate | Required proof |
|---|---|
| Config and parser tests | metadata arrays and geometry classes exact |
| `kernel-check` | final `ALL GREEN (<n> cells, <m> skipped)`, new confusion guards included |
| `tools/argmax-margin-gate.sh` | calibrated prefill-versus-decode verdict, `--decode eager` and `--decode batch` |
| `run-gen` | batched-prime and tokenwise `MATCH`; pinned tokens unchanged across refactors |
| `run-spec` | K=1..8 `SELF-CONSISTENCY PASS` for every supported drafter |
| Chunk and tick gates | naked invariant result plus a canary that breaks the same assertion |
| B>1 geometry | c=2 and c=4 equal c=1, and the logs prove the B>1 arm ran |
| `tools/serve-smoke.sh` | real API, streaming, concurrency, deterministic output, spec/plain identity |
| Architecture surface | tokenizer, template, thinking and tool goldens plus live requests |

The argmax gate judges a cell against a floor calibrated for its model, prompt, decode program and
exact device name, and against main's stored rows. Calibrate a new cell with
`tools/argmax-margin-gate.sh <model> --prompt <file> --decode <eager|batch> --calibrate`; floors
live in `tools/argmax-margin-floors.tsv` and stored rows in `tools/argmax-margin-rows/`. A cell with
no floor row reports UNCALIBRATED, and a flip there fails
([ARGMAX-GATE-GAP-SHIFT-FLOOR](decisions/ARGMAX-GATE-GAP-SHIFT-FLOOR.md),
[ARGMAX-GATE-STORED-ROWS](decisions/ARGMAX-GATE-STORED-ROWS.md)).

`tools/fast-gate/fast-gate.sh --tier 2` runs the full local battery (`tools/local-ci.sh`). A new
architecture also needs its own artifact probes until the standing battery owns them, and a
clean-tree fast-gate run must name `--probes`: an empty diff runs nothing.

Push evidence follows the change, not the lane: `python3 tools/cell_evidence.py need` lists the
cells a diff reaches ([Release qualification](RELEASE-QUALIFICATION.md)). A pack the device battery
does not cover (only `qwen35` and `qwen35_moe` have roster models in `tools/change-impact.toml`)
produces no device cell; the gate names it UNCOVERED, and the pack's own gates go in the PR body.
What to record, and how to word a failure, is
[AGENTS.md, Evidence discipline](../AGENTS.md#evidence-discipline).

## 9. Close the lane

1. Set the pack's `support` field and add its record to `docs/support-records.toml` in the same
   PR, with tracked `gates.txt` receipts for every passed gate and the pending gates listed as
   pending. Run `python3 tools/check-support-states.py`.
2. A serving qualification record (`memra-qualification-record-v1`, written by
   `tools/collect-serving-qualification.py`, which targets qwen35 today) must name the cells
   `cell.streaming`, `cell.cache`, `cell.concurrency`, `cell.cancellation` and
   `cell.long_context`. `memra model qualify <record> [--evidence-dir <dir>]` checks that each
   `passed` cell names an evidence file whose sha256 matches, and refuses a promotion past the first
   state unless every cell passed. It checks consistency; it runs nothing.
3. Write a short `docs/models/<model>.md` card: pack state with its support marker, the measured
   configurations and their gates, deeper sources. Commands go in [Cookbook](COOKBOOK.md); the card
   links them. A new hardware or workload class gets a `docs/rigs/` or `docs/workloads/` card.
4. Add the lane's row to `research/INDEX.md`.
5. A changed default, format, target or arm gets a record in [decisions/](decisions/README.md).
6. Flags go in [Flags](FLAGS.md), serving contracts in [Serving](SERVING.md), gate ownership in
   [Testing](TESTING.md). Moved published numbers follow
   [AGENTS.md, Perf board](../AGENTS.md#perf-board-and-pre-push-hook).

Before merge: the battery is green, raw receipts are committed, unsupported doors still fail
closed, generated perf surfaces pass `python3 tools/update-perf-board.py --check`, no unrelated
work is staged, and a tag follows [Releasing](RELEASING.md).

## What stays architecture-specific

| Surface | Why it cannot be generated |
|---|---|
| New attention, FFN, router or recurrence math | Metadata names dimensions; it does not define numerical semantics. |
| Tensor ownership and transforms | Shape-compatible tensors can have different roles or block locality. |
| Tokenizer and pre-tokenizer | Named or regex behavior must match the authoritative implementation byte for byte. |
| Chat, tools and thinking | A marker inventory cannot reproduce control flow, roles, whitespace or reasoning boundaries. |
| Drafter state topology and head selection | K=1..8 exactness cannot identify the right head by itself. |
| Canary seams and prompts | They must be calibrated against a real failure mechanism and execution class. |
| Performance promotion | Evidence from one card does not set the default of the device the default is keyed on. |

The kit removes repeated geometry threading and gate boilerplate. It does not automate
architecture research.

## Appendix: gate spec fields

The input spec for `tools/generate-arch-gates.py` is a JSON object with exactly the keys below;
unknown or missing keys are refused. `valid_spec()` in `tools/test_generate_arch_gates.py` is a
complete example the unit tests keep valid. Environment names must match `^[A-Z_][A-Z0-9_]*$`.
`artifact_env` and `draft_env` are variables the generated scripts read. `seam` and the
`canary_env` keys must be flags the engine reads (each has a [Flags](FLAGS.md) row), or the canary
cannot bite.

| Key | Contents |
|---|---|
| `id` | `[a-z0-9]+` |
| `artifact_env` | variable that overrides the artifact path |
| `chunk` | `label`, `prompts` (repo-relative), `chunks`, `steps`, `seam` |
| `tick` | `label`, `prompts`, `budgets` (0 allowed), `steps`, `seam`; optional `splits` |
| `batch` | `model_alias`, `canary_env`, `canary_expect_regex` (starts with `FAIL`), `required_gpus`, `pp_stages`, `pp_devices` (one per stage), `concurrency` (each at least 2), `port` (18300-18399), `receipt_dir`, `server_env`, `request` (non-empty `messages`), `liveness` (`cap_regex` with a `[0-9]+`, `cap_min`, `walk_regex`); optional `draft_path`, `draft_env` |
| `mapping` | non-empty list of rows with `path_regex`, `kernel_scope`, `base_probes`, `base_spec_probes`, `gate_families` (`chunk`, `tick`, `batch`) |

`server_env` may not set `MEMRA_ADDR`, `MEMRA_MODELS`, `MEMRA_PP_DEVICES` or `MEMRA_PP_STAGES`
(the generator owns them), and `canary_env` may not set those or any `server_env` key.
