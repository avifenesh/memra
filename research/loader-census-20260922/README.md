# Loader tensor-contract boundary (memra#541)

Verdict: both engine loaders now bind the pack's canonical tensor contract against the
checkpoint census before any CUDA upload and refuse missing, unexpected, duplicate, ambiguous,
wrong-shape and wrong-quant tensors and an undeclared tied head, with the pack and dialect named.
The served Qwen3.5-9B NVFP4 MTP GGUF binds 668 semantic tensors and loads with zero unconsumed
bound tensors, so its pack (`qwen35`) carries `tensor_consumption: Refuse`. The one audit finding
on that artifact is real and by design: its 113 modelopt `.input_scale` planes are declared by the
contract and read by no memra program.

## The gap

`memra model inspect` bound `TensorContract` and refused a malformed census; the loaders did not.
`Model::load_dense_from_source` and `HybridModel::load_from_source_impl` compiled a plan and read
string-named tensors one by one through the ggml name map (`hf_mapping`), so a map row the
contract knew and the map lacked resolved to "absent", and several load sites treat absent as a
legal shape. Absent `output.weight` fell back to `token_embd.weight` for every family. The
strict inspector's verdict was therefore never a loading guarantee (docs/ONBOARDING.md, "The name
map is a SECOND surface").

## The boundary (`crates/memra-gguf/src/checkpoint_binding.rs`)

- `bind_source(src, cfg, plan)`: `src.tensor_census()` (a source without a census is refused),
  `output_head_for` (below), the pack's `compile_tensor_contract` (or the canonical plan contract
  when no pack owns the arch), `TensorContract::bind`. Every error is
  `checkpoint refused before upload (pack <family>, <dialect>[, N census tensors]): <error>`.
- `output_head_for`: a present head tensor is the head. An absent one is the token embedding only
  when the pack declares `OutputHeadContract::TiedHeadAllowed` (qwen3, qwen35, gemma4 dense and
  MoE, llama_dense) and, for safetensors, config.json does not say `tie_word_embeddings: false`.
  `SeparateHead` packs (qwen3_moe, qwen35_moe, glm5_next, glm_dsa, deepseek_v4 both profiles,
  step35, hy3 both profiles, qwen4_exp) refuse a headless checkpoint. `ModelConfig` gained
  `tie_word_embeddings: Option<bool>` (parsed from config.json, `None` for GGUF).
- `CheckpointBinding::ggml_name(TensorId)`: the engine's request spelling for a semantic id,
  taken from the GGUF dialect of the same contract (no second hand-written table). The dense
  loader now addresses the embedding, norms, head and every attention projection by `TensorId`.
- `RecordingSource`: records every `find*` read (probes via `has` are not reads). After the
  model is built, `audit_consumption` names the bound tensors never read; `settle_consumption`
  applies the pack's `TensorConsumption` (`Report` prints one `[tensor-contract]` line, `Refuse`
  fails the load). Skipped by the loaders: vision-owned tensors (their own tower reads them), MTP
  tensors when the caller asked for no MTP, and `QuantAux InputScale` planes (`unread_by_design`).
- The same boundary now feeds the auto-parallel planner (`parallel.rs::artifact_costs`), the
  expert-bank catalog (`banked_residency/native.rs::plan_catalog`) and `model inspect`, so head
  ownership is decided once.

## Receipts (local RTX 5090)

| receipt | what |
|---|---|
| `raw/inspect-9b-artifact.lock` | `memra model inspect` of the 9B against `qwen35` on this tree: census gate passed (668 rows) |
| `raw/serve-9b-boot.log` | first boot through the boundary: `bound 668 semantic tensors (pack qwen35, Gguf, output head Separate)`; the audit REPORTED 113 unconsumed `QuantAux InputScale` planes and nothing else |
| `raw/serve-9b-boot-report-skip.log` | after `unread_by_design` excludes the `.input_scale` planes: bound line only, zero unconsumed |
| `raw/serve-9b-boot-refuse.log`, `raw/serve-9b-completion-refuse.json` | `qwen35` flipped to `Refuse`: the 9B loads and answers a completion |
| `checkpoint_binding::tests` (7, CPU) | the glm-dsa micro fixture clean; a byte-renamed trunk tensor refused (`missing tensor ... blk.0.attn_norm.weight`); the headless copy under a `SeparateHead` pack refused (`the embedding is not substituted`); the head-ownership matrix; recording source and audit; census-less source refused |
| `tests/checkpoint_contract_refusal_gpu.rs` (device) | the renamed and headless tampered copies refuse before upload through `HybridModel::load`, the clean fixture loads; PASS on the 5090 |
| `raw/local-ci/` | full `tools/local-ci.sh` on the final tree |

The `.input_scale` finding: modelopt writes a static activation scale per quantized projection.
memra's W4A16 path quantizes activations dynamically and its calibrated A4 program carries its own
`.a4_input_scale` values (`model_packs/qwen35/activation.rs`, "deliberately not `.input_scale`").
No engine code path reads the plane; it is in the census, bound by the contract (so a malformed
one still refuses), and skipped by the consumption audit with that reason in code.

## Scope and what stays open

- `Refuse` is set for `qwen35` only, on the receipt above. Every other pack reports. Flipping a
  family needs one real-artifact boot showing zero unconsumed tensors; the report line is the
  evidence to read. Expert banks read through the disk-spill tier's `gguf()` mmap path are not
  seen by the recording source, so MoE families need that path recorded before they can refuse.
- GGUF carries no tie flag, so under `TiedHeadAllowed` an absent `output.weight` in a GGUF is
  still the tied-head convention; only safetensors can contradict it through config.json.
- The hybrid loader still requests its tensors by ggml name (the recording source and the audit
  are the enforcement); only the dense loader addresses tensors by `TensorId`. Moving the hybrid
  loader's reads onto ids is the follow-up, one family at a time with its qualification gates.
