# Exact auxiliary accounting for floating inventory tensors

The scoped catalog at `1770f1bca9e48256f8ddd2290f6eec50f4b38376` accepted an undeclared
`vision_model.conv1.weight_scale` with dtype I64 and shape [13]. The source census folded
those 104 bytes into the floating parent, but canonical binding checked auxiliary names only
for quantized storage and the perception inventory did not declare an exact empty list.

Every census entry now retains its complete normalized auxiliary list, independently of its
primary storage encoding. Canonical binding checks that list against the declared schema and,
for quantized entries, against the codec list. Bound compilation additionally verifies the list
against retained physical auxiliary records. The Step perception inventory requires no sidecars.
This closes the gap even when callers invoke the canonical contract directly, without a runtime
wrapper. All fixture and source producers initialize the same field.

## Validation

- The new regression failed on the old implementation (`red` log) and passes after repair.
  Cases include integer and floating weight scales, input scales, and pre-quant scales.
- The independent reviewer's unchanged `unexpected-vision-aux-probe.rs` passes both modes:
  valid inventory accepted; the injected I64 sidecar rejected with an auxiliary mismatch.
  Its source hash is checked against the review manifest and retained in `evidence.json`.
- Fresh full-header control for
  `stepfun-ai/Step-3.7-Flash-FP8@b3d7916fccac844cca050d7520f2aaa513f9a84f`:
  1,597 physical tensors, 1,471 folded roles, 804 text roles, 667 vision inventory roles,
  zero unclaimed entries. The original headers remain in the neighboring `header-census` and
  `vision-inventory` receipt namespaces, including their original padding.
- `cargo test --locked -p memra-gguf -p memra-cli`: 339 GGUF tests passed (2 ignored),
  11 CLI tests, 1 Step integration test, and 7 inspector tests passed. Artifact-dependent early
  skips remain possible; this is not a native checkpoint qualification claim.
- `cargo clippy --locked -p memra-gguf -p memra-cli --all-targets -- -D warnings`: passed.
- `DOCS_RS=1 cargo check --locked --target x86_64-unknown-linux-gnu
  --target-dir target/541-linux-typecheck -p memra-engine --lib --bins --tests
  -p memra-server`: passed with documentation stubs, not native CUDA.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

Raw logs are losslessly gzipped; compressed and uncompressed hashes are recorded. The broader
reference numeric caveat remains as recorded in the existing validation receipts. No GPU work,
root-loader activation, identity composition, merge, or whole-issue completion is claimed.
Existing native Step vision remains available through its preserved implementation and historical
receipts; this repair concerns canonical catalog completeness.
