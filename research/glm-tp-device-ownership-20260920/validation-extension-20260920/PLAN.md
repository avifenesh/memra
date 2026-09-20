# Indexed MLA / KDA future-state validation extension

Status: implemented and CPU-checked; the new native gate has not run. This closes the
specific fixture gap identified by the independent PR #555 coverage review without
claiming full-checkpoint or performance qualification. Historical `d413747d` receipts,
including `partial_key_bytes=0`, remain unchanged.

## Integrated source

`origin/main` at `a5744c72` was merged into the issue branch in `4ec0bdf6`. The only manual
conflicts were appended sections in `docs/TESTING.md` and `research/INDEX.md`; both sides
were retained. Upstream operation-registry and spill code is preserved. This lane does
not change identity/certificate semantics, model-pack math or loader acceptance.

## New fixture and gate

The CUDA-free fixture source is derived from the canonical GLM5-next plan, its GGUF-dialect
tensor contract and Memra's deterministic reference tensors. It declares F32 and supplies
F32 bytes. It has hidden size 128, four dense trunk layers alternating KDA / indexed MLA,
no MTP, two attention/KDA heads (one per TP2 rank after real sharding), KDA head width 128,
conv kernel 4, index head width 8 and pool size 4. The native cache capacity is 8,192;
the index state uses the default physical tail ring. No external model is downloaded.
The config and sorted tensor names/shapes/bytes produce a versioned SHA-256 identity,
printed by both the CPU fixture check and native test.

Runner stage: `indexed-kda`, exactly two physical GPUs.
Test: `model_memory::native_tests::glm_indexed_mla_kda_state_materialization`.

The test uses real `shard_kda_layer` / `shard_mla_layer` sidecars and the production
`ensure_kda_tp_state`, `ensure_mla_peer_latent` and `mla_kpool_indices` allocation paths:

1. Cold cache: primary and peer obligations are both nonzero; peer charge additionally
   includes the indexed MLA state. The no-cache and fresh-cache predictions agree.
2. Materialize the first KDA layer on both ranks and both MLA peer base planes. A second
   KDA layer and both lazy key planes remain missing. Both rank obligations must be
   nonzero, and the peer's excess must equal the two positive key allocations.
3. Run the first peer indexer through a complete pool. Its real lazy key buffer appears;
   one indexed layer and the second KDA layer still owe nonzero allocations.
4. Materialize the second KDA layer, then the second indexer's keys. Remaining charges
   reach exactly zero on both ranks. A further indexer append must not reintroduce debt.

At every stage, independently counted lengths/element sizes of real cache-owned
`CudaSlice` allocations plus the remaining prediction must equal the original cold
charge for each physical rank. Temporary indexer scratch is not mislabelled as cache
state. Nonzero index/rank preconditions prevent the old non-indexed fixture from passing.
This is allocation/ownership validation, not a token-output or indexer-quality oracle.

## Local validation

- `tools/test-model-memory-fixture.sh`: 10 tests passed, no skips. Includes real plan/census
  construction, TP2 geometry and indexer presence, repeatable identity, and a negative
  control removing a required indexer tensor. `fixture-cpu.log.gz`.
- `tools/test-model-device-memory.sh`: 11 passed, no skips. `arithmetic-cpu.log.gz`.
- `python3 tools/test_qualify_model_device_memory.py`: 8 passed; 1 explicit Linux-only
  live-flock skip on macOS. New stage cardinality is checked. `runner-cpu.log.gz`.
- Full ModelPlan/compiler (`cargo test -p memra-gguf --lib` through skip census):
  280 reported passes, 2 ignored, 12 declared artifact skips; zero failures or filtering.
  Thus 268 gates exercised their bodies. `modelplan-cpu.log.gz`.
- `cargo test -p memra-tier -p memra-kv --offline --no-fail-fast`: 243 passed across ten
  test binaries/doc-test groups, zero failures/ignores. `tier-kv-cpu.log.gz`.
- `cargo test -p memra-reference --lib`: 65 passed; the unchanged Apple-arm64
  `qwen35_fixture_executes_mixed_gdn_and_full_attention_state` bit-equality failure remains.
  `reference-cpu.log.gz`. It is not represented as a passing suite.
- Focused review found an untyped closure index; `il: usize` now fixes it. No other concrete
  fixture shape/lifetime/runner issue was reported. Full engine/server compilation and
  tests still require Linux CUDA build CI; CPU fixture checks do not compile the GPU test.

## Fresh native execution plan

The coordinator supplies the non-serving PRO host and its per-card wrapper. Do not reuse
retired SSH targets or rent independently. Use the final pushed source SHA as `COMMIT`,
a new output directory as `OUT`, and an explicitly accepted CUDA 13.1+ toolkit:

```sh
python3 tools/qualify-model-device-memory.py build \
  --expected-sha "$COMMIT" --arch 120a --jobs 8 --out "$OUT"

memra-gpu-run --gpus "$GPU_A,$GPU_B" --receipt "$OUT/lease-indexed-kda" -- \
  python3 tools/qualify-model-device-memory.py run \
  --expected-sha "$COMMIT" --out "$OUT" --stage indexed-kda \
  --gpu-uuid "$GPU_A" --gpu-uuid "$GPU_B"
```

Build outside leases in the runner's fresh per-receipt Cargo target. Run each original
same-device/pair/worker regression on the integrated executable with its exact one-/two-card
set as well; retain and sync every new stage receipt promptly. The coordinator owns the
remaining integrated release exactness/artifact checks. Full-checkpoint, actual CUDA OOM,
complete HTTP/token-stream and performance work is not implied by the new fixture.

Raw CPU logs are stored losslessly with gzip; no failure or skip was removed.

## Execution receipt

This plan subsequently passed all four exact native stages at `fcb1b986`, including
positive partial key bytes and warm/append zero. See [NATIVE-RESULTS.md](NATIVE-RESULTS.md).
The preparation-time CPU record above and historical `d413747d` data remain unchanged.
