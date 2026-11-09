# DeepSeek-V4.1-Flash onboarding, step 1: config, pack and census (2026-11-09)

Verdict: the new `deepseek_v41` pack parses the 1017 config, compiles a typed plan for it and
binds every one of the 96,085 header tensors of the pinned checkpoint. Config, TokenizerTemplate
and TensorCensus pass at inspect. The pack's support stays none: no executor runs CSA2,
single-pass mHC or Engram, so TinyParity, CheckpointParity, RewriteParity and Serve are pending.
No GPU was used.

## Artifact

`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`: 48 safetensors
shards, 510,286,023,000 tensor bytes, 96,085 tensors. Headers were read by HTTP range requests;
no weight was downloaded. `inspect/artifact.lock`:

- `config_sha256=8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` (the
  `config.json` bytes; the same file is the pack fixture `fixtures/config.json`)
- `census_sha256=42e86e668ad65b03c252064d34ce424b7a575eb2927222bbb2aa1574028057c1`
  (`inspect/tensor-census.tsv.gz` unpacks to the hashed text)
- `plan_sha256=c4d9ad3bfdeac3d8152963b4c8449080b4e8a6024e2fdf7c1f67799764a31b8c`
- `binding=passed`, `tokenizer=passed`, `dialect=encoding_dsv41`

Code: commit `8e86db30fd` on `dsv41/onboarding-20261109` (base `06b6a8136a`).

## Commands

```bash
memra model inspect deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --against deepseek_v41 --out inspect
# family=deepseek_v41 tensors=48496 artifacts=inspect
memra model verify config deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --against deepseek_v41 --out verify-config
# family=deepseek_v41 verified=Config
```

The census has 48,496 rows because each quantized row carries its `.scale` plane (47,589 of
them). Rows by storage class:

| rows | bytes | storage |
|---:|---:|---|
| 47,232 | 295,997,276,160 | MXFP4 (I8-packed e2m1, e8m0 scale per 32): routed experts, trunk and DSpark |
| 2 | 202,758,032,400 | FP8 e4m3, e8m0 scale per 32 of a row: the two Engram tables |
| 355 | 7,408,601,600 | FP8 e4m3, e8m0 scale per 32x32 block: dense projections |
| 520 | 3,952,883,712 | BF16: embedding, head, norms, router, compressor, index-key projection, tower |
| 387 | 169,229,128 | F32: hyper-connection parameters, sinks, router biases |

## What the plan says

`inspect/model-plan.txt` is the compiled plan. Per layer:

| layers | CSA2 mode | compressed KV source | top-k source | candidates |
|---|---|---|---|---|
| 0, 1 | Window (ratio 0, `rope_theta` 10000, no YaRN) | - | - | - |
| 2, 8, 14 | Full, ratio 2, softmax-gated compressor | own | own | none |
| 20 | Full, ratio 1, projection compressor | own | own | produces 2048 blocks of 8 |
| 24, 28, 32, 36 | Reindex, ratio 1 | 20 | own, restricted to 20's pool | restricted |
| other 3..39 | Reuse | group's Full layer | most recent index layer | - |
| DSpark 40..42 | Window | - | - | - |

Every layer is a 384-expert top-6 (DSpark: 128 top-3) sqrtsoftplus MoE with one shared expert
and pre-clamped SwiGLU (limit 10), inside single-pass mHC (4 streams, Sinkhorn 20). Engram sits
on layers 1 and 14. The plan carries the checkpoint numerics: 32x32 FP8 dense blocks, per-32 FP8
activation quantization on every quantized linear, FP4 index query/key (32, ue8m0), FP4
compressed KV (16, e4m3), FP8 window KV and attention output (32, ue8m0).

New plan operations, all registry rows with no implemented surface:
`Fp8Block32Linear`, `Csa2Attention`, `Csa2Compressor`, `Csa2Indexer`, `Csa2CandidatePool`,
`Csa2KvReuse`, `Csa2IndexReuse`, `Csa2State`, `SinglePassHyperConnections`, `EngramLookup`.
`inspect/execution-rewrites.tsv` lists them as blockers on every surface.

## DeepSeek-V4 is unchanged

The 0731 artifact (`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`)
inspected `--against deepseek_v4_dspark` with the base binary and with this commit writes
byte-identical `artifact.lock`, `model-plan.txt`, `tensor-census.tsv`, `execution-rewrites.tsv`,
`gates.txt` and `tokenizer-contract.tsv`. Only the refusal text moved: the base refuses the
output head (`head.weight` was not a recognized spelling), this commit names the next refusal,
the upstream MXFP4 trunk experts against the pack's NVFP4 house-mint contract.

## Files

- `census_classes.py`: compresses the header census into the pack fixture
  `crates/memra-gguf/src/model_packs/deepseek_v41/fixtures/census-classes.tsv` and refuses to
  write unless the fixture expands back to every name, dtype and shape.
- `inspect/`: the inspect outputs listed above, `normalized-config.txt` included.
