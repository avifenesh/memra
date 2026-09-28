# MiMo-V2.6-Flash-RL NVFP4 mint tooling

`tools/mint_mimo_v26_nvfp4.py` exports the routed experts of the pinned
MiMo-V2.6-Flash-RL checkpoint into a ModelOpt mixed-precision checkpoint. It uses
NVIDIA ModelOpt's MXFP4 exponent-window selection, shares gate/up global scales,
and refuses any changed decoded nonzero expert weight. The source FP8/BF16
non-expert tensors and auxiliary weights retain their precision.

The tool is an artifact producer. Runtime and activation-precision qualification
are separate steps; producing a seed does not establish model quality or add a
Memra runtime-support state.

## Pinned inputs

- Model: `XiaomiMiMo/MiMo-V2.6-Flash-RL`
- Revision: `3b38d063180c3e4aed9691fdc735f3d10b266ee4`
- ModelOpt: `051d6adb204f10cd3e78d0f824f31a5a01d54831`

The source revision is verified through the Hub's file metadata: SHA256 for LFS
objects and Git blob identity for the remaining files. The complete model has
47 routed-expert layers, 256 experts per layer, and three projections per expert.
The final census requires all 36,096 projections.

## Produce a seed

Install ModelOpt from the exact source revision in an environment with a compatible
PyTorch build. The tool checks the installed distribution's VCS provenance.

```bash
python -m pip install \
  'nvidia-modelopt[torch] @ git+https://github.com/NVIDIA/Model-Optimizer.git@051d6adb204f10cd3e78d0f824f31a5a01d54831'
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4 \
  --local-dir /data/mimo-source
python tools/mint_mimo_v26_nvfp4.py \
  --source /data/mimo-source --output /data/mimo-nvfp4-seed \
  --input-scale 1.0
```

`--input-scale` is explicit because it changes activation execution. A constant
1.0 is a development seed, not a quality recommendation. A calibrated policy can
be supplied with `--activation-scales`: its JSON object must contain exactly
`model.layers.1.mlp.experts` through `model.layers.47.mlp.experts`, each with
positive `input_scale` and `down_input_scale` values. Calibrate against the actual
serving arithmetic and keep calibration data separate from held-out evaluation.

Use a new output directory for a different policy. An interrupted run resumes
only when its policy and completed output hashes still match. `--shard` produces
one checked shard for a component gate, and deliberately does not finish a model
index.

## Output and checks

Each shard is reopened after serialization. Checks cover the complete tensor map,
unchanged packed expert payloads, exact decoded scale products for nonzero weights,
valid FP32 global/input scales, and byte preservation for the other source tensors.

Outputs include:

- Standard `config.json`, `hf_quant_config.json`, and safetensors index.
- `mint-policy.json`, identifying source, tool, library, and activation policy.
- `mint-receipts/`, containing source/output hashes and projection-level proofs.
- `mint-report.json`, with the final tensor census and reconstruction verdict.
- `DRAFT-NOT-QUALIFIED.json`, keeping activation/model qualification explicit.

The source DFlash configuration at this revision has an invalid trailing comma.
The tool repairs that syntax, records the change, and preserves its values and
weights. The source model card is retained as `README_UPSTREAM.md`; a release card
must be written from the actual qualification results.

```bash
python -m pytest -q tools/test_mint_mimo_v26_nvfp4.py
```

The CPU tests cover the E2M1 grid and signs, exponent boundaries, shared gate/up
scales, zero blocks, rejected loss, tensor census, and serialized corruption. They
do not establish GPU kernel correctness, FP4 activation quality, or serving speed.
Those require the exact artifact on the named hardware and runtime.
