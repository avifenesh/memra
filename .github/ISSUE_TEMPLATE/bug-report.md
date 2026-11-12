---
name: Bug report
about: Wrong output, crash, gate failure, or performance regression
---

<!--
For a security vulnerability, do not use this template: see SECURITY.md.
GitHub CI has no GPU, so a GPU bug can only be reproduced from the details below. Paste output;
do not paraphrase it.
-->

## What happened

<!-- Expected vs observed. "argmax MISMATCH at token 12" or "run-gen panics with <exact message>"
     is actionable; "wrong output" is not. -->

## Repro

```
<exact command, with every MEMRA_* variable you set>
```

- Model: <HF id and file, e.g. hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp, or a local path and where it came from>
- Version: <output of `memra-server --version`>
- Install: <prebuilt release tag, or source commit plus MEMRA_CUDA_ARCH>

## Output

```
<the first 40 lines of the boot log, then the lines around the failure>
```

## Gate output (if applicable)

```
<kernel-check / run-gen / run-spec tail>
```

## Hardware

- GPU and VRAM:
- Driver and CUDA version (from `nvidia-smi`):
- OS and glibc:

## Checked

- [ ] I searched existing issues and, for a performance report, the [research index](https://github.com/avifenesh/memra/blob/main/research/INDEX.md)
- [ ] This reproduces on the latest release or a clean `main` checkout
