---
name: Hardware validation report
about: You ran memra on your GPU; share the results, whatever the card
---

<!--
Reports from any NVIDIA GPU help, and correctness output alone is useful.

- sm_120 cards (RTX 5090, 5080, 5070 Ti, 5070, RTX PRO 6000 Blackwell) run the prebuilt sm_120a
  build. They share the build, not the measurements: device-keyed defaults were measured on the
  cards that set them.
- GB10 (DGX Spark), H100, B200 and Ada build from source. MEMRA_CUDA_ARCH (121a, 90a, 100a, 89)
  is detected from the GPU when unset.
- A GPU with no build target (Ampere and older) cannot run memra: the installer refuses it, and
  a source build falls back to sm_120a, which does not load there.
-->

## Your hardware

- GPU: <exact model and VRAM, e.g. RTX 5090 desktop, 32 GB>
- Driver / CUDA version:
- OS:
- Install: <prebuilt release tag, or source commit plus MEMRA_CUDA_ARCH>
- Power state during runs: <e.g. stock or power-limited, and whether it stayed pinned>

## Correctness (required)

```
<kernel-check tail; expect "ALL GREEN (<n> cells, <m> skipped)">
```

```
<run-gen output on a model with a card in docs/models/, including the "verify-prefill ... MATCH" line>
```

If the model has an MTP drafter, also paste the `run-spec` line `=== SELF-CONSISTENCY PASS ===`.

## Performance cells (optional)

With the models present: `MEMRA_MODELS_DIR=<your model root> tools/local-ci.sh --perf` (needs
`jq`), then paste the per-cell verdict lines. The cells and their model file names are in
`research/tune-data/perf-cells.json`; cells whose models you do not have are skipped.

For a comparison with another engine, follow the interleaved protocol in
[research/benchmarks.md](https://github.com/avifenesh/memra/blob/main/research/benchmarks.md):
same session, both orders, N >= 3 runs per arm, one engine on the GPU at a time. Sequential
numbers drift by up to about 10% and cannot be used. State the other engine's version and flags.

## Anything that broke

Kernel launch failures, wrong output, out-of-memory at sizes that should fit: the exact output,
pasted.
