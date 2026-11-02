# Installation

The release installer is the shortest path. It downloads the latest matching prebuilt archive,
checks its SHA-256 digest against the release manifest, and installs:

- `memra-server`
- `run-gen`
- `run-spec`
- `kernel-check`

## Prebuilt binaries

Requirements:

- Linux x86_64
- NVIDIA driver 580 or newer
- CUDA 13 runtime libraries (`cudart`, `cublas`, and `cublasLt`)
- a glibc version compatible with one of the archives in the selected release

The installer reads the available glibc floors from the release manifest and chooses the newest
one the host can run. Prebuilt binaries do not require the CUDA toolkit or `nvcc`.

```bash
curl -fsSL https://raw.githubusercontent.com/avifenesh/memra/main/tools/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

The default install directory is `~/.local/bin`. Useful overrides:

```bash
MEMRA_INSTALL_DIR=/opt/memra/bin \
MEMRA_VERSION=<release-tag> \
MEMRA_CUDA_ARCH=120a \
  sh tools/install.sh
```

`MEMRA_CUDA_ARCH` normally comes from `nvidia-smi`. Published release manifests contain the
release-qualified architectures. B200 `sm_100a` is currently source-only: the installer recognizes
compute 10.0 and refuses before network access rather than downloading or suggesting an unqualified
prebuilt.

Verify the installed CUDA path:

```bash
kernel-check
```

A successful run ends with `ALL GREEN`. Model-backed cells may report `SKIP` when their optional
artifacts are not present.

## Build from source

Source builds require:

- Rust 1.85 or newer
- CUDA toolkit 13.1
- a supported NVIDIA driver and GPU target

```bash
git clone https://github.com/avifenesh/memra.git
cd memra
cargo build --release
export PATH="$PWD/target/release:$PATH"
```

The source build detects B200 compute capability 10.0 and selects `sm_100a`. The explicit form is
useful for cross-building or for making the target visible in a build receipt:

```bash
MEMRA_CUDA_ARCH=100a cargo build --release --bins
```

The release installer still refuses B200 because no `sm_100a` prebuilt is published. Source support
is hardware-qualified but model-specific: use the NVFP4 and FP8 states in [the B200 card](rigs/b200.md)
instead of treating one successful build as universal model support. For other targets, set
`MEMRA_CUDA_ARCH` only when cross-building or selecting a different configured backend.

## First model

Memra accepts a supported local GGUF file, a supported safetensors checkpoint directory, or an
`hf:owner/repo[:file-substring]` specification. `hf:` specifications download into the Hugging
Face cache on first use.

```bash
MEMRA_CHAT=1 run-gen \
  hf:Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp \
  --prompt "Explain KV caching in one sentence."
```

For server configurations that have been qualified on a named card, use the
[cookbook](COOKBOOK.md). The authoritative support matrix is [models and hardware](MODELS.md).

## Common installation failures

- **No matching release archive:** select another published architecture or build from source.
- **Host glibc is below every published floor:** build from source on that host.
- **CUDA libraries cannot be loaded:** install the CUDA 13 runtime libraries; `nvcc` is not needed
  for a prebuilt.
- **Architecture mismatch or illegal instruction:** remove an incorrect `MEMRA_CUDA_ARCH` override
  and reinstall for the detected GPU.

Open a [bug report](https://github.com/avifenesh/memra/issues/new?template=bug-report.md) with the
installer output, `nvidia-smi`, `ldd --version`, and the selected release if the failure persists.
