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

`MEMRA_CUDA_ARCH` normally comes from `nvidia-smi`. Published release manifests may contain builds
for `sm_120a`, `sm_100a`, `sm_90a`, and `sm_89`; the installer fails with the available list when
the selected release has no matching archive.

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

The build detects the local compute capability. Set `MEMRA_CUDA_ARCH` only when cross-building or
selecting a different configured backend. See [docs/FLAGS.md](FLAGS.md) for build-time variables
and [docs/TESTING.md](TESTING.md) before changing kernels or runtime paths.

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
