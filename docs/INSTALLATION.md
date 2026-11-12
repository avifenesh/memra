# Installation

How to get memra's binaries onto a GPU host: the prebuilt installer for RTX 50-series and RTX PRO
6000 Blackwell cards, source builds for every build target, the first model download and the
common failures. Launch configurations per model and card are in the [cookbook](COOKBOOK.md).

## Pick a route

A binary carries GPU code for one build target and loads only on that target's exact compute
capability. Only `120a` ships prebuilt.

| GPU | Compute capability | Build target | Route | Compiled in CI | Card |
|---|---|---|---|---|---|
| RTX 50-series, RTX PRO 6000 Blackwell | 12.0 | `120a` | prebuilt or source | yes | [5090](rigs/rtx-5090.md), [PRO 6000](rigs/rtx-pro-6000-blackwell.md) |
| DGX Spark (GB10), aarch64 host | 12.1 | `121a` | source, on the GB10 | yes | [DGX Spark](rigs/dgx-spark.md) |
| B200 | 10.0 | `100a` | source | yes | [B200](rigs/b200.md) |
| H100 | 9.0 | `90a` | source | no | [H100](rigs/h100.md) |
| Ada (RTX 40-series, L4, L40S) | 8.9 | `89` | source | no | [Ada](rigs/ada.md) |

Any other GPU has no build target. A push to `main` that reaches `120a` or `121a` needs GPU
evidence from a device of that target; the other targets are best effort.

## Prebuilt install

Requirements: Linux x86_64 with glibc 2.35 or newer; a compute capability 12.0 GPU; NVIDIA driver
580 or newer (the release requirement, not checked at run time); and the CUDA 13 runtime libraries
`libcudart.so.13`, `libcublas.so.13` and `libcublasLt.so.13` on the loader path. No CUDA toolkit or
`nvcc`. The installer itself needs `curl`, `tar`, `sha256sum` and access to github.com.

```bash
curl -fsSL https://raw.githubusercontent.com/avifenesh/memra/main/tools/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

[`tools/install.sh`](../tools/install.sh) picks the build target (`MEMRA_CUDA_ARCH`, else the
compute capability `nvidia-smi` reports, else `120a`). It stops before any download on a host that
is not Linux x86_64, on a B200, or on a GPU whose compute capability has no build target. It then
resolves the release, downloads `SHA256SUMS` with its `SHA256SUMS.sigstore.json` bundle, and
verifies the signature with a pinned cosign v3.1.3 (itself checked by SHA-256) against this
repository's release workflow at that tag. It picks the highest glibc floor the host satisfies (2.35
or 2.39), checks the archive's SHA-256 against the signed list, and installs `memra-server`,
`run-gen`, `run-spec` and `kernel-check`.

| Variable | Default | Effect |
|---|---|---|
| `MEMRA_INSTALL_DIR` | `~/.local/bin` | Install directory |
| `MEMRA_VERSION` | the latest release | Release tag, for example `v0.141.1` |
| `MEMRA_CUDA_ARCH` | detected | Build target to fetch; only `120a` has archives |

Pass them to the shell that runs the script: `curl -fsSL <url> | MEMRA_VERSION=v0.141.1 sh`.

`cargo binstall memra-server --bin memra-server` installs the server alone from the release's
`sm_120a`, glibc 2.35 archive, without the cosign check. The `--bin` is required: the archive lacks
the crate's second binary (`kv-handoff-gate`), so a plain `cargo binstall memra-server` rejects it
and compiles from source instead (checked with cargo-binstall 1.27.0 against v0.141.1).

## Check the install

```bash
kernel-check
memra-server --version
```

`kernel-check` compares each kernel with a CPU reference and ends with
`ALL GREEN (<n> cells, <m> skipped)`, or exits non-zero with `<n> kernel(s) FAILED`. Cells that
read real model weights skip when their GGUF files are absent; `MEMRA_KC_MODELS_DIR` names a
directory to find them in, and `MEMRA_KC_FAST=1` runs only the synthetic cells, in seconds.
`memra-server --version` prints the version, `system_fingerprint`, `build_id_src` and `git_sha`,
and exits.

## Build from source

Requirements:

- Rust through rustup. The checkout's `rust-toolchain.toml` pins 1.97.1; a crates.io build needs
  Rust 1.97 or newer.
- A CUDA 13 toolkit with `nvcc`. Releases build with 13.1, GB10 builds have used the aarch64 13.0
  toolkit, and CUDA 12.4 cannot compile `compute_120a`.
- A C++ host compiler and binutils (`nm`, `ar`, `objcopy`): `build-essential` on Debian and Ubuntu.
- At run time, the same driver and CUDA 13 runtime libraries as the prebuilt.

The build takes `nvcc` from `MEMRA_NVCC`, else `CUDA_HOME`, `CUDA_PATH` or `CUDA_ROOT`
(`<root>/bin/nvcc`), else the newest release among `PATH`, `/usr/local/cuda` and
`/usr/local/cuda-*`, and prints its pick (`nvcc auto-detected CUDA 13.1 at ...`). The build target
follows the first GPU's compute capability; with no GPU, or a capability no target covers, it falls
back to `120a` and warns that the result is compile-only. `MEMRA_CUDA_ARCH` (`120a`, `121a`,
`100a`, `90a` or `89`) builds for a GPU the build host does not have.

```bash
git clone https://github.com/avifenesh/memra.git
cd memra
cargo build --release -p memra-server --bin memra-server \
  -p memra-engine --bin kernel-check --bin run-gen --bin run-spec
export PATH="$PWD/target/release:$PATH"
```

`cargo build --release -p memra-server` builds the server alone. A bare `cargo build --release`
links all of the workspace's roughly 250 binaries, each with fat LTO, which takes several times
longer ([measurements](decisions/RELEASE-PROFILE.md)). The onboarding CLI runs as
`cargo run --release -p memra-cli --bin memra -- model inspect ...`.

`cargo install memra-server --locked` builds the server from crates.io instead. The crate leaves out
the generated `MEMRA_*` registry, so such a build skips the boot audit of `MEMRA_*` names
(`[env-audit] registry absent in this build`) and does not refuse a misspelled one.

Per target:

- **`120a`**: the default; nothing to set.
- **`121a`** (DGX Spark): build on the GB10 itself, an aarch64 host the installer refuses, with
  rustup and the aarch64 CUDA 13 toolkit. The build detects compute capability 12.1. `121a` runs
  the `120a` kernels and defaults compiled to its own SASS, and neither binary loads on the other's
  GPU ([decision](decisions/SM121-SPARK-TARGET.md)). Memory defaults follow unified memory
  ([card](rigs/dgx-spark.md), [serving](SERVING.md#unified-memory-devices-gb10-dgx-spark)).
- **`100a`** (B200), **`90a`** (H100), **`89`** (Ada): detected on their GPUs. H100 has its own
  kernel set and ledger ([ARCHITECTURE-H100.md](../ARCHITECTURE-H100.md)); Ada is a portable
  compatibility build, not an optimized target. What was measured on each is on its card.

## Models

`MEMRA_MODELS="name=<model>[+<draft>],..."`, `run-gen` and `run-spec` take a model as a GGUF file
(the `-00001-of-0000N.gguf` shard of a split GGUF), a Hugging Face safetensors directory
(`config.json` plus `model.safetensors` or `model.safetensors.index.json`), a memra repack
directory (`manifest.json`), or an `hf:owner/repo[:<substring>]` spec.

An `hf:` spec downloads on first use through the `hf` CLI (Hugging Face's installer:
`curl -LsSf https://hf.co/cli/install.sh | bash -s -- --exclude-skill`, which needs Python 3.10 or
newer with `venv`) into
`$MEMRA_MODELS_DIR/hf/<owner>--<repo>/`, by default `~/.cache/memra/models/hf/<owner>--<repo>/`,
and resolves offline once the files are there.

- `:<substring>` downloads the files whose names contain it and takes the one matching GGUF; more
  than one match refuses and lists them. With no substring the spec downloads the whole repo and
  takes its single GGUF, or the directory when it holds a safetensors checkpoint. Files named
  `mmproj*` are never the model.
- Only top-level `.gguf` files count. A `.txt` file, a GGUF in a subfolder and a split GGUF need
  `hf download` and a local path.
- A spec cannot pin a revision. Download the pinned files and pass the path:
  `hf download <owner>/<repo> --revision <commit> --local-dir <dir>`.
- `hf:` works in `MEMRA_MODELS` (model and `+draft`), `MEMRA_DSPARK_DRAFT`, `MEMRA_FRSPEC_TRIM`,
  `MEMRA_GLM5_DFLASH`, and the model argument of `run-gen` and `run-spec`. Other path variables,
  such as `MEMRA_DRAFT`, `MEMRA_MTP_DRAFT` and `MEMRA_GEMMA_MMPROJ`, take local paths.

## First generation

```bash
MEMRA_CHAT=1 run-gen hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp \
  --prompt "Explain KV caching in one sentence."
```

The first run downloads the 15.7 GB GGUF, which fits a 24 GB card. Before generating, `run-gen`
checks that batched prefill and token-by-token decode pick the same next token:

```text
prefill argmax=<id>  decode argmax=<id>  logit maxdiff=<x>  MATCH
batched-prime argmax=<id>  tokenwise argmax=<id>  logit maxdiff=<x>  MATCH
```

The second line appears for prompts of 16 tokens or more. A `MISMATCH` is a bug report.

## Troubleshooting

| Message (start) | Cause | Fix |
|---|---|---|
| `install.sh: prebuilt binaries are x86_64-only` | aarch64 host, such as a GB10 | Build from source on that host |
| `install.sh: B200 sm_100a is source-only` | B200 detected | Build from source on the B200 |
| `install.sh: release <tag> publishes no sm_<arch> build` | H100, Ada, or an override other than `120a` | Build from source |
| `install.sh: this host's glibc <v> is older than every published floor` | glibc below 2.35 | Build from source on that host |
| `install.sh: checksum signature verification failed`, `sha256 verification FAILED` | The checksum list or the archive failed verification | Do not install; retry, then report it |
| `error while loading shared libraries: libcudart.so.13` | No CUDA 13 runtime on the loader path | Install the CUDA 13 runtime libraries, or add their directory to `LD_LIBRARY_PATH` |
| `Unable to dynamically load the "cuda" shared library` | No NVIDIA driver loaded | Install or load the driver |
| `memra was built for sm_<t> but device <n> reports compute capability <m>` | Binary built for another target | Build from source on that machine; the message names the target it needs, or says none covers the GPU |
| `nvcc fatal   : Unsupported gpu architecture 'compute_120a'` | The build picked an old toolkit | Point `MEMRA_NVCC` or `CUDA_HOME` at a CUDA 13 toolkit |
| `` `hf` CLI not runnable `` | No `hf` command on `PATH` | `curl -LsSf https://hf.co/cli/install.sh \| bash -s -- --exclude-skill` (Python 3.10 or newer with `venv`) |
| `[env-audit] REFUSED: the environment names doors this build does not have` | A retired or misspelled `MEMRA_*` name | Unset it; [FLAGS.md](FLAGS.md) lists the live names |
| `[server] FATAL: refusing unauthenticated non-loopback bind` | `MEMRA_ADDR` is not loopback and no key is set | Set `MEMRA_API_KEY` or `MEMRA_API_KEYS` ([serving](SERVING.md)) |

If a failure persists, open a
[bug report](https://github.com/avifenesh/memra/issues/new?template=bug-report.md) with the
installer or build output, `memra-server --version`, `nvidia-smi`, `ldd --version`, every `MEMRA_*`
variable you set and the first 40 lines of the server log.
