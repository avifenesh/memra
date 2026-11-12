# Hy3 expert spill on a 24 GB card

How to run Hy3 on one 24 GB card when its expert bank exceeds both VRAM and ordinary host RAM. For
advanced users who build from source. The model, its packs and its other routes are on the
[Hy3 model card](models/hy3.md).

This is a research profile. No support record covers the spill route, no standing gate runs it, and
its throughput is not a published claim.

## How it works

memra freezes a profiled HBM-resident expert set, keeps a bounded LRU projection cache in normal
RAM, reads misses with positioned direct I/O, and can split each large read across byte-identical
copies on two NVMe devices. Experts routed to the CPU run in an optional companion library that is
memra code: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS, NVFP4, Q4_0, BF16 and F32 row dots
with a memra Q8/16 activation format. It does not compile, link or load llama.cpp, ggml or another
inference runtime; the build script refuses a library that depends on them.

## Requirements

- One 24 GB NVIDIA card (the script's defaults were measured on an RTX 5090).
- Host RAM for a 20 GiB expert cache plus 4 GiB of headroom; the cache shrinks to fit
  `MemAvailable`.
- An x86-64 CPU and a C++17 compiler with OpenMP. The companion builds with `-march=native` and
  uses its AVX2 and AVX-VNNI kernels when the host has them.
- The Hy3 Layer103.5 expert overlay ([below](#the-published-overlay)). A second NVMe device is
  optional.

## Build and check

```bash
cargo build --release
tools/build_cpu_expert_companion.sh
MEMRA_CPU_EXPERT_LIB=target/release/libmemra-cpu-experts.so \
  cargo run -p memra-engine --bin cpu_native_check
```

`cpu_native_check` compares every supported packed row dot against memra's independent Rust
dequantization oracle. The companion ABI is versioned and fails closed: the engine requires native
ABI v2, so a stale v1 library is refused. `dlopen` runs library constructors before the ABI check,
so `MEMRA_CPU_EXPERT_LIB` must always point at a trusted build.

## Run

```bash
tools/run_hy3_local_5090.sh \
  /path/to/hy3-layer103p5-dual-nvme \
  target/release/libmemra-cpu-experts.so \
  /path/to/expert-mirror/inode-alternates.tsv
```

The script runs `target/release/run-spec` (it exits 3 without a release build) and takes the model
directory, the companion library and an optional mirror map. Its defaults, each overridable by the
same variable:

| Variable | Script default |
| --- | --- |
| `MEMRA_CPU_EXPERT_CACHE_GB` / `MEMRA_CPU_EXPERT_RESERVE_GB` | 20 / 4 |
| `MEMRA_CPU_EXPERT_THREADS` / `MEMRA_CPU_EXPERT_IO_THREADS` | 8 / 8 |
| `MEMRA_CPU_EXPERT_IO`, `MEMRA_SPILL_IO` | `direct` |
| `MEMRA_CPU_EXPERT_FREEZE_CACHE`, `MEMRA_CPU_EXPERT_FREEZE_PROFILE_ADMIT` | `1` |
| `MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS` / `MEMRA_CPU_EXPERT_FREEZE_WARMUP_SPEC_K` | 128 discarded profiling tokens / 3 |
| `MEMRA_CPU_AFFINITY`, `MEMRA_CPU_PROC_AFFINITY` | cores `0-7` |
| `MEMRA_MOE_CACHE`, `MEMRA_MOE_SIZE_AWARE`, `MEMRA_MOE_LFU` | `1` |
| `MEMRA_MOE_LFU_DECAY` | 1.0 |
| `MEMRA_MOE_VRAM_FRAC` / `MEMRA_MOE_HARD_VRAM_FRAC` | 0.90 / 0.90 |
| `MEMRA_SPILL_PREAD_DEPTH` / `MEMRA_SPILL_WORKER_EXPERT_WINDOW` | 32 / 8 |
| `MEMRA_SPEC_HOST_EMBD`, `MEMRA_CHAT` | `1` |
| `MEMRA_SPEC_K` | 1 (`all` leaves it unset, so `run-spec` sweeps K) |

Contracts for these and the other `MEMRA_CPU_EXPERT_*` variables are in [FLAGS.md](FLAGS.md).
Overlapped read and compute (`MEMRA_CPU_EXPERT_PIPELINE`) stays off: serial read-then-compute
measured faster, because overlapped direct-I/O DMA slowed the compute loops.

## Dual-NVMe mirror view

`tools/build_dual_nvme_expert_view.py` and `tools/build_expert_mirror_map.py` create the verified
striped view and the alternate-path map. When the map is set, `MODEL_DIR` must be that exact
dual-NVMe view: ABI v2 pins both sides by device, inode, size and ctime and rejects a map paired
with the original source tree.

## The published overlay

The Hy3 Layer103.5 expert overlay, its receipts and the relocation tool are documented in
[`research/per-expert-quant/hy3-layer103p5-release.md`](../research/per-expert-quant/hy3-layer103p5-release.md).

## Results and history

Throughput measurements, the closed lanes and the Hopper spec sweeps are records, not current
claims. Raw runs: `research/per-expert-quant/evidence/`, the plan and study
[`research/per-expert-quant/local-5090-10toks-plan.md`](../research/per-expert-quant/local-5090-10toks-plan.md),
[`research/hy3-hopper-20260801/`](../research/hy3-hopper-20260801/),
[`research/hy3-spec-20260802/`](../research/hy3-spec-20260802/) and
[`research/hy3-accept-profile-20260802/`](../research/hy3-accept-profile-20260802/). The text this
page carried before 2026-11-12 is in [history/HY3-SPILL.md](history/HY3-SPILL.md). The expert-spill
code contracts are in `AGENTS.md` ("Hy3 expert spill") and the five-arm quantization study in
[`research/per-expert-quant/README.md`](../research/per-expert-quant/README.md).
