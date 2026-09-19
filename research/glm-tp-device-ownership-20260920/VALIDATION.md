# GLM TP device ownership — issue #544

Status: implementation under review; native GPU qualification pending. No support-state,
rewrite-certificate, loader acceptance, kernel, numerical-program or performance-default
change is claimed by this record.

Source base: `b3487a03b0ee3f833c1157e7b7d68f2cb35a3843`.
Branch: `codex/544-glm-tp-device-ownership`.
Local compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, Apple arm64 macOS.

## Change

`HybridModel::owned_engines` enumerates loaded primary, PP, Step TP/EP and GLM TP/EP
owners. Shared references collapse by engine identity; independent streams on the same
physical ordinal remain owners. Pool statistics and trims are deduplicated by physical
device, since CUDA's default async pool is per-device.

Admission charges the incoming request and every active cache's unmaterialized rank
state. The primary cache estimate retains the canonical/shadow planes. GLM adds the
sharded KDA convolution and recurrent planes and peer MLA latent/index/length/key
allocations. Materialized peers retain their transient reserve. Every distinct GLM peer also receives
a prompt-shaped prefill-workspace charge, using the conservative unsharded envelope
plus KDA core allocations; the envelope is not divided by TP degree. Missing parallel-device
headroom readings reject admission rather than falling through to an unchecked admit.

Admin trim, driver-headroom recovery and OOM reclaim use the same owner inventory.
Every owning stream is fenced before trimming physical pools. Trim reports per-device
released bytes, still-live default-pool bytes, retained cached bytes, driver free bytes,
and synchronization errors. Pinned source leases and live session state remain owned.

## CPU checks

- `tools/test-model-device-memory.sh`: production CUDA-free ownership arithmetic and
  reclaim ordering, including peer-only pressure/refill, shared-device independent streams,
  partial materialization, oversized replacement allocations, and overflow refusal.
  11 tests passed. Raw output and source/test-binary hashes: `raw/cpu-ownership.log`.
  These are CPU controls, not a CUDA reproduction. A negative control replacing KDA
  geometry `[conv, state, state]` with `[conv, state, 0]` failed
  `kda_reserves_convolution_and_both_recurrent_planes`; restoring the production source
  passed. Negative output: `raw/kda-omit-alt-red.log`.
- `cargo test -p memra-gguf -p memra-tokenizer -p memra-validate -p memra-sampling --lib -- --test-threads=1 --nocapture`:
  376 reported passes, 2 ignored, no failures. Twelve artifact-gated GGUF tests printed
  SKIP and count among libtest's reported passes; 364 tests actually exercised their gates.
  Raw output: `raw/cpu-modelplan.log`.
- `cargo test -p memra-reference --lib -- --test-threads=1 --nocapture`:
  65 passed, 1 failed: `qwen35_fixture_executes_mixed_gdn_and_full_attention_state`.
  The failure compares unequal f32 bit patterns on Apple arm64; the reference source and
  its dependencies are unchanged by this lane. Raw output: `raw/cpu-reference.log`.
- `DOCS_RS=1 cargo check -p memra-server --tests`: blocked on existing Linux-only
  `cpu_set_t`, `CPU_SET`, `sched_setaffinity`, `posix_fadvise`, `POSIX_FADV_RANDOM`,
  `O_DIRECT`, and missing `MEMRA_MMQ_ARCHIVE_HASH` in the documentation build path.
  This diagnostic does not build CUDA and is not a passing server build.
  Raw output: `raw/macos-docs-check.log`.

Formatting (`cargo fmt --all -- --check`), `git diff --check`, the flags census,
and generated performance-board freshness checks passed. Initial implementation hashes
for `cc348cce32f3d9b677f794032ac2775bb88339bc` are in `raw/source-hashes.json`;
runner/preparation changes are separately hashed in `raw/qualification-source-hashes.json`.

Linux CI passed for `cc348cce32f3d9b677f794032ac2775bb88339bc`
([run 35470038721](https://github.com/avifenesh/memra/actions/runs/35470038721)):
511 engine tests (zero hidden skips), 715 server tests (5 ignored), both CUDA builds,
Clippy, boundary/gates and package dry run. Follow-up qualification preparation adds a
native worker gate and a fail-closed per-card runner. Its local runner controls passed
7 tests with 1 explicit Linux-only skip on macOS (`raw/runner-controls.log`); a new Linux
CI run must compile the new ignored test and exercise the actual Linux flock control.
None of these results is native GPU qualification.

## Required native qualification

The coordinator is provisioning one non-serving host. Do not execute GPU work until its
host and lock wrapper are provided. The owner's current instruction supersedes the older
whole-rig lock rule for this work: each session holds exclusive locks on exactly the
physical GPUs it uses; multi-GPU sets are acquired in stable order by the shared wrapper.
Do not merge, tag or promote model support from CPU results or synthetic seam results.
The runnable protocol and resource envelope are in [QUALIFICATION.md](QUALIFICATION.md).

1. Build this exact revision natively, recording source SHA, binary SHA-256, CUDA/toolchain,
   device topology and artifact/config manifest hashes.
2. Use the runner's separate one-card `same-device` and two-card `pair` stages
   inside their exact per-card lock sets. Verify a GLM peer absent
   from Step-only enumeration is present, and lazy state charges disappear only when its
   allocations exist. Test different physical cards, not only same-card emulation.
3. Exercise GLM serving with asymmetric peer pressure: healthy primary, limiting peer;
   cold and warm caches; concurrent pending state; peer-only cache reclaim and repeated
   refill. Verify admission/defer and per-device bytes against actual allocator readings.
4. Exercise explicit admin trim and OOM teardown with a pinned prefix source lease and
   another live session. Fence all owners, reclaim only evictable state, preserve token
   bytes and live leases, and retain per-device reclaim/occupancy/error logs.
5. Run the affected exactness battery: `kernel-check`, `run-gen` argmax, and `run-spec`
   K=1..8 on their supported surfaces. Respect GLM TP's existing speculative refusal;
   this lane does not open a numerical-program transition or alter certificate semantics.

A CPU mock can establish orchestration order and arithmetic; it cannot establish CUDA
allocator behavior, cross-device synchronization, throughput or checkpoint exactness.
