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
and generated performance-board freshness checks passed. Source hashes are in
`raw/source-hashes.json`.

The full engine/server suites and native CUDA compilation are delegated to the ordinary
Linux CI jobs. Their result must be read on the PR; the CPU controls above do not substitute
for those suites.

## Required native qualification

No designated non-serving rig or lock allocation is verified for this task. Do not merge,
tag or promote model support from this CPU record. On a designated non-serving rig,
serialize with its existing canonical lock (`/tmp/memra-gpu.lock` for a PRO pair,
`/tmp/memra-5090.lock` for the local 5090 development rig):

1. Build this exact revision natively, recording source SHA, binary SHA-256, CUDA/toolchain,
   device topology and artifact/config manifest hashes.
2. Run `cargo test -p memra-engine --lib model_memory::native_tests -- --ignored --test-threads=1 --nocapture`
   inside the lock. This includes the same-device owner test and the physical-pair
   state/trim/refill test. Verify a GLM peer absent
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
