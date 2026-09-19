# Day-5 native gate — implementation, not qualification

`crates/memra-engine/src/bin/kv_tier_gate.rs` is a new executable. Its std-only argument
module has independent CPU tests; those do not compile or execute the CUDA-dependent bin.
No existing runtime, root manifest, model format, kernel or numerical program is changed.

Lead-owned `crates/memra-engine/Cargo.toml` landing fragment:

```toml
[[bin]]
name = "kv-tier-gate"
path = "src/bin/kv_tier_gate.rs"
```

Until that fragment lands Cargo's automatic binary discovery exposes `kv_tier_gate`
(underscore). Build with `cargo build --release -p memra-engine --bin kv_tier_gate -j 16`.
No external dependency added: SHA256 is the engine's existing sha2 dependency.

## Exact contract

```text
kv-tier-gate --artifact <gguf> --case baseline|active|prefix \
  --context 8192|32768 --tiers host|host,nvme --same-program --out <new-directory>
```

Unknown/duplicate/missing arguments, unsupported contexts/routes and missing same-program
refuse. Output directory must not already exist. `active` and `prefix` return exit 2 with
`REFUSED.txt` **before opening an artifact or initializing CUDA**: the native materializer
and scheduler binding does not exist. No fake restore, synthetic engagement or fallback.
An 8k baseline command is **not** the B2 active-8k gate.

Baseline uses the existing `HybridModel::load_without_mtp` trunk loader (as run-gen does),
`pp::new_cache` and `decode_step_h` for EVERY prompt and generated token. `decode_step`
itself wraps this exact function and drops the returned hidden; this gate retains the hidden
only for capture. No prefill-vs-decode, graph-vs-eager, batched or speculative crossing.
Plan/cache states outside full native q8_0 K/q5_1 V plus recurrent state refuse. No alternate
KV encoding is chosen. Optional MTP is absent, matching the existing plain trunk baseline;
this does not qualify MTP or the full artifact's auxiliary programs.

A deterministic raw-token prompt repeats a tokenizer-encoded English fixture to
`context - 128` tokens. It then greedily feeds 128 tokens, INCLUDING any EOS, to reach
exactly the requested committed context. This is a fixed-length raw-token diagnostic,
not a chat, template, EOS-serving or model-quality gate. Output text retains the full token
stream (reasoning/content are not selectively stripped).

Receipts bind artifact bytes (streamed SHA256), executable SHA256, the engine's Debug plan
serialization, exact prompt token bytes and the named native numerical path. Initial prefix
and final state manifests include per-layer presence/geometry, valid K/V bytes (no allocation
slack), host/device counters, current conv/SSM, final hidden and full logits. Spare SSM storage
is scratch, not current continuation state. Unknown latent/rank/ring/graph/draft/parked-logit
planes refuse rather than being omitted. Every generated-decision full logit row is hashed;
the final full logits and all token IDs are also retained. `BASELINE.txt` appears only after
all promised tokens and captures complete; partial/failed runs never produce it.

The gate rejects externally set runtime `MEMRA_*` override names without logging their
values; build selectors `MEMRA_NVCC`, `MEMRA_CUDA_ARCH`, and lock selector `MEMRA_GPU_LOCK`
are allowed. This adds no selectable runtime door. FLAGS documentation fragment for lead:
**kv-tier-gate is a diagnostic which refuses runtime overrides; no new MEMRA variable, no
runtime default, and no decide-by door are introduced.** Existing flag rows cover the three
allowed build/lock selectors.

## Execution / evidence boundary

GPU commands must be launched through `python3 tools/tier-battery.py --rig rtx5090
--timeout 1200 --out <new-collector-dir> --execute <absolute-bin> ...` after checking
processes and compute-apps, per BOX-ACCESS.md. Record source commit and binary hash beside
the collector outputs. Do not wrap the collector in another lock. Artifact directory is
read-only. The development filesystem is overlay; NVMe ancestry is unproven, and baseline
requests for `host,nvme` engage NEITHER tier. No storage/performance claim follows.

The initial build-launch SSH attempts failed after scratch creation; see rented evidence.
Until a native build and collector execution are recorded, this source is **UNCOMPILED as
an engine executable, BASELINE UNRUN**, not a delivered active-8k result. CPU argument and
B→D seam tests are separate evidence. PRO-pair, active reload, prefix engagement, native
materializer/fences and long-context gates remain blocking.
