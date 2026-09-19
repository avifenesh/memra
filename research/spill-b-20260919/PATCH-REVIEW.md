# HostPrefixCache first-slice patch review — day 3

Repository `avifenesh/memra`, branch `lane/spill-b-20260919`. Patch base is the
unchanged server source at merge `fb9375f24c066c006d8a2cdcb1f48fb8ecb2b7a2`
(includes integration `98e558dc`). CPU helper implementation: `fa763c97`.

**Status: reviewed source proposal, NOT APPLIED, NOT COMPILED as memra-server.**
`HOSTPREFIX-PATCH.diff` is an email-style unified patch, accepted by
`git apply --check`. The worktree's server source remains byte-identical to the
integration tip. Rustfmt parsed scratch copies with `--edition 2024
--config skip_children=true`; those scratch copies were deleted. No nvcc workaround,
CUDA stub build, live config, external runtime or serving instance was used.
Review here is this lane's source inspection, not independent lead/GPU acceptance.

## Hunk-by-hunk review (original source line numbers)

| Source hunk | Walk and invariants |
|---|---|
| `worker.rs:6121`, `:7391` | Put optional residency guards **last** in PrefixEntry/HostPrefixEntry. Rust drops payload fields before guards credit quota. Host metadata is owned, with a separate pageable charge; the identity lease alone is never called a payload lease. No asynchronous operation is introduced. |
| `worker.rs:7399` | `HostPrefixCache.tier: Option<HostTierContext>` defaults to None. Context is explicitly owner-injected, with the SAME governor and complete per-PoolKey program identities/model-generation Arcs; no private governor is constructed. There is deliberately no environment read or auto-enable bootstrap. |
| `worker.rs:7671` | Tier-enabled insert requires a residency charge and a currently valid immutable identity lease before existing layout/version, budget, tenant-delta, twin replacement and LRU operations. Unbound handoff imports refuse rather than being silently trusted as generic records. None leaves the existing sequence untouched. |
| `worker.rs:7789` (new helpers) | `tier_charge` converts existing byte accounting to the injected governor request, never treats the advisory memory verdict as a permit. Demotion is Backup, promotion AdmittedRestore. Fixed arena integration **refuses** until A's already-charged backing lease can be attached; slices must not double-charge the arena. `bind_tier_image` checks the narrow plain native surface, hashes actual completed host K/V, recurrent, logits and hidden bytes, and retains explicit presence/order/length transaction metadata. The existing token key is SHA-bound through KvBlockId with immutable epoch 0. Layout is exact bytes, no requantization. K remains q8_0, V q5_1. |
| `worker.rs:8002` | Existing successful native D2H constructor initializes new guard/metadata fields empty. Physical copy and all existing failure handling occur before identity publication. |
| `worker.rs:8191` | Only with injected context, reject TP/latent/draft surfaces not covered by the first slice; split host allocation charge into actual pinned K/V and remaining pageable bytes with checked arithmetic. Reserve BEFORE D2H. Off path does not execute this arithmetic or quota operation. Failure returns Failed, keeping a still-live source per the existing caller contract. |
| `worker.rs:8204` | Attach the completed host image's guard, then bind actual immutable bytes, then call existing insertion. Failure drops the image before releasing its charges. Source prefix charge (if any) is not released by this function: its owner still controls source lifetime. |
| `worker.rs:8494` | Existing device reconstruction initializes a guard empty; the promotion caller transfers the reservation only after success. |
| `worker.rs:8576` | Preserve native generation refusal. Acquire identity lease from the exact selected entry and reserve destination residency before synchronous native allocation/copy. `device_bytes - last_h*4` is GPU bytes; copied token/logit/hidden vectors are pageable. Host twin remains charged. The ordinary borrow retains actual host payload during synchronous H2D, separately from metadata. |
| `worker.rs:8612` | Recheck identity generation before publication; attach target guard to PrefixEntry before insert. Existing digest verification, recency-before-insertion rule, insertion pin, index repair and host twin retention are unchanged. No host vector index is retained after insert can trigger demotions. |
| `worker.rs:9632` | Legacy handoff import initializes unbound metadata/charges. Its persistent format is unchanged; tier-enabled insertion refuses it. |
| `worker.rs:10127`, `:11058` | Other ordinary PrefixEntry producers initialize None. They do not fabricate program identity or assert that the entire runtime already participates in global quotas. |
| `worker.rs:30793`, `:33250`, `:33289` | CPU test constructors initialize added fields. Existing prefix/LRU/tenant/hand-off teeth retain their old inputs. |
| `admit_memory.rs:721` | Narrow reservation helper delegates to CPU-tested ResidentCharge. Existing estimate, Tiers, decide, demote_budget_bytes, timeout and retry arithmetic are unchanged. |
| `worker/host_glm.rs:533` | Required one-line test-constructor initialization, found during cross-file inspection. Without this B-owned third-file hunk, the server tests would not compile after adding PrefixEntry's field. No GLM operation or gate is changed/enabled. |

## Review findings corrected before publishing this diff

- A wildcard contract import would shadow the closure's two-parameter `Result` with
  the frozen one-parameter alias; closure now names `std::result::Result` explicitly.
- First draft charged the whole mixed image as pinned; final diff splits pinned and
  pageable, and subtracts pageable hidden bytes from destination device accounting.
- Shape metadata originally would have been dropped after hashing; final diff retains
  it and its own pageable guard with the image.
- Cross-file constructor census found `worker/host_glm.rs:535`; final diff includes it.
- Incomplete/unsupported first-slice surfaces refuse when injected; they do not fall
  back to a different KV/compute program. The original None path remains unchanged.

## Explicitly unresolved before enabling

1. **No runtime bootstrap installs HostTierContext.** Applying this patch alone exercises
   legacy/off behavior. The lead must supply canonical artifact/serialized-plan/numeric/
   stream/tokenizer/template/tenant identities and lifecycle generation updates, and
   independently review that each injected PoolKey maps to that exact program. No new
   flag is proposed to bypass this missing integration.
2. Full-native-layout agreement with the admitted ModelPlan, quota dimensions including
   allocator capacity/metadata overhead, and bind-time scratch must be audited on rig.
   The first slice hashes a legacy image; it is not a qualified generic persistent wire
   migration. GLM/TP/draft and A fixed-arena handoff remain explicit refusals.
3. The guard is valid only for existing **synchronous** native copies. It is not an async
   DMA retirement adapter. Engine synchronization, no-tier byte identity, complete Qwen
   continuation, and last-use behavior still require real CUDA/server tests.
4. Model-scale active KV materialization/scheduler binding, corruption during active
   generation, graph/spec crossings, and generic host/NVMe engagement do not exist in
   this patch. No legacy prefix result may satisfy those gates.
5. This is NOT merge-ready runtime code. The first rig may find compiler/borrow/API
   errors that rustfmt and CPU helper tests cannot detect.

## Exact rig commands, before AND after applying

Use an isolated, approved **non-serving RTX 5090** with nvcc and an immutable,
byte-verified Qwen3.8-27B artifact that fits. Pin native q8_0 K/q5_1 V; no format flag.
Source of the commands: `docs/TESTING.md:585-600` requires the actual server suite;
`tools/kv-host-spill-identity-gate.sh:29-40` defines teeth and its internal lock;
`tools/kv-host-spill-failure-gate.sh:1-28` defines the failure battery. None ran here.

```sh
# Run this block as ARM=before, then git apply, then ARM=after in a scratch worktree.
cargo build --release -p memra-server -p memra-engine \
  --bin memra-server --bin run-gen --target-dir "target/spill-b-$ARM"
cargo test -p memra-server --lib --target-dir "target/spill-b-$ARM"

# Fitting 8k/32k baseline run-gen probes precede the short legacy prefix cells;
# rig-cells-b.py supplies immutable raw token arrays, MEMRA_MAX_CTX and MEMRA_NGEN.
# The scripts below own flock internally; NEVER wrap them in an outer flock.
MEMRA_GPU_LOCK=/tmp/memra-5090.lock bash tools/kv-host-spill-identity-gate.sh \
  "$ARTIFACT" "$PWD/target/spill-b-$ARM/release/memra-server" "$EV/$ARM-identity"
MEMRA_GPU_LOCK=/tmp/memra-5090.lock MEMRA_HOSTGATE_TEETH=1 \
  bash tools/kv-host-spill-identity-gate.sh \
  "$ARTIFACT" "$PWD/target/spill-b-$ARM/release/memra-server" "$EV/$ARM-teeth"
MEMRA_GPU_LOCK=/tmp/memra-5090.lock bash tools/kv-host-spill-failure-gate.sh \
  "$ARTIFACT" "$PWD/target/spill-b-$ARM/release/memra-server" "$EV/$ARM-failures"

# Between complete before/after blocks:
git apply --check research/spill-b-20260919/HOSTPREFIX-PATCH.diff
git apply research/spill-b-20260919/HOSTPREFIX-PATCH.diff
```

These legacy gates have a box-global `pkill -x memra-server` teardown. The runner
requires explicit non-serving/exclusive authorization, rejects existing server/GPU
processes, preserves basename `memra-server`, and serializes gates. They still cover
only their declared legacy surfaces and short prompts. Ignored CUDA/2-device tests in
host_glm are not silently counted as executed by `cargo test --lib`; the pair-owned
ignored test belongs to the later PRO qualification window.

No new MEMRA_* read, kernel, hardware default or published number: no FLAGS/KERNELS
amendment fragment is needed. Runtime application/enablement requires lead acceptance.
