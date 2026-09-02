# glm5 TP-2 serving wiring (memra #14, lane/glm5-tp-serve-wiring-20260902)

Parent: memra #12 (the B200 + 1M arc). This lane closes #14: "B200: TP serving wiring
does not exist in the worker." Worktree `/home/avifenesh/projects/wt-glm5-tp-serve`,
branch `lane/glm5-tp-serve-wiring-20260902`, cut from `origin/main` at `16c74a3f`.

No GPU was available to this session. Every gate below is host-only. The B200 box
belongs to the session that spawned this one; the copy-pasteable serving env at the
bottom is for that session (or whoever runs the box gate next) to execute.

## What was there before

The engine-side glm5_next (GLM-5.3-Flash) TP walk (`crates/memra-engine/src/glm5_tp.rs`)
already existed, byte-gated, for TP-2 and TP-4: head-sharded KDA/MLA, EP-N MoE, a
swappable transport axis (`MEMRA_GLM5_TP_TRANSPORT`). It loads through the same
`HybridModel::load` path every model family uses (`crates/memra-engine/src/hybrid.rs`,
armed at line ~4098 via `glm5_tp::prepare_glm5_tp_load`, layers sharded at ~4390-4427).

The ONLY thing stopping it from serving was one unconditional panic in
`crates/memra-server/src/worker.rs`'s `spawn()`:

```rust
if memra_engine::glm5_tp::glm5_tp_armed() {
    panic!(
        "MEMRA_GLM5_TP is set on a serving worker: the glm5 TP-2 seam is \
         engine/gate-only in v1 (serving wiring is the named tp2-lane box increment); \
         unset it"
    );
}
```

Setting `MEMRA_GLM5_TP` on `memra-server`, at ANY rank count, refused outright before any
model load started.

A real gap sat underneath that panic, independent of it: admission's device-keyed KV
cost model (`AdmissionDeviceRequirement`, `StepTpKvDeviceAdmission`,
`HybridModel::step_tp_unmaterialized_kv_bytes`) only walked `Mixer::Full` layers with a
`step_tp_qkv` sidecar (the "step" model family's own TP attention). glm5's sharded
layers live on `Mixer::Kda`/`Mixer::Mla` with a `.tp` field of a different shape
(`Glm5TpKda`/`Glm5TpMla`), so that function silently returned an empty charge vector for
a glm5 TP-armed model: peer-device VRAM for TP-sharded KDA/MLA state was invisible to
admission. The MLA half of this is a real leak, not a rounding error: peer devices hold
a FULL, per-token-growing latent KV replica (`memra-kv::Cache::glm5_tp_latent_peer`,
allocated lazily by `glm5_tp::ensure_mla_peer_latent`, geometry-cloned from the
canonical/root plane): the same per-token coefficient the root already pays via
`latent_kv_bytes_per_token_for_plan`, uncharged on every peer device.

## What changed

### 1. Engine seam: `HybridModel::glm5_tp_unmaterialized_kv_bytes`

`crates/memra-engine/src/hybrid.rs`, inserted right after
`step_tp_unmaterialized_kv_bytes` (~line 3806). Mirrors that function's "reserve until
the sidecar exists" contract for glm5's two sharded mixer classes:

- KDA-sharded layers (`Mixer::Kda` with `.tp` set): FIXED per-session bytes,
  `(conv_pad + state) * 4` (the peer's recurrent `conv_state`/`ssm_state` planes,
  `glm5_tp::ensure_kda_tp_state`, a linear-attention state that never grows with context).
- MLA-sharded layers (`Mixer::Mla` with `.tp` set): `bytes_per_token * capacity` per
  session, `bytes_per_token` read from `crate::cache::latent_kv_bytes_per_token_for_plan`
  for that layer index (the peer plane is a full geometry clone of root's, so it pays
  root's own coefficient).

Both classes charge a device only while the corresponding cache slot
(`cache.glm5_tp_recur[layer]` / `cache.glm5_tp_latent_peer[layer]`) has not yet been
materialized, exactly like the step precedent (once the sidecar exists, live CUDA
accounting sees the bytes directly and a continued reserve would double-count).

Return type is the shared `StepTpKvDeviceAdmission { device, bytes }` (device + bytes,
nothing step-specific about the shape), reused rather than inventing a parallel type.

### 2. Worker wiring: the three `[admission]` call sites

`crates/memra-server/src/worker.rs`. All three places that already call
`step_tp_unmaterialized_kv_bytes` now also call `glm5_tp_unmaterialized_kv_bytes` and
merge the charges onto the same device-keyed vector via the existing
`merge_tp_kv_charges` helper:

- `active_unmaterialized_tp_kv` (~line 5215): per-active-session pending charge.
- The per-request admission path (~line 13113 pre-edit, `request_tp_kv`): the
  request's own unmaterialized charge.
- The boot calibration probe (~line 22870 pre-edit, `tp_kv`): folded into the transient
  floor's subtracted charge so the measured floor never double-counts the reserved
  peer-rank bytes.

No new admission TYPE, no new call site: the existing Step-family device-requirement
pipeline (`parallel_device_requirements`, `AdmissionDeviceRequirement`) now sees glm5
peer-rank charges the same way it already sees Step's.

### 3. Worker spawn path: admit TP-2, refuse TP-4 by name

`crates/memra-server/src/worker.rs`, moved to run BEFORE the model-load loop starts
(previously the panic sat after all models had already loaded, which would have wasted
a full weight load before refusing an unsupported rank count). New helper in
`crates/memra-engine/src/glm5_tp.rs`:

```rust
pub fn glm5_tp_rank_count_from_raw(raw: &str) -> Result<usize, String>
```

Reads the device count out of the raw `MEMRA_GLM5_TP` value without needing the model's
trunk layer length (every spec's device list sits after `@`; every spec in one load
shares one device list per `prepare_glm5_tp_load`'s "one runtime group" invariant), so
the worker can refuse before any model is loaded. The worker then does:

- `MEMRA_GLM5_TP` unset or `0`/empty: unchanged, no-op.
- Exactly 2 devices: ADMITTED. Logs
  `[worker] MEMRA_GLM5_TP admitted for serving: ranks=2 ...` and falls through into the
  normal model-load loop, which calls `prepare_glm5_tp_load` exactly as before (every
  geometry/co-arm/transport law there is untouched).
- Any other rank count (including the engine-qualified 4): REFUSED by name, panic,
  before any model load starts.
- A malformed value (no `@`, no devices): REFUSED by name via the parse error.

## Named refusal strings (grep-able)

Serving-level (new, this lane, `crates/memra-server/src/worker.rs`):

```
[worker] MEMRA_GLM5_TP admitted for serving: ranks=2 (TP-2, general transport seam);
MEMRA_GLM5_TP names {ranks} devices: v1 SERVING admits TP-2 only
MEMRA_GLM5_TP={raw:?}: {err}
```

Already-existing, engine-level, unchanged by this lane (load-time geometry/co-arm,
`crates/memra-engine/src/glm5_tp.rs::prepare_glm5_tp_load` and friends):

```
MEMRA_GLM5_TP names {ranks} devices per layer; the qualified rank envelope is [2, 4]
MEMRA_GLM5_TP + MEMRA_PP_STAGES>1: the TP x PP composition is unwired
MEMRA_GLM5_TP + MEMRA_STEP_TP/MEMRA_STEP_EP: the step and glm5 parallel contracts never co-arm
{primary} + {flag}: unproven composition, refused   (the four decode-diet doors)
glm5-tp: {N} KDA heads do not shard across {ranks} ranks
glm5-tp: {N} MLA heads do not shard across {ranks} ranks
glm5-tp: {N} routed experts do not partition across {ranks} ranks
MEMRA_GLM5_TP rank devices must be distinct in serving
```

Already-existing, per-session refusal (spec x TP stays refused unless
`MEMRA_GLM5_SPEC_TP=1`, unrelated to and untouched by this lane):

```
cache snapshot is unwired for glm5 TP rank state (MEMRA_GLM5_TP): per-rank planes are not carried by CacheSnapshot   (memra-kv::Cache::snapshot / snapshot_into)
glm5 spec is co-refused on a MEMRA_GLM5_TP-sharded model: set MEMRA_GLM5_SPEC_TP=1 ...   (glm_spec::glm5_spec_session_new)
```

## Lifecycle points touched vs. reused as-is

| point | status |
|---|---|
| spawn admission of the flag | CHANGED (this lane): rank-count gate before load, TP-2 only |
| model load (weights, sharding) | UNCHANGED: same `HybridModel::load` path every family uses |
| readiness (`/livez`, `/readyz`) | REUSED AS-IS: family-agnostic, keyed on `worker::spawn` success/failure |
| drain (SIGTERM, in-flight) | REUSED AS-IS: family-agnostic, HTTP-inflight-gauge based, not per-device |
| admission KV accounting | CHANGED (this lane): new peer-rank charge function, wired into 3 call sites |
| spec x TP rollback | REUSED AS-IS: already named-refused at the cache layer, out of this lane's scope |

## Was an engine-side seam missing?

Yes, one: `HybridModel::glm5_tp_unmaterialized_kv_bytes` (item 1 above) did not exist.
Everything else the serving half needed (the load path, the per-rank state structures,
the cache-layer spec refusal) was already there. That one seam is implemented in this
same branch (`crates/memra-engine/src/hybrid.rs`), not deferred.

## Host-test evidence

Commands run in the worktree, no GPU:

```
cargo build -p memra-engine        # PASS, 1m26s, warnings only (nvcc/arch auto-detect)
cargo build -p memra-server        # PASS, 12.65s (workspace already warm)
cargo fmt --all -- --check         # PASS after `cargo fmt --all`
git diff --check                   # PASS, no whitespace errors
bash tools/check-flags.sh          # PASS: "runtime literal reads=800, no uncovered runtime names"
cargo test -p memra-server -p memra-engine --no-run   # PASS (binaries built)
cargo test -p memra-server -p memra-engine            # <fill in from the full run below>
```

`tools/check-flags.sh` confirms this lane introduced NO new `MEMRA_*` env read: the two
new functions (`glm5_tp_rank_count_from_raw`, `glm5_tp_unmaterialized_kv_bytes`) only
read the value that `glm5_tp_env_raw()`/`prepare_glm5_tp_load` already read, so no new
`docs/FLAGS.md` row was required for a NEW flag. The existing `MEMRA_GLM5_TP` and
`MEMRA_GLM5_TP_GATE_RED` rows were updated in the same commit to state the new serving
behavior accurately (they previously said the serving worker refuses the flag outright,
which stopped being true).

## The box gate this lane cannot run

No GPU, so none of this is exercised on real hardware yet: TP-2 model load reaching
`listening on`, a live `/v1/chat/completions` (and `/v1/completions`, `/v1/messages`,
tools) round trip, the admission KV numbers against a real artifact's actual VRAM
footprint, and the byte-identity gate against PP-2 on greedy decode (per the
`greedy-is-the-instrument-not-the-product` law: greedy is the exactness oracle, sampled
vendor-default rows are what a serving-decision claim would need on top).

Copy-pasteable serving env for the next session with the B200 pair, devices 0 and 1,
`$GLM5_ARTIFACT` pointing at the pinned NVFP4 checkpoint directory (the convention
elsewhere in `research/glm53-flash-bringup-20260827/` is
`zai/glm-5.3-flash=/root/models/glm53-nvfp4`):

```bash
GLM5_ARTIFACT=/root/models/glm53-nvfp4   # set to the pinned artifact this box holds

CUDA_VISIBLE_DEVICES=0,1 \
MEMRA_GLM5_TP=all@0,1 \
MEMRA_COMPAT=openai \
MEMRA_MODELS="zai/glm-5.3-flash=${GLM5_ARTIFACT}" \
MEMRA_ADDR=127.0.0.1:18400 \
MEMRA_CTX=131072 \
MEMRA_MAX_SESSIONS=4 \
memra-server
```

What that run needs to bank, at minimum:

1. Boot log carries `[worker] MEMRA_GLM5_TP admitted for serving: ranks=2 ...` and
   `[glm5-tp-preflight] armed ranks=2 ...` (the existing engine marker) and reaches
   `[server] listening on http://127.0.0.1:18400`.
2. `/readyz` reports ready; a plain `curl` round trip against `/v1/chat/completions`,
   `/v1/completions`, `/v1/messages`, and one tool-call request all return real
   completions (the standard-surface law: every model gets the identical full API).
3. `[admission]` log lines show a non-zero glm5 TP peer-device charge on device 1 for the
   first admitted session (proves the new accounting path actually engages, not just
   compiles), and a second, third concurrent session's charge growing the aggregate
   correctly against the box's real free VRAM (no premature 402/OOM, no over-admission
   into an OOM).
4. Greedy byte-identity: same prompt, same `max_tokens`, TP-2 vs PP-2 (or vs a
   single-device baseline if PP-2 is not simultaneously available on the box), logits or
   generated bytes compared. This is the pending gate `docs/SERVING.md`'s new
   subsection names as still open.
5. One vendor-default SAMPLED request (no explicit sampling params) with a
   spec-engagement or plain-decode receipt from the server log, per the
   never-serve-greedy-verify-sampled law, before any claim that TP-2 serving is fit to
   carry real traffic.
6. SIGTERM drain: confirm in-flight TP-2 sessions complete or the drain deadline applies
   the same as any other model, and the process exits 0.

None of this is claimed done. This LANE.md states it as the next box run's checklist.
