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

Runs from the integration branch `lane/glm5-b200-int-20260902` (coordinator note,
2026-09-02): this lane's branch merged onto `ff3dd4038`, the cuBLASLt per-device-handle
fix the PP2 boot on the B200 pair needed. Not this branch alone: the box gate below is
the integration branch's job, once merged there.

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

## Box round 1 (2026-09-02): every request failed at the first decode tick

The B200 pair ran the integration binary with this lane's wiring under
`MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 MEMRA_CTX=262144` (darklanes
`research/glm5-b200-20260902/box/tp2/boot-tp2-262144.log` and `gate-262144-rp0.log`).
What the receipts say, in order:

1. Spawn admitted the door: `[worker] MEMRA_GLM5_TP admitted for serving: ranks=2 ...`,
   `[glm5-tp-preflight] armed ranks=2 devices=[0, 1] layers=45 kda_shard=34 mla_shard=11
   moe_ep=42 kda_heads_per_rank=32 mla_heads_per_rank=32 experts_per_rank=144
   transport=host-canonical`, `[server] listening on http://127.0.0.1:18400`. Readiness and
   the pinned id answered. The `[admission]` device plan carried the new peer charge
   (`dev1 ... tp-kv 83MB`).
2. The prime ran through the TP prime walk (`[bf16-tc] ENGAGED m=155 ...` lines, no error).
3. EVERY request then died at its first decode token:
   `[engine-error] class=Engine batch step: KDA layer is glm5-TP-sharded (MEMRA_GLM5_TP):
   the plain mixer path is unwired for a head shard, only the TP decode/prime walk may
   execute it (t=1, arm decode)`. HTTP 500 on the greedy item, `completion_tokens: null` on
   all three sampled reps.
4. The box script printed `[tp2] rc=0` anyway: its exit status was the drain step's, not
   the requests'.

Root cause. `MEMRA_HYPER_BATCH` is DEFAULT ON (2026-08-31), so the worker classed the mHC
trunk as BATCHED DECODE and every decode tick went through `decode_step_batch_hyper` ->
`hyper_batch_range_decode`, whose per-session mixer loop dispatches the PLAIN
`kda_decode_cached` / `mla_attn_cached` calls. A head shard refuses those by name at the
`kda_core` choke point (the fail-closed surface working as designed). The eager per-session
walk (`hyper_range_decode`, `hyper_range_prime`) already carries the TP arms
(`glm5_tp::kda_tp_cached`, `mla_tp_attn_cached`); the batched walk never did.

### The fix: one numeric program, the batched tick refused by name

Decision: a glm5-TP-sharded model NEVER takes the batched mHC decode chunks; every step
of every session (prime, t=1, and each later tick) runs on the per-session eager TP walk.
Not the alternative (wiring the TP mixer calls into `hyper_batch_range_decode`), because:

- The only TP program with receipts is the eager walk: `glm5-tp-gate` holds decode (t=1)
  split-vs-plain BYTE identity per layer class and at model level, and the prime near-tie
  class is documented and banded. The batched composition (batched hc glue at m=B +
  per-session TP mixers + the EP MoE walk at t=B) has no gate at all:
  `glm5-hyper-batch-gate`'s fixture is unsharded.
- CLAUDE.md's one-numeric-program law names batched-vs-solo as a pair to keep honest:
  either make the crossing impossible or prove bit-identity in a serving-shape gate. A
  session whose numeric program depends on how many peers are queued is the eosclass
  failure shape. Refusing makes the crossing impossible today; the proof is the named
  follow-up.
- The serving items this round needs (readiness, identity, a greedy tape identical to
  PP-2, vendor-default sampled, a 256k prompt) are all single-program items. Concurrency
  on the eager path is per-session round-robin, one token per tick per session (the
  gemma4 eager-only precedent), which is functional, not fast.

Code:

- `HybridModel::glm5_tp_sharded()` (hybrid.rs): the model's own per-layer sharding, never
  the env (the #80 review finding). The two inline copies in `glm_spec.rs` now call it.
- `worker.rs`: `hyper_batched_decode_model` -> pure `hyper_batched_decode_route(hyper_trunk,
  hyper_batch_on, tp_sharded)`, sharded => false in both `MEMRA_HYPER_BATCH` arms (four host
  tests, both arms). A sharded model therefore lands in `eager_decode` (chunk cap 1) and
  the boot log names it after load: `[worker] <model>: EAGER-ONLY serving (glm5-TP-sharded
  trunk, MEMRA_GLM5_TP): batched mHC decode (MEMRA_HYPER_BATCH) is refused by name on a
  sharded model, every session decodes on the per-session TP walk; monolithic prefill, no
  graph promotion, no prime batching`. The spawn-time admitted line says the same.
- `decode_step_batch_hyper` (decode_batch.rs): refuses a sharded trunk by name before any
  row is touched (second fence; reaching it is a scheduler bug).
- `docs/FLAGS.md` (`MEMRA_GLM5_TP` DECODE ROUTE, `MEMRA_HYPER_BATCH` sharded clause),
  `docs/SERVING.md` (decode route + gate), `glm5_tp.rs` module doc (the stale "worker
  refuses the flag outright" sentence).

Named follow-up (not this lane): a sharded-fixture arm of `glm5-hyper-batch-gate` (B-row
tick vs solo TP decode, full-logit bit identity, red-armed) and a box A/B of eager vs
batched under TP-2, before `hyper_batched_decode_route` may admit a sharded model.

### The gate that fails when a request fails

`tools/glm5-tp2-serve-gate.py` replaces the box script's request items. It runs against
the listening server and its exit status IS the verdict: 1 on any failed item, 3 when an
item was skipped for a missing input (PARTIAL, never PASS), 0 only on 9/9. Items: I1
`/readyz`, I2 pinned id on `/v1/models`, I3 128-token greedy tape (temperature 0,
`reasoning_effort` low, streamed, sha16 over reasoning + content exactly as the floor
bench assembles it) equal to the PP-2 tape sha16, I4 the same tape twice concurrently
(both equal), I5 one vendor-default sampled request (no sampling params, 512 tokens,
completion_tokens > 0, no loop), I6 `/v1/completions` + `/v1/messages`, I7 a tool-call
request, I8 a 256k-class prompt (`prompt_tokens >= --long-min-tokens`, default 200000),
I9 boot-log route markers plus zero `[engine-error]` / `batch step:` lines appended during
the gate. One JSONL receipt row per item plus a summary row (`--out`).

Red-armed on the rig against a mock server before it shipped: PASS 9/9 (exit 0); a wrong
expected sha fails I3 and I4 (exit 1); the box's exact failure shape (HTTP 500 carrying
`batch step: KDA layer is glm5-TP-sharded`) fails every tape item (exit 1); a dead port
fails (exit 1); missing `--long-prompt`/`--boot-log` yields PARTIAL (exit 3); a missing
prompt file is a usage error (exit 2).

Box invocation (round 2):

```
MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=4 \
  memra-server ... 2>&1 | tee /root/lane/boot-tp2.log &
# wait for "[server] listening", then:
python3 tools/glm5-tp2-serve-gate.py --base http://127.0.0.1:18400 \
  --model zai/glm-5.3-flash --prompt /root/prompts/digits.txt \
  --expect-sha16 9437b599f6b9d2a9 --long-prompt /root/prompts/<256k-file>.txt \
  --boot-log /root/lane/boot-tp2.log --out /root/lane/tp2-gate-r2.jsonl
echo "gate rc=$?"     # 0 = PASS, 1 = FAIL, 3 = PARTIAL (an item was skipped)
```

`9437b599f6b9d2a9` is the PP-2 digits tape on this artifact (128 greedy tokens, effort
low, bench.py assembly). Bank `boot-tp2.log` and `tp2-gate-r2.jsonl` next to the round-1
receipts.
