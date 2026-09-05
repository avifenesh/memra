# Experimental native NVFP4 latent cache checkpoint

## Current scope: DFlash2, no native MTP

2026-09-05 update after merging main `e35041e29` (merge `72bbd416c`).
The owner requires DFlash2 with D2T vocabulary trim, not native MTP. Serving cache
creation now follows loaded-head presence via `pp::new_cache_for_model`; checkpoint
metadata declaring NextN no longer reserves state for an unloaded head. Raw config
has12 latent layers; the no-MTP allocator has11 trunk latent layers and zero head
KV/recurrent/latent planes. Raw-config allocation controls are not MTP serving tests.
The quality probe explicitly refuses a loaded MTP head or allocated head state.

Whole-verify graph replay now invalidates the shared device-status cache, matching
ordinary launch wrappers. Three server diagnostics hash encoded payload, block
scales and row macro scales with an NVFP4 format tag. Previously the entry digest
omitted compressed bytes and live digests attempted to read the absent f32 shadow.
The valid f32 digest byte sequence is unchanged.

The actual server `prefix_snapshot` / `prefix_restore` path passes a CUDA gate
for both f32 and NVFP4: populated latent/index/recurrent state, restore from8-row
capacity into16 and4 rows, absent NextN slots, and a red encoded-row mutation
that changes both state diagnostics. The fixture supplies three synthetic rows
and the indexer's pool geometry; this is not model inference or quality evidence.
Allocator controls retain state when a head is admitted and handle no-NextN config.

Independent review also found a pre-existing latent-history omission in
`pp::restore_cache_checkpoint` (the older length/recurrent-only checkpoint path).
It now refuses latent presence on either side before copying. This prevents plain
park compaction or exact-extension growth from reporting a false reuse hit with
empty history. Callers retain the original cache or cold-prime. The separate
full-state latent prefix path remains supported and is the passing path above.
Neither format may receive performance credit from the broken compaction path.

Fresh receipt namespace: `receipts/headless-r2/`. Binary identities and raw logs
there supersede the earlier hashes below only for their explicitly run gates.
Target-hardware/model-load, DFlash2+D2T engagement, PP transport, quality and
best-vs-best sampled serving measurements remain open. No MTP experiment, model
publication or production replacement is authorized by this checkpoint.

## Earlier implementation checkpoint

Issue244, 2026-09-05. This is implementation/mechanism evidence, not model-scale
quality, target-hardware performance, or production qualification.

The format uses sequential E2M1 nibbles, one E4M3 scale per16 values and a
row-local f32 macro scale. A512-value row occupies292 bytes in the three data
planes. Active cache planes share a sticky request/stage status word. DSA
index/pool and recurrent state remain unchanged. No persistent f32 history shadow.

Wired boundaries: allocation/admission, prefix-budget sizing, direct gathered
attention, transient BF16 tensor-core prefill operand, eager append, device-position
append/attention, captured replay, encoded snapshots, and completion-time error
checks for prime, decode, batched decode, verify and native MTP. Failed status
taints the cache. Snapshot format mismatch refuses. Stateless all-row forward is
explicitly unsupported under this experiment; last-row forward uses cached prime.

Validation on a non-production local SM120 device, serialized by
`flock -n /tmp/memra-gpu.lock`:

-28 memra-kv CPU tests pass.
- Engine test compilation and server test compilation pass.
- Prefix encoded-row budget CPU regression passes (one selected test, not the
  zero-test server binary target).
- Six ignored-by-default native GPU integration tests pass when explicitly run:
  actual-model plan allocation and poisoned MTP plane; CPU encoded-byte/gather
  identity; direct attention versus CPU-decoded f32 history; compressed snapshot
  with unchanged index tail/keys; captured device-position replay; and captured
  multirow verification with partial acceptance and rejected-suffix overwrite.
- The private production TC prefill chain test passes at t16 with19 existing
  prefix tokens, requires TC engagement, and compares against CPU-decoded history.
- Flag census and formatting checks pass. Read-only code-quality, security,
  performance and test-coverage reviews ran; their concrete findings were fixed.

Six-test GPU executable SHA256 at this checkpoint:
`5616be615cc4a4e5f13bcbe08550222db11e2e385e2c9d23367d9e7def039ed9`.
TC unit-test executable SHA256:
`ae79c1b0a10c72905c06e37ef5384e846f4489d8870f2ec1ff85353407421694`.
Subsequent source edits require rebuilt receipts, not inheritance of these hashes.

Model-scale quality, actual serving integration, cross-device qualification and
best-qualified baseline versus best-qualified NVFP4 tuning remain open. The
default-OFF experiment has decide-by2026-09-19 and must be removed on a losing or
neutral verdict. No speedup or quality preservation claim is made here.
