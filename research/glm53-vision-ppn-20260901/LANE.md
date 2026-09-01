# lane/glm53-vision-ppn — VISION ON THE ppN SERVING SHAPE, WITHOUT THE 3x DECODE TAX

**Opened 2026-09-01 as the named engine follow-up to the glm53 launch window's finding 7.
Worktree `~/projects/wt-glm53-vision-ppn`, branch `lane/glm53-vision-ppn-20260901`, base
`origin/main` @ `d647e7b22`. Rig: local RTX 5090, exactness only — no timing number is read from
any log in this lane.**

Trigger: the owner's "image should be default on" order for glm5 (2026-08-30) versus what the
launch measured on 2026-09-01 — vision CANNOT ship on the 3-card PP3 serving shape of v0.123.0.
The launch pinned `MEMRA_GLM5_VISION=0`, shipped the modality fact text-only, and named this
lane. (darklanes `research/glm5-serving-launch-20260901/window-20260901/WINDOW.md` finding 7,
receipts in `vision-serving-shape/receipts.json`.)

---

## 1. VERDICT

**ROOT CAUSE: the overlay's residency invariant was proxied by an ENGINE-POINTER identity that
the serving shape can never satisfy. `hybrid_forward.rs` asked `std::ptr::eq(rt.engine(0, e), e)`
— "pp stage 0 must BE the primary engine" — and on the deployed shape it is not, because the
worker's primary engine deliberately follows the LAST pp stage.**

`worker::worker_device` returns the LAST entry of `MEMRA_PP_DEVICES`, so `0,1,2` puts the primary
on **dev2** while stage 0 owns **dev0**, and `PpNRt::build` hands stage 0 its own `Engine`
whenever its device differs from the primary's. The vision tower runs on the primary
(`build_vision_overlay(engine, ..)`), so its rows sat in dev2's context while the splice needed
them in dev0's. Every image request on the serving shape returned
`prefill error: vision embedding overlay requires stage 0 on the primary device`.

That primary placement is NOT incidental and must not be "fixed" by moving it: pinning the
primary to stage 0 was the v0.72 tag-blocker-2 regressor, **112.5 -> 17.5 tok/s** on spec+PP-2
serving (`worker.rs`'s own test names it). The check was RIGHT about the hazard — a cross-context
splice reads a pointer from another address space — and WRONG about the remedy.

**FIX: publish the overlay into the engine that owns embedding intake, and make the check exact
instead of loosening it.** `EmbedOverlay` now carries the CUDA context its rows live in;
`HybridModel::vision_intake_engine` names the consuming engine for the placement;
`EmbedOverlay::new_published` puts the rows there at construction. The identity check passes
because the pointers ARE in the right domain.

**Cost: nothing per token.** Publication is ONE host round trip of `[rows, n_embd]` f32 per
SESSION, in prefill (~5 MiB for a 256-row 5120-wide image), against a tower that already
host-bounces q/k/v and its merger input in every one of its blocks. Decode never touches an
overlay. The alternative the launch had inside the tag, `MEMRA_PP_STREAMS=0`, cost **~3x decode**
(19.8-26.5 vs 75.3-77.5 tok/s sampled, same box, minutes apart).

**RESULT so far: 6/6 rig cells green at 8 PASS lines each (was 6), the span-shift RED arm bites
both new arms, publication non-vacuity asserted in every cell, clippy clean, flags census clean.
The claim this lane is allowed to make ends there** — the cross-context half needs the box, and
the battery for it is written and waiting for a window (`box/`).

---

## 2. WHY THE CHECK WAS BOTH TOO STRONG AND TOO WEAK

A device pointer is meaningful inside ONE CUDA context. That is the invariant, and neither half
of `ptr::eq(&Engine, &Engine)` measures it:

* **Too strong.** `CudaContext::new(ordinal)` RETAINS the device's primary context (cudarc 0.19,
  `primary_ctx::retain`), so two `Engine`s on one device share one address space — which is
  exactly why `PpNRt::build` can give every stage `s > 0` its own `Engine` on the primary device
  for scratch-pool isolation. `ptr::eq` refuses those, though a pointer would be perfectly
  valid.
* **Too weak in principle.** It sat at ONE call site (the hc ppN prime) and reasoned about where
  the overlay was BUILT. Any other consumer — the serial chunk walk, gemma4's masked-prefill arm
  — carried no check at all; they were safe only because their caller happened to be the same
  engine.

So the law moved to the point of use and to the real condition: `EmbedOverlay::require_resident`
compares CUcontexts, and every site that dereferences `rows` calls it. The new check REFUSES a
case the old one accidentally allowed (an overlay from a third engine in a third context) and
ALLOWS the cases that were always sound. Nothing was relaxed to make a feature pass.

`prime_chunk`'s hand-rolled splice loop — byte-identical to the shared one — now calls
`EmbedOverlay::splice_into`, so there is one implementation, one law, and no drift.

---

## 3. WHY A HOST BOUNCE, AND NOT A PEER COPY

Peer D2D would be faster and is available in principle (`PpNRt` records `peer_capable` pairs).
It is not what this lane ships, for reasons in this order:

1. **Ordering is provable without event plumbing.** `tower.dtoh` drains the producing stream at a
   HOST boundary; `intake.htod` + one `synchronize` puts the bytes down before any later stream
   can read them. The consumer may be a DIFFERENT stream of the intake context (stage 0's stage
   stream), which an event recorded on the upload stream would not cover — and getting that
   wrong is precisely the class the accrace lane just spent a lane on
   (`MEMRA_PP_EXIT_PUBLISH`: a body that published only the producing stage's stream let the
   caller allocate under queued work; the hc ppN prime stopped being a function of its inputs).
2. **It works on every placement.** No P2P capability needed, so it holds on
   `MEMRA_PP_HOST_BOUNCE=1` boxes too, where a peer path is refused by construction.
3. **The cost cannot matter.** Once per session, prefill only, on a tower whose v1 posture is
   already host-heavy by design. Optimizing it before measuring it would be inventing a
   correctness risk to buy an unmeasured millisecond.

**Named follow-up, not taken here:** a peer D2D twin behind the existing boundary transport,
with its own byte-integrity receipt and a real measurement of what it buys. It needs a box row,
not a bigger diff.

---

## 4. WHAT SHIPPED

| file | change |
|---|---|
| `crates/memra-engine/src/vision.rs` | `EmbedOverlay` carries `ctx: Arc<CudaContext>` (residency cannot be forged, and the context cannot outlive the pointer); `new`, `ctx()`, `resident_in`, `require_resident` (the law as a refusal, shared by every consumer), `new_published` (the host-bounce publication + its anatomy); `OverlayPublish` + pure `overlay_publish_resolve` + `overlay_publish_mode`; `overlay_publications()` counter so a gate arm can prove it did something; `window()` carries the parent's residency; `splice_into` enforces the law. Unit tests pin the whole arm matrix and the unrecognized-value refusal |
| `crates/memra-engine/src/hybrid_forward.rs` | the OVERLAY DEVICE LAW rewritten as the OVERLAY RESIDENCY LAW with the full diagnosis in place (including why the primary follows the last stage and why that stays); `vision_intake_engine` (the engine that owns embedding intake for the placement, mirroring the prime's own door test, with the prime's refusal as the fail-closed enforcement); `prime_chunk` routed through the shared splice; gemma4's masked-prefill arm calls `require_resident` |
| `crates/memra-server/src/worker.rs` | `build_vision_overlay` takes the intake engine and constructs through `new_published`; **placement admissibility decided at BOOT** — a loaded tower is not sufficient, so `GLM5_VISION_SERVING` now also requires that the overlay can reach intake on this placement. With `MEMRA_VISION_OVERLAY_PUBLISH=0` on a cross-context placement the boot says `IMAGE INPUT DISABLED` and image requests refuse at the HTTP waist instead of failing mid-prefill (the launch's 500). An unrecognized door value is a boot death |
| `crates/memra-engine/src/bin/glm5_hyper_ppn_gate.rs` | arms **5d** (published overlay, monolithic) and **5e** (published overlay, windowed = the serve prefill-tick shape), bit-identical against the substituted-token truth; `MEMRA_VISION_OVERLAY_PUBLISH=force` makes the path run on one card at all, and the arm ASSERTS `overlay_publications()` moved so it cannot pass as a vacuous re-run of 5b |
| `crates/memra-engine/src/bin/gemma_vision_e2e.rs` | constructor instead of a struct literal |
| `docs/FLAGS.md` | the `MEMRA_VISION_OVERLAY_PUBLISH` row (default stated as a decision with its reason, both arms, the rollback seam, the cost, the receipts pointer) and a correction to the `MEMRA_GLM5_VISION` row's "remaining precondition": the serving-shape cell WAS run at the launch and it FAILED, which is why this lane exists |

### The flag, and why its default is `auto`

`MEMRA_VISION_OVERLAY_PUBLISH` = `auto` (default) | `force` | `0`.

Default-ON-shaped behavior with a stated reason, per the new-flags law: on every shape that
already worked (one device, door shut, `MEMRA_PP_STREAMS=0`) the contexts match and `auto` takes
the zero-copy branch, so it is **byte-identical to the pre-lane program there**; on the ppN
serving shape it is the only thing that serves an image at all. The alternative default is
shipping a 500 for every image request. `force` is the rig gate's lever. `0` is the rollback seam
to the pre-lane program, and it now refuses at the waist rather than mid-prefill. An
unrecognized value refuses instead of resolving to a default.

---

## 5. RIG RECEIPTS (`receipts/rig/`)

`flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`, debug binary, final tree.

| cell | exit | `gate PASS` lines | publications |
|---|---|---|---|
| stages=2 P=6 N=8 | 0 | 8 | 1 |
| stages=3 P=6 N=8 (the deployed stage count) | 0 | 8 | 1 |
| stages=4 P=6 N=8 | 0 | 8 | 1 |
| stages=2 `MEMRA_PP_STREAMS=0` (the same-stream seam) | 0 | 8 | 1 |
| stages=2 P=16 N=24 (long) | 0 | 8 | 1 |
| stages=3 P=192 `MEMRA_PRIME_CHUNK=64` (**3-call** chunked prime, per-chunk `window()` of a published overlay) | 0 | 8 | 1 |
| **RED** `MEMRA_GLM5V_GATE_RED=span-shift` stages=3 | **1** | 0 | — |

* The PASS-line count of a green run moves **6 -> 8**. Any runner asserting 6 now fails loudly
  with a wrong-count message rather than silently passing. No runner in `tools/` asserts it;
  closed lanes' banked receipts that quote 6 were deliberately NOT edited.
* The RED arm reports **5 FAILs including BOTH new arms** (`overlay-ppn-published`,
  `overlay-ppn-published-windowed`) — the new arms have teeth, they are not decoration.
* Non-vacuity: every cell prints `overlay publication: 1 performed, rows resident dev0 (intake
  dev0)` and asserts the counter moved. Without that assertion these arms would be arm 5b under
  a new name on a one-card rig.
* `cargo fmt --all --check` clean, `cargo clippy -p memra-engine -p memra-server --all-targets`
  **zero lints**, `tools/check-flags.sh` clean at **794** runtime literal reads, no uncovered
  names, no grandfather list. (The `while let` warning this lane previously named as belonging to
  `lane/clippy-worker-whilelet-20260901` was merged to main as PR #97 while this lane ran, so the
  exception is RETIRED rather than left standing — a named item rots the moment its reason dies.)
* Unit tests, `receipts/rig/unit-vision-tests.txt` — **3 passed**:
  * `overlay_publish_arm_matrix_is_pinned` and
    `an_unrecognized_overlay_publish_value_refuses_rather_than_defaulting` — the pure resolver's
    whole arm matrix, no env mutation (the `pp::peer_probe_startup_policy` pattern).
  * `a_foreign_context_overlay_is_refused_not_dereferenced` — **the refusal path EXECUTED on one
    card.** A context is not a device: `CudaContext::new_non_primary(0, 0)` gives a genuinely
    independent context on the same 5090 (asserted `!=` the primary's `cu_ctx`, or the test would
    be vacuous), rows are allocated inside it, and both `require_resident` and `splice_into` must
    refuse by name without touching the pointer — while the same-shape primary-context twin
    splices fine, so the assertion is about residency and not about the arguments.
    **Teeth proven by mutation**: neutering the check to `if true { return Ok(()) }` turns this
    test RED (`0 passed; 1 failed`), then green again on revert.
  * These have **no CI caller, by CI's own design** — `ci.yml` runs `--lib` suites for the
    CUDA-FREE crates only (plus one name filter it criticizes in its own comment). Stated rather
    than left as an assumption; the banked rig run is their receipt.
* `tools/local-ci.sh --perf` was NOT run and is NOT on this lane's critical path:
  owner process change, 2026-09-01 ("no local rig ci", self-review + revuto + GitHub CI). Pushes
  carry `MEMRA_SKIP_PERF_CI=1`; the reason is this line.

## 6. WHAT THE RIG CANNOT PROVE, AND WHAT ANSWERS IT

One card gives the SERVING PATH one context, so on the rig `vision_intake_engine` returns the
primary engine and the failing placement **cannot be constructed** — not even with
`MEMRA_PP_DEVICES=0,0,0`, because `worker_device`'s last-entry rule then makes stage 0 the primary
again. Everything the rig says about publication is therefore about BYTES.

The REFUSAL half is now an exception to that, and it was worth finding: a context is not a device,
so `CudaContext::new_non_primary` builds a second, independent context on the same card and the
residency law can be shown biting there (§5, with a mutation proving it bites). What still cannot
be shown on one card is the positive cross-context claim — that rows published INTO another
context are dereferenceable by the stage that owns it — because no `Engine` exists on that second
context to publish from. That is the box's job and nothing else's.

**Window GRANTED 2026-09-01, second in the queue** (behind the slot-B cache battery, and after
this PR merges post-freeze), with two coordinator conditions folded in and executed on the rig
BEFORE the window rather than discovered in it:

* **The fixtures are pinned by sha256 and their shapes asserted** — the launch lane's prompt-pool
  loud-loader lesson (`battery2-prompt-pool-loud-loader.patch`) applied to this probe's own
  instrument input. My probe reads no prompt pool, but it reads the banked card3 REQUEST fixtures,
  which carry a worse third failure mode than a missing pool: the codes are the right answer only
  for those exact image bytes, so a fixture swap would silently measure a different image under
  this arm's name. All **17 refusal paths executed** —
  `receipts/probe/refusal-paths-verified.txt` — including the two the launch patch names (missing
  file explicit vs default; right json, wrong shape) and five arm-SHAPE refusals that catch what a
  deliberate pin update could break (a can't-hallucinate arm with no image part; a greedy arm that
  is not greedy; a "vendor-default" arm carrying sampling params).
* **Arm D interleaves boots in ONE window** (`box/interleave.sh`), V,T,V,T,V,T, not one arm then
  the other — a cross-boot-block comparison cannot carry a "no tax" claim. Identity is
  pgrep-clear + a per-boot nonce stamped onto every row + a PID-age assertion + each arm's own
  boot lines, and the verdict asserts both arms share a `system_fingerprint` (arms differing by
  BUILD cannot testify about a flag). **A false-green was caught here before the window**: the
  first PID-age check used `ps -o lsstart=` piped to `date -d`, and this procps build has no
  `lsstart` while `date -d ""` returns today-at-midnight, so the assertion PASSED on an empty
  string. Now `ps -o etimes=`, refusing loudly if the primitive is absent. A fix boot reporting
  `cross_context=false` is a VOID, not a pass: the window would not be testing the ppN shape.

The box battery (`box/BATTERY.md`, probe `box/probe-vision-ppn.py`, interleave driver
`box/interleave.sh`, refusal verifier `box/verify-probe-refusals.py`, window ask
`box/WINDOW-REQUEST.md`) is the other half: exact can't-hallucinate codes on greedy AND
vendor-default sampled, named refusals, a **reproducing control arm** (`publish=0` refuses at the
waist — a cell whose control stays silent cannot testify), text-only byte identity with vision
armed, and interleaved x3 decode rows against the text-only arm. Until those receipts exist,
nothing customer-facing moves and the launcher keeps its `MEMRA_GLM5_VISION=0` pin.

## 7. ALTERNATIVES CONSIDERED AND REJECTED (with their reasons)

1. **Loosen the check to a device-ordinal or drop it.** Refused outright: a vision feature fails
   FLUENTLY, and the corpus law here is that the can't-hallucinate probe is the bar. The check
   became more precise, not weaker.
2. **Move the primary engine to stage 0** so the old identity holds. Receipted regressor: 112.5
   -> 17.5 tok/s (v0.72 tag-blocker-2). Never.
3. **Load the vision tower on stage 0's device** so the rows are born in the right context.
   Plausible and maybe eventually right, but it moves ~2.1 GiB f32 of resident weights onto the
   card that also carries stage 0's trunk and the resident MoE budget (`MEMRA_MOE_RESIDENT_GB=98`),
   which is an admission/placement change needing its own box cell. The publication is contained,
   costs nothing per token, and does not touch weight placement.
4. **A repeated-device placement** (e.g. `MEMRA_PP_DEVICES=0,1,0`) makes primary == stage-0
   device and the old check pass. It is a workaround, not a fix: it puts two stages on one card
   and lands in the co-residency regime the accrace lane's blast-radius table calls the exposed
   one.
5. **`MEMRA_PP_STREAMS=0`** — the only in-tag option the launch had, at ~3x decode. That is the
   cost this lane exists to remove.

## 8. FOLLOW-UPS, NAMED

1. **Peer D2D publication twin** with a byte-integrity receipt and a real measurement of what it
   buys (§3). Needs a box row.
2. **The other families' PP+overlay refusals stay refusals.** `prime_chunk`'s
   `vision embedding overlay + PP prime unsupported (v1)` and the pipelined-prime refusals are
   untouched: those walks have no stage-0 embedding-intake seam, so publication alone would not
   make them correct. Anyone lifting them owes the same substituted-token gate the hc walk has.
3. **`hy3`/qwen/step37 towers** are unaffected (`vision_intake_engine` returns the primary for
   every non-hc model), but they inherit `require_resident` — if any of them is ever served on a
   placement whose embedding lands elsewhere, they now refuse by name instead of reading a
   foreign pointer.

## 8b. REBASED ONTO `origin/main` @ `88c7caed0` (12 commits, FREEZE LIFTED 2026-09-01)

The owner replaced the history rewrite with FRESH REPOS, so this lane lands in the old repo
normally and nothing was ever at risk. Rebase was conflict-free; the audits that matter here were
run anyway, because a clean auto-merge of `docs/FLAGS.md` is exactly where this repo has been
bitten:

* **FLAGS set containment, both directions**: this lane's `MEMRA_VISION_OVERLAY_PUBLISH` row
  present, and all **12** flag names main added in those commits enumerated from its own diff and
  grepped back out of the merged file — `MEMRA_ADMIT_PREDICT_BUDGET_MB`,
  `MEMRA_ADMIT_PREDICT_SHADOW`, `MEMRA_BUILD_ID`, `MEMRA_BUILD_ID_NOTE`, `MEMRA_BUILD_ID_SRC`,
  `MEMRA_BUILD_SHA`, `MEMRA_FIRST_TOKEN_DEADLINE_GATE`, `MEMRA_HYPER_SUFFIX_PRIME`,
  `MEMRA_KV_HOST_HANDOFF`, `MEMRA_KV_HOST_HANDOFF_MB`, `MEMRA_PARALLEL_EP_Q8_GU_PAIRED`,
  `MEMRA_PREFIX_STABLE_BOUNDARY` — **12/12 PRESENT**. `research/INDEX.md`: this lane's row and
  main's new rows both present.
* **`hybrid_forward.rs` drift is disjoint from this lane's**: main's only change in that file is
  the paired-Q8 preflight at ~line 9148; this lane edits ~1815 / ~2713 / ~3356 / ~4348.
* **Main handed this lane a better instrument, and arm D now uses it.** PR #99 made
  `system_fingerprint` a real, rewrite-proof build identity and every boot now prints
  `[server] build: memra-<ver>-<id> (id: source-tree, git: ..)`. `interleave.sh` asserts three
  things with it: the line EXISTS (an older binary cannot attribute its rows to a source tree),
  the id is NOT `degraded` (a version-only id cannot back a published claim, and these rows are
  meant to move the modality fact), and it is IDENTICAL across every boot in the window (a rebuild
  mid-window would turn the A/B into a build-vs-build comparison wearing a flag's name). All five
  branches of that assertion were executed before the window.
* Rebuilt and re-gated on the rebased tree — a pre-rebase green does not carry across 12 commits.

## 8c. PEER REVIEW OF THE MERGED PR, AND THE THREE FINDINGS FIXED (memra-next#23/#24/#25)

PR #104 merged as `77a4d1249` before the peer review returned, so these are fix-forward. The
review CONFIRMED the core fix (context equality is the right validity condition, verified against
cudarc's primary-context retain; nothing loosened; the same-device case is ordered by the existing
`fence_stages_behind`) and found three things worth fixing plus one comment to correct. All are
now closed on this tree. **Every claim below was verified against cudarc 0.19.9's source, not
taken on the reviewer's word — and two of them contradicted comments already in this repo.**

### #23 FLEET-FATAL, and two of my own comments were false

`EmbedOverlay::window` did `rows: self.rows.clone()`. cudarc 0.19.9 `impl Clone for CudaSlice` is
`try_clone().unwrap()`, and `try_clone` is `self.stream.clone_dtod(self)` (core.rs:856-865). So a
"window" was a full device allocation plus a D2D copy of every row, with an `unwrap` that PANICS
in the GPU worker thread — which exits the process and kills every in-flight session on the box
(the banked engine-panics-are-fleet-fatal law). It ran once per prefill TICK and once per prime
CHUNK, so a multi-chunk image prompt paid several whole-buffer copies.

Three things were wrong at once, and all three are fixed:

* `rows` is now `Arc<CudaSlice<f32>>`, so a window is a refcount bump **for real** — no
  allocation, no copy, no panic path — and the buffer frees once when the last window drops.
* The **false belief was corrected at every site that stated it**, not just mine: this file's two
  comments, `Engine::clone_dtod`'s doc in `crates/memra-engine/src/lib.rs` (which asserted
  "`CudaSlice::clone()` only bumps a refcount and would alias the live buffer"), and
  `crates/memra-kv/src/lib.rs`'s snapshot comment (which reached the right action from the wrong
  reason). One site already had it right (`lib.rs:28670`), which is how a repo ends up believing
  both things at once.
* **The published cost story is restated honestly.** "ONE round trip per session, nothing per
  token" was true of the publication and silent about `window()`'s copies. It is true as written
  only since this fix, and the doc comment now says so.

### #25 LIVE EXPOSURE: a family-scoped guard on a family-agnostic door

`overlay_publish_mode()?` was the FIRST line of `new_published`, which **all four** vision
families reach (qwen-VL, gemma4, glm5, step37) — while the boot-time validation ran only under
`glm5_tower.is_some()`. So `MEMRA_VISION_OVERLAY_PUBLISH=yes` on a step37 deployment — step37
serves vision in production TODAY — booted clean and 500'd mid-prefill on the first image
request: precisely the failure this lane removed for glm5, left live for the others. Fixed by
resolving the door **once at boot, unconditionally**, and threading the resolved `OverlayPublish`
through `prefill_tick` -> `build_vision_overlay` -> `new_published`. The env is no longer read
per session at all.

### #24 The residency label could LIE, and the fail-open arm was reachable

`new_published` recorded `intake.ctx()` while allocating through `intake.htod`, which resolves the
stream via the THREAD-LOCAL `STREAM_OVERRIDE` — not keyed to an engine. Inside a `PpNRt::enter(s)`
scope the upload would therefore land in STAGE s's context while the struct claimed intake's, and
`require_resident` would then PASS and vouch for a foreign pointer: this lane's own hazard with
the label inverted. My "call outside any stage scope" comment was correct and load-bearing, i.e. a
convention where an invariant was two lines away.

Fixed by taking residency from the **pointer** instead of the caller: `EmbedOverlay::new` reads
`rows.context()` (cudarc core.rs:844) and REFUSES when it disagrees with the engine the caller
believes it used, so a mislabelled overlay cannot be constructed; `new_published` routes its
uploaded buffer through the same constructor, making the label evidence rather than a claim. The
GPU test now asserts BOTH layers (construction refuses; and a fabricated mislabel still refuses at
the splice). Also fixed the reachable fail-open: the boot check's `None => true` is now
`false` — the `MEMRA_GLM5_VISION_DIR` branch loads a tower with no requirement that a glm5_next
model exists, so DIR-set-and-no-glm5-model advertised image serving on an unchecked placement.
The remaining multi-glm5-trunk limit (`find()` takes the first model while `vision_intake_engine`
keys on that model's layer count) is now STATED in the boot block rather than assumed away.

### The ordering sentence, corrected

The merged comment credited `publish_all_to` for ordering the published rows' free before the
stage-0 reads. That is wrong in a way worth recording: `publish_all_to` orders the CALLER behind
stage compute (the other direction) and is `MEMRA_PP_EXIT_PUBLISH`-gated, so the argument would
evaporate at `=0`. What actually orders it is the pipeline chain plus the host-synchronizing
`dtoh` every prime ends with in `hyper_prime_tail`. And there is no implicit net underneath:
cudarc event tracking is DISABLED in `Engine::new` unless `MEMRA_EVT=1`, so a drop carries no read
guard. The comment now says exactly that — it is the sentence a future lane will lean on.

### Re-gated on the hardened tree (`receipts/rig/SUMMARY-hardening.txt`)

6/6 cells green at 8 `gate PASS` lines with the publication asserted in each, span-shift RED exits
1 with both new arms among its 5 failures, unit tests 3/3. Plus a NEW arm the fix made possible:
**n3 with `MEMRA_VISION_OVERLAY_PUBLISH=0` in the ambient environment still publishes and passes**
— arm 5d now takes `OverlayPublish::Force` as a VALUE instead of mutating process-global env, so
the gate is immune to whatever the operator's environment says (and the env PARSING is covered by
the pure resolver's unit tests instead). fmt clean, clippy zero lints across memra-engine,
memra-server and memra-kv, check-flags clean at 795 reads.

### PORTED to the fresh repo (owner's all-clear, 2026-09-01)

The archives were deleted and `avifenesh/memra` is a NEW repo on the same name with UNRELATED
history — `git merge-base HEAD origin/main` returns EMPTY, and the roots differ
(`68b66bb7d` vs `49d1d6f65`). Nothing was merged across that boundary. The fix was re-applied as
CONTENT onto a fresh anchor off the new `origin/main` (`49d1d6f65`, "catch-up sync:
memra@469c8898e (purge re-applied)"), which already carries PR #104.

Two checks before trusting the port, because "it applied cleanly" is not the same as "it is the
same change":

* all **6 touched files were byte-identical** between the new main and my pre-port base
  (`469c8898e`), so the patch was not silently adapting to purged content; and after the port,
  the changed files are byte-identical to the pre-port versions.
* the snapshot's own tree differs from mine in **38 unrelated files** (the purge's redactions,
  e.g. `research/gateway-20260812/raw/.../models-openrouter.json`). **Zero overlap** with my
  change set, verified by set intersection rather than by eye.

Re-gated ON THE PORTED TREE — a green from the pre-port clone does not carry across an unrelated
history: same 6/6 cells at 8 `gate PASS` lines, RED still exits 1 with both new arms failing,
ambient-door immunity arm green, unit tests 3/3, fmt/clippy/check-flags clean.

## 9. STATUS LOG

* **2026-09-01 (latest)** — PR #104 MERGED as `77a4d1249` (main then `469c8898e`); merge verified
  against main's own blobs and the lane's gates re-run green on the COMBINED main. Peer review
  returned after the merge with three real findings (#23 fleet-fatal `unwrap` + false aliasing
  comments + the cost story, #25 the family-agnostic door guarded per-family with step37 live,
  #24 the residency label that could lie + a reachable fail-open) — all fixed locally and
  re-gated (§8c). Held unpushed per the owner's no-merge window; ports to the fresh repos.
* **2026-09-01 (later)** — coordinator relayed WINDOW GRANTED, second in the queue, with two
  conditions; both folded in and verified on the rig (fixture sha pins + 17 executed refusal
  paths; boot-level interleave driver whose own identity assertion was caught false-greening and
  fixed). The coordinator also recorded that this lane's diagnosis corrected the brief: the
  refusal was a REAL cross-context hazard with the wrong remedy, not a pointer-identity artifact
  to loosen. Merge is the only thing blocking the window, and it is blocked on FREEZE LIFTED.
* **2026-09-01** — lane opened on `origin/main` @ `d647e7b22`. Root cause found by reading, not
  by bisect (`worker_device`'s last-stage rule against `PpNRt::build`'s stage-0 engine rule).
  Fix + residency law + boot-time placement admissibility + rig arms 5d/5e landed; 6/6 rig cells
  green at 8 PASS lines, RED arm biting both new arms; FLAGS row + unit tests; clippy/fmt/flags
  census clean. Box battery written; window NOT claimed (coordinator's to grant). Pushed twice
  before the 2026-09-01 history-rewrite freeze; committing locally only until FREEZE LIFTED, then
  rebase onto the new history and open the PR (self-review + revuto + GitHub CI per the owner's
  process change).
