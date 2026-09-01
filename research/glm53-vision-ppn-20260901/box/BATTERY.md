# On-box battery: glm5 vision on the 3-card ppN serving shape

The rig cannot run this and no amount of rig green substitutes for it. One card is one CUDA
context, so the rig arms (`glm5-hyper-ppn-gate` 5d/5e) prove the publication is byte-exact and
nothing about whether a pointer published into ANOTHER context is dereferenceable by the stage
that owns it. That is what this battery is for, and it is the flip gate for the modality fact.

Box time is the coordinator's to grant (`WINDOW-REQUEST.md`). **WINDOW GRANTED 2026-09-01,
second in the queue** — behind the slot-B CACHE battery (the launch's gate 7, which the owner's
ship packet waits on), and after this lane's PR merges post-freeze. The coordinator attached two
conditions, both folded in below and executed on the rig before the window:

1. **The launch lane's loud-loader lesson applies to the fixtures** (`battery2-prompt-pool-loud-loader.patch`).
   My probe reads no prompt POOL, but it reads the banked card3 REQUEST fixtures, which are an
   instrument input of exactly the same class — and with a worse third failure mode: the codes
   this battery compares against are the right answer only for those exact image bytes, so a
   fixture swap would silently measure a different image. Every fixture is now **pinned by
   sha256**, its request SHAPE asserted (greedy arm really greedy; vendor-default arm really
   carrying no sampling params), and its identity written into the receipt. All 17 refusal paths
   were EXECUTED (`../receipts/probe/refusal-paths-verified.txt`) rather than reasoned about.
2. **Arm D interleaves in one boot sequence**, not across two runs — see arm D below and
   `interleave.sh`.

## What has to be true before the first request

| pin | value |
|---|---|
| memra | this lane merged to `main`, tagged, and the two-pin `serving/build-artifact.sh` binary built from the TAG (a public memra build FATALs on `MEMRA_REQUEST_LEDGER`; `serving/` is the only prod build path) |
| CUDA userspace | `/data/glm53/lib` — the BUILD-MATCHED CUDA 13.1 line. Launch finding 3: box12's cublasLt 13.4.1.3 fails the glm5 MLA TC-prefill strided-batched GEMM with rc 30014, which falls OUTSIDE the engine's 2xxxx decline window, so it errors instead of falling back. Do not "simplify" this away |
| shape | the ship recipe, unchanged: `MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 MEMRA_MOE_RESIDENT_GB=98`, spec ON with the DFlash2 drafter, `MEMRA_DSA_INDEX_RING` unset (it is a ROW COUNT, default 5120 — the launch's `=1` built a one-row ring and 500'd) |
| vision | `MEMRA_GLM5_VISION` UNSET (auto-detect, default on) for the fix arm — the launcher's `=0` pin is what this lane exists to remove |
| slot | a free slot, blue/green through `serve-deploy`; the dark stack keeps serving on its slot |
| fixtures | the banked card3 requests, **sha256-pinned in the probe** — `10-…greedy` `74c45948…`, `11-…vendor-default` `5732d261…`, `20-…placeholder` `14139fc0…`, `21-…video` `f534a178…`, `23-…fakedpad` `225437f0…`. A mismatch REFUSES before the first request; a deliberate change updates the pin in the same commit and says why |

## Arm A — the fix arm (default door)

1. **Boot receipts, before any request.** The log must carry, and the battery must grep:
   * `[server] build: memra-<ver>-<id> (id: source-tree, git: <sha>)` — asserted present, NOT
     `degraded`, and identical across every boot in the window (`interleave.sh`). A degraded id is
     version-only and cannot back a published claim; a changed one turns the A/B into a
     build-vs-build comparison.
   * `[glm5-vision] tower loaded ... GiB device delta at load`
   * `[glm5-vision] overlay intake: tower dev2 -> intake dev0 (cross_context=true) publish=Auto servable=true`
     — this line IS the diagnosis: the tower is on the primary (which follows the LAST pp
     stage) and intake is stage 0's device. `cross_context=false` here would mean the battery
     is not testing the shape it claims to test, and the run is VOID rather than green.
   * `[glm5-spec] serve route ARMED: draft source = dflash2 @ b33c0347`
2. **`probe-vision-ppn.py --arm fix`**: arms 10/11 exact codes (greedy AND vendor-default
   sampled — the shape real traffic sends), 20 plain-text placeholder, 21/23 named refusals.
3. **The publication receipt** in the server log for each vision request:
   `[vision] overlay published to the intake engine: dev2 -> dev0 rows=N elems=M MiB=X mode=Auto`.
   A vision 200 WITHOUT this line on this shape means the request took a path this lane did not
   intend; treat it as a failure, not as luck.
4. **Spec engagement** on the sampled arm: `[glm5-acc]` bursts in the log (the launch's own
   `serve-gate.py` read 384 real `[glm5-acc]` bursts as "served PLAIN" because it knew only
   `[spec-k]`/`dspark-acc`; the taught gate is the one to use).

## Arm B — the control, and why the battery is worthless without it

Re-boot the same binary with `MEMRA_VISION_OVERLAY_PUBLISH=0`.

* Boot must say `servable=false` plus the `IMAGE INPUT DISABLED` line.
* `probe-vision-ppn.py --arm control`: the two can't-hallucinate requests must come back as
  NAMED 4xx at the HTTP waist. A 500 fails this arm as hard as a fluent 200 would: deciding
  admissibility at boot is exactly what converts the launch's mid-prefill 500 into a refusal a
  customer can read.
* If arm B serves images, arm A proved nothing about this lane's code (a cell whose control
  stays silent cannot testify — the accrace lane's own rule).

## Arm C — text-only byte identity, vision armed but unused

Two boots, same binary, same greedy request (`temperature 0`, `reasoning_effort` PINNED — an
omitted effort measures think-prose, not the claim shape, and faked a fleet-wide "regression"
once):

* boot 1: vision armed (default) → `text-greedy.txt`
* boot 2: `MEMRA_GLM5_VISION=0` → `text-greedy.txt`

The two files must be BYTE-IDENTICAL, and so must the token counts. Expected by construction —
a text request never builds an overlay — which is precisely why a difference would mean
something is wrong somewhere this lane did not look.

## Arm D — no decode tax, interleaved in ONE boot sequence

Driver: `interleave.sh <reps> <out-dir>` (3 reps is the banked shape). The whole point of
publishing instead of rolling back to `MEMRA_PP_STREAMS=0` is that decode is untouched, so the
comparison is vision-armed vs text-only-pinned — NOT vs the streams-0 rollback, which the launch
already measured at 19.8-26.5 vs 75.3-77.5 tok/s.

**Why boot-level, and why that still satisfies the condition.** The arms differ by a BOOT env
(`MEMRA_GLM5_VISION` armed vs `=0`) and two 3-card resident stacks do not fit on three cards, so
the arms cannot be alive simultaneously and a request-level interleave is not physically
available. The driver therefore alternates BOOTS inside one contiguous window — V,T,V,T,V,T —
which is the interleaved-A/B law's own remedy for what it exists to stop: clock, thermal and
background drift land on both arms equally instead of on whichever ran second. Running all of one
arm and then all of the other (the shape this replaces) is what could not carry the claim.

**Arm identity, and its honest limit.** memra has no per-boot nonce surface —
`system_fingerprint` is the BUILD sha, identical across arms by design, which the verdict
ASSERTS (arms differing by build cannot testify about a flag). So identity is built from what
exists: pgrep-clear before every boot; the wrapper writes `BOOT_NONCE` as the first line of a
per-boot log and the probe stamps the same nonce on every row; the PID's elapsed age is asserted
against the boot command's own start; and each boot must print its arm's distinguishing vision
lines. A row belongs to an arm because those hold together, not because a port answered 200.

*(That age assertion was written twice. The first version used `ps -o lsstart=` piped to `date
-d`; on this rig `lsstart` does not exist in procps and `date -d ""` returns TODAY AT MIDNIGHT, so
the comparison PASSED spuriously — a false-green identity check, caught by executing it on the rig
before the window rather than in it. It now uses `ps -o etimes=` and refuses loudly if the
primitive is missing.)*

**The `cross_context=true` clause is a VOID, not a failure.** If the fix boot reports
`cross_context=false`, the tower and intake share a context and the window is not testing the ppN
serving shape at all; the driver stops and no row from it may be cited. A green run that measured
the wrong shape is worse than a red one.

**Which program the battery measures.** The hardened one, deliberately: the merged tree's
`EmbedOverlay::window` deep-copied the whole rows buffer on every prefill tick and every prime
chunk (memra-next#23), so a multi-chunk image prompt paid copies the cost story did not mention.
Fixed before the window at the coordinator's preference, so arm A's prefill numbers describe the
program that would actually ship. It does not move arm D's bar — that path is prefill, and decode
never touches an overlay — but a receipt should say which program it measured.

**Rows and the bar.** Vendor-default sampled (no sampling params), `reasoning_effort` pinned,
tok/s from each response's own `usage.elapsed_s` so the number is the server's. The verdict passes
when the between-arm median gap sits inside the arms' own within-arm spread, and it is written as
"no measurable tax at this rep count", never as a performance claim — there is also no mechanism
for a tax: publication is one host round trip per SESSION in prefill, a text request never builds
an overlay, and decode never touches one.

**The confounder to rule out first if the gap exceeds the spread:** the tower's own ~2.1 GiB on
the primary card changes that card's free memory, which can move the `[moe] resident-experts
decision` line. Compare that line across the two arms' boot logs before reading any gap as a tax.
The vision request's own decode row is reported as information only — a different prompt shape is
not comparable to the text row.

## Banking

Everything under `../receipts/box-<UTC>/`: the two boots' full logs, `receipts.json` per arm,
`text-greedy.txt` per arm plus the byte-compare result, the decode rows table, the binary
sha256, the artifact/drafter shas, and the boot lines quoted above. On-box copies under
`/data/glm53/evidence/vision-ppn-<UTC>/` — box files survive a history rewrite, local clones do
not.

## Then, and only then

The modality fact moves through the darklanes product-facts workflow (`facts.json` → render →
gates → ripple), the launcher drops its `MEMRA_GLM5_VISION=0` pin, and `gate2`'s "NO image"
assertion flips to asserting image support. Not before: the launch shipped text-only ON THIS
EVIDENCE being absent, and it is the same evidence that lets it change.
