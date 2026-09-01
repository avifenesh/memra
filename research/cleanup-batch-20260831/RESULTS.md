# Cleanup batch 2026-08-31: four named debts, one landing

Lane: `lane/cleanup-batch-20260831`, branched from `origin/main` @ `ce0a42520`.
Source: hermes repo-review sweep fingerprints + prior-lane findings. Four small debts,
fixed at the seam they live in, no scope growth.

## Debt 1: vision data-URI per-image byte cap (fingerprint 48f96cb4cd37e436)

**Debt.** `decode_data_uri` (`crates/memra-engine/src/vision_pre.rs`) was an unbounded
standard base64 decode running in the HTTP content walkers BEFORE slot admission
(`content_to_text_vision_step` and siblings). The `MAX_BODY_BYTES` comment budgets
8 images x 12 MiB raw, but nothing enforced 12 MiB per image; only the 192 MiB body
ceiling bounded it, so one oversized image could expand ~144 MiB of host bytes
pre-check. The gemma mirror (`gemma_decode_data_uri`) carried the identical hole.

**Fix.** New `vision_pre::IMG_MAX_RAW_BYTES = 12 MiB` enforced AT the decode in both
decoders via a shared payload-LENGTH check (`data_uri_payload_over_cap`): base64 emits
4 chars per 3 raw bytes and the cap is a multiple of 3, so the encoded-length bound is
exact and an oversized payload is refused before any allocation. Error names the limit
("image data exceeds 12 MiB (per-image raw limit, refused before decode)") and surfaces
through the existing image-error wrap (`image {n}: ...`) as a clean 400. The
`MAX_BODY_BYTES` comment now states the line item is enforced, not aspirational. The
8-image budget (`VISION_MAX_IMAGES`) and every documented behavior are unchanged.

**Gates.**
- Unit: `vision_pre::tests::data_uri_per_image_raw_cap`: one base64 quad past the cap
  refuses by name (both decoders); exactly-at-cap admits and decodes to 12 MiB. PASS.
- Existing vision tests: PASS (see suite line below).
- Real surface (sbox dev box, live vision-enabled server): oversized data URI returns
  the named 400; at-cap and in-budget URIs are admitted past the cap gate; the 9th
  image still refuses "too many images (max 8)". Receipts: `raw/devbox/cell1*.json`,
  cells table below.

## Debt 2: MTP_SKIP treated default-ON spec as a non-request and booted PLAIN (fingerprint baf261e2bfdae118)

**Debt.** `mtp_skip_no_drafter_verdict(None)` returned `Ok(PLAIN)`: unset
`MEMRA_SERVE_SPEC` means spec ON for serving, so a skip-on MTP model with no dspark
drafter silently served plain decode at half speed behind one boot line, the
DFlash2 2026-08-25 incident class (fluent, slow, no receipt). A unit test pinned this
as intended.

**Fix (refuse-loud).** The contract is now: only the literal `MEMRA_SERVE_SPEC=0`
(the one value `serve_spec_enabled()` treats as spec-off) boots, announcing PLAIN by
explicit choice. An explicit non-zero request refuses as before; the unset/blank
default now ALSO refuses at boot, naming the missing drafter and the overrides
(`set MEMRA_SERVE_SPEC=0`, `arm MEMRA_DSPARK_SPEC`, or `unset MEMRA_MTP_SKIP`).
Refuse-loud was chosen over WARN+metrics: the only known skip deployment shape
(q38, `research/mtp-skip-q38-20260830/`) runs dspark-armed and is untouched, and no
deployment was found that legitimately needs the silent path. The pinning test was
rewritten to the new contract (`default_spec_refuses_only_explicit_zero_boots_plain`),
and the `MEMRA_MTP_SKIP` FLAGS.md row was updated in the same commit.

**Gates.**
- Unit: explicit non-zero refuses; None/blank/" 0 " refuse naming `MEMRA_SERVE_SPEC=0`
  and `MEMRA_DSPARK_SPEC`; literal "0" boots with the PLAIN announce. PASS.
- Boot cells (rig 5090, lock-serialized, real MTP-carrying q38 artifact; tooth 3 also
  re-proven on the sbox dev box): `MEMRA_MTP_SKIP=1` with `MEMRA_SERVE_SPEC` unset
  refuses at boot with the named FATAL; `MEMRA_SERVE_SPEC=0` boots and serves plain
  quietly. Receipts: `raw/rig/tooth*.log`, cells table below.

## Debt 3: docs/SERVING.md sold the effort levels as a graded thinking-budget ladder (fingerprint ec383a1300460e6d)

**Debt.** The `reasoning_effort` paragraph said `low|medium|high` = "thinking ON at
that budget", promising a graded depth ladder. Measured truth
(`research/step37-reasoning-effort-20260829/RESULTS.md`, cell12): every level is
honored and answers sanely, but depth is NOT monotone (high inverts below medium on
non-trivial prompts) and the absent-field default is the deepest arm.

**Fix.** Paragraph rewritten: the levels are per-request token-saving dials, not a
quality ladder; absent = the model's own default and the deepest arm; naming any level
constrains the model relative to sending nothing; the measurement is cited. No em
dashes in the new text.

**Gate.** Docs-only; the rewritten copy quotes the cell12 receipts and promises
nothing the receipts contradict. Reviewed against the RESULTS.md numbers verbatim.

## Debt 4: FLAGS.md SWA_RING row contradicted the step37 family arm (review [high])

**Debt.** The `MEMRA_SWA_RING` row ended "Default remains OFF" while the step37
serving-defaults row arms it ON for SlidingGatedMoe at model load (owner flip
2026-08-27): two rows, two truths, one contradiction.

**Fix.** The SWA_RING row now states both truths: default OFF globally, ARMED ON at
model load for the SlidingGatedMoe (step37) family by `arm_step37_serving_defaults()`
since the owner flip 2026-08-27, pointing at the step37-serving-defaults row's
interleaved battery (tip b2c722e5a..7a784b67c) with `=0` as the kill switch. The
default column carries the same two truths. The stale "default-flip policy" remaining
door is removed (resolved by the flip for the family; OFF-globally is the design).

**Gate.** `tools/check-flags.sh` PASS (710 runtime literal reads, no uncovered names).

## Repo gates

- `python3 tools/check-public-boundary.py check --summary-only`: 670 matches, 670
  grandfathered, **0 new**. PASS.
- `tools/check-flags.sh`: PASS (710 runtime literal reads, no uncovered names).
- `cargo test -p memra-engine --lib vision`: 12 passed. PASS.
- `cargo test -p memra-server --lib`: 463 passed. PASS.
- Perf-CI arm of pre-push: SKIPPED KNOWINGLY (`MEMRA_SKIP_PERF_CI=1`). The engine files
  this lane touches (`vision_pre.rs`, `vision_gemma.rs`) change host-side pre-admission
  input validation only (a base64 payload-length check before decode); no kernel,
  decode, prefill, or serving path is touched, so a perf battery would measure nothing
  this lane changed.

## GPU cells: ALL GREEN

The mtp-skip teeth are correctness gates (no timing), so they ran on the rig 5090 under
`/tmp/memra-5090.lock` with the lane's release binary; the vision cells need the step37
artifact and ran on the sbox dev box. The toolchain A/B lane owned the box GPU first:
the window was taken only after 10 SUSTAINED free minutes (their inter-arm gaps ran up
to ~9 min, and two shorter 0-MiB dips were correctly not treated as a handback), with a
window note filed in lanectl control and a loud marker file on the box for the cell's
duration; the box was back at 0 MiB with no memra-server and fully cleaned at yield.
Runner: `cells.sh` (this dir). Real artifact: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (the
mtp-skip lane's own model); vision walker armed via `MEMRA_STEP_VISION_DIR` on the
step37-flash-nvfp4 artifact.

| cell | arm | result | receipt |
|---|---|---|---|
| tooth 1 (rig) | `MEMRA_MTP_SKIP=1 MEMRA_SERVE_SPEC=1`, no dspark | PASS: boot FATAL exit 1, "cannot be honored" quoted | `raw/rig/tooth1-*.log` |
| tooth 2 (rig, NEW contract) | skip, `MEMRA_SERVE_SPEC` UNSET, no dspark | PASS: boot FATAL exit 1, refusal names `MEMRA_SERVE_SPEC=0` + `MEMRA_DSPARK_SPEC` + the DFlash2 incident class | `raw/rig/tooth2-*.log` |
| tooth 3 (rig + box) | skip, `MEMRA_SERVE_SPEC=0` | PASS both hosts: boots, "serves PLAIN decode by explicit choice" announce, greedy request 200 with `.usage.spec == null` | `raw/rig/tooth3-*`, `raw/devbox/tooth3-*` |
| cell 1a (box) | data URI one base64 quad past the cap | PASS: 400 "image 1: image data exceeds 12 MiB (per-image raw limit, refused before decode)" | `raw/devbox/cell1a-oversized.json` |
| cell 1b (box) | data URI at exactly the cap (16,777,216 chars) | PASS: cap gate ADMITS; refusal is the image decoder's ("image header: The image format could not be determined") | `raw/devbox/cell1b-atcap.json` |
| cell 1c (box) | 9 valid images | PASS: 400 "too many images (max 8)"; the 8-image budget behavior unchanged (all 8 first images planned at the walker) | `raw/devbox/cell1c-nine-images.json` |
| cell 1d (box) | 1 valid small image | PASS: admitted past the cap gate (downstream refusal, where any, is the pixel decoder's CRC check, not the cap) | `raw/devbox/cell1d-one-image.json` |

Binary provenance per host in `raw/*/provenance.txt` (sha256 + `git log -1` of the
build checkout, both at 9fad380f8).
