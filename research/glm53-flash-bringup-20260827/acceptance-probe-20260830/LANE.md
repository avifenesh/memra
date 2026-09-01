# glm5_next native MTP + T-parallel verify: real-artifact acceptance PROBE (card3 lane, 2026-08-30)

**CAVEAT (in every receipt): 1-card SLRU, served-path via lane/glm5-spec-routing
(single-device admit), probe-only — serving-shape numbers wait on the ppN twin. Acceptance
numbers from this posture are PROBE evidence for the go/no-go, never the serving-decision
receipt (that A/B runs interleaved x5 on the serving shape with the cache-on 8-turn twin).
Nothing here is a timing number: this is a co-tenant count-based lane on a shared box; no
tok/s exists in these receipts by design.**

## What this lane measures

The tparallel-verify lane's step 3 + 3b probe inputs, on the REAL published NVFP4 mint
(`glm53-nvfp4`) with the native NextN/MTP head (`MEMRA_GLM5_MTP=1`) and the T-parallel
verify loop (`MEMRA_GLM5_SPEC=1`):

- acceptance-at-K for K in 1..7 (accepted drafts per verify cycle; tok/cycle = acc/cycle
  + 1 verify bonus), greedy accept rule, real prompts only.
- spec-vs-plain greedy tape byte identity on the real artifact (the property the loop
  sells, previously fixture-only).
- FR-Spec trim arm: `MEMRA_FRSPEC_TRIM=glm53-ranks-sxc32768.gguf.txt` (the agentic
  serving-default candidate from the owner ranks mint, sha256
  `1804027e6148414c46cdab1a4f8773d063b1af8435d37a231ecd31d5574a1632`) vs no-trim, same
  prompts — the trim-on/off acceptance delta (verify step 3b). Verify stays full-vocab
  by contract.

Upstream calibration reference (never our claim): native MTP acceptance 3.71-5.06
tokens/cycle, 1.36-2.05x decode at c=1 (engine-survey-20260829).

## Execution path measured (stated per the lane spec)

**[SUPERSEDED mid-lane — see the UPGRADE section below: lane/glm5-spec-routing landed
and the served path now routes spec on a single device. This section stands as the
receipt of the pre-routing state, measured live at cc718b988.]**

The SERVING path does NOT route spec for glm5_next: worker `mtp_spec_capable`
(`crates/memra-server/src/worker.rs`) requires
`plan_backend::MTP_SPEC.capabilities(plan).speculative.supported`, which the
tparallel-verify lane deliberately left false (fail-closed manifest stance). Verified
live on the box: server boot with `MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1` loads the head
but serves plain decode (receipts in `serving-path-receipts/`).

Acceptance is therefore measured at ENGINE level through
`HybridModel::generate_spec` — the same `MEMRA_GLM5_SPEC` door, the same
`generate_spec_glm5` loop the fixture gates pinned — via the probe binary
`crates/memra-engine/src/bin/glm5_card3_probe.rs` (added on this branch; count-based,
prints no timing). No engine code was modified.

**Sampled twin at this level: DOES NOT EXIST.** The glm5 loop's accept rule is greedy
longest-matching-prefix only; the sampled accept rule (spec::SpecSampling rejection
walk) is the verify lane's stated follow-up and is not implemented for the glm5 door.
This is a banked finding, not a measurement gap: rows here are greedy-instrument rows.
The vendor-default sampled twin obligation attaches to the serving-shape A/B (flip
condition step 4), which cannot run until the worker routes spec.

## Box shape (identity scrubbed)

- 1x RTX PRO 6000 Blackwell Server Edition (97,887 MiB), CUDA_VISIBLE_DEVICES pinned to
  one card of a shared 4-card box; co-tenancy protocol honored (no requests during the
  other lane's timed windows).
- memra @ cc718b988 (lane/glm53-flash-bringup consolidated head), fresh release build,
  `MEMRA_CUDA_ARCH auto-detected 120a`. NVIDIA_TF32_OVERRIDE=0 everywhere.
- Artifact: the published GLM-5.3-Flash NVFP4 mint (local copy of `glm53-nvfp4`).
- Posture: 1-card SLRU per rebaseline-and-surface-20260828/serve.sh family:
  MEMRA_ST_PINNED=1 MEMRA_MOE_RESIDENT=0, MEMRA_MOE_SLOTS recorded per boot in each
  receipt. Named deviations from that serve.sh: single device (no PP), one card,
  MEMRA_MAX_SESSIONS=2, port/addr local, MOE_SLOTS re-sized to one card.
- Prompt pool: real prompts only — the 10 decode-attribution prompts
  (`../decode-attribution-receipts/prompts.json`) + the banked gpf-ab agent pool
  (3 large agentic contexts) where noted.
- Loop-law: any degenerated/repeating row is flagged, excluded from aggregates, and
  reported separately (greedy looping is a known artifact, never a finding).

## Files

- `results-notrim.tsv`, `results-trim-sxc.tsv` — per (prompt, K) rows:
  rounds/drafted/accepted/out_len/tape_identical/accept_rate/acc_per_cycle/tok_per_cycle.
- `outputs/` — decoded plain + spec texts per row (loop-law screening evidence).
- `probe-stderr-*.log` — load receipts (`[mtp-glm5] ...loading`, `[glm5-spec] draft head
  TRIMMED to N rows (FR-Spec d2t engaged)`), VRAM after load, engagement lines.
- `serving-path-receipts/` — the server-side fail-closed receipts (boot log lines +
  request/response showing plain decode served while the flags were set).
- `vision-cell/` — CELL 1 receipts (the glm5-vision lane's box-cell can't-hallucinate
  probe + negative arms; see that LANE.md for spec).

## UPGRADE mid-lane: the served route landed (lane/glm5-spec-routing @ 19d49a0b1)

The worker spec route shipped while this lane ran; on a single device it admits, so
acceptance moved from the engine harness to the REAL SERVED PATH (the engine-ladder rows
above/below stand as the K=1..7 ladder; the served cells are the product-shaped rows).
Server binary: 19d49a0b1 build (md5 `31b7a70ee4333d95ee3025a7a1f10cd6`, run as the
lane-scoped basename). Boot receipts per cell: `[glm5-spec] serve route ARMED: MTP head
loaded; draft head FULL target vocab` / `... TRIMMED to 32768 rows (FR-Spec d2t
engaged)`; per-request `[glm5-spec] route=spec K=..` lines; per-burst `[glm5-acc]`
lines; `usage.spec` rounds/drafted/accepted in every 200 response. Fresh-boot
output-sample gate ran on every boot before rows counted.

Cells (each: 10 decode + 3 agent real prompts, greedy arm = temperature 0, vendor-default
sampled arm = NO sampling params on the wire, max_tokens 128, cold sessions):
served-notrim-k3 (policy default K=3), served-notrim-k5 (MEMRA_SPEC_K=5),
served-trim-k3, served-trim-k5, served-trim-nopin (policy receipt).

### Acceptance summary (ACCEPTANCE-SUMMARY.txt; loop-law: 0 flagged rows of 130 served files)

Engine ladder, greedy, 10 decode prompts, every tape byte-identical to plain:
acc/cycle 0.79 (K=1) -> 1.39 (K=3) -> saturates ~1.41 by K=5; tok/cycle plateau ~2.41.
Served path (12 prompts): notrim-k3 greedy 1.413 / sampled 1.410 acc/cycle; notrim-k5
greedy 1.438 / sampled 1.307; trim-k3 greedy 1.398 / sampled 1.317; trim-k5 greedy
1.423 / sampled 1.382. Trim cost on greedy is ~1% acc/cycle at matched K (10/12 prompts
byte-identical counts); sampled arms vary per boot (statistical, one draw per prompt).
All well below the upstream 3.71-5.06 tokens/cycle reference at 7 drafts — our number on
our traffic; deep drafts past K~4 buy nothing on this artifact (chain acceptance decays
fast, the q38 deep-slot pattern).

### Consistency receipts (the strongest rows in this lane)

- Cross-box: 23/23 overlapping engine-ladder rows IDENTICAL (rounds/drafted/accepted)
  across two 4-card boxes (RTX PRO 6000 Server Edition vs Workstation Edition).
- Engine-vs-served: served greedy counts equal the engine-loop counts exactly at matched
  K (e.g. p00 K=3: 50 rounds/150 drafted/78 accepted on both paths, two binaries).
- Binary: 19d49a0b1 probe bin reproduces the cc718b988 p00 ladder identically (7/7 rows).
- Reboot: trim-k3 vs trim-nopin fresh boots: 13/13 greedy rows identical.
- Streamed == non-streamed counts (p10 greedy 51/153/79 both shapes).

### Findings (banked, not patched)

1. **Trim does not move the K policy.** With the FR-Spec trim loaded and no MEMRA_SPEC_K,
   the route logs K=3 (cold-short row), not the policy table's trim=5 — the trim-aware
   depth row does not key on the glm5 FR-Spec trim. served-trim-nopin/server.log.
2. **90s non-streaming deadline + this posture caps cold long prompts.** p11 (5.5k
   tokens) sampled died `deadline_exceeded` mid-generation non-streaming (98 tokens
   billed, `usage.spec` MISSING on the error shape — the spec counters are not attached
   when finish=error); `timeout_ms` is hard-capped at 90000 (probe of 1800000 -> 400
   with the platform-ceiling explanation). Streaming fixes p11; p12 (~7k tokens) still
   408s even streamed: its cold prefill on the 1-card SLRU posture exceeds the 90s TTFT
   ceiling. Posture artifact, banked as rows; not a route bug.
3. Loader-law WARNINGs (serving-path-receipts/RECEIPT.md): `blk.*.nextn.eh_proj.weight`
   Float-2D row is new with MEMRA_GLM5_MTP=1; audit decision belongs to the engine lane.
4. Vision cell findings live in vision-cell/VERDICTS.md (text-only literal placeholder
   served; faked-pad-with-image and video refusals exact; can't-hallucinate PASS both
   sampling arms).
