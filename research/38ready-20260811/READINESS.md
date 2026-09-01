# Qwen3.8-27B same-architecture readiness ledger

Date: 2026-08-11

Status: **procedure STAGED; exact scored object MISSING. Do not start a scored evaluation or the
beside-Step A/B.** This is an artifact-readiness stop, not a measured non-fit: the frozen campaign
does not contain the exact Qwen3.8-27B target or a validated draft, so its load-bearing memory terms
are unknown (`research/beside-math-20260811/VRAM.md:3-9`).

This is a docs-only execution ledger. It makes no business, priority, timing, hardware-rental,
source-selection, purchase, or acquisition decision. It does not claim that Qwen3.8 is the same
architecture as Qwen3.6; the maintained runbook explicitly treats that as an assumption which must
be re-proved from current official metadata (`docs/ONBOARDING.md:368-372`).

## Artifact-format boundary

The lane brief asks that the absent exact Qwen3.8 GGUF be visible in the ledger. The maintained
Qwen3.8 runbook, however, defines the production candidate as the official FP8-E4M3 safetensors
directory and permits the frozen Qwen3.6 GGUF only as a kernel oracle, an A/B reference, or a
byte-verbatim MTP donor after an exact interface match; it forbids substituting Q8_0, GGUF, NVFP4,
or a community requant for missing official bytes (`docs/ONBOARDING.md:374-381`). The consolidated
prep receipt says the same and records that no Qwen3.8 model gate has run
(`research/cx-38prep-20260808/PROGRESS.md:9-19`,
`research/cx-38prep-20260808/PROGRESS.md:122-130`).

Accordingly, this ledger records both facts without making a new format decision:

- an exact Qwen3.8-27B GGUF is **MISSING** and is not supplied by the Qwen3.6 oracle;
- the only acquisition/encoding path already specified end to end is direct official FP8-ST, but
  its exact Qwen3.8 repository, immutable revision, files, manifest, and validation receipts are
  also **MISSING**; and
- if a future frozen campaign manifest literally requires a GGUF, that artifact needs its own
  named source, quantization recipe, complete hashes, and full correctness battery. Nothing in this
  ledger authorizes fabricating or substituting one.

## STAGED vs MISSING

`STAGED` means the reusable procedure/tool/control exists and has a non-Qwen3.8 receipt. It does not
mean the exact target passed. `MISSING` means the exact Qwen3.8 campaign receipt does not exist.

| Surface | State | Evidence and exact boundary |
|---|---|---|
| Maintained day-one runbook | **STAGED** | The architecture kit makes `docs/ONBOARDING.md` the single artifact-to-green sequence and identifies its Qwen3.8 worked example as executable; the two older runbooks are pointers only (`research/archkit-20260808/REPORT.md:82-98`). Its order covers identity, metadata, shards, direct-path proof, golden output, chunking, MTP, serving, and final receipt (`docs/ONBOARDING.md:420-437`). |
| Release-independent preflight and helpers | **STAGED** | The prep lane delivered the preflight, FP8 header inspector, exact HF-token comparator, same-architecture classifier, and safetensors chunk gate (`research/cx-38prep-20260808/PROGRESS.md:21-42`). The frozen run reported `PASS=54 WAIT=3 FAIL=0`; target absence remained a `WAIT` (`research/cx-38prep-20260808/preflight-20260808.log:55-66`). |
| Frozen Qwen3.6 controls | **STAGED** | Exact paths are pinned for the block-128 FP8-ST baseline, minimal architecture/tokenizer reference, model-backed GGUF oracle, and own-trim draft (`research/cx-38prep-20260808/PROGRESS.md:44-57`). They are controls only, never Qwen3.8 target evidence. |
| Declarative same-family geometry | **STAGED** | `ArchGeometryTable` centralizes mixer/head/KV/RoPE/window/gate geometry across prefill, decode, verify, MTP, cross-request prefill, and batched decode (`research/archkit-20260808/REPORT.md:35-58`). The kit also stages generated chunk/tick/B>1 gates with fail-closed validation (`research/archkit-20260808/REPORT.md:60-80`). |
| Exact Qwen3.8 same-architecture verdict | **MISSING** | The classifier and STOP matrix exist, but the exact Qwen3.8 config/tokenizer/template have not been bound and compared. A changed model type, head contract, attention cycle, GDN shape, RoPE scheme, tokenizer, or MTP interface is a new bring-up lane (`docs/ONBOARDING.md:484-539`, `docs/ONBOARDING.md:547-582`). |
| Native block-128 FP8-ST runtime path | **STAGED** | The existing class keeps one checkpoint-native E4M3 copy and uses the direct per-block prefill route by default; Qwen3.6's 208 block-128 projections supplied the measured control (`research/fp8blk-20260805/VERDICT.md:3-16`). Its control battery was `kernel-check` green, `run-spec` K=1..8 8/8, and both serve gates green (`research/fp8blk-20260805/VERDICT.md:63-66`). These are class/tooling receipts, not a Qwen3.8 pass. |
| Exact Qwen3.8 FP8-ST source and local artifact | **MISSING** | No exact target revision, indexed shard set, file hash manifest, or local target directory is frozen. The target repo and target directory were both still `WAIT` in the prep receipt (`research/cx-38prep-20260808/preflight-20260808.log:55-56`); beside-math consequently refuses to invent target bytes (`research/beside-math-20260811/VRAM.md:45-49`). |
| Exact Qwen3.8 GGUF | **MISSING** | No Qwen3.8 GGUF is in the frozen controls or scored manifest. The existing full Qwen3.6 GGUF is explicitly oracle/A-B/donor-only and must not be used to manufacture a full Qwen3.8 bridge (`research/cx-38prep-20260808/PROGRESS.md:117-126`). |
| Artifact classifier and direct-path proof procedure | **STAGED** | The header inspector accepts per-tensor and exact block-128 E4M3, rejects per-row/unknown layouts, and separates packed-U8 auxiliary scale planes (`research/cx-38prep-20260808/PROGRESS.md:27-32`). The Qwen3.6 control found 208 block-128 E4M3 weights, zero unsupported weights, and a header PASS, while explicitly leaving finite-scale, NaN-code, transform, residency, and dispatch checks to runtime (`research/cx-38prep-20260808/fp8-header-q36-baseline.log:1-11`). |
| Exact Qwen3.8 golden-output receipts | **MISSING** | The project gate is `kernel-check`, `run-gen` argmax MATCH, and `run-spec` K=1..8 (`CLAUDE.md:110-114`). No exact Qwen3.8 log exists for any of them; the prep lane explicitly disclaims a Qwen3.8 model gate (`research/cx-38prep-20260808/PROGRESS.md:122-126`). |
| Drafter construction machinery | **STAGED** | MTP extraction, draft trimming, quantization executable, prompt pack, and frozen Qwen3.6 draft are present in preflight (`research/cx-38prep-20260808/preflight-20260808.log:34-49`). The runbook specifies Qwen3.8-own generation/rank receipts and exact donor-interface checks before an external draft is built (`docs/ONBOARDING.md:871-909`, `docs/ONBOARDING.md:911-957`). |
| Exact validated Qwen3.8 drafter pairing | **MISSING** | A family-compatible-looking draft is not enough. The scored object must freeze the optional draft and its hash, and Qwen3.6 must never be paired without model-specific proof (`research/beside-plan-20260811/PROTOCOL.md:20-24`). Embedded MTP is not yet known; absent embedded tensors leave `run-spec` waiting (`docs/ONBOARDING.md:769-794`). |
| Beside-Step execution protocol | **STAGED** | The two-process/card placement, context 32,768, artifact stop, correctness preflight, raw-evidence rules, and pending matrices are already specified (`research/beside-plan-20260811/PROTOCOL.md:20-24`, `research/beside-plan-20260811/PROTOCOL.md:44-68`, `research/beside-plan-20260811/PROTOCOL.md:170-192`). |
| Exact Qwen3.8 memory fit | **MISSING** | Target/draft residency, second-process overhead, prefix cache, KV/session, learned residual, allocator pool, and Step overlap delta are all target-specific unknowns (`research/beside-math-20260811/VRAM.md:51-66`). Therefore `M38`, `S38`, `DeltaStep`, `H38`, and `c_max` are not yet values (`research/beside-math-20260811/VRAM.md:68-91`). |

## Exact-artifact acquisition and validation checklist

This is the future authorized operator's order. It is not authorization to acquire anything now.
Every STOP leaves the object unscoreable.

1. **Discover; do not assume the source id.** Query the official Qwen namespace at execution time,
   bind the exact official FP8 model id, fetch its model metadata, and freeze the immutable source
   revision. Record the returned repository id, revision, metadata response, and their hashes. If
   the official FP8 sibling is absent, STOP; do not substitute a community artifact or local bridge
   (`docs/ONBOARDING.md:442-470`).

2. **Fetch metadata only and prove the architecture before weight acquisition.** Compare the exact
   target config against the frozen Qwen3.6 reference with the mechanized architecture classifier;
   then compare tokenizer class, pre-tokenizer/regex, chat template, thinking markers, and token
   structure. Record both input hashes and the complete diff. Any hard architecture or tokenizer
   change opens a separate bring-up lane and stops this same-architecture checklist
   (`docs/ONBOARDING.md:472-539`).

3. **Freeze the quantized arm and encoding without reinterpretation.** The staged production path is
   checkpoint-native **FP8-E4M3 safetensors**, with `quant_method=fp8`, `fmt=e4m3`,
   `weight_block_size=[128,128]`, and `activation_scheme=dynamic` when the exact release confirms
   those fields (`docs/ONBOARDING.md:580-582`). Scale siblings must be one-value per-tensor or the
   exact `[ceil(out/128),ceil(in/128)]` block grid; per-row or any other grid is STOP
   (`docs/ONBOARDING.md:584-589`). Preserve the source encoding: no Q8_0 re-encode, GGUF/NVFP4
   substitution, scale folding, or community requant. Naked defaults must keep one native E4M3 copy
   and the native block route; explicit opt-in flags are not the production path
   (`docs/ONBOARDING.md:383-387`).

4. **Acquire only the pinned revision, then prove completeness.** After the preceding gates and
   separate authorization, download the exact revision. Verify that every shard named by
   `model.safetensors.index.json` exists. Record every file's byte length and full SHA-256, the
   directory size, index hash, config/tokenizer/template hashes, and one manifest hash. Do not edit
   config or template files to make a gate pass (`docs/ONBOARDING.md:591-611`). A literal GGUF, if
   later required by the frozen campaign manifest, remains a different artifact and cannot inherit
   this ST manifest.

5. **Classify the bytes before loading.** Run the header-only direct classifier. Require at least one
   E4M3 weight, zero per-row weights, and zero unsupported weights. Carry forward the explicit
   runtime obligations: finite positive scales, no refused E4M3 NaN codes, supported transforms,
   native residency, and native prefill dispatch (`docs/ONBOARDING.md:638-647`).

6. **Freeze the runtime and target surface.** Record runtime commit and cleanliness, binary path and
   SHA-256, host/GPU identity, driver/CUDA, exact environment, artifact/draft paths, prompt/template
   hashes, and the served context cap. The beside campaign additionally retains process commands,
   PIDs, ports, `/health`, `/readyz`, `/v1/models`, and `/metrics`
   (`research/beside-plan-20260811/PROTOCOL.md:170-187`).

7. **Pass `kernel-check` without a vacuous section.** Require final `ALL GREEN`. The current ST
   harness uses the frozen Qwen3.6 GGUF for real-weight 27B shapes and synthetic cells for both FP8
   scale classes; if Qwen3.8 introduces an unrepresented shape, STOP and add a kernel lane rather
   than accepting a skipped section (`docs/ONBOARDING.md:649-663`).

8. **Pass naked first light and the `run-gen` golden-output gate.** With the shipped defaults and the
   exact Qwen3.8 ST directory, require internal prefill/decode argmax `MATCH`, nonzero native
   `F8_E4M3`/`F8_E4M3_BLK` residency, and nonzero block-FP8 MMQ dispatch when the artifact is
   block-128. The rollback diagnostic must move the bank to Q8_0, proving that the naked result did
   not silently use the fallback (`docs/ONBOARDING.md:665-721`). Generate an authoritative greedy
   reference from the same exact FP8 directory, prompt bytes, rendered template, thinking mode, and
   decoding settings; require exact prompt-token and generated-token identity, not merely close
   logits (`docs/ONBOARDING.md:726-747`).

9. **Pass the execution-surface gates.** Require chunk invariance on the frozen chunks, the ST serve
   battery, and the live default/off/on Qwen thinking mapping. A new window or geometry class needs
   newly calibrated chunk/tick/B>1 gates rather than inherited Qwen canaries
   (`docs/ONBOARDING.md:749-767`, `docs/ONBOARDING.md:796-869`).

10. **Freeze and validate the exact drafter pairing.** If the artifact contains embedded MTP, hash
    those files/tensors and run the exact trunk/head pair. For an external draft, first prove the
    trunk/donor interface fields, derive ranks only from Qwen3.8's own templated generations, retain
    corpus/rank hashes, build and hash the draft, and keep the donor's role byte-verbatim and narrow
    (`docs/ONBOARDING.md:871-909`, `docs/ONBOARDING.md:911-945`). Require `run-spec` K=1..8
    self-consistency on that exact pair and record the short plus agentic-long adjacent A/B evidence
    (`docs/ONBOARDING.md:947-1006`). Never attach the frozen Qwen3.6 draft merely because dimensions
    look compatible.

11. **Freeze the scored manifest before handoff.** The manifest must name the exact artifact kind
    (ST, or a separately authorized GGUF), source id/revision, every full file hash, quantization
    encoding and scale class, tokenizer/chat-template hash, context cap, exact validated draft and
    hash, runtime commit/binary hash, prompt/settings hashes, and all raw gate paths. Require
    `kernel-check ALL GREEN`, `run-gen` argmax `MATCH` plus exact HF token identity, and `run-spec`
    K=1..8 PASS. The project defines those three as the real correctness battery
    (`CLAUDE.md:162-167`). **Until all three are green for the exact frozen object, it cannot feed any
    scored evaluation or the beside-Step A/B.** A plain-only first light is diagnostic, not a scored
    artifact in this ledger.

## First-boot measurements required by beside-math

All memory quantities below are recorded in bytes and MiB; never infer them from file size. Use the
exact frozen campaign path—plain or validated-spec—and do not mix measurements between paths.

The governing equations are (`research/beside-math-20260811/VRAM.md:68-85`):

```text
M38 = measured card-0 B-idle used-memory delta from Arm A
S38 = KV38(ctx=32768, selected path) + R38
H38(step_cell, c) = 50,672 - M38 - c*S38 - T38 - DeltaStep(step_cell)
c_max = floor((50,672 - M38 - T38 - DeltaStep(step_cell)) / S38)
```

| Unknown / receipt | Exact first-boot measurement | Required raw evidence |
|---|---|---|
| Exact object prerequisite | Freeze target and validated-draft revisions, file sizes and SHA-256s, artifact/template hashes, runtime binary hash, selected plain/spec path, and `MEMRA_CTX=32768`. | Scored artifact manifest plus secret-redacted launch environment. The protocol requires exact identity before execution (`research/beside-plan-20260811/PROTOCOL.md:20-24`, `research/beside-plan-20260811/PROTOCOL.md:72-95`). |
| Arm-A starting snapshot | Re-establish or validly adopt Step's post-warm-up card-0/card-1 used/free MiB and per-process use. The accepted arithmetic anchor is 50,672 MiB free on card 0, but adoption requires the same host, GPU/power regime, runtime/binary, Step hashes, environment, prompts, cache state, and raw receipts. | Step load-plan line, NVML snapshot, process list, environment, `/metrics`, and adoption checklist (`research/beside-plan-20260811/PROTOCOL.md:37-42`, `research/beside-plan-20260811/PROTOCOL.md:159-168`). |
| `M38` whole incremental resident cost | On the unchanged Arm-A state, start the exact Qwen3.8 process, load target plus the validated draft if selected, apply the pinned prefix-cache setting, perform the fixed content warm-up, then measure the card-0 B-idle used-memory increase over Arm A. This one delta includes target, draft, CUDA context/Engine/static allocations, prefix-cache residency, and the allocator's warmed state. | Card 0/1 used/free and per-process bytes immediately before Q38, after process creation, after target load, after draft attach, and after warm-up; allocator used/cached/reserved and pool-entry counts at every milestone. Components diagnose `M38` but must not be added to the whole delta again (`research/beside-math-20260811/VRAM.md:77-80`). |
| `KV38(ctx=32768, path)` | Capture the loaded-model admission value for exact KV bytes/token for the selected plain or validated-spec path and multiply by exactly 32,768. If both paths may be retained, record separate `KV38_plain` and `KV38_spec`; never reuse Qwen3.6's 33,408-byte proxy. | Raw server admission/log line, exact KV K/V formats, model-derived shapes, context cap, and calculation in bytes/MiB. The target value is currently unknown because it is model- and path-derived (`research/beside-math-20260811/VRAM.md:61-63`). |
| `R38` fixed per-session residual | Create a known fresh session, observe the effective-free delta after its KV allocation, and record the runtime's exact learned fixed residual in bytes. Do not use a zero-decimal `MB` log as the value. | Exact metric/counter before and after allocation, session id, pool state, and the corresponding raw log. The historical rounded `308 MB` receipt was explicitly insufficient for exact arithmetic (`research/beside-math-20260811/VRAM.md:110-113`). |
| `S38` full-cap session cost | Compute `S38 = KV38(32768, selected path) + R38`. Confirm it by creating one, then two, retained full-cap sessions and comparing the effective-free deltas; explain allocator granularity rather than replacing the model-derived value silently. | Per-session allocation trace, allocator used/cached/reserved, retained-entry count, and card/process deltas. The runtime pool limits are occupancy caps, not preallocation (`research/beside-math-20260811/VRAM.md:116-120`). |
| Prefix-cache residency | Pin the exact `MEMRA_PREFIX_CACHE_MB` value and measure realized resident bytes after load and after warm-up. Do not inherit the historical zero setting or leave the manifest placeholder unresolved. | Launch environment plus allocator/NVML deltas with prefix cache empty and initialized. The scored target setting is currently unknown (`research/beside-math-20260811/VRAM.md:61-61`). |
| Allocator pool reserved/cached/used | Record exact bytes for allocator `used`, `cached`, and `reserved` after Q38 load, after warm-up, after each session rung, at every peak, and at the final floor. Treat them as overlapping counters: cached bytes already inside reserved bytes, and the whole process total already contributes to `M38`. | `/metrics` snapshots and per-process NVML at each named milestone. Beside-math warns that file bytes cannot recover allocator alignment, warmed arenas, cached blocks, or learned residual (`research/beside-math-20260811/VRAM.md:87-91`). |
| `T38` transient reserve | Record whether the frozen path is spec-capable or plain and apply the runtime's path-specific transient rule. The spec-capable floor is 1,536 MiB; plain charges the lesser of its request cost and that floor. | Frozen path/draft setting, admission trace, and peak allocation receipt (`research/beside-math-20260811/VRAM.md:64-64`). |
| `DeltaStep(step_cell)` | For every overlapping Step cell, measure card-0 Step-related used-memory increase relative to the accepted starting snapshot; keep card 1 as the control. This is cell-specific, not one reusable scalar. | Continuous card 0/1 trace plus per-process samples around every Step request and Q38 `c` rung. The protocol requires post-load, post-warm-up, peak, and final-floor values in every load cell (`research/beside-plan-20260811/PROTOCOL.md:65-68`, `research/beside-plan-20260811/PROTOCOL.md:292-300`). |
| `H38(step_cell,c)` and `c_max` | Solve only after the exact `M38`, `S38`, `T38`, and cell-specific `DeltaStep` are recorded. First prove steady Q38 `c=2` with two persistent namespaces, then probe only `c=1/2/4/8`; stop on the captured failure rules. | All operands, calculations, session namespace/count, peak/floor snapshots, and raw stderr/exit status (`research/beside-math-20260811/VRAM.md:173-192`). |

For the complete first-boot window, retain a continuous 500 ms trace of both cards' clocks, power,
temperature, utilization, and used/free memory, and list every allowed process. Co-resident cells are
controlled overlap, not a clean window (`research/beside-plan-20260811/PROTOCOL.md:194-207`). Capture
stdout and stderr to raw files before parsing; an uncaptured death is not an OOM conclusion
(`research/beside-plan-20260811/PROTOCOL.md:189-192`).

The readiness verdict changes only when the exact target/draft manifest, all three golden-output
gates, and the target-box measurements above exist. Until then, the Qwen3.6 arithmetic remains a
planning proxy and the exact Qwen3.8 fit remains **UNKNOWN — needs measurement**, not “shown not to
fit” (`research/beside-math-20260811/VRAM.md:149-154`,
`research/beside-math-20260811/VRAM.md:194-195`).
