# cx-prefixmoney progress

## 2026-08-12 — started

- Lane: `lane/cx-prefixmoney` at base `8b2ba8c883152fdbb9f9bbd800a055ad03fe80c4`.
- Scope: establish the quoted Step PP-2 prefix-cache state, run the smallest safe cache-on PP-2 exactness/timing gate on the local RTX 5090, fix only a proven config-level unwiring, and specify the later box1 battery.
- Constraints recorded: do not touch box1; no merge, tag, push, board update, formatting sweep, clock change, or hook bypass.
- Current checkpoint: reading the required history and tracing the cache/PP-2 gates before selecting a local model and command.

## 2026-08-12 — history and steering checkpoint

- Owner steering makes cache-hit concurrency a first-class deliverable beside hit-path TTFT: the later pair battery must quantify how much 90%-cached prompt traffic changes admitted concurrency and completion goodput versus cold recompute.
- Quoted architectural refusal: Step35 sessions with `MEMRA_SWA_RING=1` reject flat-history prefix snapshots/restores; the stock ring smoke is scoped red only in its prefix-cache accounting cell.
- Quoted transport refusal: `MEMRA_PP_HOST_BOUNCE=1` disables cross-device prefix and plain-affinity snapshots because those copies still go through the primary engine rather than the explicit bounced activation boundary.
- Native-peer Step PP-2 is not generally unwired: the 2026-08-08 fanout receipt recorded one computed 1,024-token prefix plus seven cached requests and an 82.7% p50 TTFT reduction on Step-3.7 PP-2. That receipt did not provide the required cached-vs-cold output-byte identity under the later dual-PP default.
- The dual-PP soak recipe explicitly sets `MEMRA_PREFIX_CACHE_MB=0`, but its script, report, and introducing commit provide no quoted reason. Therefore the dual-PP/prefix-cache intersection is classified as untested by that battery, not refused or broken.
- The local RTX 5090 initially had another flocked GPU job. The selected first vehicle was the 15.7 GB Qwen3.6-27B GGUF, the largest known local PP-2-capable artifact.

## 2026-08-12 — local 5090 gate stopped by the binding PP refusal

- The release server built successfully with CUDA 13.1 / auto sm_120a. The raw build log is `raw/build-server.log`.
- Under the exclusive `/tmp/gpu5090.lock`, the 27B model loaded on the 24,463 MiB RTX 5090 Laptop GPU with same-device PP-2 (`MEMRA_PP_DEVICES=0,0`), native defaults, ring off, spec off, and a 512 MiB prefix-cache budget. The server explicitly logged `[prefix-cache] on: budget 537MB`.
- The first 528-token request then returned the binding error: `prime chunk pipeline refused with 2 stage streams on one device — that concurrent-stream placement remains quarantined by the deferred pp flake record. Use one device per stage or MEMRA_PRIME_PIPE=0 for the serial split.`
- Per the lane instruction, the suggested serial-split escape was not used. A smaller model would encounter the same placement predicate, so no fallback model was run. This is a PP prime-pipeline refusal before cache insertion, not evidence of a prefix-cache mismatch.
- Consequently the local timing/exactness sample size is N=0: there is no hit-path timing delta to report. The raw receipt is `raw/local5090/`, including the error, config, artifact hashes, metrics, and before/after GPU snapshots.
- The deferred box1 battery now makes the missing intersection explicit: native-peer devices 0/1, Step-3.7 trunk+drafter loaded, ring/host-bounce/spec off, 4096 MiB cache, production prime pipeline, repeated and shared-prefix byte identity, plus an interleaved 0%-vs-90.002%-hit concurrency ladder.

## 2026-08-12 — harness validation

- `DOCS_RS=1 cargo test -p memra-server prefix_ -- --nocapture`: PASS, 13 passed / 0 failed / 178 filtered. Raw output: `raw/host-prefix-tests.log`.
- Python compilation and Bash syntax checks pass for both clients and both runners.
- A stateful local fake-API protocol smoke passed both clients end to end: the exactness client observed the intended two misses, LCP learning, four hits, byte identity, and a dual-slot pair; the capacity client completed interleaved cold/hit c=1,2 cells and emitted a PASS summary. This validates harness control/accounting only and is not model or performance evidence.

## Acceptance gates

- Repeated-identical and shared-prefix cache-on runs produce the same output bytes as their cache-off goldens.
- Raw logs preserve cache activity, refusal text, timing, commands, model identity, and GPU/process state.
- `REPORT.md` distinguishes quoted refusals, unwired paths, and untested paths without inference.
- The final commit contains only this lane's evidence, report, and any narrowly proven fix.
