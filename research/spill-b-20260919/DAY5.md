# WP-B day 5 — native patch passes; 8k capture gate refuses

Repository **avifenesh/memra**, branch **lane/spill-b-20260919**. Native tested scratch
source: **b569164bf9e57aaebcdfbaf5c457ed683e10ff68**, plus the exact unchanged
**HOSTPREFIX-PATCH.diff v2** applied ONLY in `/root/wt-b`. Source/patch/binary hashes:
`rented-5090-20260919/native-r1/{source.commit,hashes.sha256}`. B lane runtime remains
unapplied. No main merge, release, deployment, support promotion or active-tier
qualification is claimed.

## Result

- HostPrefix v2 **compiled natively without edits**, as server release binary and test
  harness. Full native server suite: **711 passed, 0 failed, 5 existing ignored**.
- Existing legacy prefix **identity, tiny-pool teeth, corruption and pinned-allocation
  refusal gates passed**. This exercises legacy `tier=None`, not the generic injected
  governor/bootstrap or active materializer.
- New `kv_tier_gate` **build, three CLI tests and native clippy -D warnings passed**.
- Requested collector baseline **ran but FAILED closed at prefix capture**, after all
  8064 prompt tokens. No generated-token/state-logit success bundle exists. It is NOT
  an 8192-committed or active-8k pass. No format fallback or numeric override was used.
- B admission now calls D's actual exported CPU capacity fixture and continues into
  B's local materializer/retirement test using **one governor**. CPU interoperability,
  not CUDA/PCIe, is proven.

## Commits and publication

All milestones were committed and pushed through `tools/hooks`; no bypass or skip.
Important commits:

| Commit | Milestone |
|---|---|
| `fb870849` | Requested day-4 integration merge; memra-kv keeps workspace memra-tier dependency |
| `59bc443b` | B→D CPU seam, directed refusal/reverse grants/stale epoch/fenced materialization/Busy retirement |
| `6e3c67d6` | Native gate source, CLI tests, manifest fragment, verification runner |
| `9a8760aa` | Initial CPU receipts and interruption boundary |
| `b569164b` | Merge current integration `cc2a7df638fc4ba84d5545c7b0685a60a10229cf`; remote scratch base |
| `1c74f18f`, `c5bc4318` | Native patched server build and test-compile receipts |
| `db286e2d`–`d070e21e` | Six CPU prefix/host test filters, 33 tests |
| `9ed0d675` | Engaged legacy identity gate |
| `a98bd29d`, `2f4ba549` | Tiny-pool original assertion failure, then intended global-pool diagnostic pass |
| `98826414` | Three legacy fault cells pass |
| `b7d9d01c`, `3c49b2e0`, `c0dea8cf` | Native gate build/CLI tests/clippy |
| `fb6c4dee` | Full native server suite under collector |
| `8bcdb3c6` | Baseline refusal, prompt/plan/binary binding and collector telemetry |

A few SSH/rsync and two GitHub pushes failed transiently. Retries were bounded; a failed
connection was never treated as state change. Every completed cell was synced before
starting another GPU cell and its receipt committed/pushed. Initial interrupted-rental
receipts and setup failures remain in the ledger; see ACCESS.md. Final handoff names the
subsequent documentation-only tip and independently queried remote SHA.

## Native commands actually executed

All raw logs are losslessly gzip archived, with raw/archive hashes in
`rented-5090-20260919/raw-manifest.json`. `verify-rented.py` checks these hashes and the
collector's unchanged descriptor trees by decompressing in memory. Decompress `.log.gz`
back to the original `.log` filename before using other raw collector tooling.

| Command / cell | Exact result | Receipt under rented-5090-20260919/ |
|---|---|---|
| `cargo build --release -p memra-server -j 16` | `Finished release profile [optimized] target(s) in 3m 55s` | native-r1/server-build.log.gz |
| `cargo test --release -p memra-server --no-run -j 16` | `Finished release profile [optimized] target(s) in 23.03s`; lib + main test executables | native-r1/server-test-compile.log.gz |
| CPU filters: prefix_cache_, host_cache_, host_purge_, host_image_, host_handoff_, prefix_restore_plane_preflight | 17 + 6 + 1 + 1 + 7 + 1 passed; zero failures/ignored | native-r1/*.log.gz |
| Full server lib test executable, `--test-threads=1`, through tier-battery collector | `711 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out` | server-full-tests/ |
| Legacy host-prefix identity, device cache 256 MiB | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | legacy-identity-r3/ |
| Legacy tiny-pool teeth, 256 MiB device cache, tenant cap disabled for this global-pool cell | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=1)` | legacy-teeth-r2/ |
| Legacy pool-full, flip-demote digest corruption, alloc-fail | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | legacy-failures/ |
| `cargo build --release -p memra-engine --bin kv_tier_gate -j 16` | `Finished release profile [optimized] target(s) in 44.84s` | gate-build/ |
| `cargo test --release -p memra-engine --bin kv_tier_gate -j 16` | `3 passed; 0 failed; 0 ignored` | gate-tests/ |
| `cargo clippy --release --offline -p memra-engine --bin kv_tier_gate -j 16 --no-deps -- -D warnings` | exit 0; `Finished release profile [optimized] target(s) in 3m 29s` | gate-clippy/ |
| `tier-battery.py --rig rtx5090 --timeout 1200 ... --execute .../kv_tier_gate --artifact .../Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --case baseline --context 8192 --tiers host,nvme --same-program --out .../receipt` | exit 2, explicit capture refusal, no timeout | baseline-8192/ |

The native build used repository-pinned Rust 1.97.1 and nvcc selected at
`/usr/local/cuda/bin/nvcc`, `MEMRA_CUDA_ARCH=120a`. The full server suite's five ignored
cases remain **unrun**: external proxy fixture, one-device host generic arena, two-device
host GLM, DFlash retained GPU fault matrix, and mixed-load admission requiring ≥32 GiB
free. Existing CPU/template fixtures do not represent execution of a paused model.

### Failed setup cells retained

1. `legacy-identity/`: port guard refused because neither ss nor lsof existed. Installed
   the required **iproute2 observation utility**, not an engine/model dependency;
   installation output retained in native-tools/ and ss version checked before retry.
2. `legacy-identity-r2/`: default 1024 MiB device cache held both ~159.7/159.6 MB seeds,
   so eviction/demotion did not engage. Measured sizes justify 256 MiB (one fits, two
   do not); no model format or numerical program changed.
3. `legacy-teeth/`: default 50% tenant share cap refused before the gate's expected
   global-pool message. All zero-demotion/promotion, cold-output and byte-identity checks
   passed but the named-message assertion failed. `MEMRA_KV_HOST_TENANT_PCT=100` targets
   the intended global-pool fault; the original failed receipt stays failed.

### Lock and telemetry boundary

GPU idle/no collector/model process was checked before every cell. Lead expressly
approved the legacy scripts' **internal canonical** `/tmp/memra-5090.lock` and their
current global memra-server teardown on this dedicated **non-serving development box**.
They were NOT nested inside the collector, did not bypass a lock and did not use a third
lock. Each legacy receipt has `CELL-NOTE.json` with `lock=internal-canonical`.

Full server tests and native baseline used the collector's canonical lock and 250ms
telemetry. Legacy scripts do not provide that telemetry envelope; no performance claim
is made. Filesystem: **overlay, development, not spill speed**. Host/NVMe active tiering,
NVMe ancestry and PRO-class qualification remain unproven. `LEGACY-LOCK-PROPOSAL.md`
hands WP-D the requested text-only external-lock design, without changing scripts.

## Baseline refusal: exact boundary and next fix

Verbatim raw error:

```text
kv-tier-gate: REFUSED: capture requires full-history native q8_0 K/q5_1 V, no ring
```

Collector metadata uses its generic `died, cause unknown — repro needed` fallback;
that fallback is NOT the cause here. The captured explicit refusal is authoritative.
The immutable receipt has the 8064-token prompt, exact plan, artifact hash and binary
hash, but **no BASELINE.txt, generated-token stream or completed state/logit hashes**.

Static inspection identifies a likely gate-only issue: recorded plan has MTP layer
index 64; `pp::new_cache` allocates `cfg.n_layer` including NextN while the chosen
`load_without_mtp`/decode_step_h baseline executes only the trunk. Gate capture currently
requires every allocated KV layer's len == cache.pos; the existing server's
`zero_length_nextn_layer_is_absent_not_corrupt` test explicitly handles this case.
The raw error did not record layer/geometry, so this is a source-backed diagnosis,
not a captured failed-predicate proof. Next iteration should represent **plan-declared,
unexecuted** MTP planes as absent, retain strict trunk completeness, and emit exact
layer/geometry refusal diagnostics. Do not change the KV format or set a fallback flag.
Per stop-at-baseline instruction no additional model cell was launched.

## CPU/integration verification

`day5-checks/` at `6e3c67d6`: all 14 checks pass, including Mac and Linux-target checks,
224 tests/doctests plus three CLI tests, scoped clippy -D warnings, fmt, flags, patch
applicability, runtime-unapplied and shared-file-unchanged checks.
`day5-post-archive-checks/` at `f613c3dd`: all 14 pass again on updated integration.

`day5-integrated-checks/` preserves an intermediate diff-check failure caused by raw log
blank lines/CRs (and an inherited lead raw log), not a code/test failure. Own logs were
losslessly archived, NOT normalized; lane diff now uses exact current integration
`cc2a7df6`, excluding unrelated inherited work. The separate Mac engine attempt failed
before gate compilation due to missing nvcc; native results above supersede that build
blocker, not its historical receipt. Existing Darwin AsRawFd dependency warning stays
out of scope. Final documentation/hygiene verification is recorded at handoff.

## Numbered blockers / lead decisions

1. **Baseline capture contract:** diagnose and explicitly map inactive NextN slots;
   rerun requested 8192 baseline to obtain full state/logits/tokens. Current attempt
   is a refusal, not successful baseline capture.
2. **Generic active/prefix implementation:** native materializer/scheduler/fence binding,
   nonzero active eviction/reload and prefix engagement, full continuation/serving gates
   remain absent. Active/prefix CLI cases fail closed by design.
3. **Runtime patch enablement:** canonical program identity/bootstrap, allocator-accounting
   audit, fixed-arena/rank/draft support and independent lead approval remain required.
   Compiled `tier=None` legacy success does not qualify the injected path.
4. **Hardware/workload qualification:** 32k/full-context PRO-pair ladder, native NVMe and
   peer transport, measured policy/frontier and serving crossings are unrun. No default
   or performance decision follows from these single correctness cells.
5. **Legacy collector integration:** WP-D/lead may implement the external-lock proposal;
   today's explicitly approved internal-lock exception is not a general shared-box API.

The B lane/worktree remains open for lead integration. Remote scratch cleanup and exact
final remote SHA are recorded in the closing handoff. Only B source/tests/receipts and
requested integration merges were staged. Time: about **0.6 agent-hours** in initial
continuation plus about **1.3 agent-hours** in the replacement-rig continuation; WP-B
budget is **10 agent-days**. Prior agents' actual hours are not invented or re-estimated.
