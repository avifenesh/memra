# WP-B day 8 — native active copy/restore, no observed reclaim

Repository: **avifenesh/memra**, `lane/spill-b-20260919`. Development correctness
cell only; no merge, release, serving registration, performance/default decision,
NVMe result, or model support promotion.

## Verdict (verbatim)

```text
ACTIVE-8K copy/restore bit-identical, no reclaim — not G1 PASS
```

The native cell ran to completion (exit 0). The collector correctly retains
`executed-not-qualified`, `qualification=false`. The suspended native K/V state
was copied to pinned host memory and restored; continuation is byte-identical to
the frozen baseline. **Device free VRAM did not increase on demotion or decrease
on restoration. Accounting release is not evidence of physical reclamation.**
No reason for the zero driver delta is inferred from this cell. Investigation of
allocator reservation, retained ownership and trim behavior is the next boundary.

The instruction was to run 32k only after 8k G1 PASS and stop on a verdict.
**32k was not run.** This is not a refusal or a failed continuation test.

## Actual native evidence

Directory: `rented-5090-20260919/day8-active-8192/`.

| Observation | Actual result |
| --- | --- |
| Context | 8064 prompt positions suspended; 128 subsequent generated ids; committed=8192 |
| Model/program | Qwen3.8-27B NVFP4+Q5K artifact; native tokenwise `decode_step_h`, trunk only, no MTP; original native q8_0 K/q5_1 V |
| Demote engagement | **32** K/V planes across **16** full-history layers |
| Reload engagement | **32** planes returned to the original native cache operand type |
| Source allocation bytes | **243,269,888** |
| Logical D2H / pinned bytes | **239,468,544** |
| Backend device charge after demote | **0** (not a driver free-VRAM observation) |
| Driver free bytes before | **17,934,516,224** |
| Driver free bytes after demote | **17,934,516,224** |
| Driver free bytes after restore | **17,934,516,224** |
| Reclaimed / reacquired delta | **0 / 0 bytes** |
| Pool trim released | **0 bytes** |
| Source retirement | `retire_source(ticket)` followed by release of retained device lease; host image lives until H2D source retirement |
| Restore ownership | `take_device(&DeviceLease)` hand-back after consumer retirement; transfer budget returns to zero |
| Power envelope | **400 W limit / 600 W maximum**, all collected samples |
| Protocol | Canonical `/tmp/memra-5090.lock`, collector, 250 ms telemetry, idle compute-app snapshots before/after |
| Scope | N=1, restricted-power RTX 5090 development; no balanced thermal/performance claim |

The source-retirement and hand-back calls are in the native-tested gate source;
the actual transfer/accounting counters are in `receipt/active-reclaim.txt`.
`receipt/active-bundles.tsv` binds every plane to a serialized typed bundle hash.
The raw native stdout, sampler CSV, lock metadata and collector journal are retained,
not reduced to a summary. Filesystem remains **overlay, development, not spill speed**.

### Continuation comparison

Against `day6-baseline-8192/receipt/`, published at **ce447c98**:

| Byte surface | Matching SHA-256 |
| --- | --- |
| Prefix and restored-prefix state manifests | `ce48492d782d34cb4562cd11ca6d8b610f43efd50137f36da50f93f08b49dace` |
| Final state manifest | `ec49a40a42f7336a53f2d606cfc2167a31a5d4e0edcd15f3c5032dad8d412af4` |
| 128 generated token ids | `c7f6b2cd4fb42a2008d96832dea5260212f64a7f8397d55c9ae7b3c2af13f6ea` |
| All 129 decision/final logit rows | `7ec02bb7052e2083ec120d7491d87d6b2b83af41528c2f82a7efb9166bd135b0` |
| Full final logits | `6df198aea3b5c62ef75e87429bbbb68b95570c1d58418443e94016ed88f0cb17` |
| Prompt bytes | `079672d3960a7bd57cc472c9254cb9a0b58ea3aec1e9908ca805a0066c70677f` |
| Serialized plan debug capture | `8fc1154223424f168e69fdf10f4befec7a9af75544e385bb773759954ff427e5` |

Artifact SHA-256: `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
Native source: **c95691692949561dddcf7e0e49bd6c6ba778036b**.
Native binary SHA-256: `4ca5b5057b2505c32ed827a6dff835fe4542681331ac3229a5921b4489543e30`.
The baseline uses an older binary; artifact, plan, prompt and numerical program
identities and all captured continuation bytes match. This is not a same-binary
paired performance experiment.

## Changes and publication sequence

- **5e979a97**: merged A's final **06fda471**, including final conformance/pointer
  identity tests and native receipts, then pushed. No earlier A tip substituted.
- **c9569169**: gate-only demote/reload counts and post-restore free-VRAM evidence;
  preserve copy/restore evidence even when physical reclaim is unobserved. Both
  CLI contexts can dispatch to host active capture; 32k is eligible, **not tested**.
- **df9f6600**: native build and `cargo test --release -p memra-kv -p memra-tier`
  receipts synced, committed and pushed before the model cell.
- **3f3330ea**: full 8k receipt and negative reclaim verdict synced, committed and
  pushed before any further cell (none ran).
- **d5bf6ce7a676e1a15fba15b82c32016e75158f1d**: metadata-only correction after
  discovering the inherited pre-execution `identity.txt` still wrote
  `engaged_tier_bytes=0` and `scope=baseline-only` for active runs. Future identity
  headers say execution pending and refer to completion/engagement receipts.
  **The native raw receipt is not rewritten**: its stale identity scope/count
  fields are not the engagement evidence; `ACTIVE.txt` and `active-reclaim.txt`
  carry the measured active result. This string-only correction was CPU-checked
  but **not rebuilt/run natively** after the ControlMaster disappeared.

D's explicit refusal contract was read at **7f7bf547**. Existing CLI tests verify
`REFUSED: <reason>` is the final-line diagnostic and ordinary errors are not
promoted to refusals. The active cell itself did not refuse. No new environment
read, numeric program, runtime default or external dependency was added.

## Verification actually run

| Check | Source/scope | Result |
| --- | --- | --- |
| Native release gate build | c9569169, `cargo build --release -p memra-engine --bin kv-tier-gate -j 16` | PASS; 3m52s build log |
| Native release tests | c9569169, memra-kv + memra-tier | **243 passed**, 0 failed |
| GPU active cell | c9569169 binary above | Copy/restore matches; **zero observed reclaim**, not G1 PASS |
| `cargo fmt --all -- --check` | d5bf6ce7 | PASS |
| Mac all-target check | memra-kv + memra-tier, offline, d5bf6ce7 | PASS |
| Linux-target all-target check | same crates, `x86_64-unknown-linux-gnu`, offline | PASS; cross-check only, not execution |
| Mac `cargo test ... --offline --no-fail-fast` | memra-kv + memra-tier | **243 passed**, 0 failed |
| Scoped clippy `--all-targets --no-deps -- -D warnings` | memra-kv + memra-tier | PASS; not whole-engine clippy |
| Standalone gate CLI tests | `rustc --test`, including refusal diagnostic | **4 passed** |
| `git diff --check`; `bash tools/check-flags.sh` | d5bf6ce7 | PASS |
| `python3 research/spill-b-20260919/verify-day8.py` | local archived evidence | PASS: 50 hashed files, collector/CELL validation, baseline comparison, 5 negative verdict arms |

`day8-checks/commands.json` records exact commands, source, exits and compressed/raw
log hashes. `day8-checks-pre-metadata/` preserves the earlier green pass at
3f3330ea. `day8-manifest.json` seals both check sets, native build and GPU receipts.
The replay verifies integrity; it does not re-execute CUDA or qualify a serving path.
Full engine/server suites, PRO-pair battery and 32k were **not run**.

## Hygiene, remaining blocker and budget

Only B's local lane and native B scratch worktree were changed. Main, other lanes,
model artifacts and the shared native checkout were untouched. No hooks skipped;
each step was pushed with `core.hooksPath=tools/hooks`.

The native B scratch initially had an older detached revision and an untracked
B day-7 receipt directory that conflicted with already tracked receipt files.
It was preserved as `.day8-preserved-day7` inside that scratch; `diff -qr` confirmed
it byte-identical to the newly checked-out tracked directory before building.
The ControlMaster disappeared at final cleanup. The mandatory `ssh -O check`
failed, **no fresh connection was attempted**, and cleanup of that redundant
preserved copy remains with the lead after access restoration. The day-8 receipts
were already synced and pushed. Local task scratch is removed at close; the B lane
stays open for lead integration, not merged or abandoned.

Elapsed for this relaunch: approximately **0.4 agent-hours**, within the
**10-agent-day** WP-B budget. Prior sessions' cumulative time is not reconstructed.
Next required result is **observed physical source reclaim and restore re-acquisition**
at 8k, not another byte-copy result. Only a genuine 8k G1 PASS admits the 32k cell.
