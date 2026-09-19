# WP-B day 4 — CPU milestone pushed; native patch unapplied

Repository **avifenesh/memra**, `/Users/avifen/tiyuvta/wt-spill-b`, branch
**`lane/spill-b-20260919`**. Exact tested implementation/runner tip:
**`02483bd499728b73557a8aac50f23453678c5914`**. The receipt-only follow-up adds this
report and raw archives, not runtime code. No main merge, release, deployment,
GPU/model execution, support promotion or native qualification is claimed.

## Re-entry, commits and push

| Commit | Milestone |
|---|---|
| `09bc15b40b3b1ea87c4a1823dee556dd1e4a6521` | First requested `git merge --no-ff lane/spill-integ-20260919`, integration `020d2047`; clean initial lane, no conflicts. |
| `99965e80` | B peer-capacity admission helper, second opaque materializer, per-request restore-policy binding, program-bound residency guard and tests; D export request. |
| `3ffc27f1` | Merge lead-owned publish-census fix `200a3c66`; B did not edit shared workflow/contracts. First successful lane push through hooks. |
| `f1a2729c` | HostPrefix v2 unapplied patch/review and ordered runner; pushed and remote SHA queried. |
| `02483bd4` | Load-winner admission/retirement and epoch-zero mandatory-state regression; pushed and final full CPU battery run. |

The initial required push was refused, without bypass:

> publish-census: publishable workspace member(s) memra-tier absent from the crate list in .github/workflows/publish.yml

Reported to lead; merged their `200a3c66` fix once instructed. Subsequent pushes
ran `tools/hooks` (configured), perf board, flags, releasability, docs registry and
public-boundary checks. No `--no-verify` or skip environment was used. First repaired
push additionally printed the inherited hook limitation:

> pre-push: NOTE — engine files touched but no model dir; perf-ci not enforced here.

That is NOT a performance pass. B did not change the hook or claim GPU evidence.
`git ls-remote origin refs/heads/lane/spill-b-20260919` after the implementation
milestone returned exactly:

```text
02483bd499728b73557a8aac50f23453678c5914	refs/heads/lane/spill-b-20260919
```

The final handoff names the subsequent receipt-only commit and independently queried
remote SHA. Branch remains open for lead integration; no other lane/main was edited.

## Deliverables and scope

1. **B side of PeerCapacity:** `integration::reserve_peer_source` validates identity,
   live-state metadata, directed lookup source/consumer, byte geometry and source
   charge dimensions, then calls the frozen trait. Source-generation allocation is
   explicit; local destination admission stays separate. B's CPU fake runs advisory
   peer lookup → context/pool/link-denial refusal → independent reverse grants →
   source reservation → exact CPU copy → local fenced materializer → Busy/retry →
   zero final shared-governor usage. No automatic host bounce or remote-attention
   pointer. **D's actual fake remains private** and unchanged. `D-SEAM-REQUEST.md`
   specifies the minimal export; B-only schedule is not B→D interoperability evidence.
2. **Format-agnostic binding:** `PackedMaterializer` implements the frozen trait on
   opaque FP8-record fixtures with exact pinned RecordLayout and ProgramIdentity.
   AllPages and TrailingPages(2) round-trip every valid byte and zero padding; wrong
   identity, coverage replacement and mixed encodings refuse. Native Qwen q8_0/q5_1
   binding/tests remain intact. This is not FP8 attention or a selectable Qwen F8 KV
   program, and does not implement any new model numerical operation.
3. **Admission policy:** immutable decision + original fixture costs are retained
   on each scheduler request alongside its frozen TierAdmissionPlan (shared contract
   unchanged). Recompute winners never call lookup/admit/prefetch/load and remain
   uncharged; Load winners exercise reservation → prefetch → local ready → Busy
   retirement/retry. Explicit StateKind prevents epoch-zero active state from being
   mistaken for an optional prefix. No post-admission policy mutation API exists.
   All **48 synthetic CSV rows** feed the recorded policy tests. Inputs are fixtures,
   not measured calibration or a promoted runtime default.
4. **HostPrefix v2:** source/destination ResidentCharge retains complete ProgramIdentity;
   tenant mismatch refuses before quota changes; runtime patch injects the SAME
   governor, checks host model-generation Arc, and charges metadata capacity before
   identity publication. Patch is **UNAPPLIED**. Review covers all **19 hunks** across
   worker.rs, admit_memory.rs and worker/host_glm.rs, with explicit enablement blockers.
5. **Rig sequence:** baseline build + server tests + existing prefix identity/teeth/
   failures → patched build + same gates → baseline/patched 8k/32k fitting cells.
   Raw bytes are teed before parsing. Two CPU runner teeth check order, hashes,
   lock discipline and deliberately injected exit 7. Native qualification remains
   fail-closed; these stub commands do not compile/run a model or touch a GPU.

### Paths changed by B

- `crates/memra-kv/src/tiered/{hostprefix,integration,materializer,scheduler,tests,day4_tests}.rs`
- `research/spill-b-20260919/{D-SEAM-REQUEST.md,HOSTPREFIX-PATCH.diff,PATCH-REVIEW.md,CELLS.md}`
- `research/spill-b-20260919/{rig-cells-b.sh,rig-cells-b.py,test-rig-cells-b.py,revise-patch-v2.py,verify-day4.py}`
- This report and `day4-dev/`, `day4-checks/`, `day4-final-checks/` evidence.

No B edit to root Cargo/lock, shared contracts/tests, other lane implementation,
server/engine runtime source, kernel, new MEMRA_* read, hardware default or board
number. The shared publish workflow change was lead-owned and merged unchanged.

## Exact executed verification

Reproducer at the code tip: `python3 research/spill-b-20260919/verify-day4.py day4-final-checks`.
Use a NEW output directory on repeat: existing receipts refuse overwrite, and tracked
source must be committed. Each complete raw stdout/stderr capture is archived verbatim
before parsing; command, exit, HEAD, archive SHA256 and uncompressed SHA256 are in
**`day4-final-checks/commands.json`**. Earlier full passes at `f1a2729c` remain in
`day4-checks/`; they are not relabeled as final-tip execution.

| Command (actually ran) | Exact relevant result |
|---|---|
| `cargo fmt --all -- --check` | exit 0, empty stdout/stderr |
| `cargo check -p memra-kv -p memra-tier --offline --all-targets` | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.65s` |
| same plus `--target x86_64-unknown-linux-gnu` | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.61s`; type/cfg check only, not linked Linux test execution |
| `cargo test -p memra-kv -p memra-tier --offline --no-fail-fast` | exit 0; kv **57**, bank **35**, contracts **44**, peer **17**, placement **6**, storage **37**, doctests **4**; **200 total**, all nonempty suites `0 failed; 0 ignored; 0 measured; 0 filtered out` |
| `cargo clippy -p memra-kv -p memra-tier --offline --all-targets --no-deps -- -D warnings` | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 1.09s` |
| `git diff --check` | exit 0, empty output; integration-range `git diff --check 020d2047` also exit 0 |
| `bash tools/check-flags.sh` | exit 0; `runtime literal reads=864`; `no uncovered runtime names` |
| `git apply --check research/spill-b-20260919/HOSTPREFIX-PATCH.diff` | exit 0, empty output; shape only, NOT server compile |
| `bash -n research/spill-b-20260919/rig-cells-b.sh research/spill-b-20260919/rig-stub-b.sh` | exit 0, empty output |
| `python3 research/spill-b-20260919/test-rig-cells-b.py` | exit 0; `Ran 2 tests in 1.326s`, `OK` |
| frozen contracts/tests diff against `020d2047` | exit 0, unchanged |
| D's private `tests/peer/fake.rs` diff against `020d2047` | exit 0, unchanged |
| runtime worker/admit_memory/host_glm diff against `020d2047` | exit 0, unchanged (patch unapplied) |

Also reran `revise-patch-v2.py`, confirmed byte-identical patch output, and removed
its scratch automatically. This is source-rewrite reproducibility, not Rust runtime
compilation. Existing Darwin dependency warning remains out of scope:
`crates/memra-gguf/src/source.rs:20:5: unused import: std::os::fd::AsRawFd`.

Development red: initial opaque fixture filled storage padding with nonzero bytes;
frozen `StateBundle::verify` correctly returned **Corrupt**, before materialization.
Captured repro: `day4-dev/packed-red.log.gz`. Fixed the fixture to zero padding,
NOT the validator/contract. A later compile iteration missed one newly added
StateKind argument in a test; repaired call site before the final all-green battery.
No failure is attributed to GPU/OOM/storage hardware.

## Numbered blockers / lead decisions

1. **D test seam export:** D owner must provide the construction/owner/fence seam in
   `D-SEAM-REQUEST.md`; then run one actual B→D shared implementation schedule. Private
   D tests passing separately is not that result.
2. **Native patch review/build/bootstrap:** no local nvcc/GPU, no native server/engine
   compilation here; patch remains an unapplied source proposal. Lead must supply
   canonical identity/generation lifecycle, audit all byte accounting, and explicitly
   approve application. CI on integration is not automatically a build of this patch.
3. **Real active/prefix binding and target cells:** CPU owners are not CUDA attention
   operands. Native host/NVMe active reload, full-state/logit/token identity, graph/spec
   crossings, fixed-arena handoff and complete auxiliary state remain blocking.
   8k/32k fitting developer cells and PRO-pair 128k/262144/peer gates remain UNRUN.
4. **Measured envelope/policy:** route/copy/materialization costs, chunk sweep,
   source/artifact/binary calibration, N≥5 both-order performance and serving latency
   distributions remain unmeasured. No synthetic choice becomes a shipping default.

## Time and hygiene

Approximately **0.3 agent-hours** in this continuation (~18 minutes from first merge
through code, checks and report). WP-B's budget is **10 agent-days**; previous agents'
actual time is not re-estimated, and no hours/day conversion or GPU milestone completion
is implied. No task-created scratch worktree remains; patch/test temporary directories
are automatically removed. Only B code and receipts were staged, plus the explicitly
requested lead merge. Final `git status --short` is checked after the receipt commit.
The open lane worktree/branch remains for the lead; no merge/bank/abandon closure has
been requested, so it must not be deleted yet.
