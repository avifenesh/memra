# Native caller qualification: integrated #542 candidate

Status: executable probes are implemented; **all new CUDA cases remain UNRUN**.
The frozen `d9010b45` production correction passed independent security/performance
review. This integration/test delta requires another review before GPU execution.
Source621 native records remain historical and binary-bound.

## Integrated inputs

`integration-20260920/source-diff.json` records the integration-only tree before
these new probes. It includes main `cf4f317e68b9cedea4f4e195caee83a598db421e` and the
approved tokenizer dependency `80f0109e1051bceb1a69e531088987a407840cf3`. Relative
to d901 there are 57 changed source/build-input paths, including Gemma prime/view
code, operation-registry manifests, CudaTransfers/banked ownership, worker paths
and the Qwen2 tokenizer. This is a changed program, not receipt/head bookkeeping.

Six overlap paths were reconciled: `run_spec.rs`, `hybrid_forward.rs`, GGUF
`execution_manifest.rs`, `lib.rs`, `source.rs`, and server `worker.rs`. The GGUF
source delta is only Linux gating of `AsRawFd`; opened-source hashing is preserved.
The bank installer compares each retained expert with `host.expert_bytes()` before
attaching its approved GGUF source. INDEX conflicts retained the current spill
records, #542's pending status and the tokenizer record.

## Build before acquiring GPUs

From a clean final integration checkout, run the owned builder into a new external
directory:

```sh
python3 research/modelplan-onboarding-rewrite-identity-20260920/native_build_record.py \
  --out BUILD_DIR --cuda-arch 120a --nvcc /absolute/path/to/nvcc
```

Schema v2 binds nine production tools, including `argmax-margin-probe`,
`concat-prime-probe` and `tier-transfer-gate`, plus exact server-lib, native-repack and Gemma-prime test
executables. Test builds use a separate target directory and cannot overwrite the
production tools. Compiler inputs, commands, Cargo completion/events and executable
bytes are checked before native execution. Missing or changed test executables
refuse just like stale production tools. DOCS_RS builds are compile-only.

## Run only after the final delta is reviewed

Use the coordinator's per-physical-card wrapper, its selected UUID set, a fresh
lease receipt per attempt, and new external evidence directories. No independent
resource rental or lock-name substitution. Caller probes require exactly one
visible assigned card; the generic battery uses the designated non-serving rig.

1. Run the existing `qualify-native.py` with the final build record and pinned
   Qwen3-0.6B artifact. Its 12 scheduled commands include the common retained-eager
   executable-mapping refusal case and historical admission controls on the new
   binary. It does not inherit source621 qualification.
2. Run the bounded caller suite:

   ```sh
   python3 research/modelplan-onboarding-rewrite-identity-20260920/qualify-callers.py \
     --phase callers --model MODEL_SOURCE --build-record BUILD_DIR/build.json \
     --out NEW_CALLER_EVIDENCE
   ```

   This schedules one CPU inspection, one authentic retained-surface capture, and
   24 independent native cases. No eager receipt is copied into graph/prime or
   worker qualification. Only the artifact lock is shared across fresh worker
   bundles; each worker test executable measures its own eager receipt.
3. Run the required generic battery with the same owned tools:

   ```sh
   python3 research/modelplan-onboarding-rewrite-identity-20260920/qualify-callers.py \
     --phase battery --build-record BUILD_DIR/build.json --out NEW_BATTERY_EVIDENCE \
     --roster tools/release-roster.tsv
   ```

   The runner invokes the authoritative release battery and exact coverage checks.
   It creates only absent, verified links at `target/release`, removes only its own
   links, and refuses an existing unrelated executable. Use a fresh checkout when
   that directory is already occupied. No missing-own-model or coverage override.
4. Run the transfer correctness phase on one assigned card, using the same wrapper
   and a newly built executable from the final reviewed integration:

   ```sh
   python3 research/modelplan-onboarding-rewrite-identity-20260920/qualify-callers.py \
     --phase transfer --build-record BUILD_DIR/build.json --out NEW_TRANSFER_EVIDENCE
   ```

   This runs the exact owned `tier-transfer-gate conformance` and
   `tier-transfer-gate roundtrip` commands, without a model or release roster.
   Conformance requires all eleven native assertions exactly once, including
   source-consumer retirement and dropped-destination graph retention from the
   reviewed transfer lifetime changes. Roundtrip requires exactly one N=1 row at
   each of 4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB and 256 MiB, equal expected/actual
   SHA256, byte exactness, a freed source with live host destination, no-copy
   hand-back, and a drained governor. Missing, duplicate, malformed or incomplete
   rows and nonzero exits fail closed. Earlier eight-tool build records refuse;
   the runner cannot reuse an old executable or relabel its evidence.

   The per-card wrapper supplies the authoritative physical-card lease. The
   existing runner owns process supervision, raw stdout/stderr, 250 ms telemetry,
   cleanup, final provenance/lease checks and atomic result publication. This
   direct two-command phase checks the native gate assertions; an executed-only
   collector result is not a qualification pass. It makes no claim about the
   broader tier battery, VMM, model quality, serving parity or transfer performance.

Every case revalidates the lease while running and source/executable provenance
before/after. Environment cases run the controller in the same supervisor so lease
loss can stop its owned child, including a child suspended by SIGSTOP. Outputs,
protocol, exit status, artifact/source/build hashes, and 250 ms telemetry are kept.
The caller, transfer and battery phases have separate results; passing one does
not pass another or qualify a different serving executable.

## Actual production callers exercised

| Cases | Positive and refusal witnesses |
| --- | --- |
| GraphSession `step`, `prof_apply`, `prof_launch`, `prof_read` × library/environment | Authentic eager/graph observations in this executable; live graph creation, independent eager token comparison, ended guards, then actual named caller first after drift. Cache, resident token/position and split-argument staging must not change. `prof_apply` uses a 256-token prompt and a larger captured bucket, and requires a real observed parameter update during its positive control; a no-op cannot pass. |
| `prime_graph_run` × library/environment | Authentic CarriedPrime comparison with independent quantized-cache/ordinary-prime observations, real retained PrimeGraph and fresh destination cache; snapshot graph IO, scratch and destination state. Refuse before replay/copy/output and keep old origin revoked after restoration. |
| Worker `advance_sample_emit`, `advance_token_emit`, `step_session`, prefill form of `step_session`, `prefill_tick`, `step_session_async_chain` × library/environment | The actual private production functions run in the server lib-test executable, with a real pinned model, actual Session/LoadedModel and fresh in-executable eager qualification. Require a non-vacuous unchanged control, then no successful event/token or cache/logit/generated/fed-state mutation after drift. Error/abort metadata may change. This is not qualification of `memra-server`. |
| Native stacked and per-expert NVFP4 cache fixtures | Public HostExps loaders, cold/hit/poisoned cache, retained named writer/truncation, dropped source, actual stage_expert H2D/D2H, qmatvec_view and macro-scale output checks. Exact raw bytes/outputs and manifests are recorded. These tiny fixture transfer/arithmetic gates do not promote whole-model support. |

Each refused object is called again after restoring external state; it must still
refuse using its original origin. No fresh snapshot, validator or reinstall may
trigger refusal on the caller's behalf immediately before the tested invocation.
A genuinely changed identity, vacuous test, missing update witness, interrupted
process or absent native prerequisite fails the case.

## Real environment drift without concurrent set_var

`native_env_controller.py` launches the direct cooperative probe with MEMRA_FAST=0.
The Rust helper reports the address of that exact libc.getenv value and releases
all borrowed environment references and stdout locks. Callers first drain the CUDA
context, then the helper requests SIGSTOP. The parent waits for a completed group
stop and checks every task. Only then does it check the known byte plus NUL,
write 0→1 through `/proc/OWNED_CHILD/mem`, verify it, and resume. The second stop
restores 1→0 at the same address. No environment arrays/strings are reallocated,
no other process is targeted and no memory/environment scan is performed.

This is a Linux debugger experiment, not a portable language-level environment
mutation API. It assumes a cooperative pinned child, no environment mutators,
other debugger/SIGCONT sender, fork/exec after handshake or outside shared-VM
process. SIGSTOP alone does not stop submitted GPU work: context drains are explicit
in both probe families. The controller has bounded pipes/deadlines, exact ordered
handshakes, source/lease supervision, and kill/reap cleanup. Kernel hangs, SIGKILL of
the supervisor and escaped descendants require the external job/cgroup supervisor.

The CPU controller suite has 22 portable cases and five actual Linux process tests.
CI runs it with `--require-linux`; access denial is a failure, never a skipped pass.
If the host cannot permit parent-to-child proc memory access, use a dedicated Linux
VM/CI environment that can first pass these CPU controls. Dependency injection or a
fresh process with changed startup variables does not replace retained-object drift.

## Remaining acceptance

Native mathematical parity, CUDA state witnesses, Linux controller execution and
all 24 native cases are pending until run on the final reviewed build. Preserve
failures and stop on a real math/identity mismatch; do not widen tolerances, fake
receipts or refresh an old origin. Upstream CPU compiler/tokenizer/tier suites,
engine/server tests, formatting, the required generic battery and affected Gemma
checks remain the integration gates. No latency benchmark is needed merely to
restate the already tested nested-call count; no throughput/default claim is made.
