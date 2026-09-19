# First rented RTX 5090 hour — D day 3

**Runbook, not a rig receipt.** Minimum integration source to build: **`914229ae`**
(full `914229ae3f706c0581f5d7f7bb415a14795b1640`), which fixes the engine tier dependency
and registers `storage-bench`. Bootstrap checks that this commit is an ancestor of the
requested remote tip. D merged it at `ca974cef827564149aed1c00d749a625534b8602`.
The requested tip must additionally contain this runbook's bootstrap/collector.
No rental exists yet. Development only, never
an existing serving box. Target remains 4× RTX PRO 6000, PCIe Gen5, **no NVLink**.
One 5090 is not peer/four-tier qualification and never vetoes Step-only PRO delivery.

## Before rental / source handoff (blocking)

1. Lead must push the reviewed integration branch containing D's bootstrap/collector and
   approved shared-file amendments. This session does not push. `BRANCH` is mandatory;
   bootstrap checks the exact remote head and refuses missing/moving branches, never main.
2. **First engine/server build is blocking.** B's `_tier_identity` field in `worker.rs` has
   only CPU-side/source inspection so far. It resolves via `memra_engine::cache`, not a
   direct server `memra-kv` dependency. All approved native patches must compile on this
   same source. No CPU green substitutes for that build.
3. **The manifest defect is fixed at `914229ae`.** `memra-engine` now depends on
   `memra-tier` and explicitly registers `storage-bench`; `cargo metadata` resolves both.
   This fixes target discovery, not native compilation. The blocking first native compile
   of **memra-engine + memra-server** must validate both the new bin and B's `worker.rs` field.
4. A/B/C `rig-cells-*.sh` runners are **absent in this integrated checkout**. B/C runners
   exist on their separate day-3 branches; they have not been merged or run by D. Use the
   explicit paths below **if present** after lead integration; review their own CLI and
   lock ownership, not guessed flags or successful zero-test filters.
   The first-hour commands below use binaries that actually exist in source. A real GPU
   transfer owner, B active materializer/server adapter and C NVMe row consumers remain
   unfinished; their gates are blocked, not silently skipped.

## Bootstrap, via the operator's existing SSH identity

No keys, tokens, env files or provider credentials are embedded/read. Use an approved dev
host alias already configured by the operator; do not use a production alias. Example:

```sh
# Operator supplies DEV_HOST, BRANCH. No ssh key material is copied into the script.
: "${DEV_HOST:?approved new development host alias}"
: "${BRANCH:?pushed lane/spill-integ-* branch}"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$DEV_HOST" \
  "BRANCH='$BRANCH' bash -s -- --expected-power 600 --gap-seconds 60 --jobs 4" \
  < tools/tier-rig-bootstrap.sh
```

Use only validated `lane/spill-*` branch text in the remote command (letters/digits,
`/._-`; no quotes/metacharacters). At most two connection attempts, backoff 10 seconds;
a failed connection establishes no remote state. Read the failure receipt before rerunning
bootstrap. Do not automatically destroy/rent/reconfigure boxes from this script.

Bootstrap is repeatable: preserves dirty tracked work and old receipts; existing clean
checkout is detached at the exact requested remote head. Existing untracked receipts are
preserved. Explicit `--out` must be new. No driver/power/clock/ACS/IOMMU changes. It installs
ordinary missing Ubuntu/Debian packages; other distros work if dependencies exist, otherwise
name the missing tools and refuse. It installs/verifies rustup **stable >=1.97**; if rustup is absent it downloads the official HTTPS installer to
a temporary file and installs the minimal profile (no pipe-to-shell). CUDA resolution matches `build.rs`:
explicit `MEMRA_NVCC`, existing CUDA root, otherwise newest runnable PATH/local toolkit;
selected CUDA must be >=13 and list `compute_120a`; the accepted absolute compiler and
`120a` architecture are pinned for these builds so rustup's PATH change cannot alter selection.
Missing toolkit fails with the exact
installation requirement, not an attempted old distro CUDA or model-format fallback.

Acceptance reads **both** power.limit and power.max_limit in the first minute; both must be
>=600 W (no Max-Q/listing inference). It verifies one >=31,000 MiB RTX 5090. Under
`/tmp/memra-5090.lock`, the embedded CUDA program cudaMallocs **8 GiB**, memsets it and reads
**all 8 GiB** back in 4 MiB chunks, twice with a real >=60-second gap. NVML alone never
accepts a card. Source, binary hash, exact query output, both logs and compute snapshots
are retained. Nothing runs on the Mac GPU in `--dry-run`.

Inventory through `tools/tier-topology.py --storage`: topo matrix, full `nvidia-smi -q`
(including P2P-related fields), P2P read/write capability, lscpu full+NUMA/CPU listing,
numactl, df, lsblk JSON, free, findmnt `/scratch`. A diagnostic inventory does not prove
context/pool grants, NVMe locality, direct DMA or link bandwidth.

Builds (4 jobs, release, locked dependency graph, outside GPU lock):
- `memra-tier`, `memra-kv` libraries;
- `memra-server --bin memra-server` — **B field's first real compile**;
- `memra-engine --bin storage-bench --bin pp-transport-smoke --bin qwen4exp_gpu_gate --bin run-gen --bin run-spec`;
- `memra-engine --lib` test executable via `cargo +stable test --release --locked -j 4 -p memra-engine --lib --no-run`.

Raw destination defaults to `research/spill-d-20260919/raw/<host>-<utc>/BOOTSTRAP.json`
plus `.log`, `TOPOLOGY.json`, acceptance source/binary and `locked-run.sh`. Pre-clone
failures go to `~/spill-bootstrap-failures/<host>-<utc>/`. Keep host/raw topology identities
machine-local; sanitize names/identifiers before lead publication, retaining original hashes
privately. Bootstrap receipts are not byte-receipt schema-v1 GPU passes.

## Exact serialized sequence after bootstrap

Estimates below are **booking budgets**, not observed durations. On a CUDA-devel image with
Rust/dependencies present, target minutes 0–20 bootstrap/build, 20–25 A, 25–30 D1,
30–45 C tiny PLE, 45–60 B fitting baseline. Cold CUDA/Rust downloads or native compilation
can consume **30–90+ minutes alone**. At minute 60 stop starting new cells, preserve the
running cell's bounded timeout/result, and report the actual reached step. Never skip the
blocking build to make an invented one-hour promise. Full C tiny gate budget remains
30 minutes; full B baseline 60 minutes and active-8k 20 minutes, so their completion may
legitimately be in hour two. No performance/default decision in this smoke window.

In the remote checkout, set `BOOT` to the successful bootstrap receipt directory and
`EV` to a **new** dated D receipt directory. `QWEN` is a manifest-verified `/scratch` file,
not an hf: auto-fetch locator. Start with no concurrent compute applications.

```sh
set -euo pipefail
: "${BOOT:?successful bootstrap receipt directory}"
: "${EV:?new D receipt directory}"
mkdir "$EV"
python3 - "$BOOT/BOOTSTRAP.json" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
r=json.load(open(sys.argv[1]))
assert r['status']=='bootstrap-complete-not-tier-qualified'
assert r['cuda_acceptance']=='two-full-readbacks'
assert not r['qualification']
assert r['source_commit']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for name, pin in r['release_binaries'].items():
    h=hashlib.sha256()
    with (Path('target/release')/name).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    assert h.hexdigest()==pin['sha256'], name
PY
# Reproduce build validation if the source changed: rerun bootstrap, never reuse its success.
git rev-parse HEAD > "$EV/source.commit"
sha256sum target/release/{storage-bench,pp-transport-smoke,qwen4exp_gpu_gate,run-gen,run-spec,memra-server} > "$EV/binaries.sha256"

# A: local filesystem exactness baseline (NOT SSD→GPU performance).
# Verify /scratch is physically backed by local NVMe from lsblk/findmnt, not overlay/network.
# Create one owned directory; do not remove an existing campaign's scratch.
SCRATCH=$(mktemp -d /scratch/spill-first-hour.XXXXXXXX)
trap 'rm -rf -- "$SCRATCH"' EXIT
for size in 264 1048576 4194568; do
  python3 tools/tier-battery.py --rig rtx5090 --timeout 120 \
    --out "$EV/a-roundtrip-$size" --execute \
    target/release/storage-bench roundtrip "$SCRATCH/object-$size" "$size" buffered
  python3 tools/tier-battery.py --rig rtx5090 --timeout 120 \
    --out "$EV/a-restore-$size" --execute \
    target/release/storage-bench restore "$SCRATCH/object-$size" "$size" buffered
done

# D1: existing single-device same-context driver copy and alternating PP boundary slots.
# Clean inherited transport seams for this exact named baseline; no new flag/default.
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 \
  --out "$EV/d1-local" --execute \
  env -u MEMRA_PP_DEVICES -u MEMRA_PP_HOST_BOUNCE \
  target/release/pp-transport-smoke

# C: CPU/fixture bank tests are separate from actual GPU PLE history baseline.
# qwen4exp_gpu_gate synthesizes its own tiny checkpoint fixtures; no full PLE download.
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/c-ple-tiny" --execute \
  target/release/qwen4exp_gpu_gate "$EV/ple-tiny.tsv"

# B: fitting native checkpoint baseline, not new active demote/reload proof.
: "${QWEN:?pinned Qwen3.8-27B NVFP4+Q5_K GGUF with qualified MTP attachment}"
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/b-qwen-argmax" --execute target/release/run-gen "$QWEN"
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/b-qwen-spec" --execute \
  env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_GEN_ONLY \
  target/release/run-spec "$QWEN"
```

These commands capture merged stdout/stderr to `command.log` **before** RESULT parsing,
plus `command.capture.json`, `command.gpu.csv`, sampler stderr, and before/after/failure
compute-apps logs per cell. Native output TSV/JSONL remains adjacent; binary-specific
outputs are not rewritten into positive exactness rows automatically. Exit 0 means executed,
**not qualified**. Missing telemetry, missing actual tier counters, absent logits/state/token
hashes or insufficient thermal/window evidence forbids scoring. Failure capture quotes only
observed lines; no OOM inference without a captured OOM line and concurrent process query.
The collector owns the canonical lock across binary, sampler and snapshots. Never nest a
self-locking A/B/C shell runner under `--execute` or `locked-run.sh`.

### Cell budgets / evidence boundary

| Order | Cell / artifact | Memory and wall budget | Result requirement / raw destination |
|---|---|---|---|
| 0 | Bootstrap, no model | 8 GiB CUDA allocation +4 MiB readback; gap 60 s, each allocation timeout 180 s; build -j4 | both exact full readbacks, adequate power, exact branch, all native builds; `$BOOT/*` |
| 1 | A filesystem fixture, internally deterministic bytes | <=4,194,568 useful bytes per object; bounded 1 MiB chunks in current store; 5 min initial | roundtrip/restore exact counts/hash; `$EV/a-*`; buffered is not O_DIRECT and no H2D occurred |
| 2 | D1 PP primitive, no checkpoint | 4096/5120 f32 elements +runtime pools, <32 GB budget; 5 min | same-context `bytediff=0`, four slot roundtrips; `$EV/d1-local`; no peer pairs on single GPU |
| 3 | C PLE tiny native gate, fixture generated by binary | tiny 4-layer hidden16/expert8 geometry; reserve <=4 GiB GPU operational ceiling, stop if unexpectedly larger; 15 min first-hour slice /30 min full | PLE history +tiny loader/forward rows; `$EV/ple-tiny.tsv`, `$EV/c-ple-tiny`; does NOT prove bounded NVMe service or full-model PLE |
| 4 | B Qwen fitting baseline | 32 GiB total card; admit only measured free minus model/workspace reserve, single request; 15 min first-hour slice /60 min baseline | `run-gen` argmax and supported `run-spec` outputs; `$EV/b-*`; no invented 8k/32k context from a short-prompt baseline |
| 5 | B active-8k then prefix-8k | <=32 GiB, original q8_0 K/q5_1 V, working-set refusal required; 20 min/cell after native binding | **BLOCKED**: proposed `$GATE_BIN --artifact "$QWEN" --case active --context 8192 --tiers host,nvme --same-program --out ...` is not present; no fake substitution |

A's existing pinned-worker exact-byte test is a subsequent 5-minute cell after `--no-run`:
`cargo +stable test --release -p memra-engine --lib spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read -- --ignored --exact --nocapture`
under the 5090 lock via the collector (single run, timeout 300 seconds):

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 \
  --out "$EV/a-worker-pinned" --execute \
  cargo +stable test --release -p memra-engine --lib \
  spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read \
  -- --ignored --exact --nocapture
```

Prefer resolving the one compiled test executable and invoking it directly
when scored; avoid starting sccache/build daemons inside a held lock. This test is **not H2D**.
Linux `uncached` is **implemented but unrun here**: only macOS installs `open_uncached`;
Linux `FileBackend` instead calls `AlignedFile::open(O_DIRECT)` with aligned reads. The
inherited draft incorrectly inferred Linux Unsupported from the unused non-macOS stub.
After buffered correctness, a bounded optional A follow-up is the same roundtrip/restore
loop with `uncached` and fresh object/output directories. Filesystem rejection is a recorded
failure, never buffered fallback or O_DIRECT success. Real physical-I/O counters and native
Linux execution remain required before a direct-path claim. `storage-bench trace/replay`
are still proposed-only here.

## Artifacts to pre-stage (do not fetch in this lane)

- **Qwen3.8-27B:** `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF`, trunk
  `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` and qualified embedded/attached MTP head. Sources:
  `docs/COOKBOOK.md` Qwen section, `docs/models/qwen38-27b.md`, B `CELLS.md`.
  Lead supplies immutable revision/full byte manifest, plan, tokenizer/template, native KV
  and prompt-token pins. No guessed drafter and no switching to DFlash2 mid-request.
- **PLE/Qwen4Exp:** tiny fixture is generated locally. Full model source
  `Qwen/Qwen3.8-Flash-Next` and `tiyuvta/Qwen3.8-Flash-Next-NVFP4` mint are documented in
  `docs/models/qwen38-flash-next.md` and `research/qwen4exp-bringup-20260829/REAL-CHECKPOINT-GATE.md`.
  Short source revision in card is NOT an immutable lock; lead must supply full lock/manifest.
  Full ~174 GB mint is not a 5090 fitting model. PLE row encodings stay F32/BF16.
- **Hy3:** designated qualified HostExps artifact from C owner, original IDs/layouts/scales,
  not a newly minted/pruned arm. `docs/models/hy3.md` pins official BF16 semantic source
  `tencent/Hy3@a960ebc3da325ba167f069f76c41eb62c9280d22` and full W4A16 manifest.
  Do not equate the current four-card W4A16 qualification with the older mixed HostExps
  research arm; C/lead must select and lock the exact affected existing artifact.
- **Step:** pinned **official FP8 safetensors** from D `CELLS.md`/Step lane receipts, not
  the GGUF cookbook as a compatibility replacement. Lead supplies immutable artifact bundle.
- Durable bytes originate from owner-approved `/data`; stage hash-identically on local
  `/scratch` NVMe with complete manifest. Exact machine-local artifact inventory belongs in
  darklanes/owner handoff, not hard-coded provider or serving paths. No fetch performed here.

## After runners land / PRO-only sequence

Explicit runner paths, **if present** in the reviewed integration tip:

- A: `research/spill-a-20260919/rig-cells-a.sh` (expected handoff path; absent in D and
  in A's committed branch when checked). Keep the runnable positional storage-bench cells
  above until A supplies its exact reviewed CLI; do not fabricate trace/replay arguments.
- C: `research/spill-c-20260919/rig-cells-c.sh` (seen on C's day-3 branch, not integrated).
  Its fitting invocation is `bash research/spill-c-20260919/rig-cells-c.sh
  --non-serving-confirmed --rig 5090 --host-label development`. It builds and owns its lock;
  retain its `research/spill-c-20260919/raw/development-<utc>/` outputs and link them from D.
- B: `research/spill-b-20260919/rig-cells-b.sh` and sibling `.py` (seen on B's day-3
  branch, not integrated). It accepts `--exclusive-non-serving --artifact "$QWEN"
  --artifact-sha256 "$QWEN_SHA256" --tokens-8k "$TOKENS8K" --tokens-32k "$TOKENS32K"
  --fit-plan "$FIT_PLAN" --out "$EV/b-fitting"`. These input files and exact fit envelope
  must come from B/lead, never invented from available VRAM. Without the future
  native `--active-gate` it records active cells **BLOCKED**, not qualified.

A/B/C owners supply dependency amendments and lock ownership in their `CELLS.md`.
Record each script hash and integrate in **A → D1 → C → B** order above; maintain single
whole-box ownership. Do not put these self-locking scripts under another lock or the
collector. Their JSONL stays under their own raw namespaces, linked and hashed from D.
Missing script/target/adapter is an explicit blocker, not SKIP/PASS.

Non-serving PRO pair only (then all four cards): real directed P2P/context+pool grants and
PeerCapacity lifetime/red tests; official Step `decode-batch-gate --mode pp/ppspec` ladder;
Qwen 128k and **262144 actually consumed** active/prefix gates; full Hy3/PLE/model-consumer
spill; global pool-smaller-than-object/fairness/churn/spec/tenant corruption pressure;
full affected kernel-check/run-gen/run-spec and ModelPlan/engine/server suites. Full PLE/Hy3
must satisfy actual capacity first—pair availability alone is no permission to truncate.
Use `/tmp/memra-gpu.lock` across the entire campaign. Pair = four tiers, not four-card proof.
Finally 12 directed routes, real per-device allocation peaks and one simultaneous shared-fabric
campaign on 4× PRO. No TP/EP claim, host bounce ≠ P2P, QSA scatter guard unchanged.

Scoring later: forced ON/OFF exactness first, >=5 AB +5 BA pairs, actual 250 ms clocks/power/
VRAM/link plus instrumented route/host/NVMe/queue waits, all serving latency percentiles and
throughput, >=30 min stable pressure and <=70% **measured** route/SSD envelope. Network-volume
4 KiB faults never become spill speed. G0–G7 remain pending; no V4.1 work or paused 0731 oracle.
