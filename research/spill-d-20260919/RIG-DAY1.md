# First rented RTX 5090 hour — D day 5

**Runbook, not a rig receipt.** Minimum source: **`020d2047`**
(`020d20479cd686835c0fb7743040947d0fc2723b`) plus this day-4 bootstrap/collector.
That integrated source includes A/B/C day-3 runners and reviewed fixes. Bootstrap checks
ancestry and pins the exact remote head. Earlier `914229ae` only fixed target discovery.
The first rental ran on 2026-09-19; see the observed-duration section below.
Development only, never an existing serving box. Target remains 4× RTX PRO 6000, PCIe Gen5, **no NVLink**.
One 5090 is not peer/four-tier qualification and never vetoes Step-only PRO delivery.

## Before rental / source handoff (blocking)

1. Lead must push the reviewed integration branch containing D's bootstrap/collector and
   approved shared-file amendments. Every lane milestone now commits **and pushes** with hooks. `BRANCH` is mandatory;
   bootstrap checks the exact remote head and refuses missing/moving branches, never main.
2. **First engine/server build is blocking.** B's `_tier_identity` field in `worker.rs` has
   only CPU-side/source inspection so far. It resolves via `memra_engine::cache`, not a
   direct server `memra-kv` dependency. All approved native patches must compile on this
   same source. No CPU green substitutes for that build.
3. **The manifest defect is fixed at `914229ae`.** `memra-engine` now depends on
   `memra-tier` and explicitly registers `storage-bench`; `cargo metadata` resolves both.
   This fixes target discovery, not native compilation. The blocking first native compile
   of **memra-engine + memra-server** must validate both the new bin and B's `worker.rs` field.
4. A/B/C `rig-cells-*.sh` runners are now **present**, merged from integration `020d2047`.
   Exact invocations, lock ownership and explicit incomplete-adapter outcomes appear below.
   They are correctness launchers, not performance collectors; never nest their locks.
   Bootstrap compiles the native engine and server before any runner is admitted. A real
   GPU transfer owner, B active materializer/server adapter and C NVMe row consumers remain
   unqualified; successful baseline binaries cannot stand in for these bindings.

## Bootstrap, via the operator's existing SSH identity

No keys, tokens, env files or provider credentials are embedded/read. Use an approved dev
host alias already configured by the operator; do not use a production alias. Example:

```sh
# Operator supplies DEV_HOST, DEV_PORT, BRANCH. No ssh key material is copied into the script.
: "${DEV_HOST:?approved new development host alias}"
: "${BRANCH:?pushed lane/spill-* branch}"
: "${DEV_PORT:?provider SSH port, e.g. from the approved console connection panel}"
[[ "$DEV_PORT" =~ ^[0-9]+$ ]] && (( DEV_PORT >= 1 && DEV_PORT <= 65535 ))
ssh -p "$DEV_PORT" -o BatchMode=yes -o ConnectTimeout=10 "$DEV_HOST" \
  "BRANCH='$BRANCH' bash -s -- --provider runpod --expected-power 600 --gap-seconds 60 --jobs 4" \
  < tools/tier-rig-bootstrap.sh
```

Use only validated `lane/spill-*` branch text in the remote command (letters/digits,
`/._-`; no quotes/metacharacters). At most two connection attempts, backoff 10 seconds;
a failed connection establishes no remote state. Read the failure receipt before rerunning
bootstrap. Do not automatically destroy/rent/reconfigure boxes from this script.

**Observed SSH pitfall (2026-09-19):** the account-level Vast public key association did
not inject the key into the container. Operator command
`vastai attach ssh <instance-id> "<public-key with a NEW comment>"` succeeded; reusing the
same comment returned "already associated" without fixing access. This is public key
material only; never copy a private key. Prefer the provider's **SSH proxy** over the
flaky direct-IP route. Resolve host/port only from the gitignored `LANE-LOCAL.md` and obey
`BOX-ACCESS.md` (up to 3 tries with 30 s backoff on this rig). Never put those values or
provider ids into tracked examples. Key attachment remains an operator action, not an
automatic bootstrap API call.

Bootstrap is repeatable: preserves dirty tracked work and old receipts; existing clean
checkout is detached at the exact requested remote head. Existing untracked receipts are
preserved. Explicit `--out` must be new unless `--resume` is used. No driver/clock/ACS/IOMMU changes
or systemd dependency. `--pidfile <persistent-path>/spill-bootstrap.pid` explicitly selects
process status metadata. This is **not another GPU campaign lock**: all GPU work still uses
`/tmp/memra-5090.lock`. The pidfile's advisory lock lives for the bootstrap process lifetime;
a crash releases it, so stale/recycled PIDs cannot report "running". Do not unlink the inode
while an invocation may be active. Default: `<persistent-root>/spill-bootstrap.pid`
(dry-run: alongside its output directory).

```sh
# Read-only; no BRANCH required and no inventory/build/GPU operation is started.
bash tools/tier-rig-bootstrap.sh --status --pidfile "$PIDFILE"
# Alias: --already-running. Exit 0=active, 1=inactive/missing, 2=invalid/unreadable.
# Handle exit 1 explicitly under set -e; it is not a failed qualification cell.
```

Do **not** use `pgrep -f tier-rig-bootstrap.sh`: an SSH command containing that text
matches itself. Duplicate bootstrap invocations at the same pidfile refuse before creating
receipt directories. Builds cap `--jobs` at **16** (default 4) for box responsiveness.
`--set-power` optionally requests the expected limit under the
canonical lock after an empty compute-apps check; container refusal is logged, not fatal
when max power is adequate. No power request occurs without this explicit option. It installs
ordinary missing Ubuntu/Debian packages; other distros work if dependencies exist, otherwise
name the missing tools and refuse. It installs/verifies rustup **stable >=1.97**; if rustup is absent it downloads the official HTTPS installer to
a temporary file and installs the minimal profile (no pipe-to-shell). CUDA resolution matches `build.rs`:
explicit `MEMRA_NVCC`, existing CUDA root, otherwise newest runnable PATH/local toolkit;
selected CUDA must be >=13 and list base `compute_120` (the list omits architecture-specific
`compute_120a`); the subsequent **`-arch=sm_120a` compile probe** must succeed; the accepted absolute compiler and
`120a` architecture are pinned for these builds so rustup's PATH change cannot alter selection.
Missing toolkit fails with the exact
installation requirement, not an attempted old distro CUDA or model-format fallback.

Acceptance reads **both** power.limit and power.max_limit in the first minute. Max must be
>=600 W; a lower current cap or refused `nvidia-smi -pl` is recorded as restricted power,
not an acceptance failure or a full-power performance claim. `--set-power` re-reads both
fields even when the request is refused. No Max-Q/listing inference. It verifies one >=31,000 MiB RTX 5090. Under
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

### Container paths and interruption recovery

- **RunPod:** `--provider runpod` uses `/workspace` for the checkout/build cache and
  receipt root. Verify the selected volume survives interruption under the purchased
  storage contract; `/workspace` alone proves neither durability nor local NVMe.
- **Vast:** `--provider vast --persistent-root <operator-confirmed-volume>` is mandatory.
  Offer/container disks have provider-specific restart/destroy behavior; no guessed mount.
- Other already-approved development hosts default to the operator home. No automatic
  rental, provider API request, key installation, systemd or serving-host operation.
- Bootstrap captures `df -hT` and `lsblk -J` **before selecting paths**. Build artifacts
  stay at `<repo>/target` on the chosen persistent volume so all lane runners agree.
  `--nvme-root <mount>` requires `findmnt` + `lsblk -s` NVMe ancestry. Default refuses
  an explicitly supplied unproven mount. No selected storage leaves scratch/NVMe **null**.
  The sanctioned **development exactness only** alternative is
  `--nvme-root <actual-path> --allow-unproven-storage`. It retains failed ancestry output,
  `findmnt` and `lsblk` verbatim; sets scratch but leaves NVMe **null**; labels storage
  **"overlay/unproven — not NVMe, not spill speed"** in the bootstrap receipt. This is not
  a fallback format or a waived NVMe gate. Missing actual directories still refuse.
  Every storage collector invocation must separately receive `--storage-root "$SCRATCH"`
  and, on an unproven filesystem, `--allow-unproven-storage`; it reprobes before execution
  and copies that label into both CELL rows and capture JSON. `STORAGE.json` and raw
  ancestry diagnostics survive refusal. Local-NVMe/O_DIRECT/spill-speed gates remain pending.
- Default receipts: `<persistent-root>/spill-bootstrap-receipts/<host>-<utc>/`.
  Every external step writes its raw log there as it runs; `BOOTSTRAP.jsonl` records start
  and end incrementally (fsync), and `BOOTSTRAP.json` is atomically checkpointed. An abrupt
  kill retains the active step and partial log, not a pass. No evidence waits in `/tmp`.
- Resume with the same arguments plus `--out <prior-dir> --resume`. It reads the prior
  BOOTSTRAP receipt, checks branch/class, and writes a new `attempts/<utc>/` child. All
  hardware acceptance/source checks rerun; a prior GPU pass never survives instance loss
  as fresh qualification. Package installs, clean checkout/fetch and Cargo builds are
  repeatable. Dirty tracked work is preserved/refused. The old receipt is never replaced.
- Each collector command also writes `CELL.jsonl` start/end with run id, UTC, elapsed time,
  exit/result and optional `--hourly-cost` estimate. End rows and `command.capture.json`
  now share `started_utc`, `ended_utc`, `elapsed_seconds`; this is **collector elapsed**
  (includes compute snapshots/sampler), not CUDA device time or binary-only duration.
  A's `StorageSample.total_ns` remains its separate internal measurement. `--resume --out <prior-cell>` reads
  the last CELL and requires identical argv, then reruns in a new attempt. Only rerun
  idempotent commands; binaries creating named output files need new explicit output
  paths and a new cell. A start without end means **interrupted, outcome unknown**.
- After interruption, inspect the last completed cell and rebootstrap before continuing
  the first uncompleted step. Do not automatically repeat B's patch worktree or blindly
  continue a half-written artifact. Its existing runner has no resume switch: preserve
  its scratch/receipts, inspect source state, then start a fresh runner output directory.
- Provider id, if exposed as `RUNPOD_POD_ID` or `CONTAINER_ID`, and optional cost are raw
  **private** receipt metadata, never credentials. Keep real ids/cost/host/topology outside
  public git; sanitize before publication and retain original hashes privately. No env
  dumps or env/auth file reads. **Sync receipts after EVERY cell, including failures,
  before starting the next cell** — never only at the end of the sequence. Use the same
  operator SSH identity to pull to private storage. Run this on the operator workstation
  after each remote collector invocation (the sequence below must be segmented accordingly):

  ```sh
  : "${LOCAL_EV:?private local mirror for this campaign}"
  rsync -a --checksum -e "ssh -p $DEV_PORT -o BatchMode=yes -o ConnectTimeout=25" \
    "$DEV_HOST:$EV/" "$LOCAL_EV/"
  # Also pull $BOOT after each bootstrap stage/checkpoint when it is still running.
  # Inspect/validate the completed CELL + capture hashes in the local mirror before proceeding.
  python3 tools/tier-battery.py --validate "$LOCAL_EV"
  ```

  Failed commands can pass **integrity** validation; inspect their retained status before
  proceeding. A torn/start-only cell remains incomplete and must not be claimed successful.
  Mirror native output TSVs, manifests and goldens too, not just CELL.jsonl. No script can
  recover a provider-destroyed volume without the off-box copy.
- **Observed spot interruption, reported by the lead 2026-09-19:** the first rented box
  was preempted mid-checkpoint download (approximately 6.4 of 15.7 GB). A/D1/C receipts
  were already synced; the incomplete B artifact and scratch worktree were lost. Provider
  state was `exited/stopped`; `vastai start` refused with
  **"Required resources are currently unavailable"**. This is an operator-reported recovery
  incident, not an additional GPU result. Restart of a preempted instance can remain queued
  indefinitely: choose bounded wait versus replacement from **receipt state**, not hope.
  Inventory which completed receipts are already off-box, which uncompleted cells must
  rerun, and whether any unique bytes remain only on the old disk. Do not discard the only
  copy of unsynced evidence. Once retained results are safe, a missing reproducible scratch
  worktree or partial download is not a reason to wait indefinitely for capacity; operator
  may replace the rental under the existing approval. No automatic destroy/start loop.
- Artifacts must remain re-downloadable from an **immutable pinned locator plus complete
  byte manifest/SHA-256**, independently of the rented disk. Record the full revision and
  expected length/hash before transfer. A partial file is never an accepted artifact; resume
  only against the same immutable bytes, verify the full hash, and rebootstrap/rebuild and
  rerun pending cells on the replacement. Restore the scratch worktree from its **pushed
  branch/commit**, not from an assumed surviving local branch. B remains unrun in this
  snapshot; preserved A/D1/C results cannot stand in for B or qualify the new device.

Retain `BOOTSTRAP.json`, journal/logs, `TOPOLOGY.json`, acceptance source/binary and
`locked-run.sh`. Bootstrap is **not** a schema-v1 byte receipt or tier-qualified GPU pass.

## Observed 2026-09-19 (one run; not booking budgets)

Source `01e7b77f29c7b40fb29744e5321ca0e8b0c81f39`, one RTX 5090, 600 W current/max.
Raw: `research/spill-lead-20260919/rented-5090-20260919/receipts/` (`*run3` bootstrap,
`first-hour*` CELL/capture/driver logs). No repeated/thermal-steady-state timing claim.
PCIe inventory: host/effective max **Gen4 x16**, device max Gen5; idle sample **Gen1 x16**.
Current is not max, and neither measures bandwidth. No peer pair exists on one device.

| Stage | Observed duration | Evidence/meaning |
|---|---:|---|
| Successful bootstrap, 12:38:24–12:47:03 UTC | 518.976 s (8.65 min) | includes all following bootstrap stages; not additive to them |
| Two 8 GiB readbacks and gap | 15.852 +60.053 +15.853 s | standalone acceptance, not tier qualification |
| Rust install +stable download | 17.417 s | install/stable commands only |
| Clone | 29.133 s | source retrieval only |
| Builds 0/1/2/3 | 9.939 /294.238 /49.519 /23.664 s | libraries /server /five bins /engine lib no-run; **-j32 historical**, future cap16 |
| A 264 B roundtrip /restore | 0.074769 /0.068704 s | collector elapsed; **overlay/unproven — not NVMe, not spill speed** |
| A 1 MiB roundtrip /restore | 0.362527 /0.088735 s | same scope |
| A 4,194,568 B roundtrip /restore | 0.394961 /0.362578 s | same scope |
| D1 local | 0.645826 s | collector elapsed; same-device only |
| C first attempt /corrected attempt | 0.377596 /6.877514 s | missing-goldens failure /executed after goldens copied |

ADC driver started 13:00:57 UTC; corrected C ended 13:10:55.983 UTC. That interval
includes operator recovery/gaps, not summed GPU time. No B duration is present in this
snapshot (`driver-b.log` has only a start marker). Earlier two bootstrap failures remain
recorded (unproven ancestry; obsolete arch-list predicate). These observations replace
**no budgets** below; they are one-run observations on the original binary only.

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
assert r['qualification'] is False
assert set(r['release_binaries']) == {'storage-bench','pp-transport-smoke','qwen4exp_gpu_gate','run-gen','run-spec','memra-server'}
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
# Default: verify local NVMe ancestry. For overlay/unproven development only, explicitly
# use STORAGE_OPTIONS=(--allow-unproven-storage). Never call these numbers spill speed.
STORAGE_OPTIONS=()
: "${STORAGE_ROOT:?existing owned development storage parent; NVMe or explicit unproven mode}"
# Create one owned directory; do not remove an existing campaign's scratch.
SCRATCH=$(mktemp -d "$STORAGE_ROOT/spill-first-hour.XXXXXXXX")
trap 'rm -rf -- "$SCRATCH"' EXIT
for size in 264 1048576 4194568; do
  python3 tools/tier-battery.py --rig rtx5090 --timeout 120 \
    --storage-root "$SCRATCH" "${STORAGE_OPTIONS[@]}" \
    --out "$EV/a-roundtrip-$size" --execute \
    target/release/storage-bench roundtrip "$SCRATCH/object-$size" "$size" buffered
  python3 tools/tier-battery.py --rig rtx5090 --timeout 120 \
    --storage-root "$SCRATCH" "${STORAGE_OPTIONS[@]}" \
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
# It synthesizes tiny checkpoints, BUT requires the PRE-STREAMING byte goldens beside
# the output TSV. Never mint replacement goldens with the current loader.
mkdir -p "$EV/c-ple-tiny/receipt"
cp research/qwen4exp-bringup-20260829/gpu-eager/bank-bytes-goldens.tsv \
  "$EV/c-ple-tiny/receipt/bank-bytes-goldens.tsv"
sha256sum "$EV/c-ple-tiny/receipt/bank-bytes-goldens.tsv" > "$EV/c-ple-tiny/goldens.sha256"
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/c-ple-tiny/cell" --execute \
  target/release/qwen4exp_gpu_gate "$EV/c-ple-tiny/receipt/ple-tiny.tsv"

# B: fitting native checkpoint baseline, not new active demote/reload proof.
: "${QWEN:?pinned Qwen3.8-27B NVFP4+Q5_K GGUF with qualified MTP attachment}"
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/b-qwen-argmax" --execute target/release/run-gen "$QWEN"
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/b-qwen-spec" --execute \
  env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_GEN_ONLY \
  target/release/run-spec "$QWEN"
```

`python3 tools/tier-battery.py --validate "$EV"` validates all completed CELL/capture
hashes and outcomes (including failed commands and empty diagnostic telemetry). It is
**archive integrity only**, not a byte gate or GPU qualification. `--validate` also accepts
one `CELL.jsonl` or `*.capture.json`. Legacy first-hour records remain immutable: the
validator derives seconds from recorded ns and uses the journal UTC; it reports missing
storage labels as "not recorded; no NVMe/spill-speed claim". Torn/start-only journals fail
as incomplete. The original strict byte-receipt and telemetry validators remain separate.

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
| 3 | C PLE tiny native gate, fixture generated by binary | tiny 4-layer hidden16/expert8 geometry; reserve <=4 GiB GPU operational ceiling, stop if unexpectedly larger; 15 min first-hour slice /30 min full | PLE history +tiny loader/forward rows; `$EV/c-ple-tiny/receipt/ple-tiny.tsv`, `$EV/c-ple-tiny/cell`; does NOT prove bounded NVMe service or full-model PLE |
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
- **PLE/Qwen4Exp:** tiny checkpoints are generated locally, but pre-streaming
  `research/qwen4exp-bringup-20260829/gpu-eager/bank-bytes-goldens.tsv` is mandatory
  **beside** the output TSV. Copy the checked-in file and record its SHA-256 first;
  never use `--write-bank-goldens` to repair missing oracle input. Full model source
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

## Integrated lane runners / PRO-only sequence

All three paths below exist at the required integrated baseline. Invoke standalone in
**A → D1 → C → B** order after the successful native compile, never inside `--execute`
or `locked-run.sh` because the runners own the canonical lock themselves.

```sh
# A: whole-window /tmp/memra-5090.lock. Local NVMe proof required.
bash research/spill-a-20260919/rig-cells-a.sh "$NVME" --approved-non-serving
# D1: non-locking binary through D's collector, as above.
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 --out "$EV/d1-local" \
  --execute env -u MEMRA_PP_DEVICES -u MEMRA_PP_HOST_BOUNCE target/release/pp-transport-smoke
# C: builds before acquiring /tmp/memra-5090.lock. Before ple-tiny, its own runner
# must copy+hash bank-bytes-goldens.tsv beside $out/ple-tiny.tsv (not collector output).
# D's C-RUNNER-GOLDENS.diff is the handoff to C; apply via C owner before invoking
# an older runner. D's bounded C cell above already includes the precondition.
bash research/spill-c-20260919/rig-cells-c.sh \
  --non-serving-confirmed --rig 5090 --host-label development
# B: per-GPU-command lock; legacy self-locking gates are not nested. Exact pinned inputs.
bash research/spill-b-20260919/rig-cells-b.sh --exclusive-non-serving \
  --artifact "$QWEN" --artifact-sha256 "$QWEN_SHA256" \
  --tokens-8k "$TOKENS8K" --tokens-32k "$TOKENS32K" \
  --fit-plan "$FIT_PLAN" --out "$EV/b-fitting"
```

These are separate invocations, not an unguarded `set -e` batch: inspect each exit/raw log
before the next. A currently exits **1** after retaining unimplemented A2/M1 probe failures;
C exits **3** because native bank/row adapters remain missing; B refuses missing active
binding. These are blockers, not expected-pass waivers. Do not swallow unknown failures.
B tests a reviewed **unapplied** HOSTPREFIX patch in a temporary dedicated worktree,
building before/after and checking 8192+32768 context envelopes. This is not mutation of
D's accepted binary and needs separate receipts. Do not interpret "before/after patch"
as generic tier OFF/ON or as AB/BA N>=5 scoring.

Budget: A ≤4,194,568 bytes/object, initial five-minute characterization slice; D1 five
minutes; C tiny geometry ≤4 GiB operational ceiling, fifteen-minute first-hour slice
(thirty-minute full gate); B single request within measured 32 GiB envelope and immutable
8k/32k full-run peak fit plan, fifteen-minute first-hour slice (≥60-minute full baseline).
Cold builds may consume the hour. Standalone lane runners have broader rosters/no global
hour deadline: do **not** start them near the hour boundary expecting a five-minute bound.
Use the bounded per-binary cells above for the first-hour window; defer full runners to
reserved follow-up time. A's additional pinned-worker test is not H2D; C's tiny PLE is not
bounded NVMe consumption; B's fitting baseline is not active demote/reload.

`python3 tools/tier-battery.py --first-hour` emits `FIRST-HOUR-PLAN.json`: booked stage
order and AB then BA correctness controls for future bound ON/OFF cells. No fabricated
arm flags are passed to baseline binaries. No performance cells/medians are allowed;
later scoring requires **>=5 AB and >=5 BA pairs**, actual counters and 250 ms telemetry.

A/B/C owners supply dependency amendments and lock ownership in their `CELLS.md`.
Record each script hash and run in **A → D1 → C → B** order above; maintain single
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
