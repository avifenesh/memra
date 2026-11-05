#!/usr/bin/env python3
"""Port the DAY37 O1 harness (memra tree 3476ea4dc, the second Workstation sitting's harness) to <box>.

Extracts the harness subset with `git archive` (exact blobs) into kit/harness/, then applies the portability
patches below. Every patch asserts how many times its text occurs, so a drifted source fails loudly instead of
half-porting. Idempotent: the harness dir is rebuilt from the archive every run.

Ports only (DEVIATIONS.md lists each): no git on <box> (SOURCE.commit, PORTED.txt), no cargo on <box> (serve-smoke runs
the prebuilt lane binary), nvidia-smi restricted to the card ($CARD), the runner's canonical lock hold inherited by
the collector and the boots (KIT_LOCK_FD), a bounded idle wait for the window (KIT_IDLE_WAIT_S), and resume after a
window deadline (an executed boot is never run again; a boot cut before its exit file is asked again).

usage: port.py <harness worktree at 3476ea4dc> <kit dir>
"""
import hashlib, io, os, re, shutil, subprocess, sys, tarfile

WT, KIT = sys.argv[1], sys.argv[2]
SHA = "3476ea4dca27c4c3c738017b8429f507389f1223"
LANE = "c6f9282c26e71349c0263b88dc7beb59c4a57797"
H = os.path.join(KIT, "harness")
D = "research/spill-b-20260919"

head = subprocess.check_output(["git", "-C", WT, "rev-parse", "HEAD"], text=True).strip()
assert head == SHA, f"harness worktree at {head}, want {SHA}"
names = subprocess.check_output(["git", "-C", WT, "ls-tree", "-r", "--name-only", SHA, "--", f"{D}/"], text=True).split("\n")
keep = [n for n in names if n and (
    (os.path.dirname(n) == D and n.endswith((".py", ".sh"))) or
    (os.path.dirname(n) in (f"{D}/rtx5090-day37", f"{D}/pro-single-day37") and n.endswith(".sh")))]
paths = ["tools", "docs"] + keep
tar = subprocess.check_output(["git", "-C", WT, "archive", "--format=tar", SHA, "--", *paths])
if os.path.isdir(H):
    shutil.rmtree(H)
os.makedirs(H)
with tarfile.open(fileobj=io.BytesIO(tar)) as t:
    t.extractall(H, filter="tar")

# Serve-smoke runs from this tree instead of a worktree at the lane source: prove the files it reads are the same.
diff = subprocess.check_output(["git", "-C", WT, "diff", "--name-only", LANE, SHA, "--", "tools", "docs/SERVING.md"], text=True).strip()
assert diff == "", f"tools/ or docs/SERVING.md differ between {LANE[:12]} and {SHA[:12]}: {diff}"

ported = []


def patch(rel, subs):
    p = os.path.join(H, rel)
    src = open(p).read()
    before = hashlib.sha256(src.encode()).hexdigest()
    out = src
    for old, new, count in subs:
        n = out.count(old)
        assert n == count, f"{rel}: expected {count} of {old!r}, found {n}"
        out = out.replace(old, new)
    open(p, "w").write(out)
    blob = subprocess.check_output(["git", "-C", WT, "rev-parse", f"{SHA}:{rel}"], text=True).strip()
    ported.append((rel, blob, before, hashlib.sha256(out.encode()).hexdigest()))


SMI_SH = ('nvidia-smi --query-', 'nvidia-smi -i "$CARD" --query-')
SRC = '$(cat "$KIT_HARNESS/SOURCE.commit")'

# --- the gate set (A1) ---------------------------------------------------------------------------------------------
gates_smoke_old = '''SMOKE_DIR=$WT
if [ -f "$(dirname "$BIN")/source.commit" ]; then
  SMOKE_SHA=$(cat "$(dirname "$BIN")/source.commit")
  # On disk under the lane's own (ignored) target dir, never /tmp: the worktree builds its own target/.
  SMOKE_DIR=$WT/target/smoke-wt-${SMOKE_SHA:0:12}
  [ -d "$SMOKE_DIR" ] || git worktree add --detach "$SMOKE_DIR" "$SMOKE_SHA" >> "$ROOT/battery.log" 2>&1
  # The build runs under the rig's CPU quota before the smoke (whose own build then finds it current).
  scope=(); systemd-run --user --scope -q true 2>/dev/null && scope=(systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G)
  ( cd "$SMOKE_DIR" && "${scope[@]}" nice -n 10 cargo build --release -p memra-server -j 12 ) > "$ROOT/serve-smoke-build.log" 2>&1
  log "serve-smoke runs at $SMOKE_SHA in $SMOKE_DIR (build rc=$? $(tail -1 "$ROOT/serve-smoke-build.log" | cut -c1-80))"
fi
'''
gates_smoke_new = '''SMOKE_DIR=$WT
# kit port (<box> has no cargo and no git checkout): serve-smoke runs the prebuilt lane binary, which the kit runner links
# at $WT/target/release/memra-server. tools/ and docs/SERVING.md are identical between the lane source c6f9282c2 and
# this tree 3476ea4dc (port.py asserts it), so this tree stands in for the smoke worktree at the lane source.
log "serve-smoke runs the prebuilt lane binary from $(cat "$(dirname "$BIN")/source.commit") at $WT/target/release/memra-server (kit port: no on-box build)"
'''
gates_rm_old = '''if [ "$SMOKE_DIR" != "$WT" ]; then
  sha256sum "$SMOKE_DIR/target/release/memra-server" > "$ROOT/serve-smoke-binary.sha256" 2>/dev/null
  git worktree remove --force "$SMOKE_DIR" >> "$ROOT/battery.log" 2>&1
fi
'''
patch(f"{D}/rtx5090-day37/gates.sh", [
    ("TREE=$(git rev-parse HEAD)", f"TREE={SRC}  # kit port: no git on <box>", 1),
    (SMI_SH[0], SMI_SH[1], 3),
    (gates_smoke_old, gates_smoke_new, 1),
    (gates_rm_old, "", 1),
])
patch("tools/serve-smoke.sh", [
    ('cargo build --release -p memra-server || { echo "serve-smoke: build FAILED"; exit 1; }',
     '# kit port: no cargo on <box>; the kit runner links the prebuilt lane binary at target/release/memra-server.\n'
     '[ -x target/release/memra-server ] || { echo "serve-smoke: build FAILED (kit: no prebuilt target/release/memra-server)"; exit 1; }', 1),
])
for rel in ("tools/admit-mem-burst-gate.sh", "tools/spec-ctx-edge-gate.sh"):
    patch(rel, [('echo "commit $(git rev-parse HEAD)"', f'echo "commit {SRC}"', 1),
                ('  nvidia-smi --query-gpu=name,driver_version', '  nvidia-smi -i "$CARD" --query-gpu=name,driver_version', 1),
                ('gpu_apps() { nvidia-smi --query-compute-apps', 'gpu_apps() { nvidia-smi -i "$CARD" --query-compute-apps', 1)])
patch("tools/prefix-newest-turn-fits-gate.py", [
    ('subprocess.run(["nvidia-smi", *args,', 'subprocess.run(["nvidia-smi", "-i", os.environ["CARD"], *args,', 1),
    ('["nvidia-smi", "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version",',
     '["nvidia-smi", "-i", os.environ["CARD"], "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version",', 1),
])

# --- the collector (stage 0, A2, the gate holds) -------------------------------------------------------------------
lock_old = '''def campaign_lock(rig, inherit=False):
    require(rig in LOCKS and LOCKS[rig] is not None, "GPU-shaped campaign requires canonical rig lock")
    # Never unlink a shared lock inode: waiters and other campaigns must see the same file.
    with open(LOCKS[rig], "a") as handle:
'''
lock_new = '''def campaign_lock(rig, inherit=False):
    require(rig in LOCKS and LOCKS[rig] is not None, "GPU-shaped campaign requires canonical rig lock")
    kit_fd = os.environ.get("KIT_LOCK_FD")
    if kit_fd is not None:
        # kit port (<box> Max-Q window): the kit runner holds the canonical lock for the whole window on KIT_LOCK_FD.
        # Take the same open file description (a dup), prove it owns the lock (an independent open is excluded),
        # and never LOCK_UN it: that would release the runner's hold for every process.
        dup = os.dup(int(kit_fd))
        st, canon = os.fstat(dup), os.stat(LOCKS[rig])
        require((st.st_dev, st.st_ino) == (canon.st_dev, canon.st_ino), "KIT_LOCK_FD is not the canonical lock inode")
        with open(LOCKS[rig], "a") as independent:
            try:
                fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(independent, fcntl.LOCK_UN)
                os.close(dup)
                require(False, "KIT_LOCK_FD does not hold the canonical lock")
        with os.fdopen(dup, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield handle if inherit else LOCKS[rig]
        return
    # Never unlink a shared lock inode: waiters and other campaigns must see the same file.
    with open(LOCKS[rig], "a") as handle:
'''
patch("tools/tier-battery.py", [
    (lock_old, lock_new, 1),
    ('tee_run([smi, "--query-compute-apps=pid,process_name,used_memory",',
     'tee_run([smi, "-i", os.environ["CARD"], "--query-compute-apps=pid,process_name,used_memory",', 1),
    ('argv = [smi, "--query-gpu=timestamp,index,', 'argv = [smi, "-i", os.environ["CARD"], "--query-gpu=timestamp,index,', 1),
])
patch(f"{D}/rtx5090-day37/run-grow.sh", [
    ('git rev-parse HEAD > "$root/source.commit"', 'cat "$KIT_HARNESS/SOURCE.commit" > "$root/source.commit"  # kit port: no git on <box>', 1),
])

# --- the serving boots -----------------------------------------------------------------------------------------------
patch(f"{D}/run-day26-cell.sh", [
    ('git rev-parse HEAD > "$C/source.txt"; git status --short > "$C/dirty.txt"',
     'cat "$KIT_HARNESS/SOURCE.commit" > "$C/source.txt"; cat "$KIT_HARNESS/PORTED.txt" > "$C/dirty.txt"  # kit port: no git on <box>', 1),
    ('lock held by the collector (external)', 'lock held by the collector (external${KIT_LOCK_FD:+: the kit runner, FD $KIT_LOCK_FD})', 1),
    (SMI_SH[0], SMI_SH[1], 4),
])
boots_idle_old = '''idle() {
  flock -n "$RIG_LOCK" true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
'''
boots_idle_new = '''# kit port: the lock is the kit runner's hold (KIT_LOCK_FD, proved, never re-taken); compute apps are read on $CARD
# (KIT_IDLE_SCOPE=box reads the whole box, the registered form, for the timed slot).
smi_scope=(-i "$CARD"); [ "${KIT_IDLE_SCOPE:-card}" = box ] && smi_scope=()
idle() {
  if [ -n "${KIT_LOCK_FD:-}" ]; then
    python3 tools/tier-lock-proof.py --fd "$KIT_LOCK_FD" --lock "$RIG_LOCK" > /dev/null 2>&1 || return 1
  else
    flock -n "$RIG_LOCK" true || return 1
  fi
  [ -z "$(nvidia-smi "${smi_scope[@]}" --query-compute-apps=pid --format=csv,noheader)" ] || return 1
'''
boots_loop_old = '''  [ -x "$BIN" ] || { log "boot $name: no binary $BIN; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
'''
boots_loop_new = '''  [ -x "$BIN" ] || { log "boot $name: no binary $BIN; not run"; exit 1; }
  # kit port (resume across windows): an executed boot (its exit file is written) never runs again; a boot an earlier
  # window cut before its exit file moves to boots/cut/ and is asked again.
  if [ -e "$R/boots/$name.exit" ]; then log "boot $name: executed already (exit $(cat "$R/boots/$name.exit")); not run again"; continue; fi
  if [ -e "$R/boots/$name.arm.txt" ]; then
    mkdir -p "$R/boots/cut"; cs=$(date -u +%Y%m%dT%H%M%SZ)
    for x in "$R/boots/$name" "$R/boots/$name.arm.txt" "$R/boots/$name.launch.log"; do [ -e "$x" ] && mv "$x" "$R/boots/cut/$(basename "$x").$cs"; done
    log "boot $name: an earlier attempt left no exit file (cut); moved to boots/cut/*.$cs, asked again"
  fi
  idle_s=${KIT_IDLE_WAIT_S:-7200}
  deadline=$((SECONDS + idle_s)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after $idle_s s; not run"; exit 3; }
'''
patch(f"{D}/rtx5090-day37/boots-f.sh", [
    (boots_idle_old, boots_idle_new, 1),
    ('export LOCK=$RIG_LOCK RIGDIR="$R/boots" N=5 MEMRA_TIMEOUT_MS_MAX=3600000',
     'export LOCK=$RIG_LOCK RIGDIR="$R/boots" N=5 MEMRA_TIMEOUT_MS_MAX=3600000\n'
     '[ -n "${KIT_LOCK_FD:-}" ] && export LOCK=none  # kit port: the kit runner holds the canonical lock', 1),
    (boots_loop_old, boots_loop_new, 1),
])
PY_SMI = ('["nvidia-smi", "--query-gpu=', '["nvidia-smi", "-i", os.environ["CARD"], "--query-gpu=')
for rel in (f"{D}/day26-client.py", f"{D}/day33-client.py", f"{D}/day37-stream-client.py"):
    patch(rel, [(PY_SMI[0], PY_SMI[1], 1)])

open(os.path.join(H, "SOURCE.commit"), "w").write(SHA + "\n")
with open(os.path.join(H, "PORTED.txt"), "w") as f:
    f.write(f"harness tree {SHA} (git archive of tools/, docs/ and the {D} scripts); kit ports, no git checkout on this box\n")
    f.write("file  source_blob  sha256_before  sha256_after\n")
    for rel, blob, b, a in ported:
        f.write(f"{rel}  {blob}  {b}  {a}\n")
# A machine-checkable manifest of every shipped harness file.
rows = []
for root, _, files in os.walk(H):
    for fn in files:
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, H)
        if rel == "MANIFEST.sha256":
            continue
        rows.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {rel}")
open(os.path.join(H, "MANIFEST.sha256"), "w").write("\n".join(sorted(rows, key=lambda r: r[66:])) + "\n")
print(f"harness: {len(rows)} files, {len(ported)} ported")
for rel, *_ in ported:
    print("  ported", rel)
