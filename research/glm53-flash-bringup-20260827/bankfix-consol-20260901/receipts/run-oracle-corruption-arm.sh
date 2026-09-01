#!/usr/bin/env bash
# THE ORACLE WITH A CORRUPTION ARM — the void-gate control for the bank-defect consolidation.
#
# WHY THIS SHAPE AND NOT "run the oracle, it passes". The rig's oracle binary DID NOT TRACK KERNEL
# SOURCE (gate-craft TRAP: a gate whose binary is stale reports the last build's verdict forever),
# so a green `nvfp4-bank-oracle` on this merge would prove nothing about the merged kernel. A
# passing gate is evidence only when the same gate, on the same box, in the same sequence, has been
# SEEN TO FAIL on a deliberately corrupted kernel state. So the order is fixed and not negotiable:
#
#   1. CORRUPT FIRST. `*1.5f` on the QT_NVFP4_V2 scale read inside `kq_fetch` (the exact byte the
#      shipped defect misread). Build. Record the binary sha. All FOUR arms MUST exit 1.
#      The corruption is on the v2 side only; the v1 arm is the oracle and stays pinned, so the
#      differential gate must see it in every tile form (unlike the PRE-FIX twin of
#      research/step37-bankv3-20260901, which localized to the deep-tail form alone).
#   2. REVERT. `git checkout` the file; the tree must be clean again.
#   3. BUILD THE TRUE FIX. The binary sha MUST DIFFER from the corrupt one — that difference IS
#      the proof that the binary tracks the kernel source. Then all four arms MUST exit 0.
#
# A run that skips step 1, or whose two binary shas match, is NOT an oracle receipt.
#
# ARMS (tile form is chosen inside the C launcher, so it is DRIVEN from here, one arm per process):
#   hybrid  unset            -> cross=64, deep 3-stage tail
#   sk128   MEMRA_F16G_SK=128 -> every group on the 128x64x64 3-stage form
#   sk32    MEMRA_F16G_SK=32  -> every group on the 32x64 tail form
#   tail0   MEMRA_F16G_TAIL=0 -> the tail form's 2-stage rollback
#
# NON-VACUITY IS ASSERTED, not hoped for: every cell line must carry `nonzero_v1=<elems>` and
# `finite=true` (a bank of zeros agrees with itself bit-for-bit in any accumulation order — the
# trap `moe_tp2_repro` documents against itself), and each arm must emit exactly 2 cell lines
# (gate_up 4096x640 and down 640x4096; both give nkb>1, which the kb+1 prefetch defect requires).
set -u
cd "$(dirname "$0")/../../../.."
REPO="$(pwd)"
OUT="${BANKFIX_OUT:-/root/out-bankfix}"
BIN="$OUT/bin"
LOG="$OUT/oracle-corruption-arm.log"
CVD="${BANKFIX_CARD:-0}"
SRC=crates/memra-engine/cu/moe_f16_grouped.cu
mkdir -p "$BIN"

fails=0
say() { echo "$@" | tee -a "$LOG"; }

: >"$LOG"
say "=== oracle corruption-arm receipt: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
say "repo=$REPO lane sha=$(git rev-parse HEAD)"
say "toolchain: $(rustc --version) | $(nvcc --version | sed -n 's/^Cuda compilation tools, //p')"
say "cards before: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' | ')"
say "arm card: CUDA_VISIBLE_DEVICES=$CVD"

if [ -n "$(git status --porcelain)" ]; then
    say "REFUSED: the tree is dirty before the corruption arm — the corruption must be the ONLY diff"
    git status --porcelain | head -20 | tee -a "$LOG"
    exit 1
fi

# ---- build <label> : build the oracle, bank the binary + its kernel object under $label ----
build() {
    local label="$1"
    say "---- build $label ----"
    nice -n 5 cargo build --release --bin nvfp4-bank-oracle >>"$OUT/build-$label.log" 2>&1
    local rc=$?
    echo "BUILD_RC=$rc" >>"$OUT/build-$label.log"
    if [ "$rc" -ne 0 ]; then
        say "FAIL: build $label rc=$rc (see $OUT/build-$label.log)"
        fails=$((fails+1)); return 1
    fi
    cp -f target/release/nvfp4-bank-oracle "$BIN/oracle-$label"
    # The KERNEL OBJECT too, not only the binary: a binary sha can move for a Rust-side reason.
    # The .o is what nvcc produced from $SRC, so its sha is the tightest source-tracking receipt.
    local obj
    obj="$(find target/release/build -name moe_f16_grouped.o -newermt '-1 day' 2>/dev/null | head -1)"
    if [ -n "$obj" ]; then cp -f "$obj" "$BIN/moe_f16_grouped-$label.o"; fi
    say "$label binary sha256: $(sha256sum "$BIN/oracle-$label" | cut -d' ' -f1)"
    say "$label binary md5:    $(md5sum "$BIN/oracle-$label" | cut -d' ' -f1)"
    if [ -f "$BIN/moe_f16_grouped-$label.o" ]; then
        say "$label kernel .o md5: $(md5sum "$BIN/moe_f16_grouped-$label.o" | cut -d' ' -f1)"
    else
        say "$label kernel .o: NOT FOUND (source-tracking receipt incomplete)"
        fails=$((fails+1))
    fi
    say "$label finished: $(grep -c '^' "$OUT/build-$label.log") log lines, $(sed -n 's/^ *Finished.*in //p' "$OUT/build-$label.log" | tail -1)"
    return 0
}

# ---- arms <label> <want-rc> : the four tile-form arms, each asserted ----
arms() {
    local label="$1" want="$2"
    local armlog="$OUT/oracle-$label-4arms.log"
    : >"$armlog"
    echo "binary sha256: $(sha256sum "$BIN/oracle-$label" | cut -d' ' -f1)  lane sha: $(git rev-parse HEAD)" >>"$armlog"
    local name env
    for spec in "hybrid:" "sk128:MEMRA_F16G_SK=128" "sk32:MEMRA_F16G_SK=32" "tail0:MEMRA_F16G_TAIL=0"; do
        name="${spec%%:*}"; env="${spec#*:}"
        echo "######## $label ARM=$name ${env:-<unset>} ########" >>"$armlog"
        # CAPTURE-THEN-GATE: no pipe on the failable step, rc taken before anything judges it.
        local cell="$OUT/.arm-$label-$name.out"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$CVD" NVIDIA_TF32_OVERRIDE=0 MEMRA_MOE_F16G=2 $env \
            timeout 900 "$BIN/oracle-$label" >"$cell" 2>&1
        local rc=$?
        cat "$cell" >>"$armlog"
        echo "EXIT=$rc" >>"$armlog"
        # The verdict: exit code first.
        if [ "$rc" -ne "$want" ]; then
            fails=$((fails+1))
            say "FAIL: $label/$name exit=$rc want=$want"
            if [ "$want" = "1" ]; then
                say "      ^ THE VOID-GATE CONTROL DID NOT FIRE: a corrupted v2 scale read passed."
                say "        Either the binary does not track the kernel source, or the oracle is blind."
            fi
        else
            say "OK: $label/$name exit=$rc (want $want)"
        fi
        # Non-vacuity: exactly two GEOMETRY cell lines, each with a finite, non-degenerate v1 side.
        # Count on `finite=`, NOT on `^\[cell `: a DEVIATING cell prints a second `[cell ...] first
        # deviation at element N` line, so a raw `^\[cell ` count is 4 on a failing arm and 2 on a
        # passing one — asserting 2 there would have failed the corruption arm for succeeding.
        local cells nonvac
        cells="$(grep -c 'finite=' "$cell")"
        nonvac="$(grep -c 'finite=true' "$cell")"
        if [ "$cells" -ne 2 ] || [ "$nonvac" -ne 2 ]; then
            fails=$((fails+1))
            say "FAIL: $label/$name VACUOUS — geometry_lines=$cells finite_true=$nonvac (want 2/2)"
        fi
        # A zero bank agrees with itself in any accumulation order: elems must equal nonzero_v1 on
        # both cells. Only lines carrying BOTH fields are judged; the per-record reset is explicit.
        local zeroish
        zeroish="$(awk '/^\[cell /{ev="";nv="";
                        for(i=1;i<=NF;i++){
                          if($i~/^elems=/){split($i,a,"=");ev=a[2]}
                          if($i~/^nonzero_v1=/){split($i,b,"=");nv=b[2]}}
                        if(ev!="" && nv!="" && ev!=nv) print "MISMATCH"}' "$cell" | grep -c MISMATCH)"
        if [ "$zeroish" -ne 0 ]; then
            fails=$((fails+1)); say "FAIL: $label/$name has $zeroish cell(s) where nonzero_v1 != elems"
        fi
        grep -E '^\[cell ' "$cell" | sed "s/^/    $label\/$name /" >>"$LOG"
        rm -f "$cell"
    done
    say "arms log: $armlog"
}

# ================= STEP 1: THE CORRUPTION ARM =================
say ""
say "=== STEP 1: DELIBERATE CORRUPTION (*1.5f on the v2 scale read) — the oracle MUST FAIL ==="
python3 - "$SRC" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
old = "        r.f1 = g_ue4m3_to_float(wrow[(size_t)n_slots * 16 + g * 2 + sub]);\n"
new = "        r.f1 = 1.5f * g_ue4m3_to_float(wrow[(size_t)n_slots * 16 + g * 2 + sub]);  // DELIBERATE CORRUPTION (void-gate control)\n"
if s.count(old) != 1:
    sys.exit(f"REFUSED: expected exactly 1 v2 scale-read site, found {s.count(old)}")
open(p, "w").write(s.replace(old, new))
print("corruption applied to", p)
PY
if [ $? -ne 0 ]; then say "FAIL: could not apply the corruption"; exit 1; fi
say "$(git diff --stat | tail -1)"
git diff >>"$LOG"
n_changed="$(git diff --numstat | wc -l)"
n_ins="$(git diff --numstat | awk '{print $1}')"
if [ "$n_changed" -ne 1 ] || [ "$n_ins" -ne 1 ]; then
    say "FAIL: the corruption is not a single-line single-file diff (files=$n_changed ins=$n_ins)"
    git checkout -- "$SRC"; exit 1
fi
build corrupt || { git checkout -- "$SRC"; exit 1; }
arms corrupt 1

# ================= STEP 2: REVERT =================
say ""
say "=== STEP 2: REVERT the corruption ==="
git checkout -- "$SRC"
if [ -n "$(git status --porcelain)" ]; then
    say "FAIL: tree not clean after revert"; git status --porcelain | tee -a "$LOG"; exit 1
fi
say "tree clean after revert; $SRC back to $(git rev-parse --short HEAD)"

# ================= STEP 3: THE TRUE FIX =================
say ""
say "=== STEP 3: THE TRUE FIX STATE — binary sha MUST DIFFER, 4 arms MUST PASS ==="
build fixed || exit 1
sha_c="$(sha256sum "$BIN/oracle-corrupt" | cut -d' ' -f1)"
sha_f="$(sha256sum "$BIN/oracle-fixed"   | cut -d' ' -f1)"
if [ "$sha_c" = "$sha_f" ]; then
    fails=$((fails+1))
    say "FAIL: THE BINARY DID NOT TRACK THE KERNEL SOURCE — corrupt and fixed shas are IDENTICAL ($sha_c)."
    say "      This is the void-gate itself. Every oracle verdict from this build tree is void."
else
    say "OK: binary sha differs across the two kernel states (corrupt $sha_c != fixed $sha_f)"
fi
if [ -f "$BIN/moe_f16_grouped-corrupt.o" ] && [ -f "$BIN/moe_f16_grouped-fixed.o" ]; then
    oc="$(md5sum "$BIN/moe_f16_grouped-corrupt.o" | cut -d' ' -f1)"
    of="$(md5sum "$BIN/moe_f16_grouped-fixed.o" | cut -d' ' -f1)"
    if [ "$oc" = "$of" ]; then
        fails=$((fails+1)); say "FAIL: the KERNEL OBJECT is identical across the two states ($oc) — nvcc did not re-run"
    else
        say "OK: kernel object md5 differs (corrupt $oc != fixed $of) — nvcc re-ran on the source change"
    fi
fi
arms fixed 0

say ""
say "cards after: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' | ')"
say "=========================================================="
if [ "$fails" -eq 0 ]; then
    say "ORACLE CORRUPTION-ARM RECEIPT: VALID — corrupted state FAILED 4/4, fixed state PASSED 4/4, shas differ"
else
    say "ORACLE CORRUPTION-ARM RECEIPT: INVALID — $fails assertion(s) failed"
fi
exit "$fails"
