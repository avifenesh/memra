#!/usr/bin/env bash
# Item 2: DAY49 addendum D's serving shape on the 5090, on the post-fix-stack tree, where
# MEMRA_BATCH_OOM_RECOVER is gone and the retry is a card default. The arm is the binary:
# OFF_BIN keeps the 5090's error arm, ON_BIN keys the retry on the RTX 5090 class. Everything
# else is day49d-run.sh's: the 9B, MEMRA_CTX=65536, MEMRA_STEP_OOM_FAULT=batch:1, burst 8 x 6,144
# with no warm, N=5, day45-client.py and day45-parse.py, MEMRA_SERVE_SPEC=0 on the plain route.
# One gpulock lease per boot (LOCK=none inside it).
#   OFF_BIN=.. ON_BIN=.. WT=.. oom_boots.sh RECEIPT_ROOT name:arm ...
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:?worktree with research/spill-b-20260919}
OFF_BIN=${OFF_BIN:?}; ON_BIN=${ON_BIN:?}
mkdir -p "$R/boots"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/run.log"; }
for spec in "$@"; do
  IFS=: read -r name arm <<< "$spec"
  aenv=(MEMRA_STEP_OOM_FAULT=batch:1 MEMRA_CTX=65536 LOCK=none RIGDIR="$R/boots" N=5 CLIENT=day45-client.py
        PARSER=day45-parse.py MEMRA_TIMEOUT_MS_MAX=3600000 WT="$WT"
        CLIENT_ARGS="--warm-n 0 --burst 8 --length 6144 --max-tokens 64")
  case $arm in
    off) B=$OFF_BIN ;;
    on) B=$ON_BIN ;;
    off-plain) B=$OFF_BIN; aenv+=(MEMRA_SERVE_SPEC=0) ;;
    on-plain) B=$ON_BIN; aenv+=(MEMRA_SERVE_SPEC=0) ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  # DOOR=1: a tree that still has the env door (main before the fix stack): one binary, the door
  # names the arm.
  if [ "${DOOR:-0}" = 1 ] && [ "${arm#on}" != "$arm" ]; then aenv+=(MEMRA_BATCH_OOM_RECOVER=1); fi
  printf 'arm=%s\nbin_sha256=%s\nenv=%s\nclient_args=%s\n' "$arm" "$(sha256sum "$B" | cut -d' ' -f1)" \
    "${aenv[*]}" "--warm-n 0 --burst 8 --length 6144 --max-tokens 64" > "$R/boots/$name.arm.txt"
  log "boot $name start arm=$arm bin=$(sha256sum "$B" | cut -c1-16)"
  gpulock run --ttl 30m --wait 4h --for "memra c8r2 item 2: batch OOM retry boot $name ($arm, 9B fault cell)" -- \
    env -u MEMRA_BATCH_OOM_RECOVER -u MEMRA_ADMIT_W_RELEASE -u MEMRA_ADMIT_BY_MEMORY "${aenv[@]}" \
    bash "$WT/research/spill-b-20260919/run-day26-cell.sh" "$name" AB "$B" > "$R/boots/$name.launch.log" 2>&1
  log "boot $name rc=$? $(tail -1 "$R/boots/$name/REPORT.txt" 2>/dev/null | cut -c1-160)"
done
python3 "$WT/research/spill-b-20260919/day49d-read.py" serve rtx5090 "$R" > "$R/read-serve.log" 2>&1
cat "$R/read-serve.log"
echo OOM_BOOTS_DONE
