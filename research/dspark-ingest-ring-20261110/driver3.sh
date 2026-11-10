#!/usr/bin/env bash
# The memra #1159 GPU cells, one lease per cell. Everything rig-specific comes from the caller:
#   CELLS    output directory               REPO     a memra checkout (the serve smoke)
#   GATE     dspark_q38_gate binary         SERVER   memra-server binary
#   MODEL    Qwen3.8-27B NVFP4 + Q5_K MTP GGUF
#   DSPARK   the q38 DSpark agentic drafter  DFLASH2  the q38 DFlash2 drafter
#   LEASE    a command that runs its arguments under one GPU lease: LEASE <cell name> <command...>
#            (any rule about who goes first belongs in it)
# usage: driver3.sh gates|serves
set -u
phase=$1
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$CELLS"
L=$CELLS/driver-$phase.log
cell() { # name, then the command
  local name=$1
  shift
  echo "$(date -Is) $name start" >> "$L"
  "$LEASE" "$name" "$@" > "$CELLS/$name.log" 2>&1
  local rc=$?
  echo "$(date -Is) $name rc=$rc last: $(grep -E '^== dspark_q38_gate|^dspark-serve-smoke:|panicked at' "$CELLS/$name.log" | tail -n 1)" >> "$L"
}
gate() { # name draft env...
  local name=$1 draft=$2
  shift 2
  cell "$name" bash "$here/cell.sh" "$GATE" "$MODEL" "$draft" "$@"
}
serve() { # name env...
  local name=$1
  shift
  rm -rf "$CELLS/$name-ev"
  cell "$name" bash "$here/serve-cell.sh" "$REPO" "$SERVER" "$MODEL" "$DSPARK" "$CELLS/$name-ev" "$@"
}
case "$phase" in
  gates)
    gate fix-ladder-eager "$DSPARK" RUST_BACKTRACE=1 MEMRA_DSPARK_VT=ladder MEMRA_DSPARK_FULLG_DEBUG=1
    gate fix-ladder-vg "$DSPARK" RUST_BACKTRACE=1 MEMRA_DSPARK_VT=ladder MEMRA_DSPARK_VERIFY_GRAPH=1 MEMRA_DSPARK_FULLG_DEBUG=1
    gate fix-default-eager "$DSPARK" RUST_BACKTRACE=1 MEMRA_DSPARK_FULLG_DEBUG=1
    gate fix-dflash2-vg "$DFLASH2" RUST_BACKTRACE=1 MEMRA_DSPARK_VERIFY_GRAPH=1 MEMRA_DSPARK_FULLG_DEBUG=1
    ;;
  serves)
    serve serve-fix-ladder RUST_BACKTRACE=1 MEMRA_DSPARK_VT=ladder
    serve serve-fix-default RUST_BACKTRACE=1
    serve serve-fix-restore-ladder RUST_BACKTRACE=1 MEMRA_DSPARK_VT=ladder MEMRA_DSPARK_PREFIX_RESTORE=1
    ;;
  *)
    echo "usage: driver3.sh gates|serves" >&2
    exit 2
    ;;
esac
touch "$CELLS/$phase.done"
