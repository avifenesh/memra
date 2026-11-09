#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: round 5 (GB10 single-user decode) scout cells on
# main's Qwen3.8-27B default (the speculative path the launcher serves), main 65fb297a59's binary.
#   decode: one cold request each at ~1k, ~131k and ~260k prompt tokens, 512 max tokens, every server
#           line timestamped with MEMRA_TICK_TRACE=1 (per-round drafted / accepted)
#   nsys1k: the same server under Nsight Systems, one ~1k-token request with 512 max tokens
# usage: r5_cell.sh NAME
set -u
D=$HOME/.cache/memra-gb10-r4
BIN=$D/bins/tfb-65fb297a59
NAME=$1
OUT=$D/runs/r5-$NAME; mkdir -p "$OUT"
case "$NAME" in
  decode)
    OUT=$OUT BIN=$BIN EXTRA="MEMRA_TICK_TRACE=1" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve.sh" &
    sp=$!
    sleep 2
    spid=$(cat "$OUT/server.pid")
    waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1 || { kill -TERM "$spid"; wait $sp; echo R5-FAIL; exit 1; }
    python3 "$D/box/pf_long_memra.py" "$OUT/pf.jsonl" --targets 1024 131072 260000 --reps 1 1 1 --max-tokens 512 --tag r5-decode >> "$OUT/run.log" 2>&1
    kill -TERM "$spid"
    waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
    wait $sp
    ;;
  k[0-9])
    # K re-price probe: both cold rungs pinned to N through their own env (the served policy
    # path, not the operator pin), one ~1k and one ~131k cold request
    N=${NAME#k}
    OUT=$OUT BIN=$BIN EXTRA="MEMRA_TICK_TRACE=1 MEMRA_SPEC_K_COLD_SHORT=$N MEMRA_SPEC_K_COLD_LONG=$N" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve.sh" &
    sp=$!
    sleep 2
    spid=$(cat "$OUT/server.pid")
    waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1 || { kill -TERM "$spid"; wait $sp; echo R5-FAIL; exit 1; }
    python3 "$D/box/pf_long_memra.py" "$OUT/pf.jsonl" --targets 1024 131072 --reps 1 1 --max-tokens 512 --tag "r5-$NAME" >> "$OUT/run.log" 2>&1
    kill -TERM "$spid"
    waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
    wait $sp
    ;;
  nsys1k|nsys131k)
    T=1024; [ "$NAME" = nsys131k ] && T=131072
    sed -e 's/--targets "$TARGET" --reps 1/--targets "$TARGET" --reps 1 --max-tokens 512/' "$D/box/profile_arm2.sh" > "$D/box/profile_arm2_dec.sh"
    bash "$D/box/profile_arm2_dec.sh" "$OUT" "$BIN" "MEMRA_TICK_TRACE=1" "$T" > "$D/logs/r5-$NAME.out" 2>&1
    ;;
esac
echo "R5-DONE $NAME"
