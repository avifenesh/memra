#!/usr/bin/env bash
# CELL 3 — THE SHIP CONFIG AT DEPTH (timed, marker held by the caller). THE OWNER'S NUMBER.
#
# Spec at depth is UNMEASURED: the verify walk pays the same DSA indexer scan x(K+1), so the
# curve is what tells us whether spec survives depth or is eaten by it. Ship config, exactly
# as the fleet serves it (mv-/struct-battery serving env):
#   MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2  the PINNED b33c0347 drafter
#   MEMRA_GLM5_SPEC=1                             the serve spec route
#   MEMRA_SPEC_PMIN=0.7                           the flip-rebattery tau
#   MEMRA_SPEC_K unset                            AUTO-K (the per-request serving policy)
#   MEMRA_SPEC_STATS=1                            per-slot accept histogram -> acceptance rows
# Doors T/X/K/W stay DEFAULT ON (unset is an ON arm). Every rung carries BOTH a greedy row
# (the byte instrument, comparable with cell 2) and a VENDOR-DEFAULT sampled row (a request
# with NO sampling params - the real traffic shape; owner law: we never serve greedy, and
# "verified" is a spec-engagement receipt from the server log, never a 200).
# usage: c3_ship.sh [rung...]   default: R16K R131K R262K R525K
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c3
mkdir -p "$D"
SPEC=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7 MEMRA_SPEC_STATS=1)
# Each argument is RUNG:arms where arms is any of g (greedy) and v (vendor-default sampled),
# e.g. "R16K:gv R131K:gv R262K:g R525K:g R1M:gv". Per-rung arm choice exists because a
# vendor row costs a SECOND FULL PRIME at that depth (MEMRA_PREFIX_CACHE_MB=0 is pinned, so
# every request is an honest cold prime and nothing is reused), and at 525k/1M that is tens
# of minutes. Where a vendor row is omitted the receipt says so by name - never silently.
RUNGS=("$@"); [ ${#RUNGS[@]} -eq 0 ] && RUNGS=(R16K:gv R131K:gv R262K:g R525K:g R1M:gv)
declare -A CH=( [R16K]=64400 [R131K]=527000 [R262K]=1054000 [R525K]=2161700 [R1M]=4282700 )
declare -A MT=( [R16K]=128   [R131K]=128    [R262K]=128     [R525K]=128     [R1M]=256 )
{
date -u +%FT%TZ
echo "######## CELL 3: SHIP CONFIG AT DEPTH (timed) rungs=${RUNGS[*]} ########"
echo "arm spec: RUNG:g=greedy only, RUNG:gv=greedy + vendor-default sampled"
echo "ship extras: ${SPEC[*]}  (K unset = auto-K policy)"
bash "$OUT/serve.sh" start c3-ship "${SPEC[@]}" || { echo "C3_EXIT=BOOTFAIL"; exit 1; }
bash "$OUT/vramwatch.sh" "$D/vram.csv" 5 & VW=$!
echo "vramwatch pid $VW"
echo; echo "=== warm rung (arena populate + first spec round, arms the announces) ==="
bash "$OUT/rung.sh" c3-ship W1K 4200 32 greedy "$D"
echo; echo "=== ENGAGE gate: spec + doors T/X/K/W + MLA-TC must all announce ==="
bash "$OUT/serve.sh" engage c3-ship "${SPEC[@]}" || echo "C3_WARN=engage-red (see logs/boot-c3-ship.engage)"
for spec in "${RUNGS[@]}"; do
  r=${spec%%:*}; arms=${spec#*:}; [ "$arms" = "$spec" ] && arms=gv
  case "$arms" in *g*) ;; *) echo "=== $r: greedy OMITTED by plan ===";; esac
  case "$arms" in
    *g*) echo; echo "=== $r greedy (chars=${CH[$r]}) ==="
         bash "$OUT/rung.sh" c3-ship "$r" "${CH[$r]}" "${MT[$r]}" greedy "$D" \
           || echo "C3_WARN=$r greedy failed, the failure is the receipt" ;;
  esac
  case "$arms" in
    *v*) echo; echo "=== $r VENDOR-DEFAULT sampled (the real traffic shape) ==="
         bash "$OUT/rung.sh" c3-ship "$r" "${CH[$r]}" "${MT[$r]}" vendor "$D" \
           || echo "C3_WARN=$r vendor failed, the failure is the receipt" ;;
    *)   echo; echo "=== $r VENDOR ROW OMITTED BY BUDGET PLAN (a vendor row is a second"
         echo "    full cold prime at this depth; named here so the omission is explicit) ===" ;;
  esac
done
kill "$VW" 2>/dev/null && echo "vramwatch $VW stopped"
echo; echo "--- per-card VRAM peak over cell 3 ---"
python3 - "$D/vram.csv" <<'PY'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0] != "ts"]
for g in range(4):
    pk = max(int(r[g+1]) for r in rows)
    print(f"  gpu{g}: peak {pk} MiB ({100*pk/97887:.1f}%)")
PY
echo "--- acceptance lines across the boot ---"
grep -E "\[glm5-acc\]" "$OUT/logs/boot-c3-ship.log" | tail -25
echo "--- boot-wide error census (must be 0) ---"
grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error|OUT_OF_MEMORY" "$OUT/logs/boot-c3-ship.log"
bash "$OUT/serve.sh" stop
date -u +%FT%TZ
echo "C3_DONE"
} 2>&1 | tee "$OUT/logs/c3.log"
