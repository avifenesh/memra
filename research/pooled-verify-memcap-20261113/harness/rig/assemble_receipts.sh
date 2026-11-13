#!/usr/bin/env bash
# memra#1266: copy the lane's receipts and harness into the worktree's research directory, then
# sanitize paths and hosts. Rerunnable: it rebuilds the directory's run copies from the runs.
# usage: assemble_receipts.sh
set -u
L=<lane>
R=$L/wt/research/pooled-verify-memcap-20261113
mkdir -p "$R/harness/rig" "$R/harness/gb10" "$R/harness/build-host"
# 5090 runs (diag, agentic c=2 and c=4, normal cells, gates, bench), each kept whole (small).
for pair in "diag1:diag-5090" "ag2:agentic-5090-c2" "ag4:agentic-5090-c4" "nc:normal-5090" "bench:bench-5090"; do
  src=${pair%%:*}; dst=${pair#*:}
  [ -d "$L/runs/$src" ] || continue
  rm -rf "${R:?}/$dst"; mkdir -p "$R/$dst"
  cp -a "$L/runs/$src/." "$R/$dst/"
done
if [ -d "$L/runs/gates" ]; then
  rm -rf "${R:?}/gates-5090"; mkdir -p "$R/gates-5090"
  for f in "$L"/runs/gates/*; do
    if [ -d "$f" ]; then tar -czf "$R/gates-5090/$(basename "$f").tar.gz" -C "$L/runs/gates" "$(basename "$f")"; else cp -a "$f" "$R/gates-5090/"; fi
  done
fi
# GB10 runs, pulled from the unit.
for pair in "gag2:agentic-gb10-c2" "gnc:normal-gb10"; do
  src=${pair%%:*}; dst=${pair#*:}
  [ -d "$L/gb10-runs/$src" ] || continue
  rm -rf "${R:?}/$dst"; mkdir -p "$R/$dst"
  cp -a "$L/gb10-runs/$src/." "$R/$dst/"
done
cp -a "$L/runs/diag1/tl-01-D.txt" "$R/diag-5090/" 2>/dev/null
# CPU receipts.
mkdir -p "$R/cpu"
cp -a "$L/cpu/." "$R/cpu/" 2>/dev/null
# Harness: rig, GB10 and build-host scripts, the step files.
cp -a "$L"/tools/{diag_probe.py,rtx_cell.sh,rtx_chain.sh,normal_5090.sh,gate_5090.sh,bench_5090.sh,timeline.py,agentic_table.py,normal_table.py,round_cost.py,chain.sh,unit_cell.sh,landing_quiet.sh,sanitize_receipts.py,assemble_receipts.sh} "$R/harness/rig/"
cp -a "$L"/steps-*.txt "$R/harness/rig/"
cp -a "$L"/gb10/box/{build_gb10.sh,agentic_gb10.sh,normal_gb10.sh,serve.sh} "$R/harness/gb10/"
cp -a "$L"/tools/{dd_build.sh,dd_check.sh,red_arms.py} "$R/harness/build-host/"
for f in dd_red.sh dd_full.sh dd_bench.sh; do scp -q "build-host:.cache/memra-1266/$f" "$R/harness/build-host/$f"; done
python3 "$L/tools/sanitize_receipts.py" "$R" > "$L/sanitize.log"
du -sh "$R"
