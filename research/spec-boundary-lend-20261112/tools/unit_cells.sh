#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: memra#1221 part 2's GPU unit cells on one build's
# lib test binaries (bins/TAG/{kv,engine,server}-lib-test). Each cell runs alone (--exact
# --ignored); its log and verdict line go to runs/unit-TAG/. WANT=fail inverts the verdict (a red
# arm must fail). MEMRA_B1_MODEL adds the 27B's own geometry to the server cells.
# usage: [WANT=pass|fail] unit_cells.sh TAG [CELL...]   CELL = kv:NAME | engine:NAME | server:NAME
set -u
D=$HOME/.cache/memra-1221b; TAG=$1; shift
WANT=${WANT:-pass}
OUT=$D/runs/unit-$TAG; mkdir -p "$OUT"
export MEMRA_B1_MODEL=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
cells=${*:-kv:lend_gpu_tests::writes_below_a_live_lend_refuse_and_the_lender_hands_its_plane_on \
kv:lend_gpu_tests::a_release_keeps_the_lent_bytes_of_an_on_demand_plane \
engine:spec::lend_floor_tests::the_mtp_scratch_refuses_a_length_below_its_lend \
server:worker::tests::spec_boundary_publish_is_the_copy_program \
server:worker::tests::spec_boundary_publish_drops_a_plane_whose_copy_failed \
server:worker::tests::a_lent_plane_outlives_its_lender_and_its_last_reader_frees_it \
server:worker::tests::adopted_planes_compact_only_when_that_frees_more_than_it_copies \
server:worker::tests::b1_snapshot_and_restore_are_the_copy_program \
server:worker::tests::an_unwritten_plane_is_charged_to_the_insert_budget_reading}
{ date -Is; sha256sum "$D/bins/$TAG/"*-lib-test | sed "s|$HOME|~|"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } >> "$OUT/verdicts.txt"
rc=0
for c in $cells; do
  bin=$D/bins/$TAG/${c%%:*}-lib-test; name=${c#*:}; log=$OUT/${name##*::}.log
  timeout 600 "$bin" "$name" --exact --ignored --nocapture --test-threads 1 > "$log" 2>&1
  r=$?
  ok=$([ $r = 0 ] && echo pass || echo fail)
  echo "$name rc=$r $ok (want $WANT) $(grep -E '^test result' "$log" | head -1)" | tee -a "$OUT/verdicts.txt"
  [ "$ok" = "$WANT" ] || rc=1
  grep -h -E '^\[(specpub|lend floor|b1) ' "$log" | sed "s|$HOME|~|" >> "$OUT/cell-lines.txt"
done
echo "UNIT-CELLS-DONE $TAG rc=$rc" | tee -a "$OUT/verdicts.txt"
exit $rc
