#!/usr/bin/env bash
# Rig, inside a 5090 lease (memra#1221 part 2 review round 1): the server crate's GPU cells of the
# review head's test binary on the RTX 5090, each alone, MEMRA_B1_MODEL adding the 27B's geometry.
set -u
D=$HOME/.cache/memra-1221b; BIN=$D/bins5090/$1/server-lib-test; OUT=$D/runs/unit5090-$1
mkdir -p "$OUT"
export MEMRA_B1_MODEL=/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
{ date -Is; sha256sum "$BIN" | sed "s|$HOME|~|"; nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader; } >> "$OUT/verdicts.txt"
rc=0
for name in two_entries_of_one_session_share_their_books spec_boundary_publish_is_the_copy_program \
  spec_boundary_publish_drops_a_plane_whose_copy_failed a_lent_plane_outlives_its_lender_and_its_last_reader_frees_it \
  adopted_planes_compact_only_when_that_frees_more_than_it_copies a_failed_compaction_stops_the_pass_until_bytes_are_freed \
  b1_snapshot_and_restore_are_the_copy_program; do
  timeout 600 "$BIN" "worker::tests::$name" --exact --ignored --nocapture --test-threads 1 > "$OUT/$name.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$name rc=$r $(grep -E '^test result' "$OUT/$name.log" | head -1)" >> "$OUT/verdicts.txt"
  grep -h -E '^\[(specpub|b1) ' "$OUT/$name.log" | sed "s|$HOME|~|" >> "$OUT/cell-lines.txt"
done
echo "UNIT-CELLS-DONE $1 rc=$rc" >> "$OUT/verdicts.txt"
exit $rc
