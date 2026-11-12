#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: each red arm's GPU cell must FAIL (memra#1221 part 2):
# the head with one guard disarmed (red_patch.py), its cell's crate test binary (spark_red.sh).
# usage: red_cells.sh HEADTAG
set -u
D=$HOME/.cache/memra-1221b; TAG=$1; rc=0
K=kv:lend_gpu_tests::writes_below_a_live_lend_refuse_and_the_lender_hands_its_plane_on
for n in rollback rewind pooled adopt; do
  WANT=fail bash "$D/box/unit_cells.sh" "red-$n-$TAG" "$K" || rc=1
done
WANT=fail bash "$D/box/unit_cells.sh" "red-release-$TAG" kv:lend_gpu_tests::a_release_keeps_the_lent_bytes_of_an_on_demand_plane || rc=1
WANT=fail bash "$D/box/unit_cells.sh" "red-scratch-$TAG" engine:spec::lend_floor_tests::the_mtp_scratch_refuses_a_length_below_its_lend || rc=1
echo "RED-CELLS-DONE $TAG rc=$rc"
exit $rc
