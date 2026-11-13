#!/bin/bash
# The byte families of the ext35 gate's builds: per group, each build's list sha of the four gate files of every case
# (top_ids, top_vals, lse, argmax: as ext35_gate_inner.sh writes GROUP.sha256) and of the full-logit files (full.f32,
# the first steps' whole rows), with the earlier receipts beside them (ordertapes-k1 = the current route's first run,
# ordertapes-cutlass = 498473abd8).
# usage: bash ext35_families.sh > receipts-perf/ext35-gate/families.txt
set -u
P=/home/access/dsv41/receipts-perf
lsha() {  # DIR PATTERN...
  local d=$1; shift
  [ -d $d ] || { echo "-"; return; }
  local n
  n=$(cd $d && ls $@ 2> /dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "- (0 files)"; return; }
  echo "$(cd $d && sha256sum $@ 2> /dev/null | sort -k2 | cut -c1-64 | sha256sum | cut -c1-16) ($n files)"
}
for g in ext35 chat long long257; do
  echo "== $g"
  for b in k3 kpc emb r2 cutlass cfix; do
    echo "$b: gate $(cd $P && lsha ext35-$b/$g '*/top_ids.i32' '*/top_vals.f32' '*/lse.f64' '*/argmax.i32')  full $(cd $P && lsha ext35-$b/$g '*/full.f32')"
  done
  for o in k1 cutlass; do
    [ -d $P/ordertapes-$o/$g ] || continue
    echo "ordertapes-$o: gate $(cd $P && lsha ordertapes-$o/$g '*/top_ids.i32' '*/top_vals.f32' '*/lse.f64' '*/argmax.i32')  full $(cd $P && lsha ordertapes-$o/$g '*/full.f32')"
  done
done
