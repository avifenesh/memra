#!/usr/bin/env bash
# Runs ON the dev desk after `dd_bins.sh TAG SHA`: snapshots that build's nvcc outputs (memra-engine's build out dir:
# every .o, libmemra_mmq.a, the fatbins) to kcmp/TAG and compares them with kcmp/cfix2 (a05173ee27's build of
# 2026-11-13 21:29 dev desk time), file by file and libmemra_mmq.a member by member, plus the binaries' sha256.
# usage: dd_kcmp.sh TAG
set -u
TAG=$1
W=$HOME/.cache/dsv41-cfix2
K=$W/kcmp
OUTDIR=
for d in $(ls -td "$W"/target-rel/release/build/memra-engine-*/out); do
  if [ -f "$d/dsv41_gpu.o" ]; then OUTDIR=$d; break; fi
done
echo "out dir $OUTDIR (dsv41_gpu.o $(date -r "$OUTDIR/dsv41_gpu.o" +%T), bins $(date -r "$W/bins/$TAG/dsv41-gate" +%T))"
mkdir -p "$K/$TAG" "$K/ar-cfix2" "$K/ar-$TAG"
cp -a "$OUTDIR"/*.o "$OUTDIR"/*.a "$OUTDIR"/*.fatbin "$K/$TAG/"
(cd "$K/$TAG" && sha256sum *.o *.a *.fatbin) > "$K/$TAG.sha256"
(cd "$W/bins/$TAG" && sha256sum dsv41-gate dsv41-tpbench dsv41-run memra-server memra) > "$K/$TAG-bins.sha256"
cmp_lists() { # A B: one line per name, same or DIFFER, then names on one side only
  join -1 2 -2 2 <(sort -k2 "$1") <(sort -k2 "$2") | awk '{print ($2==$3 ? "same   " : "DIFFER ") $1}'
  join -v1 -1 2 -2 2 <(sort -k2 "$1") <(sort -k2 "$2") | awk '{print "only cfix2  " $1}'
  join -v2 -1 2 -2 2 <(sort -k2 "$1") <(sort -k2 "$2") | awk '{print "only '"$TAG"'  " $1}'
}
echo "== nvcc outputs, cfix2 vs $TAG"
cmp_lists "$K/cfix2.sha256" "$K/$TAG.sha256"
echo "== libmemra_mmq.a members, cfix2 vs $TAG"
(cd "$K/ar-cfix2" && ar x "$K/cfix2/libmemra_mmq.a" && sha256sum *.o > "$K/ar-cfix2.sha256")
(cd "$K/ar-$TAG" && ar x "$K/$TAG/libmemra_mmq.a" && sha256sum *.o > "$K/ar-$TAG.sha256")
echo "members: cfix2 $(ar t "$K/cfix2/libmemra_mmq.a" | tr '\n' ' ')"
echo "members: $TAG $(ar t "$K/$TAG/libmemra_mmq.a" | tr '\n' ' ')"
cmp_lists "$K/ar-cfix2.sha256" "$K/ar-$TAG.sha256"
echo "== binaries, cfix2 vs $TAG"
cmp_lists <(grep -v -e build.log -e SOURCE "$K/cfix2-bins.sha256") "$K/$TAG-bins.sha256"
echo "== source-tree path strings in the binaries (src-cfix2-a05173ee27 / src-$TAG-)"
for b in dsv41-gate memra-server; do
  echo "$b cfix2 $(grep -a -c src-cfix2-a05173ee27 "$W/bins/cfix2/$b") $TAG $(grep -a -c "src-$TAG-" "$W/bins/$TAG/$b")"
done
echo "KCMP-DD-DONE $(date +%T)"
