#!/bin/bash
# Laptop: build #1286's wrx binaries on the dev desk (dd_bins.sh, this lane's own dir and target) and install them on
# wrx as perf-bins/TAG; with TREE=1 also the commit's research/modelplan-onboarding-dsv41-20261109 at /home/access/dsv41/src-m1-<sha10> (m1.sh reads the
# research oracle and the experts-host wrapper from it; the shared src-m1 link is not touched). STAGED=1: the
# tarball src-TAG.tar is already on the dev desk. No wrx CPU or GPU.
# usage: cfix2_bins.sh TAG=SHA [TAG=SHA ...]   e.g. route2=<route head> cfix2=<cfix head>; prints BINS-DONE
set -u
O=/home/avifenesh/.cache/dsv41-cfix2
M=/home/avifenesh/projects/memra
W='~/.cache/dsv41-cfix2'
log() { echo "$(date +%T) $*"; }
ssh devdesk "mkdir -p $W" && scp -q $O/dd_bins.sh devdesk:.cache/dsv41-cfix2/ || exit 1
for pair in "$@"; do
  TAG=${pair%%=*}; SHA=${pair#*=}
  full=$(git -C $M rev-parse --verify "$SHA^{commit}") || { log "BINS-FAILED $TAG (no commit $SHA)"; exit 1; }
  if [ "${STAGED:-0}" != 1 ]; then
    # the build inputs (the research tree is 8.7 GB; two research files are embedded by the server and the tokenizer)
    git -C $M archive --format=tar "$full" -- .cargo Cargo.toml Cargo.lock crates docs tools \
      research/reasoning-schema-20260823 research/spill-b-20260919/fixtures | ssh devdesk "cat > $W/src-$TAG.tar" \
      || { log "BINS-FAILED $TAG (stage)"; exit 1; }
  fi
  log "$TAG ${full:0:10} staged; building"
  ssh devdesk "bash $W/dd_bins.sh $TAG $full" > $O/dd-bins-$TAG.log 2>&1
  tail -n 7 $O/dd-bins-$TAG.log
  grep -q "BUILD-DONE $TAG rc 0" $O/dd-bins-$TAG.log || { log "BINS-FAILED $TAG (build)"; exit 1; }
  ssh wrx "mkdir -p /home/access/dsv41/perf-bins/$TAG" || exit 1
  for f in dsv41-gate dsv41-tpbench dsv41-run memra-server memra SOURCE; do
    scp -q -3 devdesk:.cache/dsv41-cfix2/bins/$TAG/$f wrx:/home/access/dsv41/perf-bins/$TAG/$f || { log "BINS-FAILED $TAG (copy $f)"; exit 1; }
  done
  ssh wrx "cd /home/access/dsv41/perf-bins/$TAG && chmod +x dsv41-gate dsv41-tpbench dsv41-run memra-server memra && sha256sum dsv41-gate dsv41-tpbench dsv41-run memra-server memra | cut -c1-16 | tr '\n' ' ' && ./dsv41-gate --help > /dev/null 2>&1; echo \"help rc \$?\""
  if [ "${TREE:-0}" = 1 ]; then
    # m1.sh reads only the lane's research dir (the checkpoint oracle and the experts-host wrapper): that part alone
    git -C $M archive --format=tar "$full" research/modelplan-onboarding-dsv41-20261109 | ssh wrx "mkdir -p /home/access/dsv41/src-m1-${full:0:10} && tar -x -C /home/access/dsv41/src-m1-${full:0:10}" \
      && log "tree ${full:0:10} -> wrx src-m1-${full:0:10}"
  fi
  log "$TAG installed on wrx"
done
log BINS-DONE
