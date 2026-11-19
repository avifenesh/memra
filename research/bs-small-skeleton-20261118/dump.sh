#!/usr/bin/env bash
# The checking machine (any box with cuobjdump 13.1; the build host has none): fetch a build host build's objects
# (obj-TAG) and dump each one's SASS and resource usage.
# usage: dump.sh TAG
set -u
BUILD=${BUILD:?set BUILD to the build host (ssh alias)}
TAG=$1
B=$HOME/.cache/bs-small
mkdir -p "$B/obj/$TAG" "$B/cmp/$TAG"
rsync -a "$BUILD:.cache/bs-small/obj-$TAG/*.o" "$BUILD:.cache/bs-small/obj-$TAG/objs.sha256" "$B/obj/$TAG/" || exit 1
for o in "$B/obj/$TAG"/*.o; do
  u=$(basename "$o" .o)
  /usr/local/cuda-13.1/bin/cuobjdump -sass "$o" > "$B/cmp/$TAG/$u.sass" || exit 1
  /usr/local/cuda-13.1/bin/cuobjdump -res-usage "$o" > "$B/cmp/$TAG/$u.res" || exit 1
done
wc -l "$B/cmp/$TAG"/*.sass
grep -c 'Function :' "$B/cmp/$TAG"/*.sass
