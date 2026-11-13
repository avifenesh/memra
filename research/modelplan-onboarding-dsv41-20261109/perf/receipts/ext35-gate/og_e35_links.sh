#!/bin/bash
# Dev desk: the vendor side of the 53-tape gate as one directory per kind (symlinks), for og_e35.sh.
set -u
cd ~/.cache/dsv41-og/e35 || exit 1
mkdir -p all/tapes all/ref all/floor b
for f in ../tapes/*.pt; do ln -sfn ../../$f all/tapes/$(basename $f); done
for k in ref floor; do
  for f in ../x/dec/$k/*.pt; do ln -sfn ../../$f all/$k/$(basename $f); done
done
for k in tapes ref floor; do
  for f in $k/*.pt; do ln -sfn ../../$f all/$k/$(basename $f); done
done
for k in tapes ref floor; do echo "$k: $(ls all/$k | wc -l) ($(find -L all/$k -name '*.pt' -type f | wc -l) resolve)"; done
