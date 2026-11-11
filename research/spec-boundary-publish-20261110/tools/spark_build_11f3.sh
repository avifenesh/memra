#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's build lease, CPU only: the sm_121a servers of main
# 11f3c0920b (B') and of the rebased head's code d117bf8607 (W') from one source tree. The old
# tree (main 2628ece940) moves to src-2628ece940; the new tree is unpacked at the same path so the
# target directory's third-party crates are reused. An arm is that tree with the arm's worker.rs.
# usage: spark_build_11f3.sh
set -u
R=$HOME/.cache/memra-specpub
cd "$R" || exit 1
if [ ! -f "$R/src-11f3c0920b.done" ]; then
  [ -d "$R/src-2628ece940" ] || mv "$R/src" "$R/src-2628ece940" || exit 1
  mkdir -p "$R/src" && tar -xzf "$R/arms/src-11f3c0920b.tar.gz" -C "$R/src" || exit 1
  touch "$R/src-11f3c0920b.done"
fi
bash "$R/box/spark_build_arm.sh" base-11f3c0920b worker.main11f3.rs server || exit 1
bash "$R/box/spark_build_arm.sh" head-d117bf8607 worker.w11f3.rs server || exit 1
grep -q BUILD-DONE "$R/logs/build-base-11f3c0920b.log" && grep -q BUILD-DONE "$R/logs/build-head-d117bf8607.log"
