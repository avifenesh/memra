#!/usr/bin/env bash
# Build host, memra#1266: arm M3 = main with memra#1256's worker.rs (the emulated landing) from
# the F3 tree: main's four changed crate files and env registry swapped in (fresh mtimes), built
# with dd_build.sh into bins5090/M3, then the F3 files put back.
set -u
R=<lane>; S=$R/src; K=$R/f3-keep; mkdir -p "$K"
E=crates/memra-engine/src; V=crates/memra-server/src
cp "$S/$E/spec.rs" "$K/spec.rs"; cp "$S/$E/lib.rs" "$K/elib.rs"; cp "$S/$V/lib.rs" "$K/slib.rs"; cp "$S/$V/worker.rs" "$K/worker.rs"; cp "$S/crates/memra-engine/env-registry.txt" "$K/reg.txt"
cp "$R/main3/memra-engine-spec.rs" "$S/$E/spec.rs"; cp "$R/main3/memra-engine-lib.rs" "$S/$E/lib.rs"
cp "$R/main3/memra-server-lib.rs" "$S/$V/lib.rs"; cp "$R/main3/memra-server-worker.rs" "$S/$V/worker.rs"
cp "$R/main3/crates-env-registry.txt" "$S/crates/memra-engine/env-registry.txt"
bash "$R/dd_build.sh" M3
cp "$K/spec.rs" "$S/$E/spec.rs"; cp "$K/elib.rs" "$S/$E/lib.rs"; cp "$K/slib.rs" "$S/$V/lib.rs"; cp "$K/worker.rs" "$S/$V/worker.rs"; cp "$K/reg.txt" "$S/crates/memra-engine/env-registry.txt"
echo "M3-DONE $(date -Is)" >> "$R/build-M3.log"
