#!/usr/bin/env bash
# LAW:rebuild-after-checkout-attribution + the two wrong-binary handoffs of 08-29: a binary
# is trusted because it is NEWER than every source it was built from and because its OWN
# STRINGS carry the levers under test - never because of checkout provenance. A 0.04s
# "Finished" after a checkout is a FAILED-CHECKOUT alarm, so the build elapsed is asserted too.
set -uo pipefail
W=/root/memra-1m
BIN=$W/target/release/memra-server
fail=0
echo "=== git identity of the built tree ==="
git -C "$W" log -1 --format="%H%n%s%n%ci"
echo "  dirty files: $(git -C "$W" status --porcelain | wc -l)"
echo "  PINNED: lane/glm53-flash-bringup 92ea07376 (dedup 5848b3d0c NOT merged into bringup at fetch)"
echo
echo "=== binary identity ==="
ls -l --time-style=full-iso "$BIN"
echo "  sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
echo "  sha16  $(sha256sum "$BIN" | cut -c1-16)"
echo
echo "=== build attribution ==="
grep -E "BUILD_HEAD|BUILD_DIRTY|BUILD_START|BUILD_END|BUILD_ELAPSED_S|BUILD_RC" /root/out-1m/logs/01-build.log || true
EL=$(grep -oP 'BUILD_ELAPSED_S=\K[0-9]+' /root/out-1m/logs/01-build.log | tail -1)
if [ -n "${EL:-}" ] && [ "$EL" -lt 20 ]; then
  echo "  *** ALARM: build elapsed ${EL}s < 20s = the failed-checkout/no-op signature ***"; fail=1
else
  echo "  build elapsed ${EL:-?}s: a real compile"
fi
echo
echo "=== FRESHNESS: binary mtime vs newest source mtime ==="
NEWEST=$(find "$W/crates" -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.toml' \) -printf '%T@ %p\n' | sort -rn | head -1)
NEWEST_T=${NEWEST%% *}; NEWEST_P=${NEWEST#* }
BIN_T=$(stat -c %Y "$BIN")
echo "  newest source: $NEWEST_P at $(date -u -d @${NEWEST_T%.*} +%FT%TZ)"
echo "  binary at $(date -u -d @"$BIN_T" +%FT%TZ)"
if [ "$BIN_T" -gt "${NEWEST_T%.*}" ]; then
  echo "  FRESH: binary is newer than every source"
else
  echo "  *** STALE: binary is NOT newer than its sources. REBUILD BEFORE TRUSTING. ***"; fail=1
fi
echo
echo "=== strings census: the levers this window measures must be IN this binary ==="
# each marker, with why it must be present (or absent)
while IFS='|' read -r want marker why; do
  n=$(strings "$BIN" 2>/dev/null | grep -c -- "$marker")
  ok="ok"
  if [ "$want" = "gt0" ] && [ "$n" -lt 1 ]; then ok="*** MISSING ***"; fail=1; fi
  if [ "$want" = "eq0" ] && [ "$n" -ne 0 ]; then ok="*** PRESENT, should be absent ***"; fail=1; fi
  printf "  %-6s %-42s %-5s %s  (%s)\n" "$want" "$marker" "$n" "$ok" "$why"
done <<'MARKERS'
gt0|mla-tc-prefill|the re-priced prefill lever, default ON
gt0|MEMRA_MLA_TC_PREFILL|its rollback seam is readable
gt0|moe-grouped-prefill|the demo's lever 1, now default ON
gt0|bf16-tcols-wide|door T
gt0|bf16-tcols-x1|door X
gt0|topk-shards|door K
gt0|glm5-verify-ws|door W
gt0|MEMRA_TIMEOUT_MS_MAX|the measurement deadline override the 1M prime needs
gt0|MEMRA_PP_SPLITS|the PP4 uneven-splits door
gt0|MEMRA_MOE_SLOTS|the capped-SLRU arena knob
gt0|verify walk BATCHED per layer|the batched verify walk (spec at depth)
gt0|glm5-phase-v|trace=2 per-layer verify sub-split (cell 4)
gt0|draft source = dflash2|the pinned DFlash2 drafter path
MARKERS
echo
echo "=== drafter artifact pin ==="
find /root/models/glm53-dflash2 -type f | sort | head -10
echo "  drafter tree sha16: $(find /root/models/glm53-dflash2 -type f | sort | xargs sha256sum | sha256sum | cut -c1-16)"
echo
echo "=== corpus pin (must equal the 1m-demo banked sha for comparability) ==="
sha256sum /root/corpus-1m/corpus-1m.txt
echo "  demo banked: a07d4fcd595b57bd3019bb4a16a1a99137c3d04e15b79091183af22141a5d868"
sha256sum /root/corpus-1m/corpus-1m.txt | grep -q a07d4fcd595b57bd3019bb4a16a1a99137c3d04e15b79091183af22141a5d868 \
  && echo "  CORPUS MATCHES the demo: every rung is a prefix of the same immutable file" \
  || { echo "  *** CORPUS MISMATCH: demo rows are NOT comparable ***"; fail=1; }
echo
[ "$fail" -eq 0 ] && echo "VERIFY_BINARY=GREEN" || echo "VERIFY_BINARY=RED"
exit "$fail"
