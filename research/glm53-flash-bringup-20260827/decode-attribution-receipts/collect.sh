#!/bin/bash
for f in ~/s-A1.log ~/s-A2PIN.log ~/s-A2LOW.log ~/s-A3BF16.log ~/s-A1R.log ~/s-A2R.log ~/s-A4BEST.log; do
  [ -f "$f" ] || continue
  echo "### $f"
  head -1 "$f"
  grep -E '"STEADY"|"SEA"' "$f"
done
