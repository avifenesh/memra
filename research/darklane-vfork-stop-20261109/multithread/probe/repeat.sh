#!/bin/bash
# Five repeats of each failing shape and its tgkill counterpart; one summary line per run.
set -u
P=$WORK/review/probe
O=$WORK/review/results-repeat
mkdir -p "$O"
for i in 1 2 3 4 5; do
  for m in main-spawn-nocldstop main-spawn-maskchld two-block-nocldstop two-block-nocldstop+tgkill main-spawn-nocldstop+tgkill single thread-spawn; do
    python3 -I "$P/driver.py" "$P/leader" "$O/run-$i" "$m" > "$O/run-$i-$m.txt" 2>&1
    python3 -I - "$O/run-$i/$m.jsonl" "$i" "$m" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
end = [r for r in rows if r.get("event") == "end-3s"][0]
pre = [r for r in rows if r.get("event") == "pre"][0]
print(f"run {sys.argv[2]} {sys.argv[3]:28s} first_ack_ms={end['first_ack_ms']} "
      f"counter pre={pre['worker_counter']} at_ack={end['counter_at_first_ack']} at_3s={end['worker_counter']} "
      f"threads_3s={end['threads']} sigstop_in_shdpnd_3s={end['sigstop_in_shdpnd']}")
EOF
  done
done
