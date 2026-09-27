#!/usr/bin/env bash
# Day-31 counts over a receipts root (DAY31.md quotes this output; nothing is counted by hand).
#   bash counts.sh pro-single-day31/box > pro-single-day31/box/counts.log     (run from research/spill-a-20260919)
#   bash counts.sh rtx5090-day31 > rtx5090-day31/counts.log
set -uo pipefail
R=${1:?receipts root}

echo "## completion lines (A3): copy-complete batch totals, D2H receipt KV items, receipt span suffixes"
grep -rhoE 'items=[0-9]+ \([0-9]+ KV, [0-9]+ f32 spans\)' --include='*.log' --exclude=counts.log "$R" | sort | uniq -c
grep -rhoE 'D2H receipt: .* items=[0-9]+ \([0-9]+ KV planes(, draft)?\)' --include='*.log' --exclude=counts.log "$R" \
  | grep -oE 'items=[0-9]+ \([0-9]+ KV planes(, draft)?\)' | sort | uniq -c
grep -rhoE '; [0-9]+ f32 spans landed under the ticket and taken back before the retire' --include='*.log' --exclude=counts.log "$R" | sort | uniq -c

echo "## demote submissions (tokens, MB, items)"
grep -rhoE 'demote submitted off the tick: [0-9]+ tokens, [0-9.]+MB, ticket seq=[0-9]+, [0-9]+ items' --include='*.log' --exclude=counts.log "$R" \
  | sed -E 's/ticket seq=[0-9]+, //' | sort | uniq -c

echo "## logs where submissions exceed copy completions"
for f in $(grep -rlE 'demote submitted off the tick' --include='*.log' --exclude=counts.log "$R"); do
  s=$(grep -c 'demote submitted off the tick' "$f"); c=$(grep -c 'demote copy complete off the tick' "$f")
  [ "$s" != "$c" ] && echo "$f submitted=$s copy_complete=$c"
done
echo "## demote failures, verbatim"
grep -rh 'demote failed' --include='*.log' --exclude=counts.log "$R" | sort | uniq -c
echo "## span refusals (the typed refusal line)"
grep -rh 'tier D2H spans refused' --include='*.log' --exclude=counts.log "$R" | wc -l

echo "## demote ledger lines per top-level cell dir: pre-submit, helper, owner in-completion (A2 reads the double-park steady set only)"
for d in $(find "$R" -mindepth 1 -maxdepth 1 -type d | sort); do
  grep -rhoE 'demote digests landed off the tick: .*' --include='*.log' --exclude=counts.log "$d" | python3 -c '
import re, statistics, sys
rows = [(float(re.search(r"pre-submit ([\d.]+)", l).group(1)), float(re.search(r"hashed in ([\d.]+)ms", l).group(1)),
         float(re.search(r"owner in-completion ([\d.]+)ms", l).group(1))) for l in sys.stdin]
if rows:
    out = []
    for k, name in enumerate(("pre-submit", "helper", "owner in-completion")):
        x = [r[k] for r in rows]
        out.append(f"{name} median={statistics.median(x):.2f} min={min(x):.2f} max={max(x):.2f}")
    print(f"{sys.argv[1]}: N={len(rows)} " + "; ".join(out))
' "$d"
done

echo "## hits parked on a Hashing entry, per server log"
grep -rc 'hit parked on a Hashing entry' --include='*server.log' --exclude=counts.log "$R" | grep -v ':0$' | sort

echo "## verdict lines per gate log"
for f in $(find "$R" -maxdepth 3 -name '*.log' ! -name counts.log | sort); do
  v=$(grep -hE 'GATE: |PREFIX-NEWEST-TURN-FITS:' "$f" 2>/dev/null | tail -1)
  [ -n "$v" ] && echo "$f: ok=$(grep -c '^ *ok' "$f") | ${v:0:160}"
done

echo "## GPU telemetry per collector cell (command.gpu.csv)"
for c in $(find "$R" -maxdepth 2 -name command.gpu.csv | sort); do
python3 - "$c" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
h = [x.strip() for x in rows[0]]
body = rows[1:]
def col(name):
    i = [k for k, x in enumerate(h) if name in x][0]
    return [float(r[i].split()[0]) for r in body if len(r) > i and r[i].strip() not in ("", "[N/A]")]
t, p, m = col("temperature"), col("power.draw"), col("memory.used")
print(f"{sys.argv[1]}: {len(body)} samples, {body[0][0].strip()} to {body[-1][0].strip()}, "
      f"{min(t):.0f} to {max(t):.0f} C, {min(p):.1f} to {max(p):.1f} W, {min(m):.0f} to {max(m):.0f} MiB")
PY
done

if [ -d "$R/double-park/ev" ]; then
  echo "## double-park replays and receipts"
  echo "replays.log STALL REPLAY: PASS lines: $(grep -c 'STALL REPLAY: PASS' "$R/double-park/ev/replays.log")"
  grep -rhoE 'errors=[0-9]+' --include=receipt.json "$R/double-park/ev" | sort | uniq -c
  echo "compute apps before/after (data rows): $(tail -n +2 "$R/double-park/ev/compute-apps.before.csv" | wc -l)/$(tail -n +2 "$R/double-park/ev/compute-apps.after.csv" | wc -l)"
fi
echo "## collector cells (CELL.jsonl status, qualification, exit_code of the closing row)"
for c in $(find "$R" -maxdepth 2 -name CELL.jsonl | sort); do
  python3 -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])][-1]; print(sys.argv[1], r.get('status'), r.get('qualification'), r.get('exit_code'))" "$c"
done
echo "## lock retries: $(find "$R" -name lock-retries.log | wc -l) file(s)"

if [ -d "$R/double-park/ev" ]; then
  echo "## double-park ON boots: helper, copy and wall by demote index of the boot (readings, not clauses)"
  python3 - "$R/double-park/ev" <<'PY'
import glob, os, re, statistics, sys
L = re.compile(r"demote digests landed off the tick: ticket seq=\d+, \d+ payloads \([\d.]+MB\) hashed in ([\d.]+)ms .*"
               r"pre-submit ([\d.]+), .*wall ([\d.]+)ms t0 to publication")
C = re.compile(r"demote copy complete off the tick: ticket seq=\d+ complete after \d+ poll\(s\), ([\d.]+)ms from submission")
by = {}
for log in sorted(glob.glob(os.path.join(sys.argv[1], "o*", "b*-on", "server.log"))):
    n = c = 0
    for ln in open(log, errors="replace"):
        m = C.search(ln)
        if m:
            c += 1
            by.setdefault(("copy", min(c, 4)), []).append(float(m.group(1)))
        m = L.search(ln)
        if m:
            n += 1
            k = min(n, 4)
            by.setdefault(("helper", k), []).append(float(m.group(1)))
            by.setdefault(("wall", k), []).append(float(m.group(3)))
for what in ("helper", "copy", "wall"):
    parts = []
    for k in (1, 2, 3, 4):
        xs = by.get((what, k), [])
        label = f"demote {k}" if k < 4 else "steady (4+)"
        if xs:
            parts.append(f"{label} N={len(xs)} median={statistics.median(xs):.2f}")
    print(f"{what}: " + "; ".join(parts))
PY
fi

echo "## day 31: span staging fills (1b), span-refusal lines (1c), slot-class tallies (1d)"
grep -rhoE 'tier span staging: [0-9]+ fresh pinned buffer\(s\), [0-9]+ bytes charged' --include='*.log' --exclude=counts.log "$R" | sort | uniq -c
grep -rhoE 'demote failed \(tier D2H spans refused: [^;]*; nothing demoted' --include='*.log' --exclude=counts.log "$R" | sort | uniq -c
grep -rhoE 'by slot class: conv [0-9]+ \([0-9]+ B\), ssm [0-9]+ \([0-9]+ B\), hidden [0-9]+ \([0-9]+ B\), logits [0-9]+ \([0-9]+ B\)' --include='*.log' --exclude=counts.log "$R" | sort | uniq -c
