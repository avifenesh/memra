#!/usr/bin/env bash
# Day-32 counts over a receipts root (DAY32.md quotes this output; nothing is counted by hand). The pre-H2D run of the
# double-park cell under <root>/pre is excluded from every section but the last two, which read it by name.
#   bash counts.sh pro-single-day32/box > pro-single-day32/box/counts.log     (run from research/spill-a-20260919)
#   bash counts.sh rtx5090-day32 > rtx5090-day32/counts.log
set -uo pipefail
R=${1:?receipts root}

echo "## completion lines (A3): copy-complete batch totals, D2H receipt KV items, receipt span suffixes"
grep -rhoE 'items=[0-9]+ \([0-9]+ KV, [0-9]+ f32 spans\)' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
grep -rhoE 'D2H receipt: .* items=[0-9]+ \([0-9]+ KV planes(, draft)?\)' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" \
  | grep -oE 'items=[0-9]+ \([0-9]+ KV planes(, draft)?\)' | sort | uniq -c
grep -rhoE '; [0-9]+ f32 spans landed under the ticket and taken back before the retire' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c

echo "## demote submissions (tokens, MB, items)"
grep -rhoE 'demote submitted off the tick: [0-9]+ tokens, [0-9.]+MB, ticket seq=[0-9]+, [0-9]+ items' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" \
  | sed -E 's/ticket seq=[0-9]+, //' | sort | uniq -c

echo "## logs where submissions exceed copy completions"
for f in $(grep -rlE 'demote submitted off the tick' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R"); do
  s=$(grep -c 'demote submitted off the tick' "$f"); c=$(grep -c 'demote copy complete off the tick' "$f")
  [ "$s" != "$c" ] && echo "$f submitted=$s copy_complete=$c"
done
echo "## demote failures, verbatim"
grep -rh 'demote failed' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
echo "## span refusals (the typed refusal line)"
grep -rh 'tier D2H spans refused' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | wc -l

echo "## demote ledger lines per top-level cell dir: pre-submit, helper, owner in-completion (A2 reads the double-park steady set only)"
for d in $(find "$R" -mindepth 1 -maxdepth 1 -type d | sort); do
  grep -rhoE 'demote digests landed off the tick: .*' --include='*.log' --exclude=counts.log --exclude-dir=pre "$d" | python3 -c '
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
grep -rhoE 'tier span staging: [0-9]+ fresh pinned buffer\(s\), [0-9]+ bytes charged' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
grep -rhoE 'demote failed \(tier D2H spans refused: [^;]*; nothing demoted' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
grep -rhoE 'by slot class: conv [0-9]+ \([0-9]+ B\), ssm [0-9]+ \([0-9]+ B\), hidden [0-9]+ \([0-9]+ B\), logits [0-9]+ \([0-9]+ B\)' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c

echo "## day 32: the H2D half (promote staging fills, submissions with spans, H2D receipt span terms, promote span refusals)"
grep -rhoE 'promote staging fill off the tick: [0-9]+ tokens, [0-9]+ f32 planes \([0-9.]+MB\)' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
grep -rhoE 'promote submitted off the tick: [0-9]+ tokens, [0-9.]+MB, ticket seq=[0-9]+, [0-9]+ items on the contracts door.s copy stream, [0-9]+ f32 spans' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" \
  | sed -E 's/ticket seq=[0-9]+, //' | sort | uniq -c
echo "submissions without spans: $(grep -rh 'promote submitted off the tick' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | grep -vc 'f32 spans from the staging fill')"
grep -rhoE 'contracts door H2D receipt: ticket issuer=[0-9]+ seq=[0-9]+ .* items=[0-9]+ \([0-9]+ KV planes(, draft)?\)' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" \
  | grep -oE 'items=[0-9]+ \([0-9]+ KV planes(, draft)?\)' | sort | uniq -c
grep -rhoE 'published retired acknowledged; [0-9]+ f32 spans landed under the ticket and taken back before the retire' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
echo "H2D receipts without a span term: $(grep -rhE 'contracts door H2D receipt: ticket issuer=' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | grep -vc 'f32 spans landed under the ticket')"
grep -rhoE 'promote refused \(contracts door\): tier H2D spans refused: [^;]*; serving without the host entry' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
grep -rhoE 'promote (failed|dropped) \((tier staging fill|tier hash helper gone during|the host entry left)[^)]*\)' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | sort | uniq -c
echo "verify ok lines: $(grep -rh 'verify ok: promoted state digest matches demote digest' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | wc -l); VERIFY FAILED lines: $(grep -rh 'VERIFY FAILED' --include='*.log' --exclude=counts.log --exclude-dir=pre "$R" | wc -l)"
echo "## day 32: the promote lines per top-level cell dir (owner segment, promote in-ms), readings"
for d in $(find "$R" -mindepth 1 -maxdepth 1 -type d | sort); do
  grep -rhE 'promote submitted off the tick: .*owner segment|\[prefix-host\] promote: ' --include='*.log' --exclude=counts.log "$d" | python3 -c '
import re, statistics, sys
own = [float(m.group(1)) for l in sys.stdin if (m := re.search(r"owner segment ([\d.]+)ms", l))]
if own:
    print(f"{sys.argv[1]}: owner segment N={len(own)} median={statistics.median(own):.2f} min={min(own):.2f} max={max(own):.2f}")
' "$d"
done
if [ -d "$R/pre/double-park/ev" ]; then
  echo "## the pre-H2D double-park run (B2 baseline): replays, receipts, compute apps"
  echo "replays.log STALL REPLAY: PASS lines: $(grep -c 'STALL REPLAY: PASS' "$R/pre/double-park/ev/replays.log")"
  grep -rhoE 'errors=[0-9]+' --include=receipt.json "$R/pre/double-park/ev" | sort | uniq -c
  echo "compute apps before/after (data rows): $(tail -n +2 "$R/pre/double-park/ev/compute-apps.before.csv" | wc -l)/$(tail -n +2 "$R/pre/double-park/ev/compute-apps.after.csv" | wc -l)"
  echo "binary: $(cat "$R/pre/double-park/ev/binary.sha256")"
fi
