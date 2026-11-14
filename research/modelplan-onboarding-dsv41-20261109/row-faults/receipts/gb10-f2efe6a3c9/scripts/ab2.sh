#!/usr/bin/env bash
# GB10 serving-step A/B of memra#1292: main's memra-server (arm A) against the row-map head's (arm B), runs
# interleaved A B B A B A, one server boot a run, the 2-layer rehearsal trunk (MEMRA_DSV41_TRUNK_LAYERS=2: the
# step's fixed costs, where a per-step addition weighs most), no fault door. A run: two greedy ignore_eos
# requests of 64 tokens sharing batched calls (the second sent after the first's prefill step), then one alone.
# Every step's `[dsv41-step]` line is kept; the summary is each run's median step_s over the 2-row decode
# steps and over the 1-row decode steps. The checkpoint's page cache is dropped before every boot (GB10's memory
# is the host's: cached pages count as used, and the first run refused its fifth boot at 3.0 GiB free).
# usage: ab2.sh TAG_A TAG_B   Done-file out/ab2/ab.txt, marker AB-DONE.
set -u
TA=$1; TB=$2
R=$HOME/.cache/dsv41-fu1113
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/ab2; mkdir -p $O; : > $O/ab.txt
ADDR=127.0.0.1:18394
KEY=dsv41-row-fault-ab
sha256sum $R/bins/$TA/memra-server $R/bins/$TB/memra-server >> $O/ab.txt
drop_cache() {
  python3 - "$C" <<'PY'
import os, sys, glob
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
}
ask() {  # ask RUN NAME PROMPT MAX
  local body="{\"model\":\"v41\",\"messages\":[{\"role\":\"user\",\"content\":\"$3\"}],\"max_tokens\":$4,\"temperature\":0,\"ignore_eos\":true}"
  curl -s -o $O/resp-$1-$2.json -w "$1 $2 http %{http_code}\n" -H "Authorization: Bearer $KEY" \
    -H 'Content-Type: application/json' -d "$body" --max-time 600 http://$ADDR/v1/chat/completions >> $O/ab.txt
}
i=0
for arm in A B B A B A; do
  i=$((i + 1))
  if [ $arm = A ]; then TAG=$TA; else TAG=$TB; fi
  RUN=r$i-$arm
  drop_cache
  free -g | head -2 | tail -1 > $O/mem-$RUN.txt
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader > $O/gpu-$RUN.txt 2>&1
  MEMRA_MODELS=v41=$C MEMRA_ADDR=$ADDR MEMRA_API_KEY=$KEY MEMRA_DSV41_TOKEN_MAP=$TM MEMRA_CTX=8192 \
    MEMRA_DSV41_TRUNK_LAYERS=2 $R/bins/$TAG/memra-server > $O/server-$RUN.log 2>&1 &
  SRV=$!
  ready=0
  for _ in $(seq 1 450); do
    if ! kill -0 $SRV 2>/dev/null; then break; fi
    if curl -fs -o /dev/null http://$ADDR/readyz; then ready=1; break; fi
    sleep 2
  done
  echo "$RUN ($TAG) ready=$ready" >> $O/ab.txt
  if [ $ready = 1 ]; then
    ask $RUN p1 "What is the capital of France? Answer in one word." 64 & a=$!
    for _ in $(seq 1 300); do [ "$(grep -ac '^\[dsv41-step\]' $O/server-$RUN.log)" -ge 1 ] && break; sleep 0.2; done
    ask $RUN p2 "Name three primary colors." 64 & b=$!
    wait $a $b
    ask $RUN solo "What is the capital of France? Answer in one word." 64
  fi
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> $O/gpu-$RUN.txt 2>&1
  kill -TERM $SRV 2>/dev/null
  for _ in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
  kill -KILL $SRV 2>/dev/null
  wait $SRV 2>/dev/null
done
python3 - "$O" >> $O/ab.txt <<'EOF'
import glob, os, re, statistics, sys
o = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{o}/server-r*.log")):
    run = os.path.basename(f)[len("server-"):-len(".log")]
    by = {1: [], 2: []}
    for l in open(f, errors="replace"):
        if not l.startswith("[dsv41-step]"):
            continue
        m = dict(re.findall(r"(\w+)=(\S+)", l))
        if m.get("prefill") == "0" and int(m.get("decode_rows", 0)) in by:
            by[int(m["decode_rows"])].append(float(m["step_s"]) * 1e3)
    rows[run] = by
    print(run, " ".join(f"rows{k}: n={len(v)} median={statistics.median(v):.3f} ms" if v else f"rows{k}: n=0"
                        for k, v in by.items()))
for k in (1, 2):
    for arm in ("A", "B"):
        meds = [statistics.median(v[k]) for r, v in rows.items() if r.endswith(arm) and v[k]]
        if meds:
            print(f"arm {arm} rows{k}: run medians {', '.join(f'{x:.3f}' for x in meds)} ms; median of medians {statistics.median(meds):.3f} [{min(meds):.3f}, {max(meds):.3f}]")
EOF
echo "AB-DONE $(date +%T)" >> $O/ab.txt
