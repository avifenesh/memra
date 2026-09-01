#!/usr/bin/env bash
# btemp-power.sh — measure the POWER of spec-on-cache-hit-gate.sh's boundary probe on a
# given artifact, and print the arithmetic that turns it into a false-fail rate.
#
# WHY THIS TOOL EXISTS. The gate's boundary probe asserts "at least one of K seeds draws a
# FIRST token different from the greedy argmax". That is a Bernoulli experiment whose success
# probability p is a property of THE FIXTURE AND THE ARTIFACT, not of the code under test:
#
#     power        = 1 - (1-p)^K
#     false-fail   =     (1-p)^K
#
# At the gate's shipped BTEMP=1.5, p on the 27B production artifact measured 1/12 = 0.083, so
# the assertion false-failed on (1-0.083)^3 = 77% of runs — a red that carried no information,
# which is strictly worse than no assertion at all because it teaches operators to skip reds.
# The default is 4.0 as of 2026-08-19 on this measurement. Re-run this tool for any new
# artifact before trusting the probe on it; the gate itself prints an ATTRIBUTION line pointing
# here when the probe fails while the traced-draw instrument passes.
#
# NOTE ON PROVENANCE: the 96GB window cited "tools/btemp-power.sh (new, banked)" but the file
# was never committed to any branch — only its JSON output survived
# (darklanes research/spec-cache-20260818/box1-96gb/ev/task1/btemp-power/btemp-power.json).
# This is that instrument, banked for real. A cited tool that is not in git is not banked.
#
# The fixture is byte-for-byte the gate's: same PROMPT, max_tokens=1, one PC-ISO namespace per
# request so no cell can hit another's entry, and a fixed seed set so a rung is reproducible.
#
#   usage: btemp-power.sh <model.gguf> <server_bin> <evidence_dir> [temps...]
#          MEMRA_BTP_SEEDS=<n>   seeds per rung (default 12)
#          MEMRA_BTP_DRAFT=<f>   attach the real external MTP drafter (MEMRA_MTP_DRAFT — the
#                                QWEN seam; MEMRA_DRAFT is gemma's and attaches nothing here)
#   default rungs: 1.5 2.5 4.0 8.0
#
# One boot, one lock hold: temperature is a per-request field, so every rung shares a server
# and the comparison cannot be a cross-boot artifact.
set -uo pipefail
MODEL=$1; BIN=$2; EV=$3; shift 3
TEMPS=${*:-"1.5 2.5 4.0 8.0"}
SEEDS_N=${MEMRA_BTP_SEEDS:-12}
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_BTP_PORT:-18098}
HERE=$(cd "$(dirname "$0")" && pwd)
DRAFT=${MEMRA_BTP_DRAFT:-}
mkdir -p "$EV"

echo "btemp-power: $SEEDS_N seeds per rung, rungs: $TEMPS"
echo "bin   $(sha256sum "$BIN" | cut -c1-16)"
echo "model $(sha256sum "$MODEL" | cut -c1-16)"

stop() {
    pkill -x memra-server 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || return 0
        sleep 2
    done
    pkill -9 -x memra-server 2>/dev/null || true; sleep 3
}
stop
trap stop EXIT

MTP=()
[ -n "$DRAFT" ] && MTP=("MEMRA_MTP_DRAFT=$DRAFT")
flock -w 300 "$GPU_LOCK" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
    "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
    "${MTP[@]}" "$BIN" > "$EV/server.log" 2>&1 &
SPID=$!
up=0
for _ in $(seq 1 240); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -20 "$EV/server.log"; exit 1; }
    sleep 5
done
[ "$up" = 1 ] || { echo "server never came up"; tail -20 "$EV/server.log"; exit 1; }
if [ -n "$DRAFT" ]; then
    "$HERE/assert-drafter-attached.sh" "$EV/server.log" "$DRAFT" || exit 1
fi

python3 - "$PORT" "$EV" "$SEEDS_N" "$HERE/spec-on-cache-hit-gate.sh" $TEMPS <<'PY'
import json, math, re, sys, urllib.request
port, ev, nseeds, gate = (int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
temps = [float(t) for t in sys.argv[5:]]
K = 3  # the gate's own seed count — the K in (1-p)^K

# The fixture prompt is READ OUT OF THE GATE, not copied. A power measurement taken on a
# different prompt is a measurement of a different probe, and a copy silently rots the first
# time someone edits the gate's fixture. Bash line continuations (`\` + newline inside a
# double-quoted assignment) collapse to nothing, which is what the join below reproduces.
src = open(gate).read()
m = re.search(r'^PROMPT="((?:[^"\\]|\\\n)*)"', src, re.M)
if not m:
    raise SystemExit(f"could not read PROMPT= out of {gate} — fixture drift, refusing to run")
PROMPT = m.group(1).replace("\\\n", "")
print(f"fixture prompt: {len(PROMPT)} chars, read from {gate}")


def one(temp, seed, salt):
    body = {"model": "gate", "prompt": PROMPT, "max_tokens": 1,
            "temperature": temp, "cache_salt": salt}
    if temp > 0:
        body["seed"] = seed
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        b = json.load(r)
    return (b.get("choices") or [{}])[0].get("text") or ""


greedy = one(0.0, 0, "btp-greedy")
print(f"greedy argmax first token: {greedy!r}")
seeds = [7 + 13 * i for i in range(nseeds)]
ladder = []
for t in temps:
    toks = [one(t, s, f"btp-{t}-{s}") for s in seeds]
    differ = sum(1 for x in toks if x != greedy)
    p_hat = differ / len(toks)
    # Clopper-Pearson ONE-SIDED 95% lower bound on p.
    #   x = n (every seed deviated):  P(X=n) = p^n >= 0.05  =>  p >= 0.05^(1/n)
    #   x < n:                        solved numerically on the Beta/binomial tail
    n = len(toks)
    if differ == n:
        lo = 0.05 ** (1.0 / n)
    elif differ == 0:
        lo = 0.0
    else:
        def tail(p):  # P(X >= differ | p)
            return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                       for k in range(differ, n + 1))
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if tail(m) < 0.05:
                a = m
            else:
                b = m
        lo = a
    ladder.append({"temp": t, "n": n, "differ": differ, "p_hat": round(p_hat, 4),
                   "p_lo95": round(lo, 4),
                   "false_fail_at_p_hat": round((1 - p_hat) ** K, 4),
                   "false_fail_bound": round((1 - lo) ** K, 4),
                   "distinct": len(set(toks)), "tokens": toks})

print(f"\n{'BTEMP':>7}{'differ/n':>10}{'p_hat':>8}{'p>=(95%)':>10}"
      f"{'P(false fail) K=3':>20}{'bound':>9}{'distinct':>10}")
for r in ladder:
    print(f"{r['temp']:>7}{str(r['differ']) + '/' + str(r['n']):>10}{r['p_hat']:>8}"
          f"{r['p_lo95']:>10}{r['false_fail_at_p_hat']:>20}{r['false_fail_bound']:>9}"
          f"{r['distinct']:>10}")
print("\nP(false fail) = (1-p)^K with K=3 (the gate's seed count). 'bound' uses the 95%")
print("one-sided Clopper-Pearson LOWER bound on p, i.e. the pessimistic reading.")

# A usable default is one whose PESSIMISTIC false-fail bound is small. 0.05 is the bar: a
# gate that false-fails 1 run in 20 is still believed; 1 in 1.3 (BTEMP 1.5's measured 0.771)
# is not, and gets ignored, which is the whole defect.
BAR = 0.05
usable = [r["temp"] for r in ladder if r["false_fail_bound"] <= BAR]
print(f"\nrungs with false-fail bound <= {BAR}: {usable if usable else 'NONE'}")
json.dump({"greedy": greedy, "K": K, "bar": BAR, "usable_temps": usable,
           "ladder": ladder}, open(f"{ev}/btemp-power.json", "w"), indent=1)
sys.exit(0 if usable else 1)
PY
rc=$?
stop
exit $rc
