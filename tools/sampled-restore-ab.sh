#!/usr/bin/env bash
# sampled-restore-ab.sh — does the sampled restore PAY, on the SOLD SHAPE?
#
# This is the prod-sized twin of lane/sampled-hit-spec §3's interleaved A/B, which measured
# 1.210x on a small shape (9B, 106-token prompt, laptop 5090) and explicitly deferred the
# sold-shape number to the 96GB window: "that row is 27B on a 96GB card with a long shared
# prefix, where the plain path also loses the draft-plane-warmed decode over a much deeper
# context. Read this as ... The sold-shape number stays the 96GB window's job (§7.3)."
# It was never banked as a tool — the 1.210x cell was ad hoc — so it is banked here.
#
# WHY THIS SHAPE OF A/B. The two arms must differ ONLY in whether the entry they hit carries
# a draft plane, which is exactly the v0.93.0 downgrade. Flipping MEMRA_SPEC_RESTORE_SAMPLED
# cannot do that inside one boot (env is read at boot) and a cross-boot median is forbidden
# by the box-clock-drift law, so the arms are two PC-ISO namespaces in ONE boot:
#
#   arm A  ns=ab-spec   leader is SAMPLED  -> publishes a draft-plane entry -> the sampled
#                       repeat re-arms spec from the restored carrier.
#   arm B  ns=ab-plain  leader is GREEDY + repetition_penalty 1.1, which is spec-INELIGIBLE
#                       upstream (worker.rs `greedy_penalized`), so it serves plain and
#                       publishes a PLANE-LESS entry -> the sampled repeat refuses and takes
#                       the plain path. That is the v0.93.0 behaviour, reproduced on demand
#                       inside a binary that has the fix.
#
# Both arms send the SAME prompt_ids, so cached_tokens is identical and the prefill discount
# cancels: the delta is decode. The cell asserts that equality rather than assuming it.
#
# Measured repeats are interleaved A,B,A,B,... inside one server lifetime and one lock hold
# (interleaved-A/B protocol law, N=5 default).
#
# VERDICT RULE, pre-registered:
#   PASS iff  spec engaged on 5/5 arm-A repeats and 0/5 arm-B repeats
#        AND  cached_tokens equal across every measured pair (delta is decode-only)
#        AND  median(A tok/s) / median(B tok/s) >= 1.05
#   The ratio itself is the deliverable; the engagement and cached-token clauses are what
#   make it a measurement of the lever rather than of two unrelated requests.
#
# tok/s is end-to-end (completion_tokens / wall), the same instrument as the banked 1.210x
# row, so the two numbers are comparable. Prefill is cached and identical in both arms.
#
# usage: sampled-restore-ab.sh <model.gguf> <server_bin> <evidence_dir>
set -uo pipefail
MODEL=$1; BIN=$2; EV=$3
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_AB_PORT:-18096}
HERE=$(cd "$(dirname "$0")" && pwd)
# The QWEN attach is MEMRA_MTP_DRAFT; MEMRA_DRAFT is the GEMMA seam and attaches NOTHING on a
# qwen model while still flipping wkv_on()/fa_f16pv_on()/the MMQ-SK form (docs/FLAGS.md 'SEAM
# TRAP'). When set, the boot ASSERTS the attach line — a silent no-drafter run must FAIL.
MTP_DRAFT=${MEMRA_GATE_MTP_DRAFT:-}
PREFIX_TOKENS=${MEMRA_AB_PREFIX:-4860}
OUT_TOKENS=${MEMRA_AB_OUT:-192}
REPS=${MEMRA_AB_REPS:-5}
TEMP=${MEMRA_AB_TEMP:-0.8}
mkdir -p "$EV"

echo "sampled-restore-ab: prefix $PREFIX_TOKENS tok, out $OUT_TOKENS tok, temp $TEMP, N=$REPS"
echo "bin   $(sha256sum "$BIN" | cut -c1-16)"
echo "model $(sha256sum "$MODEL" | cut -c1-16)"

stop() {
    pkill -x memra-server 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || return 0
        sleep 2
    done
}
stop

# One boot, one lock hold, production spec config (no doors touched).
MTP=()
[ -n "$MTP_DRAFT" ] && MTP=("MEMRA_MTP_DRAFT=$MTP_DRAFT")
flock -w 300 "$GPU_LOCK" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    MEMRA_COMPAT=openai "MEMRA_MODELS=ab=$MODEL" \
    "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=4 \
    "${MTP[@]}" "$BIN" > "$EV/server.log" 2>&1 &
SPID=$!
up=0
for _ in $(seq 1 240); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -20 "$EV/server.log"; exit 1; }
    sleep 5
done
[ "$up" = 1 ] || { echo "server never came up"; tail -20 "$EV/server.log"; stop; exit 1; }
if [ -n "$MTP_DRAFT" ]; then
    "$HERE/assert-drafter-attached.sh" "$EV/server.log" "$MTP_DRAFT" || { stop; exit 1; }
fi

python3 - "$PORT" "$EV" "$PREFIX_TOKENS" "$OUT_TOKENS" "$REPS" "$TEMP" <<'PY'
import json, statistics, sys, time, urllib.error, urllib.request
port, ev, ptoks, otoks, reps, temp = (
    int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), float(sys.argv[6]))
IDS = [11] * ptoks

def call(ns, tag, temperature, seed=None, rep_pen=None):
    body = {"model": "ab", "prompt_ids": IDS, "max_tokens": otoks,
            "temperature": temperature, "cache_salt": ns}
    if seed is not None:
        body["seed"] = seed
    if rep_pen is not None:
        body["repetition_penalty"] = rep_pen
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            b = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{tag}: HTTP {e.code} {e.read()[:400]!r}")
    wall = time.monotonic() - t0
    u = b.get("usage") or {}
    sp = u.get("spec") or {}
    ct = u.get("completion_tokens", 0)
    txt = (b.get("choices") or [{}])[0].get("text") or ""
    # A comparator whose inputs are empty agrees perfectly. Refuse empty rows outright.
    if ct == 0 or not txt:
        raise SystemExit(f"{tag}: VOID — empty completion (ctok={ct}, text {len(txt)}B)")
    return {"tag": tag, "ns": ns, "wall_s": round(wall, 3), "completion_tokens": ct,
            "tok_s": round(ct / wall, 2),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "prompt_tokens": u.get("prompt_tokens", 0),
            "accepted": sp.get("accepted"), "drafted": sp.get("drafted"),
            "engaged": bool(sp and (sp.get("accepted") or 0) > 0),
            "text_len": len(txt)}

rows = []
# Leaders. A: sampled -> draft-plane entry. B: greedy+penalized -> plane-less entry.
rows.append(call("ab-spec", "leader-A", temp, seed=1))
rows.append(call("ab-plain", "leader-B", 0.0, rep_pen=1.1))
# Measured repeats, interleaved A,B,A,B,... in one boot / one lock hold.
for r in range(1, reps + 1):
    rows.append(call("ab-spec", f"A{r}", temp, seed=1000 + r))
    rows.append(call("ab-plain", f"B{r}", temp, seed=1000 + r))

A = [x for x in rows if x["tag"].startswith("A") and x["tag"] != "leader-A"]
B = [x for x in rows if x["tag"].startswith("B") and x["tag"] != "leader-B"]
mA, mB = statistics.median(x["tok_s"] for x in A), statistics.median(x["tok_s"] for x in B)
ratio = mA / mB if mB else 0.0
engA, engB = sum(x["engaged"] for x in A), sum(x["engaged"] for x in B)
cached_eq = all(a["cached"] == b["cached"] for a, b in zip(A, B))

print(f"{'rep':<5}{'A tok/s':>10}{'B tok/s':>10}{'A cached':>10}{'B cached':>10}"
      f"{'A acc':>8}{'B acc':>8}")
for i, (a, b) in enumerate(zip(A, B), 1):
    print(f"{i:<5}{a['tok_s']:>10}{b['tok_s']:>10}{a['cached']:>10}{b['cached']:>10}"
          f"{str(a['accepted']):>8}{str(b['accepted']):>8}")
print(f"median{mA:>10}{mB:>10}")
print(f"A/B = {ratio:.3f}x   spec engaged A {engA}/{len(A)}, B {engB}/{len(B)}   "
      f"cached equal per pair: {cached_eq}")

json.dump({"rows": rows, "median_A_tok_s": mA, "median_B_tok_s": mB,
           "ratio": ratio, "engaged_A": engA, "engaged_B": engB,
           "cached_equal": cached_eq, "prefix_tokens": ptoks,
           "out_tokens": otoks, "temp": temp, "reps": reps},
          open(f"{ev}/ab.json", "w"), indent=1)

fails = []
if engA != len(A): fails.append(f"arm A engaged {engA}/{len(A)} (need all)")
if engB != 0: fails.append(f"arm B engaged {engB}/{len(B)} (need 0 — plane-less entry leaked a plane)")
if not cached_eq: fails.append("cached_tokens differ across arms — delta is not decode-only")
if ratio < 1.05: fails.append(f"ratio {ratio:.3f}x < pre-registered 1.05x")
print("SAMPLED-RESTORE A/B:", "FAIL — " + "; ".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
PY
rc=$?
stop
exit $rc
