#!/usr/bin/env python3
"""MLA TC prefill A/B on the 3-card resident serving shape (coordinator order 2026-08-30).

Spec source: research/glm53-flash-bringup-20260827/mla-tc-prefill-20260830/LANE.md par.4,
adapted to the 3-card shape per the coordinator: arm = the window's exact 3-card resident
env +/- MEMRA_MLA_TC_PREFILL=1. NAMED DEVIATION from "exact": MEMRA_PREFIX_CACHE_MB=0
(the lane's pre-registered arm-C recipe pins cache 0 so cold primes carry no capture or
insert side-work; the battery envs used 2000/2048).

Protocol: interleaved x5 FRESH BOOTS per arm (OFF,ON per round; the interleaved-A/B law),
arm identity proven per boot from /proc/<pid>/environ (boot-nonce lesson: health-200
proves a listener, not which server), pgrep-clear asserted (exactly one memra-server).

Rows per boot: greedy cold prime A4630 + C6470 (streamed; TTFD = first content byte,
prefill tok/s = prompt_tokens/TTFD; first-token text banked for the argmax gate),
one vendor-default SAMPLED twin prime (A4630, no sampling params), one decode row
(WARM prompt, 128 greedy tokens; decode ms/token must match across arms - decode never
enters the door). Engagement receipts per boot from the boot's own log: ON needs
"[mla-tc-prefill] engaged" + dispatch counter lines and ZERO cuBLASLt DECLINE lines
(a DECLINE invalidates the row); OFF needs zero mla-tc lines.

usage: mla_ab.py <outdir> [rounds=5]
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

OUT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
BIN = "/root/memra/target/release/memra-server"
SERVE = "/root/mla-tc-ab-3c-serve-scoped.sh"  # port-scoped PID-verified stop (02:47Z co-tenant incident)
POOL = json.load(open("/root/l3-ab/prompts.json"))
os.makedirs(OUT, exist_ok=True)

BASE_ENV = [
    "MEMRA_SPILL_STATS=1", "MEMRA_MOE_RESIDENT_GB=98", "MEMRA_MOE_SLOTS=16",
    # Owner-accepted serving recipe pins (BRINGUP.md adopted recipe; coordinator
    # correction 2026-08-30). The f32-trunk A/B (out-mla-ab-3c) lacked these.
    "MEMRA_BF16_MMV=1", "MEMRA_PP_BF16=1", "MEMRA_MOE_GROUPED_PREFILL=1",
    "MEMRA_PP_STAGES=3", "MEMRA_PP_SPLITS=15,30", "MEMRA_PP_DEVICES=0,1,2",
    "CUDA_VISIBLE_DEVICES=0,1,2", "MEMRA_COMPAT=openai",
    "MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4",
    "MEMRA_ADDR=127.0.0.1:18400", "MEMRA_MAX_SESSIONS=4",
    "NVIDIA_TF32_OVERRIDE=0", "MEMRA_CTX=8192", "MEMRA_PREFIX_CACHE_MB=0",
]
RESULTS = {"rounds": ROUNDS, "boots": [], "rows": [], "violations": []}


def sh(cmd, timeout=900):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def boot(arm, rnd):
    log = f"{OUT}/serve-{arm}-r{rnd}.log"
    env = BASE_ENV + (["MEMRA_MLA_TC_PREFILL=1"] if arm == "on" else [])
    r = sh(f"bash {SERVE} {BIN} {log} " + " ".join(env), timeout=900)
    ready = "READY" in r.stdout
    # pgrep-clear + arm identity from the process environment, not from a 200.
    pids = [p for p in sh("pgrep -x memra-server").stdout.split() if p]
    ident = {"arm": arm, "round": rnd, "ready": ready, "pids": pids, "log": log}
    if len(pids) != 1:
        RESULTS["violations"].append(f"boot {arm} r{rnd}: pgrep-clear failed: {pids}")
    else:
        environ = open(f"/proc/{pids[0]}/environ", "rb").read().decode(errors="replace")
        has_flag = "MEMRA_MLA_TC_PREFILL=1" in environ
        ident["environ_flag"] = has_flag
        ident["nonce"] = pids[0] + ":" + sh(f"stat -c %Y /proc/{pids[0]}").stdout.strip()
        if has_flag != (arm == "on"):
            RESULTS["violations"].append(
                f"boot {arm} r{rnd}: ARM IDENTITY MISMATCH environ_flag={has_flag}")
    if not ready:
        RESULTS["violations"].append(f"boot {arm} r{rnd}: NOT READY")
        print(r.stdout[-2000:], r.stderr[-2000:])
    RESULTS["boots"].append(ident)
    return ready and len(pids) == 1, log


def prime(text, name, greedy, max_tokens=32):
    body = {"model": MODEL, "prompt": text, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if greedy:
        body["temperature"] = 0.0
    req = urllib.request.Request(EP + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    ttfd, usage, chunks, err = None, None, [], None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for rl in r:
                s = rl.decode().strip()
                if not s.startswith("data:"):
                    continue
                pay = s[5:].strip()
                if pay == "[DONE]":
                    break
                o = json.loads(pay)
                if o.get("usage"):
                    usage = o["usage"]
                c = (o.get("choices") or [{}])[0]
                if c.get("text"):
                    if ttfd is None:
                        ttfd = round(time.time() - t0, 3)
                    chunks.append(c["text"])
                if "error" in o:
                    err = json.dumps(o["error"])[:200]
    except Exception as e:  # noqa: BLE001 - the failure is the receipt
        err = f"{type(e).__name__}: {e}"
    u = usage or {}
    wall = round(time.time() - t0, 3)
    text_out = "".join(chunks)
    row = {"name": name, "ttfd_s": ttfd, "wall_s": wall,
           "prompt_tokens": u.get("prompt_tokens"),
           "completion_tokens": u.get("completion_tokens"),
           "prefill_tok_s": round(u["prompt_tokens"] / ttfd, 1)
           if ttfd and u.get("prompt_tokens") else None,
           "decode_ms_tok": round((wall - ttfd) * 1000 / (u["completion_tokens"] - 1), 2)
           if ttfd and (u.get("completion_tokens") or 0) > 1 else None,
           "first_chunk": text_out[:24], "out_text": text_out, "error": err}
    return row


def log_census(log, upto=None):
    txt = open(log, errors="replace").read()
    return {
        "engaged_lines": len(re.findall(r"mla-tc-prefill.*engaged", txt)),
        "dispatch_lines": len(re.findall(r"MLA_TC_PREFILL_DISPATCH", txt)),
        "dispatch_raw": re.findall(r".*(?:mla-tc-prefill|MLA_TC_PREFILL).*", txt)[:40],
        "decline_lines": len(re.findall(r"(?i)^.*(mla|cublas)[^\n]*decline.*$|^.*decline[^\n]*(mla|cublas).*$", txt, re.M)),
        "engine_errors": len(re.findall(r"engine-error", txt)),
    }


for rnd in range(ROUNDS):
    for arm in ("off", "on"):
        ok, log = boot(arm, rnd)
        if not ok:
            print(f"# boot {arm} r{rnd} FAILED - skipping rows", flush=True)
            continue
        rows = []
        for key, greedy, tag in (("A4630", True, "greedy"), ("C6470", True, "greedy"),
                                 ("A4630", False, "sampled")):
            row = prime(POOL[key], f"{arm}-r{rnd}-{key}-{tag}", greedy)
            row.update(arm=arm, round=rnd, prompt=key, mode=tag, kind="prime")
            rows.append(row)
        drow = prime(POOL["WARM"], f"{arm}-r{rnd}-WARM-decode", True, max_tokens=128)
        drow.update(arm=arm, round=rnd, prompt="WARM", mode="greedy", kind="decode")
        rows.append(drow)
        census = log_census(log)
        for row in rows:
            row["log_census"] = None  # per-boot census attached once below
            out = {k: v for k, v in row.items() if k != "out_text"}
            RESULTS["rows"].append(out)
            open(f"{OUT}/{row['name']}.json", "w").write(json.dumps(row, indent=1))
            print(f"  {row['name']}: ttfd={row['ttfd_s']} wall={row['wall_s']} "
                  f"ptok={row['prompt_tokens']} pf_tok_s={row['prefill_tok_s']} "
                  f"dec_ms={row['decode_ms_tok']} first={row['first_chunk']!r} "
                  f"err={row['error']}", flush=True)
        RESULTS["boots"][-1]["census"] = census
        # Engagement contract per arm, asserted per boot.
        if arm == "on":
            if census["engaged_lines"] == 0 and census["dispatch_lines"] == 0:
                RESULTS["violations"].append(
                    f"ON r{rnd}: no engagement lines after primes (census {census})")
            if census["decline_lines"] > 0:
                RESULTS["violations"].append(
                    f"ON r{rnd}: {census['decline_lines']} DECLINE line(s) - rows invalid")
                for row in RESULTS["rows"]:
                    if row["arm"] == "on" and row["round"] == rnd:
                        row["invalid_decline"] = True
        else:
            if census["engaged_lines"] or census["dispatch_lines"]:
                RESULTS["violations"].append(
                    f"OFF r{rnd}: mla-tc lines present on the OFF arm ({census})")
        print(f"# boot {arm} r{rnd} census: {census['engaged_lines']} engaged, "
              f"{census['dispatch_lines']} dispatch-lines, {census['decline_lines']} "
              f"decline, {census['engine_errors']} engine-errors", flush=True)

# First-token argmax gate across arms, per greedy prompt.
flips = []
for key in ("A4630", "C6470"):
    firsts = {}
    for row in RESULTS["rows"]:
        if row.get("prompt") == key and row.get("mode") == "greedy" and row.get("kind") == "prime":
            firsts.setdefault(row["arm"], set()).add(row["first_chunk"])
    if len(firsts.get("off", set()) | firsts.get("on", set())) > 1:
        flips.append({"prompt": key, "off": sorted(firsts.get("off", [])),
                      "on": sorted(firsts.get("on", []))})
RESULTS["first_token_flips"] = flips
json.dump(RESULTS, open(f"{OUT}/ab-summary.json", "w"), indent=1)
print("#" * 70, flush=True)
print(f"# A/B DONE rounds={ROUNDS} violations={len(RESULTS['violations'])} "
      f"flips={len(flips)}", flush=True)
for v in RESULTS["violations"]:
    print(f"#  VIOLATION: {v}", flush=True)
for f in flips:
    print(f"#  FLIP: {f}", flush=True)
sys.exit(2 if RESULTS["violations"] else 0)
