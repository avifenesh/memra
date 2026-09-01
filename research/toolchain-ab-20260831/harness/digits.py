#!/usr/bin/env python3
# Sealed digits protocol, loopback vantage, toolchain A/B edition.
# Shape: 512-token streamed completion, vendor-default sampling (NO sampling params;
# models.toml temp=0.5/top_p=0.9 governs), banked digits prompt + fresh salt per rep,
# wall clock incl TTFT, token counts from the stream's own usage block, spec receipts
# from usage.spec. Per boot: 1 smoke (spec-engagement gate), 1 discarded warmup,
# 8 measured reps. Rows appended as JSON lines.
# Usage: digits.py <arm> <boot> <rows.jsonl>
import json, os, secrets, sys, time, urllib.request

ARM, BOOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
URL = "http://127.0.0.1:18620/v1/chat/completions"
MODEL = "stepfun/step-3.7-flash"
D = "/home/ubuntu/toolchain-ab"
PROMPT = open(D + "/digits.txt").read().strip()
receipt = dict(l.strip().split("=", 1) for l in open(D + "/receipts/boot-%s.receipt" % ARM)
               if "=" in l and not l.startswith(" "))

def stream_rep(rep, salt, maxtok=512):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT + "\n\n[session %s]" % salt}],
            "max_tokens": maxtok, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; usage = None; fp = None; finish = None; nchunk = 0
    r = urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=900)
    while True:
        line = r.readline()
        if not line: break
        s = line.decode("utf-8", "replace").strip()
        if not s.startswith("data:"): continue
        p = s[5:].strip()
        if p == "[DONE]": continue
        try: j = json.loads(p)
        except Exception: continue
        fp = j.get("system_fingerprint") or fp
        if j.get("usage"): usage = j["usage"]
        for ch in j.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                nchunk += 1
                if first is None: first = time.perf_counter() - t0
            if ch.get("finish_reason"): finish = ch["finish_reason"]
    r.close()
    wall = time.perf_counter() - t0
    u = usage or {}; sp = u.get("spec") or {}
    ct = u.get("completion_tokens"); pt = u.get("prompt_tokens")
    row = {"arm": ARM, "boot": BOOT, "rep": rep, "salt": salt,
           "prompt_tokens": pt, "completion_tokens": ct,
           "full_tokens": ct == maxtok, "finish_reason": finish,
           "ttft_s": round(first, 4) if first is not None else None,
           "wall_s": round(wall, 4),
           "decode_tok_s": round((ct - 1) / (wall - first), 2) if (ct and first is not None and wall > first) else None,
           "wall_tok_s": round(ct / wall, 2) if (ct and wall > 0) else None,
           "spec_acc": sp.get("acceptance_rate"), "spec_rounds": sp.get("rounds"),
           "spec_drafted": sp.get("drafted"), "spec_accepted": sp.get("accepted"),
           "fingerprint": fp, "chunks": nchunk,
           "bin_md5": receipt.get("bin_md5"), "boot_nonce": receipt.get("boot_nonce"),
           "built_from": receipt.get("built_from"),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return row

# smoke: spec-engagement receipt from the body, not just a 200
smoke = stream_rep(0, secrets.token_hex(4), maxtok=256)
engaged = (smoke.get("spec_rounds") or 0) > 0
print("SMOKE", "SPEC-ENGAGED" if engaged else "SPEC-MISSING", json.dumps(
    {k: smoke[k] for k in ("spec_acc", "spec_rounds", "wall_tok_s", "fingerprint")}))
if not engaged:
    sys.exit(9)

with open(OUT, "a", buffering=1) as out:
    smoke["kind"] = "smoke"; out.write(json.dumps(smoke) + "\n")
    w = stream_rep(0, secrets.token_hex(4))
    w["kind"] = "warmup"; out.write(json.dumps(w) + "\n")
    print("WARMUP (discarded)", w["wall_tok_s"], "tok/s")
    vals = []
    for rep in range(1, 9):
        time.sleep(1)
        row = stream_rep(rep, secrets.token_hex(4))
        row["kind"] = "rep"; out.write(json.dumps(row) + "\n")
        vals.append(row["wall_tok_s"])
        print("rep %d wall_tok_s=%s ttft=%s ct=%s acc=%s finish=%s" % (
            rep, row["wall_tok_s"], row["ttft_s"], row["completion_tokens"],
            row["spec_acc"], row["finish_reason"]))
    good = sorted(v for v in vals if v)
    med = (good[len(good)//2] if len(good) % 2 else
           (good[len(good)//2 - 1] + good[len(good)//2]) / 2) if good else None
    print("BOOT_MEDIAN arm=%s boot=%s wall_tok_s=%s n=%d" % (ARM, BOOT, med, len(good)))
