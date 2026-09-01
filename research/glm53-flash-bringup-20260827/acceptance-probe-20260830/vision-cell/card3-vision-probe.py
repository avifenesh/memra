#!/usr/bin/env python3
"""card3-lane CELL 1: glm5 vision can't-hallucinate probe + negative arms.

Count/content-based only, never timed. Polls /root/TIMING-IN-FLIGHT before every
request (co-tenancy protocol: no inference while another agent's timed cells run).
Banks every request/response pair as JSON under /root/card3-lane/out/vision/.

Ground truth for text448.png: the codes "ZK5465 QV4655 XR0818" — a fluent answer
WITHOUT the exact codes is a FAIL (vision failure is fluent hallucination).
"""
import base64, json, os, sys, time, urllib.request, urllib.error

BASE = os.environ.get("CARD3_BASE", "http://127.0.0.1:18500")
MODEL = "zai/glm-5.3-flash"
OUT = "/root/card3-lane/out/vision"
FIX = "/root/memra-card3/research/glm5-vision-20260830/fixtures"
CODES = "ZK5465 QV4655 XR0818"
os.makedirs(OUT, exist_ok=True)

def wait_marker():
    waited = False
    while os.path.exists("/root/TIMING-IN-FLIGHT"):
        if not waited:
            print("[hold] TIMING-IN-FLIGHT present, pausing...", flush=True)
            waited = True
        time.sleep(10)
    if waited:
        print("[hold] marker cleared, resuming", flush=True)

def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def post(name, body, timeout=900):
    wait_marker()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    status, text = None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode()
    except Exception as e:
        status, text = -1, repr(e)
    rec = {"name": name, "status": status, "request": body, "response_raw": text}
    try:
        rec["response"] = json.loads(text)
        rec.pop("response_raw")
    except Exception:
        pass
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(rec, f, indent=1)
    content = ""
    try:
        m = rec["response"]["choices"][0]["message"]
        content = (m.get("content") or "") + "\n[reasoning] " + (m.get("reasoning") or "")
    except Exception:
        content = str(text)[:400]
    print(f"=== {name}: HTTP {status}\n{content[:700]}\n", flush=True)
    return status, content

def img_msg(uri, text):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": text},
    ]}]

def main():
    arms = sys.argv[1:] or ["gate", "probe", "neg", "multiturn"]
    t448 = data_uri(f"{FIX}/text448.png")
    d112 = data_uri(f"{FIX}/det112.png")

    if "gate" in arms:
        # Fresh-boot output-sample gate: short text prompt, sane fluent output required
        # before ANY cell counts (a booting server that 200s garbage invalidates everything).
        post("00-boot-gate-greedy", {
            "model": MODEL, "temperature": 0, "max_tokens": 48,
            "messages": [{"role": "user", "content": "Name three primary colors, one word each."}],
        })
        post("01-boot-gate-sampled-vendor-default", {
            "model": MODEL, "max_tokens": 48,
            "messages": [{"role": "user", "content": "Name three primary colors, one word each."}],
        })

    if "probe" in arms:
        s, c = post("10-cant-hallucinate-greedy", {
            "model": MODEL, "temperature": 0, "max_tokens": 512,
            "messages": img_msg(t448, "Transcribe the text in this image exactly."),
        })
        print(f"VERDICT greedy codes-present: {CODES in c}", flush=True)
        s, c = post("11-cant-hallucinate-sampled-vendor-default", {
            "model": MODEL, "max_tokens": 512,
            "messages": img_msg(t448, "Transcribe the text in this image exactly."),
        })
        print(f"VERDICT sampled codes-present: {CODES in c}", flush=True)
        s, c = post("12-det112-content-pin-greedy", {
            "model": MODEL, "temperature": 0, "max_tokens": 512,
            "messages": img_msg(d112, "Describe the colors and shapes in this image briefly."),
        })

    if "neg" in arms:
        # literal placeholder token in user text -> named refusal
        post("20-neg-literal-placeholder", {
            "model": MODEL, "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": "What does <|image|> mean here?"}],
        })
        # video_url -> named refusal (out of lane scope by design)
        post("21-neg-video-url", {
            "model": MODEL, "temperature": 0, "max_tokens": 32,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": t448}},
                {"type": "text", "text": "Describe this video."},
            ]}],
        })

    if "flagoff" in arms:
        # run against a boot WITHOUT MEMRA_GLM5_VISION: image request must refuse, not 200.
        s, c = post("30-neg-flag-off-image", {
            "model": MODEL, "temperature": 0, "max_tokens": 32,
            "messages": img_msg(t448, "Transcribe the text in this image exactly."),
        })
        print(f"VERDICT flag-off refused (non-200): {s != 200}", flush=True)

    if "multiturn" in arms:
        m = img_msg(t448, "Transcribe the text in this image exactly.")
        s, c = post("40-multiturn-t1-image", {
            "model": MODEL, "temperature": 0, "max_tokens": 512, "messages": m})
        m = m + [{"role": "assistant", "content": c.split("\n[reasoning]")[0]},
                 {"role": "user", "content": "Repeat only the middle code from your transcription."}]
        s2, c2 = post("41-multiturn-t2-textonly", {
            "model": MODEL, "temperature": 0, "max_tokens": 512, "messages": m})
        print(f"VERDICT multiturn t2 middle code (QV4655) present: {'QV4655' in c2}", flush=True)

if __name__ == "__main__":
    main()
