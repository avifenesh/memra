#!/usr/bin/env python3
"""step37 vision e2e gates through the REAL /v1/chat/completions surface.

Decisive can't-hallucinate probes (vision doctrine: a broken tower produces fluent
hallucination, not errors): every image is generated FRESH per run with randomized
content (background color + one large filled shape of another color, quadrant
randomized), so the model cannot answer from the text. Assertions name the actual
content. Vendor-default sampling (NO sampling params) on every answer probe.

Usage: e2e_gates.py <port> [--skip-spec]
Prints one GATE line per probe; exits nonzero on any FAIL.
"""
import base64
import io
import json
import random
import sys
import urllib.request

from PIL import Image, ImageDraw

PORT = sys.argv[1]
SKIP_SPEC = "--skip-spec" in sys.argv
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

COLORS = {
    "red": (220, 30, 30),
    "green": (30, 180, 60),
    "blue": (40, 70, 220),
    "yellow": (235, 220, 40),
    "orange": (240, 140, 20),
    "purple": (140, 40, 180),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
SHAPES = ["circle", "triangle", "square"]
rng = random.SystemRandom()

fails = []


def gate(name, ok, detail=""):
    print(f"GATE {name}: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        fails.append(name)


def make_image(w, h, bg, shape, fg, center=None):
    img = Image.new("RGB", (w, h), COLORS[bg])
    d = ImageDraw.Draw(img)
    r = min(w, h) // 4
    cx, cy = center or (w // 2, h // 2)
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS[fg])
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=COLORS[fg])
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[fg])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pick():
    bg, fg = rng.sample(list(COLORS), 2)
    return bg, SHAPES[rng.randrange(3)], fg


def post(payload, timeout=1800):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def chat(messages, **kw):
    payload = {"model": "step37", "messages": messages, "max_tokens": 3000}
    payload.update(kw)
    return post(payload)


def answer(body):
    return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


# Shape-class synonyms: a square IS a rectangle and a circle drawn into a resized
# canvas may honestly read as an ellipse/oval — the decisive content is the color
# pair plus the shape CLASS, not the exact noun.
SHAPE_SYNONYMS = {
    "square": ("square", "rectangle"),
    "circle": ("circle", "ellipse", "oval"),
    "triangle": ("triangle",),
}


def named(text, fg, shape, bg):
    t = text.lower()
    return (
        fg.lower() in t
        and bg.lower() in t
        and any(s in t for s in SHAPE_SYNONYMS[shape])
    )


QUESTION = (
    "This image shows one large filled shape on a solid background. "
    "Answer with exactly: <shape color> <shape> on <background color>. Nothing else."
)

# ---- 1. can't-hallucinate, no-tiling shape (640x640 square -> window 0) ----
bg, shape, fg = pick()
uri = make_image(640, 640, bg, shape, fg)
st, body = chat(
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                  {"type": "text", "text": QUESTION}]}]
)
a = answer(body)
gate("img-single-640", st == 200 and named(a, fg, shape, bg), f"http={st} answer={a!r} truth={fg} {shape} on {bg}")
u1 = body.get("usage", {})

# prompt-token accounting: same request with the image part dropped costs exactly 171 less
st2, body2 = chat([{"role": "user", "content": [{"type": "text", "text": QUESTION}]}], max_tokens=8)
u2 = body2.get("usage", {})
delta = u1.get("prompt_tokens", 0) - u2.get("prompt_tokens", 0)
gate("usage-171-per-image", st2 == 200 and delta == 171, f"delta={delta} (img={u1.get('prompt_tokens')}, plain={u2.get('prompt_tokens')})")

# ---- 2. can't-hallucinate under TILING (1600x900 -> 6 crops + main = 670 tokens) ----
# shape pinned to triangle: the main view's square resize destroys aspect, so circles/
# squares honestly read as ellipses/rectangles; the triangle survives the squish.
bg, fg = rng.sample(list(COLORS), 2)
shape = "triangle"
uri = make_image(1600, 900, bg, shape, fg)
st, body = chat(
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                  {"type": "text", "text": QUESTION}]}]
)
a = answer(body)
gate("img-tiled-1600x900", st == 200 and named(a, fg, shape, bg), f"http={st} answer={a!r} truth={fg} {shape} on {bg}")
delta = body.get("usage", {}).get("prompt_tokens", 0) - u2.get("prompt_tokens", 0)
gate("usage-670-tiled", delta == 670, f"delta={delta}")

# ---- 3. multi-image ----
bg1, shape1, fg1 = pick()
bg2, shape2, fg2 = pick()
while (bg2, shape2, fg2) == (bg1, shape1, fg1):
    bg2, shape2, fg2 = pick()
st, body = chat(
    [{"role": "user", "content": [
        {"type": "text", "text": "Two images follow. First:"},
        {"type": "image_url", "image_url": {"url": make_image(600, 600, bg1, shape1, fg1)}},
        {"type": "text", "text": "Second:"},
        {"type": "image_url", "image_url": {"url": make_image(600, 600, bg2, shape2, fg2)}},
        {"type": "text", "text": "For each image, answer: <shape color> <shape> on <background color>. Number them 1 and 2."},
    ]}]
)
a = answer(body)
gate(
    "img-multi",
    st == 200 and named(a, fg1, shape1, bg1) and named(a, fg2, shape2, bg2),
    f"http={st} answer={a!r} truth1={fg1} {shape1} on {bg1} truth2={fg2} {shape2} on {bg2}",
)
delta = body.get("usage", {}).get("prompt_tokens", 0)
gate("usage-multi-342", delta >= 342, f"prompt_tokens={delta} (>= 2x171 floor)")

# ---- 4. image mid-conversation after text turns (spec-configured binary) ----
bg, shape, fg = pick()
st, body = chat(
    [
        {"role": "user", "content": "Say READY and nothing else."},
        {"role": "assistant", "content": "READY"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": make_image(700, 700, bg, shape, fg)}},
            {"type": "text", "text": QUESTION},
        ]},
    ]
)
a = answer(body)
gate("img-mid-conversation", st == 200 and named(a, fg, shape, bg), f"http={st} answer={a!r} truth={fg} {shape} on {bg}")

# ---- 5. malformed inputs refuse loudly ----
st, body = chat([{"role": "user", "content": [
    {"type": "text", "text": "look <im_patch><im_patch> here"},
    {"type": "image_url", "image_url": {"url": make_image(600, 600, "red", "circle", "blue")}},
    {"type": "text", "text": QUESTION}]}], max_tokens=8)
gate("faked-pads-400", st == 400, f"http={st} body={json.dumps(body)[:160]}")
st, body = chat([{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}], max_tokens=8)
gate("http-url-400", st == 400, f"http={st}")
st, body = chat([{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": "data:video/gif;base64,AAAA"}}]}], max_tokens=8)
gate("video-400", st == 400, f"http={st}")

# ---- 6. text request on the same binary still answers (spec engagement read from log) ----
if not SKIP_SPEC:
    st, body = chat(
        [{"role": "user", "content": "List three prime numbers greater than 100, comma-separated."}]
    )
    gate("text-on-vision-binary", st == 200 and len(answer(body)) > 3, f"http={st} answer={answer(body)[:80]!r}")

print(f"E2E {'PASS' if not fails else 'FAIL'} ({len(fails)} failing: {fails})")
sys.exit(1 if fails else 0)
