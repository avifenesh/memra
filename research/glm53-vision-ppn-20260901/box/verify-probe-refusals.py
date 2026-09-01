#!/usr/bin/env python3
"""Execute EVERY refusal path in probe-vision-ppn.py's fixture loader and shape asserts.

The loud-loader patch this folds in was itself verified by running each path
(`battery2-loud-loader-verification.txt`), and its own first draft had a bug that only executing
it found. A refusal that has never been executed is a comment. This runs on any machine — no GPU,
no server, no box — because every one of these refusals happens BEFORE the first request.

Run from the lane dir; banked output is the receipt:
  python3 box/verify-probe-refusals.py <path to the banked fixture dir>
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("probe", os.path.join(HERE, "probe-vision-ppn.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

GOOD = sys.argv[1] if len(sys.argv) > 1 else None
if not GOOD or not os.path.isdir(GOOD):
    sys.exit("usage: verify-probe-refusals.py <banked fixture dir>")

results = []


def expect_refusal(label, fn, must_contain):
    try:
        fn()
    except SystemExit as e:
        msg = str(e)
        ok = all(s in msg for s in must_contain)
        results.append((label, "REFUSED" if ok else "REFUSED-BUT-UNCLEAR", msg))
        assert ok, f"{label}: refusal message missing {must_contain!r}: {msg}"
        return
    raise AssertionError(f"{label}: NO refusal — the path is unguarded")


def expect_ok(label, fn):
    fn()
    results.append((label, "PROCEEDS", ""))


def sha_of(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---- the happy path first, so a refusal below cannot be an artifact of a broken fixture dir ----
expect_ok("good fixture dir (explicit)", lambda: probe.load_fixtures(GOOD))

# ---- (a) missing directory, explicit vs default ----
expect_refusal(
    "missing dir (explicit)",
    lambda: probe.load_fixtures("/nonexistent/vision-fixtures"),
    ["fixture directory not found", "VISION_FIXTURES"],
)
saved_default = probe.DEFAULT_FIXTURES
probe.DEFAULT_FIXTURES = "/nonexistent/default-vision-fixtures"
expect_refusal(
    "missing dir (DEFAULT path — must say it was a default)",
    lambda: probe.load_fixtures(""),
    ["the DEFAULT path", "pass --fixtures/VISION_FIXTURES explicitly"],
)
probe.DEFAULT_FIXTURES = saved_default

# ---- (b) one fixture missing ----
with tempfile.TemporaryDirectory() as d:
    for n in probe.FIXTURE_SHA256:
        if n != "21-neg-video-url":
            shutil.copy(os.path.join(GOOD, n + ".json"), d)
    expect_refusal(
        "one fixture missing",
        lambda: probe.load_fixtures(d),
        ["21-neg-video-url.json", "is missing"],
    )

# ---- (c) sha drift: the dangerous one (a different image under the same arm name) ----
with tempfile.TemporaryDirectory() as d:
    for n in probe.FIXTURE_SHA256:
        shutil.copy(os.path.join(GOOD, n + ".json"), d)
    p = os.path.join(d, "10-cant-hallucinate-greedy.json")
    doc = json.load(open(p))
    doc["request"]["max_tokens"] = 999  # any byte change at all
    json.dump(doc, open(p, "w"))
    expect_refusal(
        "sha256 drift",
        lambda: probe.load_fixtures(d),
        ["sha256", "!= pinned", "ARE the instrument"],
    )

# ---- (d) unreadable json / wrong top-level shape (pins updated so sha is not what bites) ----
with tempfile.TemporaryDirectory() as d:
    for n in probe.FIXTURE_SHA256:
        shutil.copy(os.path.join(GOOD, n + ".json"), d)
    p = os.path.join(d, "20-neg-literal-placeholder.json")
    open(p, "w").write("{not json")
    saved = dict(probe.FIXTURE_SHA256)
    probe.FIXTURE_SHA256["20-neg-literal-placeholder"] = sha_of(p)
    expect_refusal(
        "not readable json",
        lambda: probe.load_fixtures(d),
        ["is not readable json"],
    )
    # right json, no "request" key — the launch pool's own (b) failure mode
    json.dump({"prompt": "oops"}, open(p, "w"))
    probe.FIXTURE_SHA256["20-neg-literal-placeholder"] = sha_of(p)
    expect_refusal(
        "no 'request' key (prints the actual top-level keys)",
        lambda: probe.load_fixtures(d),
        ["has no 'request' key", "prompt"],
    )
    probe.FIXTURE_SHA256.clear()
    probe.FIXTURE_SHA256.update(saved)

# ---- (e) SHAPE asserts: what a deliberate pin update could silently break ----
expect_refusal(
    "can't-hallucinate arm with no image part",
    lambda: probe.assert_shape(
        "10-cant-hallucinate-greedy",
        {"temperature": 0, "messages": [{"role": "user", "content": "what code?"}]},
    ),
    ["carries NO image_url part", "memorized answer"],
)
expect_refusal(
    "greedy arm that is not greedy",
    lambda: probe.assert_shape(
        "10-cant-hallucinate-greedy",
        {"temperature": 0.7,
         "messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    ),
    ["is not greedy", "byte-deterministic"],
)
expect_refusal(
    "vendor-default arm carrying sampling params",
    lambda: probe.assert_shape(
        "11-cant-hallucinate-sampled-vendor-default",
        {"temperature": 0.6, "top_p": 0.95,
         "messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    ),
    ["carries sampling params", "VENDOR-DEFAULT", "NO sampling params"],
)
expect_refusal(
    "video arm with no video part",
    lambda: probe.assert_shape(
        "21-neg-video-url",
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    ),
    ["carries no video_url part"],
)
expect_refusal(
    "text-only placeholder arm carrying an image",
    lambda: probe.assert_shape(
        "20-neg-literal-placeholder",
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    ),
    ["must be TEXT-ONLY"],
)
# the real fixtures must PASS their own shape asserts (or the asserts are wrong, not the fixtures)
for name in probe.FIXTURE_SHA256:
    req = json.load(open(os.path.join(GOOD, name + ".json")))["request"]
    expect_ok(f"banked fixture {name} satisfies its shape contract",
              lambda r=name, q=req: probe.assert_shape(r, q))

print()
print("=" * 78)
for label, verdict, msg in results:
    print(f"{verdict:20s} {label}")
    if msg:
        print(f"{'':20s}   {msg.splitlines()[0][:150]}")
print("=" * 78)
print(f"{len(results)} paths executed; every refusal fired with a named reason and every banked "
      "fixture satisfies its own contract.")
