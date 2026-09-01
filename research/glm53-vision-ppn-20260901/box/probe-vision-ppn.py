#!/usr/bin/env python3
"""glm5 vision on the ppN serving shape: the can't-hallucinate battery for lane/glm53-vision-ppn.

WHY THIS EXISTS. The 2026-09-01 launch window proved vision unservable on the deployed 3-card
PP3 shape (every image request 500'd on the OVERLAY DEVICE LAW), pinned MEMRA_GLM5_VISION=0 and
shipped facts text-only. This lane publishes the overlay into pp stage 0's CUDA context instead.
Rig arms (glm5-hyper-ppn-gate 5d/5e) prove the publication moves the rows without changing a
byte, and a foreign-context unit test proves the residency law refuses; NEITHER can prove that a
pointer published across two CUDA contexts is dereferenceable by the other stage, because one
card has one serving context. THIS probe is that half, and it is the flip gate for any product
surface.

A VISION FEATURE FAILS FLUENTLY. A 200 with a confident description of the wrong thing is the
normal failure mode (darklanes memory: decisive-probes-for-side-channel-features), so no arm here
is satisfied by a 200: the image carries codes the model cannot guess, and the arm compares them
exactly.

THE FIXTURES ARE AN INSTRUMENT INPUT, AND THEY ARE PINNED BY sha256. Folded in from the launch
lane's prompt-pool loud-loader patch (`battery2-prompt-pool-loud-loader.patch`, coordinator
instruction 2026-09-01: "a battery that silently falls back to a missing/default prompt file is
the self-measuring-instrument trap"). The same three failure modes apply here, and the third is
the dangerous one: the CODES this file compares against are only the right answer for the exact
image bytes the card3 cell used, so a fixture swap would silently measure a different image.
Every fixture's sha256 is asserted against a pin AND written into the receipt, the arms' request
SHAPES are asserted before the first request (greedy arm really greedy, vendor-default arm really
carrying no sampling params), and every failure is a named refusal before any request is sent.

ARMS
  fix arm (default door, MEMRA_VISION_OVERLAY_PUBLISH unset = auto)
     10 can't-hallucinate GREEDY            -> 200 + all codes exactly
     11 can't-hallucinate VENDOR-DEFAULT    -> 200 + all codes exactly (no sampling params in
                                               the request: the shape real traffic sends)
     20 text-only literal <|image|>         -> 200, encoded as plain text, no injection
     21 video_url                           -> named 4xx (we serve text+image, never video)
     23 faked pad with image                -> named 4xx
  control arm (MEMRA_VISION_OVERLAY_PUBLISH=0, the pre-lane program)
     10/11                                  -> NAMED 4xx at the HTTP waist (boot decided the
                                               placement cannot deliver an overlay), never a 500
                                               and never a fluent answer
  every arm
     40 text greedy row  -> banked for the byte-identity compare across arms
     41 decode rows      -> vendor-default sampled, appended to a JSONL the interleave driver
                            merges. Arm/rep/boot-nonce travel WITH each row: a health 200 proves
                            a listener, not which server (ab-arm-identity-not-liveness).

usage (one boot; `interleave.sh` drives the boot-level alternation for the no-tax arm):
  GLM53_KEY=... VISION_FIXTURES=<dir> probe-vision-ppn.py \
      --base https://glm53-api.tiyuvta.ai --arm fix --nonce <boot-nonce> --out <dir>
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

CODES = ["ZK5465", "QV4655", "XR0818"]

# The card3 cell's request bytes, pinned. A deliberate fixture change updates these in the same
# commit; a surprise change stops the battery. (Computed 2026-09-01 from the launch lane's
# window-20260901/vision-serving-shape/.)
FIXTURE_SHA256 = {
    "10-cant-hallucinate-greedy": "74c45948fa2634b65931425ee71770ce8e56894967e287d1a25935eaa5d4e144",
    "11-cant-hallucinate-sampled-vendor-default": "5732d261e9c3024c78cd6ef970ff6e8295ac99873951ac82566bd772fb4cee44",
    "20-neg-literal-placeholder": "14139fc0e9ca4b01c540efbdc40091206f9127585707d99a1ebe73c46571be36",
    "21-neg-video-url": "f534a1781e885e61da5259a4eba929a59834baa4d77b927c874b7d306c0a46bc",
    "23-neg-fakedpad-with-image": "225437f08b542b2d7d78eed32f9516120e6fd60b251780ef14056475c7d5378f",
}

# Sampling params whose presence means a request is NOT the vendor-default shape. `max_tokens` is
# a length bound, not a sampling knob, and is deliberately absent from this set.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)

# The text arm for byte identity + the decode row. Fixed, boring and real (never synthetic
# filler): greedy for identity, vendor-default for the perf row.
TEXT_PROMPT = (
    "Explain, in about 400 words, why a pipeline-parallel server must publish a buffer "
    "across a stage boundary before the next stage reads it. Use a concrete example."
)

DEFAULT_FIXTURES = (
    "~/projects/darklanes/research/glm5-serving-launch-20260901/window-20260901/"
    "vision-serving-shape"
)


def refuse(msg):
    sys.exit("REFUSE: " + msg)


def part_types(request):
    out = []
    for m in request.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            out += [p.get("type") for p in content]
    return out


def assert_shape(name, request):
    """The arms' request SHAPES are part of the instrument. Checked before the first request,
    because an arm that quietly stopped being the shape it is named after is the launch's own
    serve-gate lesson in miniature."""
    types = part_types(request)
    if name in ("10-cant-hallucinate-greedy", "11-cant-hallucinate-sampled-vendor-default"):
        if "image_url" not in types:
            refuse(f"fixture {name!r} carries NO image_url part (parts: {types}) — the "
                   "can't-hallucinate arm would be a text question with a memorized answer")
    if name == "10-cant-hallucinate-greedy" and request.get("temperature") != 0:
        refuse(f"fixture {name!r} is not greedy (temperature={request.get('temperature')!r}); "
               "greedy is the INSTRUMENT arm and must be byte-deterministic")
    if name == "11-cant-hallucinate-sampled-vendor-default":
        present = [k for k in SAMPLING_KEYS if k in request]
        if present:
            refuse(f"fixture {name!r} carries sampling params {present} — then it is not the "
                   "VENDOR-DEFAULT shape, and the never-serve-greedy law wants exactly that "
                   "shape probed (a request with NO sampling params)")
    if name == "21-neg-video-url" and "video_url" not in types:
        refuse(f"fixture {name!r} carries no video_url part (parts: {types})")
    if name == "20-neg-literal-placeholder" and "image_url" in types:
        refuse(f"fixture {name!r} must be TEXT-ONLY (parts: {types})")


def load_fixtures(path_arg):
    """Loud loader: every failure named BEFORE the first request, identity into the receipt."""
    explicit = bool(path_arg)
    path = os.path.expanduser(path_arg or DEFAULT_FIXTURES)
    if not os.path.isdir(path):
        refuse(
            f"fixture directory not found at {path!r} "
            + ("(VISION_FIXTURES/--fixtures)" if explicit
               else "(the DEFAULT path — pass --fixtures/VISION_FIXTURES explicitly)")
            + ". The can't-hallucinate claim rests on the card3 cell's exact request bytes; "
              "it does not run on a guess."
        )
    requests, meta = {}, {}
    for name, want_sha in FIXTURE_SHA256.items():
        f = os.path.join(path, name + ".json")
        if not os.path.exists(f):
            refuse(f"fixture {f!r} is missing (the battery needs all "
                   f"{len(FIXTURE_SHA256)}: {sorted(FIXTURE_SHA256)})")
        raw = open(f, "rb").read()
        got_sha = hashlib.sha256(raw).hexdigest()
        if got_sha != want_sha:
            refuse(
                f"fixture {f!r} sha256 {got_sha} != pinned {want_sha}. These bytes ARE the "
                f"instrument: the codes {CODES} are the right answer only for the card3 image. "
                "If the fixture changed deliberately, update FIXTURE_SHA256 in the same commit "
                "and say why; do not measure a different image under this arm's name."
            )
        try:
            doc = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            refuse(f"fixture {f!r} is not readable json: {type(e).__name__}: {e}")
        if not isinstance(doc, dict) or "request" not in doc:
            keys = list(doc)[:8] if isinstance(doc, dict) else type(doc).__name__
            refuse(f"fixture {f!r} has no 'request' key (top-level: {keys}); the banked shape is "
                   '{"request": {...}, ...}')
        assert_shape(name, doc["request"])
        requests[name] = doc["request"]
        meta[name] = {"sha256": got_sha, "bytes": len(raw)}
    print(f"[fixtures] {path} ({'explicit' if explicit else 'DEFAULT'}) — "
          f"{len(requests)} pinned, shapes asserted, codes={CODES}")
    return requests, {"path": path, "explicit": explicit, "files": meta, "codes": CODES}


def post(base, key, body, timeout=600):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode()), time.monotonic() - t0
        except Exception:
            return e.code, {"raw": "unparseable"}, time.monotonic() - t0


def content_of(resp):
    try:
        return resp["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


def reasoning_of(resp):
    try:
        return resp["choices"][0]["message"].get("reasoning") or ""
    except Exception:
        return ""


def named_refusal(code, resp):
    err = json.dumps(resp.get("error") or {})
    return code in (400, 415, 422) and len(err) > 10


def image_arms(base, key, requests, model, checks, receipts, want_serve):
    for name in sorted(requests):
        body = dict(requests[name])
        if model:
            body["model"] = model
        code, resp, _ = post(base, key, body)
        receipts[name] = {"status": code, "response": resp}
        codes_arm = name.startswith(("10-", "11-"))
        if codes_arm and want_serve:
            content, reasoning = content_of(resp), reasoning_of(resp)
            hit = [c for c in CODES if c in content or c in reasoning]
            ok = code == 200 and len(hit) == len(CODES)
            checks[name] = ok
            print(f"{name}: {code} codes={len(hit)}/{len(CODES)} "
                  f"{'PASS' if ok else 'FAIL'} head={content[:70]!r}")
        elif codes_arm:
            # CONTROL: a refusal is the PASS, and a 500 fails as hard as a fluent 200 — deciding
            # admissibility at boot is exactly what turns the launch's mid-prefill 500 into a
            # refusal a customer can read.
            ok = named_refusal(code, resp)
            checks[name + "-control-refusal"] = ok
            print(f"{name} [control]: {code} named_refusal={ok} "
                  f"err={json.dumps(resp.get('error') or {})[:120]}")
        elif name.startswith("20-"):
            ok = code == 200 and "<|image|>" not in content_of(resp)
            checks[name] = ok
            print(f"{name}: {code} plain_text={ok}")
        else:
            ok = named_refusal(code, resp)
            checks[name] = ok
            print(f"{name}: {code} named_refusal={ok} "
                  f"err={json.dumps(resp.get('error') or {})[:110]}")


def text_identity_row(base, key, model, max_tokens):
    """Greedy text row: the instrument, not the product. `reasoning_effort` is PINNED — omitting
    it measures think-prose instead of the claim shape, which faked a fleet-wide regression once."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
    }
    code, resp, _ = post(base, key, body)
    return code, resp, content_of(resp)


def decode_row(base, key, model, max_tokens, arm, rep, nonce):
    """Vendor-default sampled row: NO sampling params (the real traffic shape). tok/s comes from
    the response's own usage, so the number is the server's, not the client's."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
    }
    code, resp, wall = post(base, key, body)
    usage = resp.get("usage") or {}
    comp = usage.get("completion_tokens") or 0
    elapsed = usage.get("elapsed_s") or wall
    return {
        "arm": arm,
        "rep": rep,
        "boot_nonce": nonce,
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "status": code,
        "completion_tokens": comp,
        "elapsed_s": elapsed,
        "tok_s": round(comp / elapsed, 2) if elapsed else 0.0,
        "spec": usage.get("spec"),
        "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        "fingerprint": resp.get("system_fingerprint"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://" + os.environ.get("GLM53_HOST", "glm53-api.tiyuvta.ai"))
    ap.add_argument("--arm", choices=["fix", "control", "text-only"], default="fix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("GLM53_MODEL", "zai/glm-5.3-flash"))
    ap.add_argument("--fixtures", default=os.environ.get("VISION_FIXTURES", ""))
    ap.add_argument("--nonce", default=os.environ.get("BOOT_NONCE", ""),
                    help="this boot's identity; travels with every decode row")
    ap.add_argument("--reps", type=int, default=1, help="decode rows in THIS boot (the driver interleaves boots)")
    ap.add_argument("--rows-jsonl", default="", help="append decode rows here for the interleave driver")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    key = os.environ["GLM53_KEY"]
    if not args.nonce:
        refuse("--nonce/BOOT_NONCE is required: a health 200 proves a listener, not WHICH server, "
               "and a decode row without its boot's identity cannot be attributed to an arm")
    os.makedirs(args.out, exist_ok=True)
    checks, receipts = {}, {}
    fixture_meta = None

    # text-only boots have no vision at all, so the image arms do not apply to them.
    if args.arm in ("fix", "control"):
        requests, fixture_meta = load_fixtures(args.fixtures)
        image_arms(args.base, key, requests, args.model, checks, receipts,
                   want_serve=(args.arm == "fix"))

    code, resp, text = text_identity_row(args.base, key, args.model, args.max_tokens)
    receipts["40-text-greedy"] = {"status": code, "response": resp}
    checks["40-text-greedy-200"] = code == 200 and len(text) > 0
    with open(os.path.join(args.out, "text-greedy.txt"), "w") as f:
        f.write(text)
    print(f"40-text-greedy: {code} chars={len(text)} sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}")

    rows = [decode_row(args.base, key, args.model, args.max_tokens, args.arm, i, args.nonce)
            for i in range(args.reps)]
    receipts["41-decode-rows"] = rows
    if args.rows_jsonl:
        with open(args.rows_jsonl, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    served = [r["tok_s"] for r in rows if r["status"] == 200 and r["completion_tokens"] > 0]
    checks["41-decode-rows-served"] = len(served) == args.reps
    spec_engaged = all((r.get("spec") or {}).get("rounds", 0) > 0
                       for r in rows if r["status"] == 200)
    checks["41-spec-engaged"] = spec_engaged
    print(f"41-decode-rows[{args.arm} nonce={args.nonce}]: {[r['tok_s'] for r in rows]} "
          f"median={round(statistics.median(served), 2) if served else 'n/a'} "
          f"spec_engaged={spec_engaged}")

    out = {
        "arm": args.arm,
        "base": args.base,
        "model": args.model,
        "boot_nonce": args.nonce,
        "fixtures": fixture_meta,
        "text_greedy_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "checks": checks,
        "receipts": receipts,
    }
    with open(os.path.join(args.out, "receipts.json"), "w") as f:
        json.dump(out, f, indent=1)
    failed = [k for k, v in checks.items() if not v]
    print(f"VISION-PPN-{args.arm.upper()}", "PASS" if not failed else f"FAIL {failed}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
