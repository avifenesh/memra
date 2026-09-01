#!/usr/bin/env python3
"""Build the probe's scoring prompts from the banked gpf-ab pool (real agent transcripts).

Each pool transcript is cut at structural boundaries (nearest blank line to 35/60/85/100%
of the text) so one transcript yields several decode starting points, which is the shape
of real agent traffic: the model continues mid-session. Every prefix is rendered through
the artifact's OWN chat_template.jinja exactly the way memra's build_chat_request does
(gen_surface_fixtures.py environment: keep_trailing_newline, loopcontrols+do, tojson
ensure_ascii=False), reasoning_effort pinned low (house law), add_generation_prompt on,
then tokenized with the artifact tokenizer.

PARITY GATE (must print PASS): the full A4630 render must tokenize to exactly the
prompt_tokens the serving binary reported for the same chat request (4626), which proves
the raw prompt_ids path and the chat path see the same token stream.
"""
import hashlib
import json

import jinja2
from tokenizers import Tokenizer

ART = "/root/models/glm53-nvfp4"
POOL = json.load(open("/root/gpf-ab/prompts.json"))
OUT = "/root/dfp2/scoring_prompts.json"
A4630_SERVER_PROMPT_TOKENS = 4626  # measured on the serving binary, this box, 2026-08-29

tok = Tokenizer.from_file(f"{ART}/tokenizer.json")


def _tojson(x, indent=None, ensure_ascii=False, **kw):
    return json.dumps(x, indent=indent, ensure_ascii=ensure_ascii, **kw)


env = jinja2.Environment(
    keep_trailing_newline=True,
    extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"],
)
env.filters["tojson"] = _tojson
tmpl = env.from_string(open(f"{ART}/chat_template.jinja", encoding="utf-8").read())


def render(user_content):
    return tmpl.render(
        messages=[{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        reasoning_effort="low",
    )


def encode(text):
    return tok.encode(text, add_special_tokens=False).ids


def cut_points(text, fracs):
    """Snap each fraction of len(text) to the nearest blank-line boundary."""
    bounds = []
    i = text.find("\n\n")
    while i != -1:
        bounds.append(i + 2)
        i = text.find("\n\n", i + 2)
    outs = []
    for f in fracs:
        want = int(len(text) * f)
        if f >= 1.0 or not bounds:
            outs.append(len(text))
            continue
        best = min(bounds, key=lambda b: abs(b - want))
        outs.append(best)
    return outs


# Parity gate first: full A4630 through the template must match the serving binary count.
full_a = render(POOL["A4630"])
ids_a = encode(full_a)
gate = "PASS" if len(ids_a) == A4630_SERVER_PROMPT_TOKENS else "FAIL"
print(f"parity-gate A4630: local render+tokenize = {len(ids_a)} ids, "
      f"server chat prompt_tokens = {A4630_SERVER_PROMPT_TOKENS} -> {gate}")
if gate == "FAIL":
    raise SystemExit(2)

prompts = []
for base in ["A4630", "B5550", "C6470"]:
    text = POOL[base]
    for frac, cut in zip([0.35, 0.60, 0.85, 1.00], cut_points(text, [0.35, 0.60, 0.85, 1.00])):
        prefix = text[:cut]
        rendered = render(prefix)
        ids = encode(rendered)
        prompts.append({
            "name": f"{base}-f{int(frac * 100)}",
            "base": base,
            "frac": frac,
            "cut_chars": cut,
            "n_ids": len(ids),
            "render_sha16": hashlib.sha256(rendered.encode()).hexdigest()[:16],
            "ids": ids,
        })
w = render(POOL["WARM"])
prompts.append({
    "name": "WARM-f100", "base": "WARM", "frac": 1.0, "cut_chars": len(POOL["WARM"]),
    "n_ids": len(encode(w)),
    "render_sha16": hashlib.sha256(w.encode()).hexdigest()[:16],
    "ids": encode(w),
})

json.dump(prompts, open(OUT, "w"))
for p in prompts:
    print(f'{p["name"]}: {p["n_ids"]} ids  render_sha16={p["render_sha16"]}')
print(f"wrote {OUT}: {len(prompts)} scoring prompts")
