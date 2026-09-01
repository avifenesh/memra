#!/usr/bin/env python3
"""Render Qwen3.8's OWN chat template with jinja2 to produce prompt-byte goldens.

The goldens are the rendering LAW for `ModelCaps::qwen_effort`: memra's Rust renderer must
reproduce them byte-for-byte. Same method the step35 arm used
(`research/step37-p2-20260806/render_step35_template.py`) — `trim_blocks`/`lstrip_blocks` are
the settings HF transformers and llama.cpp's minja both parse chat templates with, so this is
what the deployed template actually does, not an interpretation of it.

Why this file exists rather than a hand-written expectation: the lane it belongs to
(lane/reasoning-schema-20260823) found that `reasoning_effort: low|medium|high` was accepted
and then discarded on every qwen3.8 request, because the `effort_levels` cap probed for
`reasoning_effort is defined` and this template spells its input `reasoning_effort|default('xhigh')`.
Fixing that changes prompt bytes, so the fix is pinned against the vendor's own jinja.

Usage:  python3 render_qwen38_goldens.py [--out goldens]
"""

import argparse
import json
import pathlib

import jinja2

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "qwen38-27b.chat_template.jinja"

# One user turn, no system turn, no tools: the shape that isolates the effort sentence.
PLAIN = [{"role": "user", "content": "hi"}]
# A leading system turn: the sentence must PREPEND it with a blank line, not become its own turn.
WITH_SYSTEM = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "hi"},
]
# A system turn with NO content: the template `|trim`s system turns into `merged_system` and only
# joins with a blank line when that is non-empty, so the effort sentence must render ALONE here.
# An unconditional separator would emit a stray `\n\n` before `<|im_end|>`.
EMPTY_SYSTEM = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "hi"},
]

# TWO leading system turns — the shape this server produces itself, because it normalizes OpenAI's
# `developer` role to `system` before rendering. The template MERGES the whole leading run into one
# system turn (joined with `\n`) and its body loop raises on a later system message, so a renderer
# that emits one turn per message diverges here. Found by review of this lane's first cut.
TWO_SYSTEM = [
    {"role": "system", "content": "rules"},
    {"role": "system", "content": "dev rules"},
    {"role": "user", "content": "hi"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def raise_exception(msg):
    raise jinja2.TemplateError(msg)


def render(env, template, **kwargs):
    kwargs.setdefault("add_generation_prompt", True)
    return template.render(raise_exception=raise_exception, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "goldens"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "separators": (", ", ": ")}
    template = env.from_string(TEMPLATE.read_text())

    cases = {}
    # THE BINARY AXIS. `enable_thinking` unset = the template's own default (thinking ON).
    cases["plain_default"] = dict(messages=PLAIN)
    cases["plain_off"] = dict(messages=PLAIN, enable_thinking=False)
    # THE LADDER, thinking on. Note `medium` injects NO sentence — the template's own
    # zero-steering middle rung — and `high` is NOT in the template's accepted set (it raises);
    # Qwen's own hosted API maps high->xhigh, which is what our mint's template alias does.
    for level in ("xhigh", "medium", "low"):
        cases[f"plain_{level}"] = dict(messages=PLAIN, reasoning_effort=level)
        cases[f"system_{level}"] = dict(messages=WITH_SYSTEM, reasoning_effort=level)
        cases[f"tools_{level}"] = dict(messages=PLAIN, reasoning_effort=level, tools=TOOLS)
    cases["system_default"] = dict(messages=WITH_SYSTEM)
    cases["empty_system_low"] = dict(messages=EMPTY_SYSTEM, reasoning_effort="low")
    cases["two_system_xhigh"] = dict(messages=TWO_SYSTEM, reasoning_effort="xhigh")
    cases["two_system_off"] = dict(messages=TWO_SYSTEM, enable_thinking=False)
    cases["two_system_tools_low"] = dict(
        messages=TWO_SYSTEM, reasoning_effort="low", tools=TOOLS
    )
    cases["tools_default"] = dict(messages=PLAIN, tools=TOOLS)
    # A thinking-OFF request carries no effort sentence even when a level is named: the
    # template wraps the whole instruction block in `enable_thinking is undefined or is true`.
    cases["plain_off_with_level"] = dict(
        messages=PLAIN, enable_thinking=False, reasoning_effort="low"
    )
    cases["system_off"] = dict(messages=WITH_SYSTEM, enable_thinking=False)
    # MULTI-TURN (lane/dflash2-session-reuse): the template's preserve_thinking DEFAULT
    # (kwarg absent) REPLAYS every prior assistant turn's <think> block —
    # `<think>\n{reasoning_content|trim}\n</think>\n\n` before the content, EMPTY when the
    # client sent no reasoning. memra's first renderer cut emitted content only, which put
    # every multi-turn q38 prompt off the vendor's bytes AND kept the reuse pools' text
    # tier from ever matching a parked stream (the generation prompt ends in a <think>
    # block, so the live stream carries bytes the re-render lacked).
    MULTITURN = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": "again"},
    ]
    MULTITURN_REASONED = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello there",
            "reasoning_content": "the user greets; greet back",
        },
        {"role": "user", "content": "again"},
    ]
    cases["multiturn_off"] = dict(messages=MULTITURN, enable_thinking=False)
    cases["multiturn_default"] = dict(messages=MULTITURN)
    cases["multiturn_reasoned_off"] = dict(messages=MULTITURN_REASONED, enable_thinking=False)

    manifest = {}
    for name, kwargs in sorted(cases.items()):
        text = render(env, template, **kwargs)
        (out / f"{name}.txt").write_text(text)
        manifest[name] = {"kwargs": {k: v for k, v in kwargs.items() if k != "messages"},
                          "messages": kwargs["messages"],
                          "bytes": len(text.encode())}
        print(f"{name:26s} {len(text.encode()):5d} bytes")

    # And prove the template REFUSES what it does not define, so our renderer's refusal is
    # the vendor's refusal rather than our own invention.
    refused = {}
    for level in ("high", "none", "minimal", "banana", ""):
        try:
            render(env, template, messages=PLAIN, reasoning_effort=level)
            refused[level] = None
        except jinja2.TemplateError as exc:
            refused[level] = str(exc)
    manifest["_refused_by_template"] = refused
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("\nlevels the template REFUSES (raise_exception):")
    for level, err in refused.items():
        print(f"  {level!r:10s} -> {err if err else 'ACCEPTED'}")


if __name__ == "__main__":
    main()
