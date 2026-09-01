#!/usr/bin/env python3
"""Byte-parity oracle for the GLM-5.3-Flash (`glm5_next`) chat-template arm.

Renders the checkpoint's OWN `chat_template.jinja` (banked beside this script, byte-identical
to the artifact's file) for every fixture case, writing
`surface-fixtures/NN-name/{input.json,expected.txt}` pairs. The Rust side
(`crates/memra-server/src/main.rs`, tests `glm5_fixtures_match_the_vendor_jinja` and
`glm5_tools_flow_through_build_chat_request`) runs each `input.json` through the REAL request
pipeline (`build_chat_request` -> `chat::apply_chat_template_tools_ex`) and asserts byte
equality against `expected.txt`.

ORACLE LAW: the jinja is the law; these bytes define the arm. The environment is the one
`transformers` uses, not jinja2's bare default:

  - `keep_trailing_newline=True`, `trim_blocks`/`lstrip_blocks` left False (jinja2 defaults);
  - extensions `loopcontrols` (this template uses `{% break %}`) and `do`;
  - `tojson` overridden to `json.dumps(..., ensure_ascii=False)` — the template passes
    `ensure_ascii=False`, which jinja2's stock `tojson` does not accept.

That is the same environment `pretok-ref-glm4.py` already uses in this lane to produce the
pre-tokenizer parity corpus, so both gates render the template identically.

PIPELINE MIRROR — everything this generator does to a request before handing it to jinja is
something `build_chat_request` does before handing turns to the renderer. Each has a named
counterpart in the Rust path:

  - `tool_calls[].function.arguments` JSON strings -> objects (`render_req_tool_call`);
  - assistant `reasoning` -> the template's `reasoning_content` (`TmplTurn::reasoning`);
  - `content: null` -> `""` (`content_to_text_vision` flattens absent content to empty; jinja
    would otherwise take `visible_text`'s else-arm and render the literal `None`);
  - role `developer` -> `system` (OpenAI's o-series rename, normalized for every dialect);
  - `reasoning_effort` -> the template's `reasoning_effort` kwarg through memra's canonical
    ladder: absent stays absent, `low`/`medium` -> `low` (the documented clamp-down: the
    template has no medium rung and its `else` arm is MAX, so falling through would answer a
    request to reason less with the model's deepest setting), `high` -> `high`,
    `xhigh`/`max`/`ultra` -> `max`. See `chat::glm5_effort_level`.

NOT COVERED (the arm documents each; no OpenAI/Anthropic/Responses request can express them):
the native `tool_reference` content type, the list-of-outputs tool-message shape
(`m.content[i].output`), and the image/video/audio `visible_text` arms.

Run: python3 gen_surface_fixtures.py   (from this directory; jinja2 required)
"""

import json
import os
import sys

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "chat_template.jinja")
FIXDIR = os.path.join(HERE, "surface-fixtures")

# memra's canonical reasoning_effort ladder -> this template's kwarg (chat::glm5_effort_level).
EFFORT = {
    "low": "low",
    "medium": "low",
    "high": "high",
    "max": "max",
    "xhigh": "max",
    "ultra": "max",
}


def mirror_messages(messages):
    """Everything build_chat_request does to a message before the renderer sees it."""
    out = []
    for m in messages:
        m = json.loads(json.dumps(m))  # deep copy, insertion order kept
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("content") is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = json.loads(args) if args.strip() else {}
            elif args is None:
                fn["arguments"] = {}
        # OpenRouter-shaped `reasoning` on an assistant turn is the template's
        # `reasoning_content`; an empty string expresses none at all (the pipeline filters it).
        reasoning = m.pop("reasoning", None)
        if reasoning:
            m["reasoning_content"] = reasoning
        out.append(m)
    return out


def _tojson(x, indent=None, ensure_ascii=False, **kw):
    return json.dumps(x, indent=indent, ensure_ascii=ensure_ascii, **kw)


def render(request):
    src = open(TEMPLATE, encoding="utf-8").read()
    env = jinja2.Environment(
        keep_trailing_newline=True,
        extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"],
    )
    env.filters["tojson"] = _tojson
    tmpl = env.from_string(src)
    ctx = {
        "messages": mirror_messages(request["messages"]),
        "add_generation_prompt": True,
    }
    if request.get("tools"):
        ctx["tools"] = request["tools"]
    effort = request.get("reasoning_effort")
    if effort is not None:
        ctx["reasoning_effort"] = EFFORT[effort]
    return tmpl.render(**ctx)


def msg(role, content, **kw):
    m = {"role": role, "content": content}
    m.update(kw)
    return m


def call(cid, name, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def tool(name, desc, props, required=None, **extra):
    fn = {"name": name, "description": desc}
    fn.update(extra)
    fn["parameters"] = {"type": "object", "properties": props}
    if required:
        fn["parameters"]["required"] = required
    return {"type": "function", "function": fn}


WEATHER = tool(
    "get_weather",
    "Get the current weather for a city",
    {"city": {"type": "string", "description": "City name"},
     "unit": {"type": "string", "enum": ["c", "f"]}},
    ["city"],
)
SEARCH = tool("search", "Search the web", {"q": {"type": "string"}}, ["q"])

CASES = []


def case(name, request):
    CASES.append((name, request))


def base(messages, **kw):
    r = {"model": "zai/glm-5.3-flash", "messages": messages}
    r.update(kw)
    return r


case("01-plain-user", base([msg("user", "What is the capital of France?")]))
case(
    "02-system-and-multiturn",
    base([
        msg("system", "You are terse."),
        msg("user", "café résumé, 2026-08-28"),
        msg("assistant", "Noted: 1,234 items."),
        msg("user", "中文 and 🚀 too?"),
    ]),
)
case("03-effort-low", base([msg("user", "hi")], reasoning_effort="low"))
case("04-effort-medium-clamps-low", base([msg("user", "hi")], reasoning_effort="medium"))
case("05-effort-high", base([msg("user", "hi")], reasoning_effort="high"))
case("06-effort-max", base([msg("user", "hi")], reasoning_effort="max"))
case("07-tools-declaration", base([msg("user", "weather in Paris?")], tools=[WEATHER, SEARCH]))
case(
    "08-tools-with-strict-key",
    base(
        [msg("user", "weather in Paris?")],
        tools=[tool("get_weather", "Get weather", {"city": {"type": "string"}}, ["city"],
                    strict=True)],
    ),
)
case(
    "09-single-call-cycle",
    base(
        [
            msg("user", "What is the weather in Paris?"),
            msg("assistant", None, tool_calls=[call("call_1", "get_weather", {"city": "Paris"})]),
            msg("tool", '{"temp_c": 21, "sky": "sunny"}', tool_call_id="call_1"),
        ],
        tools=[WEATHER],
    ),
)
case(
    "10-parallel-calls-ordered",
    base(
        [
            msg("user", "Paris and Rome?"),
            msg("assistant", "Checking both.", tool_calls=[
                call("c1", "get_weather", {"city": "Paris"}),
                call("c2", "get_weather", {"city": "Rome"}),
            ]),
            msg("tool", "paris:21", tool_call_id="c1"),
            msg("tool", "rome:27", tool_call_id="c2"),
        ],
        tools=[WEATHER],
    ),
)
case(
    "11-parallel-calls-out-of-order",
    base(
        [
            msg("user", "Paris and Rome?"),
            msg("assistant", "", tool_calls=[
                call("c1", "get_weather", {"city": "Paris"}),
                call("c2", "get_weather", {"city": "Rome"}),
            ]),
            msg("tool", "rome:27", tool_call_id="c2"),
            msg("tool", "paris:21", tool_call_id="c1"),
        ],
        tools=[WEATHER],
    ),
)
case(
    "12-results-without-ids",
    base(
        [
            msg("user", "Paris?"),
            msg("assistant", "", tool_calls=[
                {"type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            ]),
            msg("tool", "paris:21"),
        ],
        tools=[WEATHER],
    ),
)
case(
    "13-assistant-reasoning-replay",
    base([
        msg("user", "a"),
        msg("assistant", "A", reasoning="I considered a."),
        msg("user", "b"),
        msg("assistant", "B"),
        msg("user", "c"),
    ]),
)
case(
    "14-assistant-inline-think-split",
    base([
        msg("user", "a"),
        msg("assistant", "<think>inline reasoning</think>final answer"),
        msg("user", "b"),
    ]),
)
case(
    "15-multi-cycle-agentic",
    base(
        [
            msg("user", "Weather in Paris, then search for it."),
            msg("assistant", None, reasoning="Call the weather tool first.",
                tool_calls=[call("c1", "get_weather", {"city": "Paris"})]),
            msg("tool", "paris:21", tool_call_id="c1"),
            msg("assistant", "Now searching.", tool_calls=[call("c2", "search", {"q": "Paris"})]),
            msg("tool", "results...", tool_call_id="c2"),
        ],
        tools=[WEATHER, SEARCH],
    ),
)
case(
    "16-non-string-arguments",
    base(
        [
            msg("user", "book it"),
            msg("assistant", "", tool_calls=[call("c1", "book", {
                "days": 3, "ok": True, "note": None, "tags": ["a", "b"],
                "opts": {"x": 1}, "who": "Ada",
            })]),
            msg("tool", "booked", tool_call_id="c1"),
        ],
        tools=[tool("book", "Book a trip", {"days": {"type": "integer"}})],
    ),
)
case(
    "17-untrimmed-user-trimmed-assistant",
    base([
        msg("user", "  spaced user  "),
        msg("assistant", "  spaced assistant  "),
        msg("user", "\ttabbed\n"),
    ]),
)
case(
    "18-developer-role-normalized",
    base([
        msg("developer", "Follow the house style."),
        msg("user", "go"),
    ]),
)
case(
    "19-tools-and-leading-system",
    base(
        [msg("system", "You are a weather bot."), msg("user", "Paris?")],
        tools=[WEATHER],
        reasoning_effort="high",
    ),
)
case(
    "20-effort-low-with-call-cycle",
    base(
        [
            msg("user", "Paris?"),
            msg("assistant", None, tool_calls=[call("c1", "get_weather", {"city": "Paris"})]),
            msg("tool", "paris:21", tool_call_id="c1"),
        ],
        tools=[WEATHER],
        reasoning_effort="low",
    ),
)


# The two turns of the LIVE agentic round-trip receipt (surface-receipts/roundtrip-*). Their
# `expected.txt` bytes are what the driver POSTs to /v1/completions verbatim, so the prompt on
# the wire is exactly what the shipped Rust arm renders — pinned by the fixture test, not
# asserted by the driver.
RT_TOOLS = [tool(
    "get_weather",
    "Get the current weather for a city",
    {"city": {"type": "string", "description": "City name"}},
    ["city"],
)]
case(
    "21-roundtrip-turn1-ask",
    base([msg("user", "What is the weather in Paris right now? Use the tool.")],
         tools=RT_TOOLS),
)
case(
    "22-roundtrip-turn2-after-result",
    base(
        [
            msg("user", "What is the weather in Paris right now? Use the tool."),
            msg("assistant", None,
                tool_calls=[call("call_rt1", "get_weather", {"city": "Paris"})]),
            msg("tool", '{"temp_c": 21, "sky": "sunny"}', tool_call_id="call_rt1"),
        ],
        tools=RT_TOOLS,
    ),
)


# IMAGE ARM (lane/glm5-vision, 2026-08-30). Upstream shape: the template's emit_image()
# renders <|begin_of_image|><|image|><|end_of_image|> for a typed image part, and
# Glm5NextProcessor.replace_image_token then expands the single <|image|> to one per
# merged token (grid/4). memra's shape: content_to_text_vision renders the ALREADY
# EXPANDED run inline as message text before the template sees it. The two must be
# byte-identical for the same grid — asserted below at generation time, so the committed
# fixture (flattened-string input, the shape memra's pipeline feeds the renderer) carries
# an expected.txt that IS the upstream bytes. Grid: det112 (8x8 patches, 16 merged
# tokens, research/glm5-vision-20260830).
IMG_N_TOKENS = 16
IMG_RUN = "<|begin_of_image|>" + "<|image|>" * IMG_N_TOKENS + "<|end_of_image|>"
_typed = base([
    msg("user", [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,IGNORED"}},
        {"type": "text", "text": "Transcribe the text in this image exactly."},
    ]),
])
_flat = base([
    msg("user", IMG_RUN + "Transcribe the text in this image exactly."),
])
_upstream = render(_typed).replace("<|image|>", "<|image|>" * IMG_N_TOKENS)
assert render(_flat) == _upstream, (
    "memra's inline image-run splice diverged from the template's typed-part arm + "
    "processor expansion"
)
case("23-image-message-16tok", _flat)


def main():
    os.makedirs(FIXDIR, exist_ok=True)
    for name, request in CASES:
        d = os.path.join(FIXDIR, name)
        os.makedirs(d, exist_ok=True)
        expected = render(request)
        with open(os.path.join(d, "input.json"), "w", encoding="utf-8") as f:
            json.dump({"request": request}, f, indent=2, ensure_ascii=False)
            f.write("\n")
        with open(os.path.join(d, "expected.txt"), "w", encoding="utf-8") as f:
            f.write(expected)
        print(f"{name}: {len(expected)} bytes")
    print(f"{len(CASES)} fixtures -> {FIXDIR}")


if __name__ == "__main__":
    sys.exit(main())
