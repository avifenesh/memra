#!/usr/bin/env python3
"""DeepSeek-V4.1 chat-encoding fixtures for memra's `encoding_dsv41` renderer and parser.

The vendor's own `encoding/encoding.py` at the pinned revision (ref/, checked against
ref/SHA256SUMS) IS the oracle. This script runs it over a case set that reaches every branch of
`encode_messages` and `parse_message_from_completion_text`, including every case of the vendor's
`test_encoding.py` and its `encoding/tests` golden pairs, and writes:

  fixtures/<case>/input.json    the case, in the shape memra's harness reads
  fixtures/<case>/expected.txt  encode_messages(...) bytes (absent for a refusal case)
  fixtures/tokenization.json    case -> token ids of expected.txt under ref/tokenizer.json,
                                taken with transformers AutoTokenizer.encode exactly as the
                                vendor's inference/generate.py does, and cross-checked against
                                the `tokenizers` library
  parse/<case>.json             a completion text and the vendor's parse of it (or its refusal)

input.json fields: "messages" (OpenAI-shape messages as the vendor takes them, request tools
excluded), "req_tools" (OpenAI tool list or null; attached to the leading system message, or to
a synthetic empty system message when there is none, which is memra's serve-surface law for
DeepSeek dialects), "think" ("thinking" | "chat"), "reasoning_effort" (memra's string form:
null, a level name, or a decimal budget), "add_generation_prompt" (always true: the vendor has no
switch), "expected_images" (the vendor's media["images"] in prompt order, by url), "wire" (the
case is expressible on memra's OpenAI chat-completions surface), "error" (present on a refusal
case: the vendor's exception), "source" (where the case comes from).

Run from the repository root (CPU only, about a second):

  uv run --with 'transformers==5.22.2' --with 'tokenizers==0.23.3' \\
      python3 -I -B research/modelplan-onboarding-dsv41-20261109/encoding/gen_fixtures.py

`-I` keeps the interpreter from importing anything beside the vendor file this script names, and
`-B` keeps it from writing bytecode into ref/.
"""

import copy
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
FIX = os.path.join(HERE, "fixtures")
PARSE = os.path.join(HERE, "parse")


def verify_ref():
    """Refuse to run against vendor files that are not the pinned bytes."""
    with open(os.path.join(REF, "SHA256SUMS")) as f:
        rows = [line.split() for line in f if line.strip()]
    for digest, rel in rows:
        with open(os.path.join(REF, rel), "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != digest:
            raise SystemExit(f"ref/{rel}: sha256 {got} != pinned {digest}")
    return len(rows)


N_REF = verify_ref()
sys.path.insert(0, os.path.join(REF, "encoding"))
import encoding as enc  # noqa: E402

IMG = enc.IMAGE_PLACEHOLDER


# ---------------------------------------------------------------------------------------------
# shared message pieces (test_encoding.py's own helpers, verbatim where they exist)
# ---------------------------------------------------------------------------------------------


def make_tool():
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


def make_tool_call_messages():
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "reasoning_content": "  reason  ",
            "content": "summary",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"value","limit":2}',
                    },
                }
            ],
        },
    ]


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression \"exactly\"\nReturns a number.",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string", "description": "表达式"}},
            "required": ["expr"],
        },
    },
}


def call(cid, name, arguments, **extra):
    c = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if cid is not None:
        c["id"] = cid
    c.update(extra)
    return c


def image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def text(t):
    return {"type": "text", "text": t}


TOOLCYCLE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Weather in Beijing?"},
    {
        "role": "assistant",
        "reasoning_content": "Use get_weather.",
        "content": "",
        "tool_calls": [call("call_001", "get_weather", '{"location": "Beijing", "unit": "celsius"}')],
    },
    {"role": "tool", "tool_call_id": "call_001", "content": '{"temperature": 22, "condition": "sunny"}'},
    {"role": "assistant", "reasoning_content": "Format the answer.", "content": "Sunny, 22C."},
]
MULTI = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "reasoning_content": "The user said hello.", "content": "Hi there!"},
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "reasoning_content": "It is Paris.", "content": "Paris."},
]
GEN = MULTI[:4]


# ---------------------------------------------------------------------------------------------
# the case set
# ---------------------------------------------------------------------------------------------

CASES = []


def add(name, messages, think="thinking", effort=None, req_tools=None, source="matrix", alt=None):
    """`effort` is the vendor argument (None, a level name or an int); memra gets its string.
    `alt` = [(messages, context)] vendor-shape inputs that must render the same suffix and the
    same trailing image records (test_encoding's content_blocks / tool_content_blocks / context
    variants), proving the memra-shape case stands for them."""
    CASES.append(
        dict(
            name=name,
            messages=copy.deepcopy(messages),
            think=think,
            effort=effort,
            req_tools=copy.deepcopy(req_tools),
            source=source,
            alt=alt or [],
        )
    )


T = "test_encoding.py::"

# ---- test_encoding.py, case by case ---------------------------------------------------------
add(
    "vt-renders-images",
    [{"role": "user", "content": [text("inspect"), image("/unused/image.png")]}],
    "chat",
    source=T + "test_v41_renders_images",
)
for role in ("user", "tool"):
    add(
        f"vt-rejects-placeholder-in-{role}",
        [{"role": role, "content": f"hi {IMG}"}],
        "chat",
        source=T + f"test_v41_rejects_image_placeholder_in_text[{role}]",
    )
# test_v41_tool_result_images_follow_prompt_order: four input formats, one prompt.
_hist = [
    {"role": "user", "content": [image("history.png")]},
    {
        "role": "assistant",
        "tool_calls": [call(n, n, "{}") for n in ("a", "b")],
    },
]
_results = [
    {
        "type": "tool_result",
        "tool_use_id": n,
        "content": [text(f"result {n}"), image(f"{n}.png")],
    }
    for n in ("b", "a")
]
add(
    "vt-tool-result-images-follow-prompt-order",
    _hist
    + [{"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]} for b in _results],
    "chat",
    source=T + "test_v41_tool_result_images_follow_prompt_order[tool_messages]",
    alt=[
        (_hist + [{"role": "user", "content_blocks": _results}], None),
        (
            _hist
            + [
                {"role": "tool", "tool_call_id": b["tool_use_id"], "content_blocks": b["content"]}
                for b in _results
            ],
            None,
        ),
        ([{"role": "user", "content_blocks": _results}], _hist),
    ],
)
for effort, budget in [(None, 75), ("low", 50), ("high", 75), ("max", 100), (1, 1), (42, 42), (100, 100)]:
    add(
        f"vt-effort-{effort if effort is not None else 'default'}",
        [{"role": "user", "content": "question"}],
        "thinking",
        effort,
        source=T + f"test_v41_maps_reasoning_effort_to_1_100_budget[{effort}-{budget}]",
    )
add(
    "vt-effort-only-first-thinking-message",
    [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
    "thinking",
    100,
    source=T + "test_v41_only_adds_reasoning_effort_to_first_thinking_message",
)
add(
    "vt-effort-only-first-chat-message",
    [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
    "chat",
    100,
    source=T + "test_v41_only_adds_reasoning_effort_to_first_thinking_message",
)
add(
    "vt-chat-mode-no-effort-no-system-token",
    [{"role": "user", "content": "hello"}],
    "chat",
    "max",
    source=T + "test_v41_chat_mode_has_no_reasoning_effort_or_system_token",
)
for effort in (-1, 0, 101):
    add(
        f"vt-rejects-effort-{effort}",
        [{"role": "user", "content": "question"}],
        "thinking",
        effort,
        source=T + f"test_v41_rejects_out_of_range_or_unknown_reasoning_effort[{effort}]",
    )
# Of the vendor's [-1, 0, 101, "medium"] and [True, False, 1.5], "medium" renders the default
# budget in memra (the never-corrupt clamp the encoding_dsv4 arm applies; README "Where memra
# departs") and no bool reaches memra's renderer (the wire carries reasoning_effort as a string
# and refuses a JSON bool), so those three are documented divergences, not fixtures. The float
# reaches memra as its decimal string, which the renderer refuses as the vendor refuses 1.5.
add(
    "vt-rejects-effort-type-1.5",
    [{"role": "user", "content": "question"}],
    "thinking",
    1.5,
    source=T + "test_v41_rejects_non_string_non_integer_effort_types[1.5]",
)
add(
    "vt-leading-system-token",
    [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hello"}],
    "chat",
    source=T + "test_v41_leading_system_message_uses_system_token",
)
add(
    "vt-mid-conversation-system",
    [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "system", "content": "mid sys"},
    ],
    "thinking",
    88,
    source=T + "test_v41_mid_conversation_system_message",
)
add(
    "vt-tool-instructions-chat",
    [{"role": "system", "content": "system", "tools": [make_tool()]}, {"role": "user", "content": "question"}],
    "chat",
    source=T + "test_v41_tool_instructions_use_spaced_dsml_tags_in_chat_mode",
)
add(
    "vt-spaced-dsml-assistant",
    make_tool_call_messages(),
    "thinking",
    source=T + "test_v41_renders_spaced_dsml_with_v4_assistant_semantics",
)
for location in ("tool", "function"):
    for ns_name, ns in (("str", "search"), ("obj", {"name": "search", "description": "Search tools."})):
        tool = make_tool()
        (tool if location == "tool" else tool["function"])["namespace"] = copy.deepcopy(ns)
        add(
            f"vt-namespaced-schema-{location}-{ns_name}",
            [{"role": "system", "content": "system", "tools": [tool]}],
            "chat",
            source=T + f"test_v41_renders_namespaced_tool_schemas[{ns_name}-{location}]",
        )
for mode in ("chat", "thinking"):
    for location in ("tool", "function", "qualified_name"):
        msgs = make_tool_call_messages()
        c = msgs[1]["tool_calls"][0]
        if location == "qualified_name":
            c["function"]["name"] = "search::lookup"
        else:
            (c if location == "tool" else c["function"])["namespace"] = "search"
        add(
            f"vt-namespaced-call-{location}-{mode}",
            msgs,
            mode,
            source=T + f"test_v41_namespaced_tool_calls_roundtrip[{location}-{mode}]",
        )
_tools, _calls = [], []
for ns in (None, "search", "files"):
    tl = make_tool()
    cl = call(None, "lookup", '{"query":"value"}')
    if ns is not None:
        tl["namespace"] = {"name": ns}
        cl["namespace"] = ns
    _tools.append(tl)
    _calls.append(cl)
add(
    "vt-same-named-tools-separate-namespaces",
    [
        {"role": "system", "content": "system", "tools": _tools},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "summary", "tool_calls": _calls},
    ],
    "chat",
    source=T + "test_v41_keeps_same_named_tools_in_separate_namespaces",
)
_tl = make_tool()
_tl["function"]["name"] = "search::lookup"
_tl["namespace"] = {"name": "search", "description": "Search tools."}
_msgs = make_tool_call_messages()
_msgs[1]["tool_calls"][0]["function"]["name"] = "search::lookup"
_msgs[1]["tool_calls"][0]["namespace"] = "search"
add(
    "vt-no-duplicate-qualified-namespace",
    [{"role": "system", "content": "system", "tools": [_tl]}] + _msgs,
    "thinking",
    source=T + "test_v41_does_not_duplicate_a_qualified_namespace",
)
for i, (name, ns) in enumerate(
    [("search::lookup", "files"), ("search::nested::lookup", None), ("lookup", "search::nested")]
):
    tl = make_tool()
    tl["function"]["name"] = name
    tl["namespace"] = ns
    add(
        f"vt-rejects-ambiguous-namespace-schema-{i}",
        [{"role": "system", "content": "system", "tools": [tl]}],
        "chat",
        source=T + "test_v41_rejects_ambiguous_tool_namespaces",
    )
    msgs = make_tool_call_messages()
    c = msgs[1]["tool_calls"][0]
    c["function"]["name"] = name
    c["namespace"] = ns
    add(
        f"vt-rejects-ambiguous-namespace-call-{i}",
        msgs,
        "thinking",
        source=T + "test_v41_rejects_ambiguous_tool_namespaces",
    )
_drop = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "reasoning_content": "r2", "tool_calls": [call("c1", "lookup", "{}")]},
    {"role": "tool", "tool_call_id": "c1", "content": "value"},
]
add("vt-drop-thinking-without-tools", _drop, "thinking", source=T + "test_v41_drop_thinking_without_tools")
add(
    "vt-drop-thinking-next-turn",
    _drop + [{"role": "user", "content": "q3"}],
    "thinking",
    source=T + "test_v41_drop_thinking_without_tools",
)
add(
    "vt-task-query",
    [{"role": "user", "content": "classify me", "task": "query"}],
    "chat",
    source=T + "test_v41_task_sp_token",
)

# ---- the vendor's encoding/tests golden pairs and inference/examples -----------------------
VENDOR_PAIRS = []
for n in range(1, 6):
    case = enc.load_cases(os.path.join(REF, "encoding", "tests", f"test_input_{n}.json"))[0]
    with open(os.path.join(REF, "encoding", "tests", f"test_output_{n}.txt")) as f:
        golden = f.read()
    prompt, _ = enc.encode_case(case, thinking_mode="chat")
    assert prompt == golden, f"vendor golden {n} does not reproduce under its own encoding.py"
    VENDOR_PAIRS.append(n)
    add(
        f"vendor-golden-{n}",
        case["messages"],
        case.get("thinking_mode") or "chat",
        case.get("reasoning_effort"),
        source=f"encoding/tests/test_input_{n}.json",
    )
with open(os.path.join(REF, "inference", "examples", "example.txt")) as f:
    _txt = f.read().rstrip("\n").split("\n\n")
for i, line in enumerate(_txt):
    add(
        f"vendor-example-txt-{i + 1}",
        [{"role": "user", "content": enc.parse_tagged_text(line)}],
        "chat",
        source="inference/examples/example.txt (parse_tagged_text)",
    )
for i, case in enumerate(enc.load_cases(os.path.join(REF, "inference", "examples", "example_harmony.json"))):
    for mode, effort in (("chat", None), ("thinking", 75)):
        add(
            f"vendor-example-harmony-{i + 1}-{mode}",
            case["messages"],
            mode,
            effort,
            source="inference/examples/example_harmony.json",
        )

# ---- the generated matrix: every branch of render_message and its preprocessing -------------
for mode in ("thinking", "chat"):
    for effort in (None, "low", "high", "max", 1, 63, 100):
        if mode == "chat" and effort not in (None, "max"):
            continue
        tag = effort if effort is not None else "default"
        add(f"m-{mode}-{tag}-single", [{"role": "user", "content": "What is 2+2?"}], mode, effort)
        add(f"m-{mode}-{tag}-system", MULTI[:2], mode, effort)
        add(f"m-{mode}-{tag}-multiturn", MULTI, mode, effort)
        add(f"m-{mode}-{tag}-genprompt", GEN, mode, effort)
        add(f"m-{mode}-{tag}-tools", MULTI[:2], mode, effort, [WEATHER_TOOL])
        add(f"m-{mode}-{tag}-toolcycle", TOOLCYCLE, mode, effort, [WEATHER_TOOL])
        add(f"m-{mode}-{tag}-toolresult-gen", TOOLCYCLE[:4], mode, effort, [WEATHER_TOOL])

add("m-tools-no-system", [{"role": "user", "content": "Weather in Oslo?"}], "thinking", None, [WEATHER_TOOL])
add("m-tools-two-decl", MULTI[:2], "thinking", None, [WEATHER_TOOL, CALC_TOOL])
add("m-empty-conversation", [], "thinking")
add("m-empty-user", [{"role": "user", "content": ""}], "thinking")
add("m-empty-system-chat", [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}], "chat")
add(
    "m-typed-args",
    [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather for 3 days, metric."},
        {
            "role": "assistant",
            "reasoning_content": "typed args",
            "content": "",
            "tool_calls": [
                call(
                    "c1",
                    "get_weather",
                    '{"location": "Beijing \\"north\\"\\n", "days": 3, "metric": true, "ratio": 0.5,'
                    ' "none": null, "tags": ["a", "b"], "opts": {"k": "v", "n": [1, 2]}, "city": "北京"}',
                )
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ],
    "thinking",
    None,
    [WEATHER_TOOL],
)
add(
    "m-empty-args",
    [
        {"role": "user", "content": "Ping."},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "ping", "{}")]},
    ],
    "chat",
    None,
    [WEATHER_TOOL],
)
add(
    "m-double-encoded-args",
    [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("c1", "get_weather", json.dumps('{"location": "Oslo"}'))],
        },
    ],
    "chat",
    None,
    [WEATHER_TOOL],
)
add(
    "m-non-object-args-string",
    [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", "Oslo please")]},
    ],
    "chat",
    None,
    [WEATHER_TOOL],
)
add(
    "m-non-object-args-array",
    [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", '["Oslo", 2]')]},
    ],
    "chat",
    None,
    [WEATHER_TOOL],
)
_two = [
    {"role": "system", "content": "Use tools."},
    {"role": "user", "content": "Weather and calc."},
    {
        "role": "assistant",
        "reasoning_content": "two calls",
        "content": "Calling both.",
        "tool_calls": [
            call("cA", "get_weather", '{"location": "Beijing"}'),
            call("cB", "calculate", '{"expr": "2+2"}'),
        ],
    },
    {"role": "tool", "tool_call_id": "cB", "content": "4"},
    {"role": "tool", "tool_call_id": "cA", "content": "sunny"},
]
add("m-two-results-sorted", _two, "thinking", None, [WEATHER_TOOL, CALC_TOOL])
add(
    "m-two-results-unknown-id",
    _two[:3]
    + [{"role": "tool", "tool_call_id": "zz", "content": "4"}, {"role": "tool", "tool_call_id": "cA", "content": "sunny"}],
    "thinking",
    None,
    [WEATHER_TOOL, CALC_TOOL],
)
add(
    "m-tool-result-then-user",
    _two + [{"role": "user", "content": "And tomorrow?"}],
    "thinking",
    None,
    [WEATHER_TOOL, CALC_TOOL],
)
add(
    "m-tool-result-merged-into-user",
    [
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "x", "content": "orphan result"},
    ],
    "thinking",
)
add(
    "m-consecutive-users",
    [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
    "thinking",
)
add(
    "m-user-parts-text-only",
    [{"role": "user", "content": [text("part one"), text("part two")]}],
    "thinking",
)
add(
    "m-images-interleaved",
    [
        {"role": "system", "content": "You see images."},
        {"role": "user", "content": [text("first"), image("a.png"), text("then"), image("b.png")]},
    ],
    "thinking",
    "high",
)
add("m-image-only-user", [{"role": "user", "content": [image("only.png")]}], "chat")
add(
    "m-image-internal-block",
    [{"role": "user", "content": [{"type": "image", "url": "internal.png"}, text("what is this?")]}],
    "chat",
)
add(
    "m-images-in-system-and-assistant",
    [
        {"role": "system", "content": [text("reference:"), image("sys.png")]},
        {"role": "user", "content": "compare"},
        {"role": "assistant", "content": [text("I drew"), image("drawn.png")]},
        {"role": "user", "content": [image("user.png")]},
    ],
    "chat",
)
add(
    "m-images-multiturn-thinking",
    [
        {"role": "user", "content": [image("one.png"), text("describe")]},
        {"role": "assistant", "reasoning_content": "look", "content": "A cat."},
        {"role": "user", "content": [text("and this?"), image("two.png")]},
    ],
    "thinking",
)
add(
    "m-tool-result-image-only",
    [
        {"role": "user", "content": "screenshot please"},
        {"role": "assistant", "content": "", "tool_calls": [call("s1", "shot", "{}")]},
        {"role": "tool", "tool_call_id": "s1", "content": [image("shot.png")]},
    ],
    "thinking",
    None,
    [WEATHER_TOOL],
)
add(
    "m-rejects-placeholder-in-text-part",
    [{"role": "user", "content": [text(f"see {IMG}")]}],
    "chat",
)
add(
    "m-rejects-placeholder-in-reasoning",
    [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": f"r {IMG}"},
    ],
    "thinking",
)
add(
    "m-rejects-placeholder-in-assistant",
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": f"a {IMG}"}],
    "chat",
)
add(
    "m-mid-system-then-user",
    [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "q2"},
    ],
    "thinking",
)
add(
    "m-mid-system-chat",
    [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "be brief"},
    ],
    "chat",
)
add(
    "m-mid-system-with-tools",
    [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
        {"role": "system", "content": "tools now", "tools": [WEATHER_TOOL]},
        {"role": "user", "content": "weather?"},
    ],
    "thinking",
)
add(
    "m-mid-system-after-tool-result",
    [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "r", "content": "", "tool_calls": [call("c1", "lookup", "{}")]},
        {"role": "tool", "tool_call_id": "c1", "content": "v"},
        {"role": "system", "content": "summarize"},
    ],
    "thinking",
)
add(
    "m-system-only-first",
    [{"role": "system", "content": "only system"}],
    "thinking",
)
add(
    "m-assistant-first",
    [{"role": "assistant", "content": "I start."}, {"role": "user", "content": "ok"}],
    "thinking",
)
add(
    "m-latest-reminder",
    [
        {"role": "system", "content": "Assistant."},
        {"role": "latest_reminder", "content": "2026-11-09,Tel Aviv,App,English"},
        {"role": "user", "content": "date?"},
    ],
    "thinking",
)
add(
    "m-user-then-reminder",
    [{"role": "user", "content": "q"}, {"role": "latest_reminder", "content": "late reminder"}],
    "thinking",
)
add(
    "m-reminder-first-thinking",
    [{"role": "latest_reminder", "content": "r"}, {"role": "user", "content": "q"}],
    "thinking",
)
add(
    "m-response-format",
    [
        {
            "role": "system",
            "content": "Reply in JSON.",
            "tools": [WEATHER_TOOL],
            "response_format": {"type": "json_object", "schema": {"answer": "string"}},
        },
        {"role": "user", "content": "hi"},
    ],
    "chat",
)
add(
    "m-response-format-no-tools",
    [
        {"role": "system", "content": "Reply in JSON.", "response_format": {"type": "json_object"}},
        {"role": "user", "content": "hi"},
    ],
    "thinking",
)
add(
    "m-wo-eos-prefill",
    [{"role": "user", "content": "Write a poem."}, {"role": "assistant", "content": "Roses are", "wo_eos": True}],
    "chat",
)
add(
    "m-wo-eos-thinking",
    [
        {"role": "user", "content": "Write a poem."},
        {"role": "assistant", "reasoning_content": "rhyme", "content": "Roses are", "wo_eos": True},
    ],
    "thinking",
)
for task in ("action", "query", "authority", "domain", "read_url"):
    for mode in ("thinking", "chat"):
        add(
            f"m-task-{task}-{mode}",
            [{"role": "system", "content": "Router."}, {"role": "user", "content": "best pizza in Rome", "task": task}],
            mode,
        )
add(
    "m-task-title",
    [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi!", "task": "title"}],
    "chat",
)
add(
    "m-task-action-answered-thinking",
    [
        {"role": "user", "content": "world famous hot springs", "task": "action"},
        {"role": "assistant", "content": "Search"},
    ],
    "thinking",
)
add(
    "m-task-then-user",
    [
        {"role": "user", "content": "q", "task": "query"},
        {"role": "assistant", "content": "query text"},
        {"role": "user", "content": "next"},
    ],
    "thinking",
)
add(
    "m-task-user-blocks-merge",
    [{"role": "user", "content": "q", "task": "query"}, {"role": "user", "content": "more"}],
    "chat",
)
add(
    "m-rejects-unknown-task",
    [{"role": "user", "content": "q", "task": "summarize"}],
    "chat",
)
add(
    "m-tools-on-user-disable-drop",
    [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
        {"role": "user", "content": "q2", "tools": [WEATHER_TOOL]},
    ],
    "thinking",
)
add(
    "m-namespace-object-no-description",
    [
        {
            "role": "system",
            "content": "s",
            "tools": [dict(make_tool(), namespace={"name": "files", "description": ""})],
        },
        {"role": "user", "content": "q"},
    ],
    "thinking",
)
_nodesc = make_tool()
del _nodesc["function"]["description"]
_nodesc["namespace"] = {"name": "files", "description": "File tools."}
add(
    "m-namespace-description-without-own",
    [{"role": "system", "content": "s", "tools": [_nodesc]}, {"role": "user", "content": "q"}],
    "thinking",
)
_nullns = make_tool()
_nullns["function"]["namespace"] = None
add(
    "m-namespace-null-dropped",
    [{"role": "system", "content": "s", "tools": [_nullns]}, {"role": "user", "content": "q"}],
    "chat",
)
_override = make_tool()
_override["function"]["namespace"] = "inner"
_override["namespace"] = "outer"
add(
    "m-namespace-tool-level-wins",
    [{"role": "system", "content": "s", "tools": [_override]}, {"role": "user", "content": "q"}],
    "chat",
)
add(
    "m-namespace-call-function-level",
    [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "namespace": "search", "arguments": '{"query": "x"}'},
                }
            ],
        },
    ],
    "chat",
)
add("m-rejects-developer-role", [{"role": "developer", "content": "dev"}, {"role": "user", "content": "q"}], "chat")
add(
    "m-rejects-effort-numeric-chat",
    [{"role": "user", "content": "q"}],
    "chat",
    0,
)


# ---------------------------------------------------------------------------------------------
# rendering through the oracle
# ---------------------------------------------------------------------------------------------


def oracle_messages(messages, req_tools):
    """memra's serve law: request tools ride the leading system message, or a synthetic one."""
    msgs = copy.deepcopy(messages)
    if req_tools:
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["tools"] = copy.deepcopy(req_tools)
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": copy.deepcopy(req_tools)})
    return msgs


def image_id(record):
    for key in ("url", "source", "data"):
        if record.get(key):
            v = record[key]
            return v if isinstance(v, str) else json.dumps(v, sort_keys=True)
    raise AssertionError(record)


def stable_error(err):
    """The vendor's refusal text, with any listed set sorted: `list(VALID_TASKS)` follows the
    interpreter's string hashing, which `-I` leaves randomized, so the raw text is not stable."""
    import re

    def sort_list(m):
        items = sorted(i.strip() for i in m.group(1).split(","))
        return "[" + ", ".join(items) + "]"

    return re.sub(r"\[('[^\]]*')\]", sort_list, f"{type(err).__name__}: {err}")


def memra_effort(effort):
    return None if effort is None else str(effort)


WIRE_ROLES = {"system", "user", "assistant", "tool"}
VENDOR_ONLY_KEYS = {"task", "wo_eos", "response_format", "tools", "content_blocks", "mask"}


def string_or_absent(container, key):
    return container.get(key) is None or isinstance(container[key], str)


def on_wire(case):
    """Expressible on memra's OpenAI chat surface: plain roles, string or text-part content (no
    images: memra has no V4.1 vision program and refuses image parts), no vendor-only fields,
    arguments that are JSON objects, a reasoning_effort the surface names. Namespaces ride the
    wire as the vendor spells them: a request tool's `namespace` (tool or function level, string
    or object), and a history call's `ns::name`, call-level or function-level string namespace
    (memra refuses a non-string call namespace by name)."""
    if case["effort"] not in (None, "low", "high", "max"):
        return False
    if case["think"] == "chat" and case["effort"] is not None:
        return False  # the surface turns reasoning off with a level, so chat carries none
    for m in case["messages"]:
        if m["role"] not in WIRE_ROLES or VENDOR_ONLY_KEYS & set(m):
            return False
        content = m.get("content")
        if isinstance(content, list) and any(p.get("type") != "text" for p in content):
            return False
        if isinstance(content, str) and IMG in content:
            return False
        if IMG in (m.get("reasoning_content") or ""):
            return False
        for tc in m.get("tool_calls", []):
            if not (string_or_absent(tc, "namespace") and string_or_absent(tc["function"], "namespace")):
                return False
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                return False
            if not isinstance(args, dict):
                return False
    return True


def render(case):
    msgs = oracle_messages(case["messages"], case["req_tools"])
    mode = case["think"]
    try:
        prompt, media = enc.encode_messages(
            msgs, thinking_mode=mode, reasoning_effort=case["effort"], return_multi_modal_data=True
        )
    except (AssertionError, ValueError, NotImplementedError, KeyError, TypeError) as err:
        return None, None, stable_error(err)
    for alt_messages, alt_context in case["alt"]:
        got, got_media = enc.encode_messages(
            copy.deepcopy(alt_messages),
            thinking_mode=mode,
            context=copy.deepcopy(alt_context),
            reasoning_effort=case["effort"],
            return_multi_modal_data=True,
        )
        assert prompt.endswith(got), f"{case['name']}: a vendor-shape variant renders differently"
        n = len(got_media["images"])
        assert media["images"][len(media["images"]) - n :] == got_media["images"], case["name"]
    return prompt, [image_id(r) for r in media["images"]], None


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------------------------
# parse fixtures (parse_message_from_completion_text)
# ---------------------------------------------------------------------------------------------

D = enc.dsml_token
EOS = enc.eos_token


def dsml_block(*invokes):
    body = "\n".join(
        f'<{D} invoke name="{name}">\n'
        + "\n".join(f'<{D} parameter name="{k}" string="{s}">{v}</{D} parameter>' for k, s, v in params)
        + f"\n</{D} invoke>"
        for name, params in invokes
    )
    return f"\n\n<{D} calls>\n{body}\n</{D} calls>"


V41_TOOL_CALL_OUTPUT = (
    "  reason  </think>summary"
    + dsml_block(("lookup", [("query", "true", "value"), ("limit", "false", "2")]))
    + EOS
)
PARSES = [
    ("p-vendor-tool-call-thinking", V41_TOOL_CALL_OUTPUT, "thinking"),
    ("p-vendor-namespaced-thinking", V41_TOOL_CALL_OUTPUT.replace('name="lookup"', 'name="search::lookup"'), "thinking"),
    (
        "p-vendor-namespaced-chat",
        V41_TOOL_CALL_OUTPUT.replace('name="lookup"', 'name="search::lookup"').split("</think>", 1)[1],
        "chat",
    ),
    (
        "p-vendor-rejects-unspaced-v4",
        V41_TOOL_CALL_OUTPUT.replace(f"{D} calls", f"{D}tool_calls")
        .replace(f"{D} invoke", f"{D}invoke")
        .replace(f"{D} parameter", f"{D}parameter"),
        "thinking",
    ),
    ("p-thinking-plain", "Simple arithmetic.</think>2 + 2 = 4." + EOS, "thinking"),
    ("p-chat-plain", "Hello there." + EOS, "chat"),
    ("p-thinking-empty-reasoning", "</think>Direct." + EOS, "thinking"),
    ("p-chat-call-no-content", dsml_block(("get_weather", [("location", "true", "Oslo")])) + EOS, "chat"),
    (
        "p-two-invokes",
        "Let me check both.</think>"
        + dsml_block(
            ("get_weather", [("location", "true", "Beijing"), ("unit", "true", "celsius")]),
            ("calculate", [("expr", "true", "2+2")]),
        )
        + EOS,
        "thinking",
    ),
    (
        "p-typed-values",
        "</think>"
        + dsml_block(
            (
                "f",
                [
                    ("n", "false", "3"),
                    ("b", "false", "true"),
                    ("x", "false", "0.5"),
                    ("z", "false", "null"),
                    ("arr", "false", '["a", 1]'),
                    ("obj", "false", '{"k": {"n": [1, 2]}}'),
                    ("s", "true", 'quote " and\nnewline'),
                    ("u", "true", "北京"),
                ],
            )
        )
        + EOS,
        "thinking",
    ),
    ("p-no-params", "</think>" + dsml_block(("ping", [])) + EOS, "thinking"),
    ("p-content-with-newline-before-calls", "</think>line one\n" + dsml_block(("ping", [])) + EOS, "thinking"),
    ("p-rejects-missing-think-end", "never closed" + EOS, "thinking"),
    ("p-rejects-duplicate-param", "</think>" + dsml_block(("f", [("a", "true", "1"), ("a", "true", "2")])) + EOS, "thinking"),
    ("p-rejects-nested-namespace", "</think>" + dsml_block(("a::b::c", [("q", "true", "v")])) + EOS, "thinking"),
]


# ---- namespaces on memra's wire --------------------------------------------------------------
# Request tools carrying namespaces (tool level as the vendor README leads with, an object with a
# description, a string, and the function-level spelling), same-named tools in three namespaces,
# and a history whose calls are the vendor parser's own output (`function.name` bare, `namespace`
# beside it), with their results: the next turn must re-render the names the model generated.
_ns_tools = []
for _ns in (None, {"name": "search", "description": "Search tools."}, "files"):
    _t = make_tool()
    if _ns is not None:
        _t["namespace"] = copy.deepcopy(_ns)
    _ns_tools.append(_t)
_fn_ns_tool = make_tool()
_fn_ns_tool["function"]["namespace"] = "docs"
add(
    "w-namespaced-request-tools",
    [{"role": "user", "content": "question"}],
    "thinking",
    req_tools=_ns_tools + [_fn_ns_tool],
)
add(
    "w-namespaced-request-tools-chat",
    [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
    "chat",
    req_tools=_ns_tools,
)
for _mode in ("thinking", "chat"):
    _completion = ("Pick the right lookup.</think>" if _mode == "thinking" else "") + "Looking." + dsml_block(
        ("search::lookup", [("query", "true", "x")]),
        ("files::lookup", [("query", "true", "y"), ("limit", "false", "2")]),
        ("lookup", [("query", "true", "z")]),
    ) + EOS
    _parsed = enc.parse_message_from_completion_text(_completion, thinking_mode=_mode)
    _calls = copy.deepcopy(_parsed["tool_calls"])
    assert [c.get("namespace") for c in _calls] == ["search", "files", None], _calls
    for _i, _c in enumerate(_calls):
        _c["id"] = f"call_{_i}"
    _history = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": _parsed["content"],
            "reasoning_content": _parsed["reasoning_content"],
            "tool_calls": _calls,
        },
    ] + [{"role": "tool", "tool_call_id": f"call_{_i}", "content": f"result {_i}"} for _i in range(3)]
    add(f"w-vendor-shape-namespaced-history-{_mode}", _history, _mode, req_tools=_ns_tools)
    _qualified = copy.deepcopy(_history)
    for _c in _qualified[1]["tool_calls"]:
        if _c.get("namespace"):
            _c["function"]["name"] = f"{_c.pop('namespace')}::{_c['function']['name']}"
    add(f"w-qualified-namespaced-history-{_mode}", _qualified, _mode, req_tools=_ns_tools)


def parse_case(text_, mode):
    try:
        msg = enc.parse_message_from_completion_text(text_, thinking_mode=mode)
    except (AssertionError, ValueError) as err:
        return {"error": stable_error(err)}
    calls = []
    for tc in msg["tool_calls"]:
        calls.append(
            {
                "name": tc["function"]["name"],
                "namespace": tc.get("namespace"),
                "arguments": json.loads(tc["function"]["arguments"]),
                "arguments_text": tc["function"]["arguments"],
            }
        )
    # the round trip the vendor test asserts: re-encoding the parse reproduces the completion.
    reencoded = enc.encode_messages(
        [msg], thinking_mode=mode, context=[{"role": "user", "content": "question"}]
    )
    return {
        "expected": {
            "reasoning_content": msg["reasoning_content"],
            "content": msg["content"],
            "tool_calls": calls,
        },
        "reencodes_to_completion": reencoded == text_,
    }


# ---------------------------------------------------------------------------------------------


def main():
    from transformers import AutoTokenizer
    import tokenizers
    import transformers

    hf = AutoTokenizer.from_pretrained(REF)
    raw = tokenizers.Tokenizer.from_file(os.path.join(REF, "tokenizer.json"))

    names = [c["name"] for c in CASES]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, dupes
    for d in (FIX, PARSE):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    ids = {}
    n_ok = n_err = n_wire = 0
    for case in CASES:
        prompt, images, error = render(case)
        d = os.path.join(FIX, case["name"])
        os.makedirs(d)
        inp = {
            "source": case["source"],
            "messages": case["messages"],
            "req_tools": case["req_tools"],
            "think": case["think"],
            "reasoning_effort": memra_effort(case["effort"]),
            "add_generation_prompt": True,
        }
        if error is not None:
            inp["error"] = error
            n_err += 1
        else:
            inp["expected_images"] = images
            inp["wire"] = on_wire(case)
            n_wire += inp["wire"]
            n_ok += 1
            with open(os.path.join(d, "expected.txt"), "w") as f:
                f.write(prompt)
            a = hf.encode(prompt)
            b = raw.encode(prompt, add_special_tokens=True).ids
            assert a == b, f"{case['name']}: transformers and tokenizers disagree"
            assert a.count(hf.convert_tokens_to_ids(IMG)) == len(images), case["name"]
            ids[case["name"]] = a
        write_json(os.path.join(d, "input.json"), inp)
    write_json(os.path.join(FIX, "tokenization.json"), ids)

    n_parse = 0
    for name, text_, mode in PARSES:
        out = {"completion": text_, "think": mode}
        out.update(parse_case(text_, mode))
        write_json(os.path.join(PARSE, f"{name}.json"), out)
        n_parse += 1

    write_json(
        os.path.join(HERE, "generated.json"),
        {
            "oracle": "ref/encoding/encoding.py",
            "ref_files_verified": N_REF,
            "vendor_golden_pairs_reproduced": VENDOR_PAIRS,
            "render_cases": n_ok,
            "render_cases_on_wire": n_wire,
            "refusal_cases": n_err,
            "parse_cases": n_parse,
            "python": sys.version.split()[0],
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
    )
    print(
        f"wrote {n_ok} render fixtures ({n_wire} on the wire), {n_err} refusals, "
        f"{n_parse} parse fixtures; {len(VENDOR_PAIRS)} vendor goldens reproduced"
    )


if __name__ == "__main__":
    main()
