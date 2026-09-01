#!/usr/bin/env python3
"""
DeepSeek-V4 template fixture generator (lane 5, 2026-08-18; 0731 re-gate 2026-08-18).

The official encoding_dsv4.encode_messages IS the oracle (house GGUF template-mint law:
byte-parity fixtures are the only acceptance). This renders a matrix of inputs across the
three think modes and the message shapes {single-turn, multi-turn, with-system, with-tools,
with-tool-results, tasks}, and writes input.json + expected.txt pairs. The Rust
`dsv4_template_fixtures` / `dsv4_0731_fixtures` tests feed each input.json through the
memra dsv4 arm and assert the rendered bytes equal expected.txt.

TWO ORACLES, selected by --encoding (they differ only in the reasoning-effort ladder —
ENCODING-DIFF.md):
  - preview (default): ref/encoding/encoding_dsv4.py      -> ./fixtures/
  - 0731:              ref-0731/encoding/encoding_dsv4.py -> ./fixtures-0731/
The 0731 run renders the SAME base matrix (efforts valid under 0731 semantics: high is a
real prefix, max the new stronger text) PLUS a systematic mode x {low,high,max} x shape
effort matrix — the remapped ladder is the whole point of the re-gate.

Each input.json carries: {"turns": [...serve-shape turns...], "think": "chat|thinking",
"reasoning_effort": null|"low"|"high"|"max", "req_tools": [...]|null,
"add_generation_prompt": bool} where a turn is {"role","content"?,"reasoning"?,
"tool_calls"?,"tool_call_id"?,"name"?,"task"?,"tools"?}. The generator ALSO runs
encode_messages with the equivalent OpenAI-shape messages to produce expected.txt.

Run: python3 gen_fixtures.py [--encoding preview|0731]
Requires encoding_dsv4.py on the path (copied to ref/encoding/ and ref-0731/encoding/).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ENCODING = "preview"
if "--encoding" in sys.argv:
    ENCODING = sys.argv[sys.argv.index("--encoding") + 1]
if ENCODING not in ("preview", "0731"):
    raise SystemExit(f"--encoding must be preview|0731, got {ENCODING!r}")

REF_DIR = "ref" if ENCODING == "preview" else "ref-0731"
sys.path.insert(0, os.path.join(HERE, REF_DIR, "encoding"))
import encoding_dsv4 as enc  # noqa: E402

FIX = os.path.join(HERE, "fixtures" if ENCODING == "preview" else "fixtures-0731")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
}
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string", "description": "The expression"}},
            "required": ["expr"],
        },
    },
}


def mode_args(think, effort):
    """(thinking_mode, reasoning_effort) for encode_messages."""
    tm = "chat" if think == "chat" else "thinking"
    re = effort if effort else None
    return tm, re


# Each case: (name, turns, think, effort, req_tools_or_None, add_gen_prompt)
# `turns` are memra serve-shape turns; `messages` (OpenAI shape for the oracle) is derived.
CASES = []


def add(name, turns, think="thinking", effort=None, req_tools=None, agp=True):
    CASES.append((name, turns, think, effort, req_tools, agp))


# ---- single-turn, three modes ----
add("01-single-user-thinking", [{"role": "user", "content": "What is 2+2?"}], "thinking")
add("02-single-user-chat", [{"role": "user", "content": "What is 2+2?"}], "chat")
add(
    "03-single-user-thinking-max",
    [{"role": "user", "content": "Prove sqrt(2) is irrational."}],
    "thinking",
    "max",
)
add(
    "04-single-user-thinking-high",
    [{"role": "user", "content": "Explain gravity."}],
    "thinking",
    "high",
)

# ---- with system, three modes ----
add(
    "05-system-user-thinking",
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    "thinking",
)
add(
    "06-system-user-chat",
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    "chat",
)
add(
    "07-system-user-thinking-max",
    [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
    ],
    "thinking",
    "max",
)

# ---- multi-turn (drop_thinking behavior), thinking + chat ----
MULTI = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {
        "role": "assistant",
        "reasoning": "The user said hello, greet back.",
        "content": "Hi there!",
    },
    {"role": "user", "content": "Capital of France?"},
    {
        "role": "assistant",
        "reasoning": "It is Paris.",
        "content": "Paris.",
    },
]
add("08-multiturn-thinking-drop", MULTI, "thinking")
add("09-multiturn-chat", MULTI, "chat")
# multi-turn ending in a user (generation-prompt shape), thinking + chat + max
GEN = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "reasoning": "greet", "content": "Hello!"},
    {"role": "user", "content": "Tell me a joke."},
]
add("10-multiturn-genprompt-thinking", GEN, "thinking")
add("11-multiturn-genprompt-chat", GEN, "chat")
add("12-multiturn-genprompt-max", GEN, "thinking", "max")

# ---- with tools (declaration only; generation-prompt shape) ----
add(
    "13-tools-decl-thinking",
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Weather in Beijing?"},
    ],
    "thinking",
    None,
    [WEATHER_TOOL],
)
add(
    "14-tools-decl-chat",
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Weather in Beijing?"},
    ],
    "chat",
    None,
    [WEATHER_TOOL],
)
add(
    "15-tools-two-decl-thinking",
    [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather then compute 2+2."},
    ],
    "thinking",
    None,
    [WEATHER_TOOL, CALC_TOOL],
)
# tools with NO system turn (synthetic empty system turn on the serve surface)
add(
    "16-tools-no-system-thinking",
    [{"role": "user", "content": "Weather in Oslo?"}],
    "thinking",
    None,
    [WEATHER_TOOL],
)

# ---- with tool-calls + tool-results (full agentic cycle) ----
TOOLCYCLE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Weather in Beijing?"},
    {
        "role": "assistant",
        "reasoning": "Use get_weather.",
        "content": "",
        "tool_calls": [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Beijing", "unit": "celsius"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_001",
        "content": '{"temperature": 22, "condition": "sunny"}',
    },
    {
        "role": "assistant",
        "reasoning": "Format the answer.",
        "content": "It is sunny and 22C in Beijing.",
    },
]
add("17-toolcycle-thinking", TOOLCYCLE, "thinking", None, [WEATHER_TOOL])
# tool cycle ending in a tool result (generation prompt for the next assistant)
TOOLCYCLE_GEN = TOOLCYCLE[:4]
add("18-toolcycle-genprompt-thinking", TOOLCYCLE_GEN, "thinking", None, [WEATHER_TOOL])

# ---- non-str argument types + numbers/bools (string="false" wire) ----
TYPED_CALL = [
    {"role": "system", "content": "Use tools."},
    {"role": "user", "content": "Weather in Beijing for 3 days, metric on."},
    {
        "role": "assistant",
        "reasoning": "call get_weather with typed args",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Beijing", "days": 3, "metric": true, "tags": ["a", "b"]}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
]
add("19-typed-args-thinking", TYPED_CALL, "thinking", None, [WEATHER_TOOL])

# ---- two tool results (sort by call order) ----
TWO_RESULTS = [
    {"role": "system", "content": "Use tools."},
    {"role": "user", "content": "Weather and calc."},
    {
        "role": "assistant",
        "reasoning": "two calls",
        "content": "",
        "tool_calls": [
            {
                "id": "cA",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Beijing"}'},
            },
            {
                "id": "cB",
                "type": "function",
                "function": {"name": "calculate", "arguments": '{"expr": "2+2"}'},
            },
        ],
    },
    # returned OUT of order: cB first, cA second — the sort must reorder to cA, cB.
    {"role": "tool", "tool_call_id": "cB", "content": "4"},
    {"role": "tool", "tool_call_id": "cA", "content": "sunny"},
    {"role": "assistant", "reasoning": "done", "content": "Done."},
]
add("20-two-results-sorted-thinking", TWO_RESULTS, "thinking", None, [WEATHER_TOOL, CALC_TOOL])

# ---- chat-mode multi-turn with tools (drop disabled, chat mode -> no think blocks) ----
add("21-tools-cycle-chat", TOOLCYCLE, "chat", None, [WEATHER_TOOL])

# ---- task heads (chat + thinking) ----
add(
    "22-task-action-chat",
    [
        {"role": "system", "content": "Router."},
        {"role": "user", "content": "world famous hot springs", "task": "action"},
    ],
    "chat",
)
add(
    "23-task-query-thinking",
    [
        {"role": "system", "content": "Query gen."},
        {"role": "user", "content": "best pizza in Rome", "task": "query"},
    ],
    "thinking",
)
add(
    "24-task-title-chat",
    [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi!", "task": "title"},
    ],
    "chat",
)

# ---- latest_reminder passthrough ----
add(
    "25-reminder-thinking",
    [
        {"role": "system", "content": "Assistant."},
        {"role": "latest_reminder", "content": "2026-08-18,Guangzhou,App,中文"},
        {"role": "user", "content": "date?"},
    ],
    "thinking",
)

# ---- 0731 effort-ladder matrix (re-gate): modes x {low, high, max} x shapes ----
# "low" is INVALID under the preview oracle (its assert rejects the string), and the
# high/max rungs are what the 0731 remap changed — so the systematic matrix only renders
# under the 0731 encoding. The base cases above already cover effort=None everywhere.
if ENCODING == "0731":
    EFFORT_SHAPES = {
        "single": ([{"role": "user", "content": "What is 2+2?"}], None),
        "multiturn": (GEN, None),
        "system": (
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            None,
        ),
        "tools": (
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Weather in Beijing?"},
            ],
            [WEATHER_TOOL],
        ),
        "toolresults": (TOOLCYCLE_GEN, [WEATHER_TOOL]),
    }
    for mode in ("thinking", "chat"):
        for effort in ("low", "high", "max"):
            for shape, (turns, req_tools) in EFFORT_SHAPES.items():
                add(f"matrix-{mode}-{effort}-{shape}", turns, mode, effort, req_tools)


def to_oracle_messages(turns, req_tools):
    """Serve-shape turns -> encode_messages OpenAI-shape messages.
    Request tools attach to the leading system turn (or a synthetic empty one), matching the
    memra serve surface AND the artifact fixtures."""
    msgs = []
    for t in turns:
        m = {"role": t["role"]}
        if "content" in t:
            m["content"] = t["content"]
        if "reasoning" in t:
            m["reasoning_content"] = t["reasoning"]
        if "tool_calls" in t:
            m["tool_calls"] = t["tool_calls"]
        if "tool_call_id" in t:
            m["tool_call_id"] = t["tool_call_id"]
        if "name" in t:
            m["name"] = t["name"]
        if "task" in t:
            m["task"] = t["task"]
        if t.get("tools"):
            m["tools"] = t["tools"]
        msgs.append(m)
    if req_tools:
        # attach to leading system turn; synthesize one if absent (serve surface law).
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["tools"] = req_tools
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": req_tools})
    return msgs


def main():
    os.makedirs(FIX, exist_ok=True)
    for name, turns, think, effort, req_tools, agp in CASES:
        tm, re = mode_args(think, effort)
        msgs = to_oracle_messages(turns, req_tools)
        expected = enc.encode_messages(msgs, thinking_mode=tm, reasoning_effort=re)
        d = os.path.join(FIX, name)
        os.makedirs(d, exist_ok=True)
        inp = {
            "turns": turns,
            "think": think,
            "reasoning_effort": effort,
            "req_tools": req_tools,
            "add_generation_prompt": agp,
        }
        with open(os.path.join(d, "input.json"), "w") as f:
            json.dump(inp, f, ensure_ascii=False, indent=2)
        with open(os.path.join(d, "expected.txt"), "w") as f:
            f.write(expected)
    print(f"wrote {len(CASES)} fixtures to {FIX}")


if __name__ == "__main__":
    main()
