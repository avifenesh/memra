#!/usr/bin/env python3
"""Byte-parity oracle for the gemma4 tools chat-template arm.

Renders the OFFICIAL Google tooluse jinja (official-tooluse-template.jinja, extracted
from the official Q8_0-MTP GGUF — byte-identical to the served trunk's embedded
template) plus the local QAT trunk's variant (qat-trunk-template.jinja: official + a
closed-thought-channel generation tail) for every fixture case, writing
fixtures/NN-name/{input.json,expected.txt} pairs. The Rust side
(crates/memra-server/src/main.rs `gemma4_tools_fixtures` test) runs each input.json
through the REAL request pipeline (build_chat_request -> apply_chat_template_tools)
and asserts byte equality against expected.txt.

ORACLE LAW: jinja2 with the DEFAULT Environment() is the oracle (the engine never
executes jinja; these bytes define the arm). Notable default-env semantics this
template leans on, verified empirically (jinja2 3.1.6):
  - `{{ none }}` renders "None" (so a JSON null argument value renders bare `None`);
  - `dictsort` is case-INSENSITIVE and stable (sorted by key.lower(), insertion order
    preserved on ties);
  - `x.get('k') | default(d)` does NOT apply the default when the dict lacks 'k'
    (.get returns None, and `default` only replaces UNDEFINED) — so an unmatched,
    unnamed role:"tool" message would crash the oracle ('response:' + None). The Rust
    arm falls back to "unknown" there instead of crashing; fixtures always resolve.

CONVERSION MIRROR: the serve pipeline parses OpenAI `tool_calls[].function.arguments`
JSON strings into objects before rendering (render_req_tool_call); this generator does
the same before handing messages to jinja, so both sides render the mapping branch.

KNOWN PIPELINE DIVERGENCES (not fixture-coverable; kept out of the cases):
  - content-parts on user/system messages: the pipeline flattens parts to one string
    before dialect dispatch (raw concat + whole-string trim); the jinja trims each part
    (and for the system turn appends ' ' per part). Equal only when parts carry no
    whitespace edges — tool-result parts are exact (both sides raw-concat).
  - mid-conversation `developer` role: the pipeline normalizes developer->system for
    every dialect; the jinja would render a literal `<|turn>developer` body turn.
  - null role:"tool" content: the pipeline flattens null to "" (renders
    `{value:<|"|>{empty}<|"|>}`); the jinja renders `{value:None}`.

Run: python3 gen_fixtures.py   (from this directory; jinja2 required)
"""

import json
import os
import sys

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = {
    "official": os.path.join(HERE, "official-tooluse-template.jinja"),
    "qat": os.path.join(HERE, "qat-trunk-template.jinja"),
}
FIXDIR = os.path.join(HERE, "fixtures")


def mirror_messages(messages):
    """The render_req_tool_call mirror: arguments JSON strings -> objects; ''/null -> {}."""
    out = []
    for m in messages:
        m = json.loads(json.dumps(m))  # deep copy, insertion order kept
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = json.loads(args) if args.strip() else {}
            elif args is None:
                fn["arguments"] = {}
        out.append(m)
    return out


def render(template_key, request):
    src = open(TEMPLATES[template_key], encoding="utf-8").read()
    env = jinja2.Environment()  # DEFAULT env — the oracle setting; see module docstring
    tmpl = env.from_string(src)
    effort = request.get("reasoning_effort")
    enable_thinking = effort in ("low", "medium", "high")
    ctx = {
        "messages": mirror_messages(request["messages"]),
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        # encode(add_special) supplies BOS on the serve path; the renderer never re-emits it.
        "bos_token": "",
    }
    tools = request.get("tools") or []
    if tools:
        ctx["tools"] = tools
    return tmpl.render(**ctx)


# ---------------------------------------------------------------------------
# Shared tool schemas
# ---------------------------------------------------------------------------

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

SHELL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Runs a shell command and returns its output",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "array", "items": {"type": "string"},
                            "description": "The command to execute"},
                "timeout_ms": {"type": "number", "description": "Timeout in milliseconds",
                               "nullable": True},
                "with_escalated_permissions": {"type": "boolean"},
            },
            "required": ["command"],
        },
    },
}

# nested objects + array-of-objects items (with its own required) + enum + nullable
BOOKING = {
    "type": "function",
    "function": {
        "name": "book_trip",
        "description": "Book a multi-leg trip",
        "parameters": {
            "type": "object",
            "properties": {
                "traveler": {
                    "type": "object",
                    "description": "Who travels",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer", "nullable": True},
                    },
                    "required": ["name"],
                },
                "legs": {
                    "type": "array",
                    "description": "Trip legs in order",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frm": {"type": "string"},
                            "to": {"type": "string"},
                        },
                        "required": ["frm", "to"],
                    },
                },
                "class": {"type": "string", "enum": ["economy", "business"]},
                "flexible": {"type": "boolean"},
            },
            "required": ["traveler", "legs"],
        },
    },
}

# empty properties: OpenAI clients send {"type":"object","properties":{}} for no-arg tools
PING = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Liveness probe",
        "parameters": {"type": "object", "properties": {}},
    },
}


def msg(role, content, **kw):
    m = {"role": role, "content": content}
    m.update(kw)
    return m


def call(cid, name, args):
    return {"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = []


def case(name, request, template="official"):
    CASES.append((name, template, request))


case("01-system-tools-basic", {
    "model": "g4",
    "messages": [msg("system", "You are a terse weather bot."),
                 msg("user", "Weather in Paris?")],
    "tools": [WEATHER],
})

case("02-no-system-tools", {
    "model": "g4",
    "messages": [msg("user", "Weather in Oslo?")],
    "tools": [WEATHER],
})

case("03-nested-schema-declaration", {
    "model": "g4",
    "messages": [msg("user", "Book me a trip.")],
    "tools": [BOOKING, PING, SHELL],
})

case("04-single-call-cycle", {
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        msg("assistant", None,
            tool_calls=[call("call_1", "get_weather",
                             {"location": "Paris", "unit": "celsius"})]),
        msg("tool", '{"temp_c": 21, "sky": "clear"}', tool_call_id="call_1"),
    ],
    "tools": [WEATHER],
})

case("05-parallel-calls", {
    "model": "g4",
    "messages": [
        msg("user", "Compare Paris and Oslo weather."),
        msg("assistant", None,
            tool_calls=[
                call("call_a", "get_weather", {"location": "Paris"}),
                call("call_b", "get_weather", {"location": "Oslo"}),
            ]),
        msg("tool", "21C clear", tool_call_id="call_a"),
        msg("tool", "9C rain", tool_call_id="call_b"),
    ],
    "tools": [WEATHER],
})

case("06-dangling-call", {
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        msg("assistant", None,
            tool_calls=[call("call_1", "get_weather", {"location": "Paris"})]),
    ],
    "tools": [WEATHER],
})

case("07-multi-cycle-agentic", {
    "model": "g4",
    "messages": [
        msg("system", "Use tools when needed."),
        msg("user", "Run echo hello, then tell me what it printed."),
        msg("assistant", None,
            tool_calls=[call("c1", "shell",
                             {"command": ["echo", "hello"], "timeout_ms": 5000})]),
        msg("tool", "hello\n", tool_call_id="c1"),
        msg("assistant", "It printed: hello"),
        msg("user", "Now run echo bye."),
        msg("assistant", None,
            tool_calls=[call("c2", "shell", {"command": ["echo", "bye"]})]),
        msg("tool", "bye\n", tool_call_id="c2"),
    ],
    "tools": [SHELL],
})

case("08-native-mapping-response", {
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        msg("assistant", None,
            tool_calls=[call("c1", "get_weather", {"location": "Paris"})],
            tool_responses=[{"name": "get_weather",
                             "response": {"temp_c": 21.5, "sky": "clear",
                                          "Wind": {"kph": 12, "dir": "NW"}}}]),
    ],
    "tools": [WEATHER],
})

case("09-native-nonmapping-responses", {
    "model": "g4",
    "messages": [
        msg("user", "Ping twice."),
        msg("assistant", None,
            tool_calls=[call("c1", "ping", {}), call("c2", "ping", {})],
            tool_responses=[{"name": "ping", "response": 42},
                            {"name": "ping", "response": "pong"}]),
    ],
    "tools": [PING],
})

case("10-content-parts-tool-result", {
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        msg("assistant", None,
            tool_calls=[call("c1", "get_weather", {"location": "Paris"})]),
        msg("tool",
            [{"type": "text", "text": "21C, "}, {"type": "text", "text": "clear sky"}],
            tool_call_id="c1"),
    ],
    "tools": [WEATHER],
})

case("11-thinking-on-tools", {
    "model": "g4",
    "messages": [msg("system", "Be helpful."), msg("user", "Weather in Paris?")],
    "tools": [WEATHER],
    "reasoning_effort": "high",
})

case("12-reasoning-rerender-and-strip", {
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        # history content carrying an old thought channel: strip_thinking removes it
        msg("assistant", "<|channel>thought\nold plan\n<channel|>I checked already."),
        msg("user", "Check again."),
        # a tool_calls-carrying assistant AFTER the last user: reasoning re-renders
        msg("assistant", None, reasoning="The user wants a fresh reading.",
            tool_calls=[call("c1", "get_weather", {"location": "Paris"})]),
        msg("tool", "20C", tool_call_id="c1"),
    ],
    "tools": [WEATHER],
})

case("13-assistant-continuation", {
    "model": "g4",
    "messages": [
        msg("user", "Two answers please."),
        msg("assistant", "First answer."),
        msg("assistant", "Second answer."),
        msg("user", "Thanks."),
    ],
    "tools": [WEATHER],
})

case("14-nested-args-call", {
    "model": "g4",
    "messages": [
        msg("user", "Book it."),
        msg("assistant", None,
            tool_calls=[call("c1", "book_trip", {
                "traveler": {"name": "Avi", "age": 30},
                "legs": [{"frm": "TLV", "to": "CDG"}, {"frm": "CDG", "to": "OSL"}],
                "class": "business",
                "flexible": True,
            })]),
        msg("tool", "booked: ref 88f3", tool_call_id="c1"),
    ],
    "tools": [BOOKING],
})

case("15-history-only-no-tool-defs", {
    # tool_choice:none strips the defs but history still carries the cycle: the arm
    # renders calls/responses with no <|tool> block and no system turn at all.
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        msg("assistant", None,
            tool_calls=[call("c1", "get_weather", {"location": "Paris"})]),
        msg("tool", "21C", tool_call_id="c1"),
        msg("assistant", "It is 21C."),
        msg("user", "Summarize the conversation in one line."),
    ],
    "tools": [],
})

case("16-content-plus-calls", {
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        # content AND calls on one message: jinja order is calls, responses, content
        msg("assistant", "Let me check.",
            tool_calls=[call("c1", "get_weather", {"location": "Paris"})]),
        msg("tool", "21C", tool_call_id="c1"),
    ],
    "tools": [WEATHER],
})

case("17-arg-value-shapes", {
    "model": "g4",
    "messages": [
        msg("user", "Shapes."),
        msg("assistant", None,
            tool_calls=[call("c1", "shell", {
                "command": ["printf", "a{b,c}:d\ne"],
                "timeout_ms": 1500.5,
                "with_escalated_permissions": False,
                "workdir": None,
            })]),
        msg("tool", "ok", tool_call_id="c1"),
    ],
    "tools": [SHELL],
})

case("18-qat-closed-tail", {
    "model": "g4",
    "messages": [msg("system", "Be terse."), msg("user", "Weather in Paris?")],
    "tools": [WEATHER],
}, template="qat")

case("19-qat-dangling-no-tail", {
    # QAT template + dangling call: the closed-tail lives INSIDE the suppressed
    # generation-turn branch, so a dangling <|tool_response> gets no tail either.
    "model": "g4",
    "messages": [
        msg("user", "Weather in Paris?"),
        msg("assistant", None,
            tool_calls=[call("c1", "get_weather", {"location": "Paris"})]),
    ],
    "tools": [WEATHER],
}, template="qat")

case("20-dictsort-arguments", {
    "model": "g4",
    "messages": [
        msg("user", "Sort these."),
        msg("assistant", None,
            tool_calls=[call("c1", "shell", {
                "zeta": "z", "Alpha": "A", "beta": 2, "alPha2": True,
            })]),
        msg("tool", "done", tool_call_id="c1"),
    ],
    "tools": [SHELL],
})


def main():
    os.makedirs(FIXDIR, exist_ok=True)
    written = 0
    for name, template, request in CASES:
        d = os.path.join(FIXDIR, name)
        os.makedirs(d, exist_ok=True)
        expected = render(template, request)
        with open(os.path.join(d, "input.json"), "w", encoding="utf-8") as f:
            json.dump({"template": template, "request": request}, f,
                      indent=2, ensure_ascii=False)
            f.write("\n")
        with open(os.path.join(d, "expected.txt"), "w", encoding="utf-8") as f:
            f.write(expected)
        written += 1
    print(f"wrote {written} fixture pairs into {FIXDIR}")


if __name__ == "__main__":
    sys.exit(main())
