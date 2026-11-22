"""Oracle fixtures for the Inkling 2 chat dialect, tokenizer and response parser.

Three oracles over the pinned artifact (thinkingmachines/Inkling-2-Mini@ed40e7da625f, files
under artifact/, sha256 checked below):

- render: transformers 5.23.0 `render_jinja_template` over `chat_template.jinja`, the call
  `apply_chat_template` makes (jinja2 3.1.6);
- ids: the `tokenizers` library over `tokenizer.json`, for every render
  (add_special_tokens=False, as apply_chat_template tokenizes), for a raw corpus in both
  modes, and as per-chunk digests over every Unicode scalar value;
- parse: transformers 5.23.0 `parse_response` over the `response_template` in
  `tokenizer_config.json`.

    uv run --no-project --with transformers==5.23.0 --with tokenizers==0.23.3 \
        --with jinja2==3.1.6 --with regex python3 -I gen_fixtures.py

Writes fixtures/renders.json, fixtures/parses.json, fixtures/scalar-digests.tsv, corpus.tsv and
ref-ids.tsv next to this file.
"""

import hashlib
import json
import pathlib
import struct
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifact"
OUT = HERE / "fixtures"
PINS = {
    "chat_template.jinja": "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc",
    "tokenizer.json": "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6",
    "tokenizer_config.json": "bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636",
    "generation_config.json": "a08737c07a65a8fdf69f7cf086576210e5aad4b241437090b26d7b0c4f7905f7",
}
for name, digest in PINS.items():
    assert hashlib.sha256((ART / name).read_bytes()).hexdigest() == digest, f"{name} pin"
TEMPLATE = (ART / "chat_template.jinja").read_text()
CONFIG = json.loads((ART / "tokenizer_config.json").read_text())
RESPONSE_TEMPLATE = CONFIG["response_template"]

import transformers  # noqa: E402
import tokenizers  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from transformers.utils.chat_parsing import parse_response  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

assert transformers.__version__ == "5.23.0", transformers.__version__
assert tokenizers.__version__ == "0.23.3", tokenizers.__version__
TOK = Tokenizer.from_file(str(ART / "tokenizer.json"))

USER = "<|message_user|>"
MODEL = "<|message_model|>"
TEXT = "<|content_text|>"
THINK = "<|content_thinking|>"
CALL = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer"},
                "metric": {"type": "boolean"},
            },
            "required": ["city"],
        },
    },
}
# Keys out of order at every level, non-ASCII, and the characters jinja's own tojson would
# escape (< > & ') but transformers' does not.
SEARCH = {
    "function": {
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "top_k": {"type": "number", "default": 5, "minimum": 0.5},
                "query": {"type": "string", "description": "Text <b>&</b> 'quotes' \"double\" \\ back"},
                "Filters": {"type": "array", "items": {"type": "string", "enum": ["b", "a", "C"]}},
                "opts": {"type": ["object", "null"], "additionalProperties": True},
            },
        },
        "name": "search",
        "description": "Search the web: returns JSON — “quoted” 中文 \n second line\t tab.",
        "strict": False,
    },
    "type": "function",
}
# Every declared-type shape the response parser's cast reads.
TYPED = {
    "type": "function",
    "function": {
        "name": "t",
        "parameters": {
            "type": "object",
            "properties": {
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "o": {"type": "object"},
                "a": {"type": "array"},
                "u": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "s": {"type": ["string", "integer"]},
                "n0": {"type": "integer", "nullable": 0},
                "ne": {"type": "integer", "nullable": ""},
                "nl": {"type": "integer", "nullable": []},
                "nt": {"type": "integer", "nullable": True},
                "n1": {"type": "integer", "nullable": 1},
            },
        },
    },
}
# A bare function object (no wrapper), no description, no parameters.
BARE = {"name": "ping"}
# A wrapper with its own type and empty description and parameters.
CUSTOM = {"type": "custom", "function": {"name": "now", "description": "", "parameters": {}}}


NUMBERS = {
    "a": 1e-07,
    "b": 1e-05,
    "c": 123456789012345678901234567890,
    "d": 1e16,
    "e": 1.5e300,
    "f": 5.0,
    "g": -0.0,
    "h": 0.1,
    "i": 1e15,
    "j": 123456789.123,
    "k": -2.5e-10,
    "l": 9007199254740993,
    "m": 1e22,
    "n": 0.0001,
    "o": 5e-324,
    "p": 1.7976931348623157e308,
    "q": 123456789012345680.0,
    "r": -18446744073709551616,
    "s": 0,
    "t": 2.5,
}


def call(name, args, cid=None):
    tc = {"type": "function", "function": {"name": name, "arguments": args}}
    if cid is not None:
        tc["id"] = cid
    return tc


U = {"role": "user", "content": "What is the weather in Paris?"}
RENDER_CASES = [
    ("plain", [{"role": "user", "content": "Hello!"}], None, {}),
    ("plain-no-generation-prompt", [{"role": "user", "content": "Hello!"}], None, {"add_generation_prompt": False}),
    (
        "multi-turn",
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello. How can I help?"},
            {"role": "user", "content": "Tell me a joke.\nA short one."},
            {"role": "assistant", "content": "Why did the GPU cross the bus? To get to the other SIMD."},
            {"role": "user", "content": "Another — with emoji 🙂 and 中文"},
        ],
        None,
        {},
    ),
    ("system", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}], None, {}),
    (
        "two-leading-systems",
        [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ],
        None,
        {},
    ),
    (
        "system-mid-conversation",
        [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Switch to French."},
            {"role": "user", "content": "Again"},
        ],
        None,
        {},
    ),
    ("system-only", [{"role": "system", "content": "Only a system turn."}], None, {}),
    ("no-messages", [], None, {}),
    ("empty-user-content", [{"role": "user", "content": ""}], None, {}),
    ("assistant-last", [U, {"role": "assistant", "content": "Sunny."}], None, {}),
    ("assistant-empty-content", [U, {"role": "assistant", "content": ""}], None, {}),
    ("assistant-named", [U, {"role": "assistant", "name": "planner", "content": "Plan."}], None, {}),
    ("effort-none", [U], None, {"reasoning_effort": "none"}),
    ("effort-minimal", [U], None, {"reasoning_effort": "minimal"}),
    ("effort-low", [U], None, {"reasoning_effort": "low"}),
    ("effort-medium", [U], None, {"reasoning_effort": "medium"}),
    ("effort-high", [U], None, {"reasoning_effort": "high"}),
    ("effort-xhigh", [U], None, {"reasoning_effort": "xhigh"}),
    ("effort-max", [U], None, {"reasoning_effort": "max"}),
    ("effort-padded-name", [U], None, {"reasoning_effort": "  low\n"}),
    ("effort-number-0", [U], None, {"reasoning_effort": 0}),
    ("effort-number-0.0", [U], None, {"reasoning_effort": 0.0}),
    ("effort-number-0.5", [U], None, {"reasoning_effort": 0.5}),
    ("effort-number-0.99", [U], None, {"reasoning_effort": 0.99}),
    ("effort-number-0.25", [U], None, {"reasoning_effort": 0.25}),
    ("effort-number-0.1+0.2", [U], None, {"reasoning_effort": 0.1 + 0.2}),
    ("effort-number-1e-05", [U], None, {"reasoning_effort": 0.00001}),
    ("effort-number-0.0001", [U], None, {"reasoning_effort": 0.0001}),
    ("effort-number-1.5e-07", [U], None, {"reasoning_effort": 1.5e-7}),
    ("effort-number-1e-10", [U], None, {"reasoning_effort": 1e-10}),
    ("effort-with-system", [{"role": "system", "content": "Be brief."}, U], None, {"reasoning_effort": "low"}),
    ("invalid-effort-name", [U], None, {"reasoning_effort": "ultra"}),
    ("invalid-effort-case", [U], None, {"reasoning_effort": "High"}),
    ("invalid-effort-numeric-string", [U], None, {"reasoning_effort": "0.5"}),
    ("invalid-effort-empty", [U], None, {"reasoning_effort": ""}),
    ("invalid-effort-number-1", [U], None, {"reasoning_effort": 1}),
    ("invalid-effort-number-1.0", [U], None, {"reasoning_effort": 1.0}),
    ("invalid-effort-number-negative", [U], None, {"reasoning_effort": -0.01}),
    (
        "reasoning-and-content",
        [U, {"role": "assistant", "reasoning_content": "Paris is in France.\nCheck.", "content": "Sunny."}, U],
        None,
        {},
    ),
    ("reasoning-empty-content", [U, {"role": "assistant", "reasoning_content": "Only thought.", "content": ""}], None, {}),
    ("reasoning-no-content-key", [U, {"role": "assistant", "reasoning_content": "Only thought."}], None, {}),
    ("reasoning-empty-string", [U, {"role": "assistant", "reasoning_content": "", "content": "Sunny."}], None, {}),
    ("tools-declared", [U], [WEATHER], {}),
    ("tools-declared-sorted-and-raw", [U], [SEARCH, WEATHER, BARE, CUSTOM], {}),
    ("tools-with-system", [{"role": "system", "content": "Use tools."}, U], [WEATHER], {"reasoning_effort": "medium"}),
    ("tools-no-messages", [], [WEATHER], {}),
    (
        "tool-call-and-result",
        [
            U,
            {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Paris"}, "call_1")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21}"},
            {"role": "assistant", "content": "It is 21 °C in Paris."},
            {"role": "user", "content": "Thanks"},
        ],
        [WEATHER],
        {},
    ),
    (
        "tool-call-with-reasoning-and-content",
        [
            U,
            {
                "role": "assistant",
                "reasoning_content": "I should call the tool.",
                "content": "Let me check.",
                "tool_calls": [call("get_weather", {"metric": True, "city": "Paris", "days": 3}, "c1")],
            },
            {"role": "tool", "name": "get_weather", "content": "sunny"},
        ],
        [WEATHER],
        {"reasoning_effort": "high"},
    ),
    (
        "tool-calls-parallel-sorted-args",
        [
            U,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call("search", {"top_k": 3, "query": "météo “Paris” <b>&</b> 'x'", "Filters": ["b", "a"], "opts": {"z": 1, "a": {"y": None, "b": [1.5, False]}}}, "a"),
                    call("get_weather", {"city": "Paris", "_x": 1, "Z": 2, "a": 3, "B": 4}, "b"),
                    call("ping", {}, "c"),
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "second"},
            {"role": "tool", "tool_call_id": "a", "content": "first"},
            {"role": "tool", "tool_call_id": "c", "content": ""},
        ],
        [SEARCH, WEATHER, BARE],
        {},
    ),
    (
        "tool-call-no-arguments-key",
        [U, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "ping"}}]}],
        [BARE],
        {},
    ),
    (
        "tool-result-unresolved",
        [U, {"role": "tool", "tool_call_id": "nope", "content": "orphan"}, {"role": "tool", "content": "bare"}],
        None,
        {},
    ),
    (
        "tool-result-name-wins-over-id",
        [
            U,
            {"role": "assistant", "content": "", "tool_calls": [call("ping", {}, "p")]},
            {"role": "tool", "name": "other", "tool_call_id": "p", "content": "x"},
        ],
        None,
        {},
    ),
    (
        "tool-result-id-matches-twice",
        [
            U,
            {"role": "assistant", "content": "", "tool_calls": [call("ping", {}, "p")]},
            {"role": "tool", "tool_call_id": "p", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [call("now", {}, "p")]},
        ],
        None,
        {},
    ),
    (
        "assistant-named-tool-call",
        [U, {"role": "assistant", "name": "ping", "content": "calling", "tool_calls": [call("ping", {}, "p")]}],
        None,
        {},
    ),
    (
        "invalid-assistant-name-mismatch",
        [U, {"role": "assistant", "name": "planner", "content": "", "tool_calls": [call("ping", {}, "p")]}],
        None,
        {},
    ),
    (
        "invalid-string-arguments",
        [U, {"role": "assistant", "content": "", "tool_calls": [call("ping", "{\"a\": 1}", "p")]}],
        None,
        {},
    ),
    ("invalid-role", [{"role": "developer", "content": "x"}], None, {}),
    (
        "text-parts",
        [
            {"role": "system", "content": [{"type": "text", "text": "Sys part."}]},
            {"role": "user", "content": [{"type": "text", "text": "One."}, {"type": "input_text", "text": "Two."}, {"text": "Three."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Reply."}]},
        ],
        None,
        {},
    ),
    ("empty-parts", [{"role": "user", "content": []}, {"role": "user", "content": "after"}], None, {}),
    (
        "image-part",
        [{"role": "user", "content": [{"type": "text", "text": "Describe."}, {"type": "image"}]}],
        None,
        {},
    ),
    ("invalid-image-in-assistant", [U, {"role": "assistant", "content": [{"type": "image_url"}]}], None, {}),
    ("invalid-audio-part", [{"role": "user", "content": [{"type": "input_audio"}]}], None, {}),
    ("invalid-part-type", [{"role": "user", "content": [{"type": "video"}]}], None, {}),
    # null or missing content: the template writes no message for it
    ("null-user-content", [{"role": "user", "content": None}, U], None, {}),
    ("missing-user-content", [{"role": "user"}, U], None, {}),
    ("null-system-content", [{"role": "system", "content": None}, U], None, {}),
    ("null-tool-content", [U, {"role": "tool", "name": "ping", "content": None}], None, {}),
    ("null-assistant-content", [U, {"role": "assistant", "content": None}, U], None, {}),
    (
        "null-assistant-content-with-call",
        [U, {"role": "assistant", "content": None, "tool_calls": [call("ping", {}, "p")]}],
        None,
        {},
    ),
    # numbers as Python prints them, whatever the request's JSON spelled
    (
        "tool-json-numbers",
        [U, {"role": "assistant", "content": "", "tool_calls": [call("nums", NUMBERS, "n")]}],
        [{"type": "function", "function": {"name": "nums", "parameters": {"type": "object", "properties": {"x": {"type": "number", "enum": list(NUMBERS.values())}}}}}],
        {},
    ),
]


def render(messages, tools, kwargs):
    kw = dict(kwargs)
    agp = kw.pop("add_generation_prompt", True)
    try:
        text, _ = render_jinja_template(
            conversations=[messages],
            tools=tools,
            chat_template=TEMPLATE,
            add_generation_prompt=agp,
            **kw,
        )
        return text[0], None
    except Exception as error:  # the template's raise_exception
        return None, f"{type(error).__name__}: {error}"


def has_image(messages):
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image", "input_image", "image_url"):
                    return True
    return False


renders = []
for name, messages, tools, kwargs in RENDER_CASES:
    text, error = render(messages, tools, kwargs)
    case = {
        "name": name,
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": kwargs.get("add_generation_prompt", True),
    }
    if "reasoning_effort" in kwargs:
        case["reasoning_effort"] = kwargs["reasoning_effort"]
    if error is None:
        case["expected"] = text
        case["ids"] = TOK.encode(text, add_special_tokens=False).ids
        # This engine serves text only: a request with an image part is refused, whatever the
        # template renders for it.
        if has_image(messages):
            case["memra"] = "refuses-image"
    else:
        assert name.startswith("invalid-"), (name, error)
        case["error"] = error
    assert (error is None) != name.startswith("invalid-"), (name, text)
    renders.append(case)

# The template never reads add_generation_prompt: the model opens its own message.
by_name = {case["name"]: case for case in renders}
assert by_name["plain"]["expected"] == by_name["plain-no-generation-prompt"]["expected"]

# ---------------------------------------------------------------- parse oracle
PROMPT = by_name["tools-declared"]["expected"]


def msg(kind, body, name="", closed=True):
    return f"{MODEL}{name}{kind}{body}" + (END if closed else "")


def jcall(name, args):
    return json.dumps({"name": name, "args": args}, ensure_ascii=False, separators=(",", ":"))


PARSE_CASES = [
    ("text", msg(TEXT, "Sunny in Paris.") + EOS, None),
    ("text-unterminated", msg(TEXT, "Sunny in", closed=False), None),
    ("text-no-eos", msg(TEXT, "Sunny."), None),
    ("thinking-then-text", msg(THINK, "Paris is in France.\n") + msg(TEXT, "Sunny.") + EOS, None),
    ("thinking-only-unterminated", msg(THINK, "Still thinking", closed=False), None),
    ("two-thinking-two-text", msg(THINK, "a") + msg(TEXT, "1") + msg(THINK, "b") + msg(TEXT, "2") + EOS, None),
    ("whitespace-is-kept", msg(THINK, "  pad \n") + msg(TEXT, "\n  body  \n") + EOS, None),
    ("empty-text", msg(TEXT, "") + EOS, None),
    ("no-model-marker", f"{THINK}t{END}{TEXT}c{END}{EOS}", None),
    ("stray-text-outside-messages", "lead " + msg(TEXT, "kept") + " between " + msg(TEXT, "!") + " tail" + EOS, None),
    ("header-text-before-kind", msg(TEXT, "named", name="planner") + EOS, None),
    ("tool-error-and-tool-text-are-content", msg("<|content_tool_error|>", "E") + msg("<|content_invoke_tool_text|>", "T") + EOS, None),
    ("marker-inside-a-message-is-text", msg(TEXT, f"a {THINK} b {MODEL} c {EOS} d") + EOS, None),
    ("call", msg(CALL, jcall("get_weather", {"city": "Paris"}), name="get_weather") + EOS, [WEATHER]),
    ("call-no-tools-given", msg(CALL, jcall("get_weather", {"city": "Paris", "days": "3"}), name="get_weather") + EOS, None),
    (
        "call-typed-arguments",
        msg(CALL, jcall("get_weather", {"city": "Paris", "days": 3, "metric": True}), name="get_weather") + EOS,
        [WEATHER],
    ),
    (
        "call-string-arguments-coerced-by-schema",
        msg(CALL, jcall("get_weather", {"city": "75001", "days": "3", "metric": "true"}), name="get_weather") + EOS,
        [WEATHER],
    ),
    (
        "call-coercions",
        msg(
            CALL,
            jcall(
                "search",
                {"query": 5, "top_k": "2.50", "Filters": "[\"a\"]", "opts": "null", "unknown": "7"},
            ),
            name="search",
        )
        + EOS,
        [SEARCH],
    ),
    (
        "call-coercion-failures-stay-strings",
        msg(CALL, jcall("get_weather", {"days": "three", "metric": "yes"}), name="get_weather")
        + msg(CALL, jcall("get_weather", {"days": "3.0", "metric": "0"}), name="get_weather")
        + msg(CALL, jcall("get_weather", {"days": "3.5", "metric": " TRUE "}), name="get_weather")
        + msg(CALL, jcall("search", {"top_k": "7", "Filters": "{}", "opts": "[1]"}), name="search")
        + msg(CALL, jcall("search", {"top_k": "nan", "opts": "{\"k\": 1}"}), name="search")
        + msg(CALL, jcall("search", {"top_k": "1e3", "opts": " None "}), name="search")
        + EOS,
        [WEATHER, SEARCH],
    ),
    (
        "thinking-text-and-two-calls",
        msg(THINK, "Need both.")
        + msg(TEXT, "Checking.")
        + msg(CALL, jcall("get_weather", {"city": "Paris"}), name="get_weather")
        + msg(CALL, jcall("search", {"query": "météo 中文 🙂", "opts": {"a": [1, {"b": None}]}}), name="search")
        + EOS,
        [WEATHER, SEARCH],
    ),
    ("call-header-name-is-ignored", msg(CALL, jcall("get_weather", {"city": "Paris"}), name="other") + EOS, [WEATHER]),
    ("call-json-with-whitespace", msg(CALL, ' \n{ "name" : "ping" ,\n "args" : { } }\n ', name="ping") + EOS, [BARE]),
    ("call-extra-keys", msg(CALL, '{"args":{"a":1},"id":7,"name":"ping"}', name="ping") + EOS, None),
    ("call-unterminated-complete-json", msg(CALL, jcall("ping", {}), name="ping", closed=False), None),
    ("call-end-message-inside-a-string", msg(CALL, jcall("ping", {"s": f"a{END}b"}), name="ping") + EOS, None),
    ("call-args-not-an-object", msg(CALL, '{"name":"ping","args":[1,2]}', name="ping") + EOS, None),
    ("call-name-not-a-string", msg(CALL, '{"name":7,"args":{}}', name="ping") + EOS, None),
    ("call-malformed-json", msg(TEXT, "before") + msg(CALL, '{"name":"ping","args":{', name="ping") + msg(TEXT, "after") + EOS, None),
    ("call-missing-args", msg(CALL, '{"name":"ping"}', name="ping") + EOS, None),
    ("call-missing-name", msg(CALL, '{"args":{}}', name="ping") + EOS, None),
    ("call-json-not-an-object", msg(CALL, '["ping"]', name="ping") + EOS, None),
    ("call-unterminated-partial-json", msg(CALL, '{"name":"ping","ar', name="ping", closed=False), None),
    # numbers: exact integers of any size, and the model's own spelling of a float
    (
        "call-numbers-keep-their-value",
        msg(CALL, '{"name":"t","args":{"big":123456789012345678901234567890,"neg":-18446744073709551616,"small":1e-07,"exp":1E5,"frac":2.50}}', name="t") + EOS,
        [TYPED],
    ),
    ("call-number-overflow", msg(CALL, '{"name":"t","args":{"n":1E400,"m":-1e999}}', name="t") + EOS, [TYPED]),
    ("call-nan-literal", msg(CALL, '{"name":"t","args":{"n":NaN,"m":Infinity}}', name="t") + EOS, [TYPED]),
    (
        "call-duplicate-keys",
        msg(CALL, '{"name":"a","name":"ping","args":{"k":1,"j":2,"k":3}}', name="ping") + EOS,
        None,
    ),
    ("call-deep-nesting", msg(CALL, '{"name":"ping","args":{"d":' + "[" * 200 + "]" * 200 + "}}", name="ping") + EOS, None),
    # string arguments cast to the declared types
    (
        "call-casts",
        "".join(
            msg(CALL, jcall("t", args), name="t")
            for args in (
                {"i": " 42 ", "n": "-0", "u": "null", "s": "7", "b": "TRUE", "o": "{\"k\":[1]}", "a": "[1,\"x\"]"},
                {"i": "+7", "n": "1e3", "u": "12", "s": "x", "b": " 0 "},
                {"i": "007", "n": "2.50", "u": " None ", "b": "1"},
                {"i": "-0.0", "n": "1E2", "o": "[1]", "a": "{}"},
                {"i": "9007199254740993", "n": "0.1", "u": "3.0"},
                {"i": "1e20", "n": "1e400", "u": "nan"},
                {"i": "123456789012345678901234567890", "n": "-inf", "u": "3.5"},
            )
        )
        + EOS,
        [TYPED],
    ),
    (
        "call-casts-python-spellings",
        msg(CALL, jcall("t", {"i": "1_000", "n": "1_0.5", "u": "１２３", "s": "٣"}), name="t")
        + msg(CALL, jcall("t", {"n": "12345678901234567890"}), name="t")
        + EOS,
        [TYPED],
    ),
    (
        "call-nullable-truthiness",
        msg(CALL, jcall("t", {"n0": "null", "ne": "null", "nl": "null", "nt": "null", "n1": "null"}), name="t") + EOS,
        [TYPED],
    ),
    (
        "call-list-items",
        msg(CALL, jcall("t", {"i": ["1", "2", 3, "x", None], "a": ["1"], "b": ["true", "no"]}), name="t") + EOS,
        [TYPED],
    ),
    # the header a call takes with it
    ("call-long-header", msg(CALL, jcall("ping", {}), name="h" * 5000) + EOS, None),
    ("call-long-header-malformed", msg(CALL, '{"name":"ping"', name="h" * 5000) + EOS, None),
    ("call-header-at-the-cap-malformed", msg(CALL, '{"name":"ping"', name="h" * 4096) + EOS, None),
    ("call-ends-after-marker", f"{MODEL}ping{CALL}", None),
    ("call-empty-body", msg(CALL, "", name="ping") + EOS, None),
]

parses = []
for name, output, tools in PARSE_CASES:
    case = {"name": name, "prefix": PROMPT, "output": output, "tools": tools}
    try:
        message = parse_response(output, RESPONSE_TEMPLATE, prefix=PROMPT, tools=tools)
        try:
            text = json.dumps(message, allow_nan=False)
            if text.count("[[[[") >= 30:
                # nested deeper than this engine's reader goes: recorded as a count
                case["vendor"] = f"parsed; arguments nested {text.count('[')} deep"
            else:
                case["message"] = message
        except ValueError:
            # inf or nan: not JSON, so the vendor's message is recorded as Python prints it
            case["vendor"] = repr(message)
    except Exception as error:
        case["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
    parses.append(case)

# What this engine does where it does not follow the vendor parser, each with its reason.
# The vendor raises on a call body it cannot shape; the serve path never fails a finished
# generation on its content, so the span surfaces verbatim as content (the house policy of
# crates/memra-server/src/toolcall.rs). The OpenAI wire carries a string name and an object of
# arguments, and the arguments are JSON.
HEADER_CAP = 4096


def verbatim(output):
    start = output.index(CALL)
    head = output.rfind(MODEL, 0, start)
    header = output[head + len(MODEL) : start] if head >= 0 else None
    # the header the open pattern takes with the marker, when it is one and within the cap
    if header is None or "<" in header or len(header) > HEADER_CAP:
        head = start
    close = output.find(END, start)
    span = output[head:] if close < 0 else output[head : close + len(END)]
    content = ("before" if "before" in output else "") + span + ("after" if "after" in output else "")
    return {"role": "assistant", "content": content}


def calls(*pairs):
    return {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": name, **rest}} for name, rest in pairs],
    }


RAISES = "malformed span surfaces verbatim; the vendor raises"
MEMRA = {
    "call-args-not-an-object": ("the OpenAI wire's arguments are an object", verbatim),
    "call-name-not-a-string": ("the OpenAI wire's name is a string", verbatim),
    "call-malformed-json": (RAISES, verbatim),
    "call-missing-args": (RAISES, verbatim),
    "call-missing-name": (RAISES, verbatim),
    "call-json-not-an-object": (RAISES, verbatim),
    "call-unterminated-partial-json": ("unterminated span surfaces verbatim; the vendor raises", verbatim),
    "call-end-message-inside-a-string": (
        "the first end marker closes the span, as in the vendor parser, which then raises",
        verbatim,
    ),
    "call-long-header-malformed": (RAISES + "; a header past the cap is not kept", verbatim),
    "call-header-at-the-cap-malformed": (RAISES, verbatim),
    "call-ends-after-marker": (RAISES, verbatim),
    "call-empty-body": (RAISES, verbatim),
    "call-nan-literal": ("NaN and Infinity are not JSON; the vendor reads them as floats", verbatim),
    "call-deep-nesting": ("nesting past 128 levels is refused before it can exhaust the stack", verbatim),
    "call-number-overflow": (
        "a number literal passes through as the model wrote it; the vendor reads inf, which JSON cannot carry",
        lambda output: calls(("t", {"arguments_json": '{"n":1E400,"m":-1e999}'})),
    ),
    "call-casts-python-spellings": (
        "digit separators and non-ASCII digits are Python numerals, not JSON Schema numbers, and stay "
        "strings; a digit string cast to number keeps every digit, where the vendor rounds it through a float",
        lambda output: calls(
            ("t", {"arguments": {"i": "1_000", "n": "1_0.5", "u": "１２３", "s": "٣"}}),
            ("t", {"arguments": {"n": 12345678901234567890}}),
        ),
    ),
}
by_parse = {case["name"]: case for case in parses}
for name, (reason, expect) in MEMRA.items():
    case = by_parse[name]
    case["memra"] = {"reason": reason, "message": expect(case["output"])}
    if "message" in case:
        assert case["message"] != case["memra"]["message"], name
for case in parses:
    assert ("message" not in case) <= ("memra" in case), case["name"]

# ------------------------------------------------------------- raw id corpus
corpus = []
STEP = HERE.parents[1] / "step-sku-20260807" / "corpus.tsv"
for line in STEP.read_text().splitlines():
    name, hexed = line.split("\t")
    corpus.append(("raw-" + name, bytes.fromhex(hexed).decode()))

OWN = {
    "case-camel": "camelCase PascalCase HTTPServer XMLHttpRequest getHTTPResponseCode iPhone eBay McDonald's",
    "case-runs": "ABC abc ABc aBC AbC a A aA Aa AAa aaA ÀÉÎõü ǅungla ǅ ǈ ǲ ΑΒΓαβγ АБВабв Straße STRASSE ʰʷ ªº",
    "case-uncased": "漢字かなカナ한글 אבג ابج ไทย देवनागरी Aあa aあA 漢A字a ერთი ᲔᲠᲗᲘ ᏣᎳᎩ ꭰꭱ",
    "marks": "é É à́b ́ ́a ́A .́a क्षत्रिय กำลัง ñÑ Ǻx Z̶a̶l̶g̶o̶",
    "contractions": "don't DON'T I'll I'LL we'Re they'VE it's IT'S I'm I'M he'd HE'D 'tis x'ſ X'ſ y'all o'clock rock'n'roll 's 'S a's'd l'eau d'abord can'tx 12's ''s 你's",
    "numbers": "0 12 123 1234 12345 1234567 3.14159 1,000,000 1e10 0x1F 1_000 ٣٤٥٦٧ ①②③④ ½¾ ⅧⅨ 一二三 １２３４ ௧௨௩௪ 2026-11-21T02:30:00Z v1.2.3-rc4 a1b22c333d4444",
    "punct": "a/b a//b http://x.y/z?q=1&r=2#f !!!/// .../\n ---\n\n ;;;/\r\n x //\n/ / // /// (a)/[b]/{c} \"/\" <|not_a_token|> <|message_user |> $€£¥ ©®™ +-*/=<>!&|^~% _under_ __init__ a...b",
    "whitespace": "a b  c   d    e\tf\t\tg \t h\n i \n j\n\nk\n\n\nl \n\n m\r\nn\r\n\r\no\rp \r q\n \n r\n\t\ns t　u\u2028v\u2029w\u0085x ​ y  z",
    "whitespace-edges": "  lead",
    "whitespace-trail": "trail  ",
    "whitespace-only-spaces": "     ",
    "whitespace-only-newlines": "\n\n\n",
    "whitespace-mixed-tail": "x \n \n  ",
    "whitespace-space-newline-space": " \n ",
    "whitespace-tabs-then-word": "\t\t\tword",
    "whitespace-crlf-run": "\r\n\r\n\r\n",
    "emoji": "🙂 🙂🙂 👍🏽 👨‍👩‍👧‍👦 🇮🇱🇫🇷 ❤️ ☃ ✈️ 1️⃣ #️⃣ 🏳️‍🌈 a🙂b A🙂B 1🙂2 🙂's 🤷‍♂️/🤷‍♀️",
    "cjk": "今天天气很好。我们去公园吧！「こんにちは」、世界。한국어도 됩니다. 中文English混排123数字，ＦＵＬＬｗｉｄｔｈ。",
    "code-python": "def fib(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n\nprint([fib(i) for i in range(10)])  # 0..34\n",
    "code-rust": "pub fn split_gpt4o(text: &str) -> Vec<String> {\n    let cpts: Vec<char> = text.chars().collect();\n    // TODO(x): handle '\\r\\n'\n    cpts.iter().map(|&c| c.to_string()).collect::<Vec<_>>()\n}\n",
    "code-c": "#include <stdio.h>\nint main(int argc, char **argv) {\n\tprintf(\"%d\\n\", argc >> 1 ? 0xFF : -1);\n\treturn 0;\n}\n",
    "code-json": "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3,\"metric\":true,\"x\":null,\"y\":[1.5,-2e-3]}}",
    "code-shell": "$ git log --oneline -n5 | grep -E '^[0-9a-f]{7,}' >/dev/null 2>&1 && echo \"ok\" || exit 1\n",
    "code-html": "<!DOCTYPE html>\n<html lang=\"en\"><head><title>T&amp;C</title></head><body class='a b'>\n  <p>Hi&nbsp;there</p>\n</body></html>\n",
    "markdown": "# Title\n\n- item **bold** _it_ `code`\n- [link](https://example.com/a/b?c=d)\n\n```py\nx = 1\n```\n\n| a | b |\n|---|---|\n",
    "prose": "The quick brown fox jumps over the lazy dog. It's 5 o'clock; isn't it? \"Yes,\" she said — 'no'… (maybe).",
    "mixed-scripts": "Привет, мир! Γειά σου Κόσμε. مرحبا بالعالم. שלום עולם. नमस्ते दुनिया. สวัสดีชาวโลก. Xin chào thế giới. İstanbul ığdır ǆ",
    "control": "a\x00b\x01c\x7fd‍e﻿ f� g\U000e0001 h\U0010ffff",
    "single-apostrophe": "'",
    "single-space": " ",
    "single-newline": "\n",
    "single-letter": "a",
    "single-upper": "A",
    "single-digit": "7",
    "single-mark": "́",
    "long-word": "Pneumonoultramicroscopicsilicovolcanoconiosis" * 3,
    "long-upper": "A" * 70 + "b",
    "long-digits": "9" * 40,
    "long-spaces": " " * 70 + "x",
}
for name, text in OWN.items():
    corpus.append(("own-" + name, text.replace("`", chr(96))))

ADDED = json.loads((ART / "tokenizer.json").read_text())["added_tokens"]
assert len(ADDED) == 60
for token in ADDED:
    content = token["content"]
    corpus.append((f"special-{token['id']}", content))
    corpus.append((f"special-in-text-{token['id']}", f"a{content}b {content}\n{content}{content} A"))
corpus.append(("special-all", "".join(token["content"] for token in ADDED)))
corpus.append(("special-broken", "<|message_user|<|message_user|>|> <|content_text| > <|end_message|"))

# Every vocabulary piece the merge table does not build: `model.ignore_merges` is what makes
# such a word encode as its one piece (the tokenizers BPE looks a whole word up first).
_raw = json.loads((ART / "tokenizer.json").read_text())
assert _raw["model"]["ignore_merges"] is True
_raw["model"]["ignore_merges"] = False
MERGES_ONLY = Tokenizer.from_str(json.dumps(_raw))
_pieces = [TOK.decode([i]) for i in range(len(_raw["model"]["vocab"]))]
_with = TOK.encode_batch(_pieces, add_special_tokens=False)
_without = MERGES_ONLY.encode_batch(_pieces, add_special_tokens=False)
unmerged = [i for i, (a, b) in enumerate(zip(_with, _without)) if a.ids != b.ids]
assert len(unmerged) == 337 and all(_with[i].ids == [i] for i in unmerged), len(unmerged)
assert _pieces[87550] == "kwargs" and 87550 in unmerged
for i in unmerged:
    corpus.append((f"unmerged-{i}", _pieces[i]))
del _raw

# Both ends of every maximal run of one class of the split (the classes of
# gen_unicode_classes.py; the tables are equal to this interpreter's unicodedata, which that
# script checks), each in the positions the regex tells apart.
WS = {9, 10, 11, 12, 13, 32, 133, 160, 5760, *range(8192, 8203), 8232, 8233, 8239, 8287, 12288}


def split_class(c):
    cat = unicodedata.category(chr(c))
    if cat in ("Lu", "Lt"):
        return 1
    if cat == "Ll":
        return 2
    if cat in ("Lm", "Lo"):
        return 3
    if cat[0] == "M":
        return 4
    if cat[0] == "N":
        return 5
    return 6 if c in WS else 0


def contexts(c):
    ch = chr(c)
    return f"x{ch}y X{ch}Y {ch}{ch}'S 1{ch}2 a'{ch}e a'{ch}l a'r{ch} .{ch}/\n{ch}\r\n {ch}"


assert unicodedata.unidata_version == "16.0.0", unicodedata.unidata_version
edges = []
previous = None
for c in range(1, 0x110000):
    if 0xD800 <= c <= 0xDFFF:
        previous = None
        continue
    k = split_class(c)
    if k != previous:
        if edges and edges[-1] != c - 1 and not 0xD800 <= c - 1 <= 0xDFFF:
            edges.append(c - 1)
        edges.append(c)
        previous = k
edges.append(0x10FFFF)
edges = sorted(set(edges))
for i in range(0, len(edges), 16):
    chunk = edges[i : i + 16]
    corpus.append((f"edge-{chunk[0]:06x}", " ".join(contexts(c) for c in chunk)))

for case in renders:
    if "expected" in case:
        corpus.append(("tmpl-" + case["name"], case["expected"]))
for case in parses:
    corpus.append(("out-" + case["name"], case["output"]))

assert len({name for name, _ in corpus}) == len(corpus), "corpus names are unique"

# ------------------------------------------------------- every scalar value
# The same contexts over every Unicode scalar value, 4096 scalars a chunk; the ids are too many
# to commit, so each chunk is committed as the SHA-256 of its ids (little-endian u32).
SCALARS = [c for c in range(1, 0x110000) if not 0xD800 <= c <= 0xDFFF]
digests = []
for i in range(0, len(SCALARS), 4096):
    chunk = SCALARS[i : i + 4096]
    text = " ".join(contexts(c) for c in chunk)
    ids = TOK.encode(text, add_special_tokens=False).ids
    digests.append((chunk[0], chunk[-1], len(ids), hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()))

OUT.mkdir(exist_ok=True)
(OUT / "renders.json").write_text(json.dumps(renders, ensure_ascii=False, indent=1) + "\n")
(OUT / "parses.json").write_text(json.dumps(parses, ensure_ascii=False, indent=1) + "\n")
with open(OUT / "scalar-digests.tsv", "w") as f:
    for first, last, count, digest in digests:
        f.write(f"{first:06x}\t{last:06x}\t{count}\t{digest}\n")
with open(HERE / "corpus.tsv", "w") as f:
    for name, text in corpus:
        f.write(f"{name}\t{text.encode().hex()}\n")
with open(HERE / "ref-ids.tsv", "w") as f:
    for name, text in corpus:
        a = TOK.encode(text, add_special_tokens=True).ids
        b = TOK.encode(text, add_special_tokens=False).ids
        f.write(f"{name}\t{','.join(map(str, a))}\t{','.join(map(str, b))}\n")
errors = sum("error" in case for case in renders)
print(
    f"renders={len(renders)} (refusals={errors}) parses={len(parses)} corpus={len(corpus)} "
    f"edges={len(edges)} scalar-chunks={len(digests)}"
)
