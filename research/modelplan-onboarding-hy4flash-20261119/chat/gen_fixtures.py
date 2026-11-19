"""Oracle fixtures for the Hy4-flash chat dialect, tokenizer and tool-call parser.

Three oracles, all at pinned revisions:

- render: transformers 5.23.0 `render_jinja_template` over the artifact's `chat_template.jinja`
  (Avifenesh/Hy4-flash-NVFP4@f9eac06cf022, sha256 63ddd5de...), the same call
  `apply_chat_template` makes;
- ids: the `tokenizers` library over the artifact's `tokenizer.json` (sha256 2e3f3d22...), for
  every render (add_special_tokens=False, as apply_chat_template tokenizes) and for a raw
  corpus in both modes;
- parse: `HYV4ToolExtractor.extract_tool_calls` (strict, as `HYV4ToolParser` builds it) from
  vLLM PR #68296 at 8cbb80efa397 (`vllm/tool_parsers/hy_v4_tool_parser.py`, git blob checked
  below), with its vLLM-only imports stubbed: the extractor itself is pure Python.

    uv run --no-project --with transformers==5.23.0 --with tokenizers==0.23.3 \
        --with jinja2==3.1.6 --with regex python3 -I gen_fixtures.py

Writes fixtures/renders.json, fixtures/parses.json, corpus.tsv and ref-ids.tsv next to this file.
"""

import hashlib
import json
import pathlib
import sys
import types
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifact"
OUT = HERE / "fixtures"
TEMPLATE = (ART / "chat_template.jinja").read_text()
assert hashlib.sha256(TEMPLATE.encode()).hexdigest().startswith("63ddd5de294df36f"), "template pin"
assert hashlib.sha256((ART / "tokenizer.json").read_bytes()).hexdigest().startswith(
    "2e3f3d22065c2be5"
), "tokenizer pin"

import transformers  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

assert transformers.__version__ == "5.23.0", transformers.__version__
TOK = Tokenizer.from_file(str(ART / "tokenizer.json"))

S = ":opensource"
THINK_END = f"</think{S}>"
CALLS = f"<tool_calls{S}>"
CALLS_END = f"</tool_calls{S}>"
CALL = f"<tool_call{S}>"
CALL_END = f"</tool_call{S}>"
KEY = f"<arg_key{S}>"
KEY_END = f"</arg_key{S}>"
VAL = f"<arg_value{S}>"
VAL_END = f"</arg_value{S}>"

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
        "strict": True,
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web: returns JSON — “quoted” 中文.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"},
                "limit": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}
DEFERRED = {
    "type": "function",
    "function": {
        "name": "lazy_tool",
        "description": "Loaded on demand.",
        "parameters": {"type": "object", "properties": {}},
        "defer_loading": True,
    },
}


def call(name, args, cid=None):
    out = {"type": "function", "function": {"name": name, "arguments": args}}
    if cid is not None:
        out["id"] = cid
    return out


RENDER_CASES = [
    ("plain-default-effort", [{"role": "user", "content": "Hello!"}], None, {}),
    ("plain-no-think", [{"role": "user", "content": "Hello!"}], None, {"reasoning_effort": "no_think"}),
    ("plain-high", [{"role": "user", "content": "Hello!"}], None, {"reasoning_effort": "high"}),
    ("plain-max", [{"role": "user", "content": "Hello!"}], None, {"reasoning_effort": "max"}),
    ("no-generation-prompt", [{"role": "user", "content": "Hi"}], None, {"add_generation_prompt": False}),
    (
        "system-user",
        [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Why is the sky blue?"}],
        None,
        {"reasoning_effort": "max"},
    ),
    ("empty-system", [{"role": "system", "content": ""}, {"role": "user", "content": "x"}], None, {}),
    ("empty-user", [{"role": "user", "content": ""}], None, {"reasoning_effort": "no_think"}),
    (
        "multiturn-reasoning-history",
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "Add them."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6", "reasoning_content": "Add again."},
        ],
        None,
        {"reasoning_effort": "high"},
    ),
    (
        "multiturn-no-think-drops-reasoning",
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "Add them."},
            {"role": "user", "content": "And 3+3?"},
        ],
        None,
        {"reasoning_effort": "no_think"},
    ),
    (
        "reasoning-field-precedence",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning": "from reasoning", "reasoning_content": "from rc"},
        ],
        None,
        {"add_generation_prompt": False},
    ),
    (
        "last-assistant-no-gen-prompt",
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        None,
        {"reasoning_effort": "max"},
    ),
    (
        "second-system-turn",
        [
            {"role": "system", "content": "Lead."},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "Later rule."},
            {"role": "user", "content": "u2"},
        ],
        None,
        {},
    ),
    ("tools-no-system", [{"role": "user", "content": "Weather in Paris?"}], [WEATHER, SEARCH], {}),
    (
        "tools-with-system",
        [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"}],
        [WEATHER],
        {"reasoning_effort": "no_think"},
    ),
    (
        "tools-empty-system",
        [{"role": "system", "content": ""}, {"role": "user", "content": "Weather?"}],
        [WEATHER],
        {},
    ),
    ("tools-defer-loading-skipped", [{"role": "user", "content": "go"}], [WEATHER, DEFERRED, SEARCH], {}),
    (
        "toolcall-roundtrip-ordered",
        [
            {"role": "user", "content": "Weather in Paris and Rome, and search news."},
            {
                "role": "assistant",
                "content": "Checking.",
                "reasoning_content": "Need two calls.",
                "tool_calls": [
                    call("get_weather", {"city": "Paris", "days": 3, "metric": False}, "c1"),
                    call("search", {"query": "news", "filters": {"lang": "en", "n": [1, 2]}, "limit": 2.5}, "c2"),
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "{\"results\": []}"},
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 21C"},
        ],
        [WEATHER, SEARCH],
        {},
    ),
    (
        "toolcall-responses-unmatched-ids",
        [
            {"role": "user", "content": "Two cities."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("get_weather", {"city": "Oslo"}, "a"), call("get_weather", {"city": "Bergen"})],
            },
            {"role": "tool", "tool_call_id": "zz", "content": "cold"},
            {"role": "tool", "tool_call_id": "a", "content": "colder"},
            {"role": "user", "content": "Thanks"},
        ],
        [WEATHER],
        {"reasoning_effort": "max"},
    ),
    (
        "toolcall-preserved-thinking-before-last-user",
        [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "kept with tools",
                "tool_calls": [call("get_weather", {"city": "Lima"}, "x1")],
            },
            {"role": "tool", "tool_call_id": "x1", "content": "warm"},
            {"role": "assistant", "content": "It is warm.", "reasoning_content": "summarize"},
            {"role": "user", "content": "Second"},
        ],
        [WEATHER],
        {},
    ),
    (
        "toolcall-no-think",
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "dropped at no_think",
                "tool_calls": [call("get_weather", {"city": "Kyiv", "days": 1})],
            },
            {"role": "tool", "content": "rain"},
        ],
        [WEATHER],
        {"reasoning_effort": "no_think"},
    ),
    (
        "unicode-and-escapes",
        [
            {"role": "system", "content": "שלום 世界 \U0001f600"},
            {"role": "user", "content": "Tabs\tand\nnewlines \"quoted\" \\back\\ <think:opensource> literal"},
        ],
        [SEARCH],
        {"reasoning_effort": "high"},
    ),
    ("invalid-effort-low", [{"role": "user", "content": "x"}], None, {"reasoning_effort": "low"}),
    ("invalid-effort-medium", [{"role": "user", "content": "x"}], None, {"reasoning_effort": "medium"}),
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


renders = []
for name, messages, tools, kwargs in RENDER_CASES:
    text, error = render(messages, tools, kwargs)
    case = {
        "name": name,
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": kwargs.get("add_generation_prompt", True),
        "reasoning_effort": kwargs.get("reasoning_effort"),
    }
    if error is None:
        case["expected"] = text
        case["ids"] = TOK.encode(text, add_special_tokens=False).ids
    else:
        case["error"] = error
    renders.append(case)

# ---------------------------------------------------------------- parse oracle
VLLM_PR_HEAD = "8cbb80efa3974d10bb09c87580627bd874a3606b"
url = (
    "https://raw.githubusercontent.com/jeejeelee/vllm/"
    f"{VLLM_PR_HEAD}/vllm/tool_parsers/hy_v4_tool_parser.py"
)
src = urllib.request.urlopen(url, timeout=60).read()
blob = hashlib.sha1(b"blob %d\0" % len(src) + src).hexdigest()


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


class _AnyMeta(type):
    def __getattr__(cls, _name):
        return cls


class _Any(metaclass=_AnyMeta):
    """Stand-in for the vLLM types the extractor's module names but the extractor never uses."""

    def __init__(self, *a, **k):
        pass


for pkg in [
    "vllm",
    "vllm.entrypoints",
    "vllm.entrypoints.generate",
    "vllm.entrypoints.generate.base",
    "vllm.entrypoints.openai",
    "vllm.entrypoints.openai.chat_completion",
    "vllm.entrypoints.openai.responses",
    "vllm.renderers",
    "vllm.tool_parsers",
]:
    stub(pkg)
stub("vllm.envs", VLLM_ENFORCE_STRICT_TOOL_CALLING=False)
stub(
    "vllm.entrypoints.generate.base.protocol",
    DeltaFunctionCall=_Any, DeltaMessage=_Any, DeltaToolCall=_Any,
    ExtractedToolCallInformation=_Any, FunctionCall=_Any, ToolCall=_Any,
)
stub("vllm.entrypoints.openai.chat_completion.protocol", ChatCompletionRequest=_Any)
stub("vllm.entrypoints.openai.responses.protocol", ResponsesRequest=_Any)


class _Logger:
    def __getattr__(self, _name):
        return lambda *a, **k: None


stub("vllm.logger", init_logger=lambda _name: _Logger())
stub("vllm.renderers.chat_utils", make_tool_call_id=lambda: "id")
stub("vllm.tokenizers", TokenizerLike=_Any)
stub("vllm.tool_parsers.abstract_tool_parser", ToolParser=_Any)
stub("vllm.tool_parsers.tool_strict_level", ToolStrictLevel=_Any)
stub("vllm.tool_parsers.utils", partial_tag_overlap=lambda *a, **k: 0)
parser_mod = types.ModuleType("hy_v4_tool_parser")
exec(compile(src, "hy_v4_tool_parser.py", "exec"), parser_mod.__dict__)

vocab = TOK.get_vocab(with_added_tokens=True)
extractor = parser_mod.HYV4ToolExtractor(vocab, S, strict=True)
assert extractor.token_suffix == S


def schemas(tools):
    return [
        {"name": t["function"]["name"], "parameters": t["function"].get("parameters")}
        for t in tools
    ]


def calls_text(*calls):
    out = CALLS
    for name, pairs in calls:
        out += CALL + name
        for k, v in pairs:
            out += KEY + k + KEY_END + VAL + v + VAL_END
        out += CALL_END
    return out + CALLS_END


PARSE_CASES = [
    ("no-call", "Plain answer.", [WEATHER]),
    ("one-call-typed", "Checking." + calls_text(("get_weather", [("city", "Paris"), ("days", "3"), ("metric", "false")])), [WEATHER]),
    ("two-calls", calls_text(("get_weather", [("city", "Rome")]), ("search", [("query", "q"), ("limit", "2.5"), ("tags", "[\"a\", \"b\"]"), ("filters", "{\"x\": 1}")])), [WEATHER, SEARCH]),
    ("bool-case-insensitive", calls_text(("get_weather", [("city", "Oslo"), ("metric", "True")])), [WEATHER]),
    ("int-not-parsed-stays-string", calls_text(("get_weather", [("city", "Oslo"), ("days", "three")])), [WEATHER]),
    ("string-number-stays-string", calls_text(("get_weather", [("city", "123")])), [WEATHER]),
    ("unicode-value", calls_text(("search", [("query", "中文 \U0001f600 \"q\"")])), [SEARCH]),
    ("no-args", calls_text(("search", [])), [SEARCH]),
    ("unknown-function-json-value", calls_text(("other", [("a", "{\"k\": [1, 2]}"), ("b", "plain text")])), [WEATHER]),
    ("whitespace-around-name", calls_text((" get_weather\n", [("city", "Nice")])), [WEATHER]),
    ("content-after-calls", calls_text(("get_weather", [("city", "Nice")])) + "trailing", [WEATHER]),
    ("unterminated-block", CALLS + CALL + "get_weather" + KEY + "city" + KEY_END + VAL + "Paris", [WEATHER]),
    ("unbalanced-call-tags", CALLS + CALL + "get_weather" + CALLS_END, [WEATHER]),
    ("payload-outside-args", CALLS + CALL + "get_weather" + KEY + "city" + KEY_END + VAL + "Paris" + VAL_END + "junk" + CALL_END + CALLS_END, [WEATHER]),
    ("empty-name", calls_text(("", [("city", "Paris")])), [WEATHER]),
]
parses = []
for name, text, tools in PARSE_CASES:
    result = extractor.extract_tool_calls(text, schemas(tools))
    parses.append(
        {
            "name": name,
            "output": text,
            "tools": tools,
            "tools_called": result["tools_called"],
            "content": result["content"],
            "tool_calls": [
                {"name": c["name"], "arguments": json.loads(c["arguments"])} for c in result["tool_calls"]
            ],
        }
    )

# ------------------------------------------------------------- raw id corpus
STEP = HERE.parents[1] / "step-sku-20260807" / "corpus.tsv"
corpus = []
for line in STEP.read_text().splitlines():
    name, hexed = line.split("\t")
    if not name.startswith("tmpl-"):
        corpus.append(("raw-" + name, bytes.fromhex(hexed).decode()))
for token in [
    "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>x<｜hy_end:opensource｜>",
    "<think:opensource></think:opensource><tool_calls:opensource><tool_call:opensource>f<arg_key:opensource>k</arg_key:opensource><arg_value:opensource>v</arg_value:opensource></tool_call:opensource></tool_calls:opensource>",
    "<｜reasoning_mode:opensource｜>reasoning_effort:no_think",
    "<tool_response:opensource>r</tool_response:opensource><｜hy_pad:opensource｜><｜hy_EOT｜>",
    "<｜hy_place▁holder▁no▁813｜><｜hy_image▁pad:opensource｜>",
    "<think:opensource without close and <｜hy_start partial",
    "x<think:opensource>y</think:opensource>z",
]:
    corpus.append((f"special-{len(corpus)}", token))
for case in renders:
    if "expected" in case:
        corpus.append(("tmpl-" + case["name"], case["expected"]))

OUT.mkdir(exist_ok=True)
(OUT / "renders.json").write_text(json.dumps(renders, ensure_ascii=False, indent=1) + "\n")
(OUT / "parses.json").write_text(
    json.dumps({"parser_blob": blob, "vllm_pr_head": VLLM_PR_HEAD, "cases": parses}, ensure_ascii=False, indent=1) + "\n"
)
with open(HERE / "corpus.tsv", "w") as f:
    for name, text in corpus:
        f.write(f"{name}\t{text.encode().hex()}\n")
with open(HERE / "ref-ids.tsv", "w") as f:
    for name, text in corpus:
        a = TOK.encode(text, add_special_tokens=True).ids
        b = TOK.encode(text, add_special_tokens=False).ids
        f.write(f"{name}\t{','.join(map(str, a))}\t{','.join(map(str, b))}\n")
print(f"renders={len(renders)} parses={len(parses)} corpus={len(corpus)} parser_blob={blob}")
