#!/usr/bin/env python3
"""Own-gen RANK-CORPUS prompt pack for the qwen4_exp FR-Spec draft trim (mtp9).

DRAFT-REGIME.md law 1: rank files are a distribution artifact of the EXACT serving
model, derived from ITS OWN generations, with the CHAT TEMPLATE ON when you serve
chat, and the corpus must cover every prompt CLASS you serve. This script produces
only the PROMPTS (input); the counted distribution is whatever the engine emits
(`qwen4exp_real_gate --owngen`).

Provenance, stated because it bounds the artifact: the owner SXC prompt pools are not
on this box, so the pack is composed HERE from real prompt shapes — the four banked
goldens continuations (raw shape) plus chat-template renders of realistic developer /
agent turns, including the tools render and the thinking-kwargs matrix that
research/qwen4exp-bringup-20260829/goldens/template-goldens.json banks. No synthetic
filler text: every prompt is a task a real caller would send.

HELD-OUT SPLIT, and it is load-bearing. Acceptance measured on a prompt whose own
continuation was counted into the ranks is optimistic by construction (frspec-owngen keeps
PROMPTS_EVAL out of its ranks for exactly this reason). So:

  - The FOUR banked goldens prompts are the perf/gate prompts (realgate/dump/prompts.tsv).
    They are NOT in the corpus at all — held out by construction. They are also raw
    continuations while the corpus is chat-shaped, so per law 1 ("ranks inherit their
    corpus MIX") they are the CONSERVATIVE cell: a trim measured on a class it was not
    derived from can only understate itself.
  - A further six chat-shaped prompts (2 code, 2 agentic-tools, 2 reasoning) go to the
    heldout file instead of the corpus, so there is also an IN-CLASS held-out cell.

OWNER POOLS. The owner directive (2026-08-14) is that FR-Spec rank corpora prompts come
from the SXC corpora + owner agent-session pools. Those live on the RIG, not on this box,
so extract-sxc-prompts.py runs there and its `pool<TAB>text` output is passed here as the
optional 4th argument: each line is rendered through the SAME chat template and appended
as class `sxc_<pool>`, AFTER the composed rows. Composed indices 0..N-1 are therefore
stable when owner prompts are added, which is what lets the own-gen resume ledger keep
every generation already banked instead of regenerating it.

Usage: python3 make-corpus-prompts.py <artifact_dir> <corpus.tsv> <heldout.tsv>
                                       [owner-prompts.tsv]
Output: corpus  = `index<TAB>class<TAB>ids-csv` (the counted set)
        heldout = `index<TAB>ids-csv<TAB>ids-csv` (the real-gate prompts.tsv shape; the
                  third column is a placeholder — the spec instruments compare plain
                  against spec, not against a golden)
"""
import json
import sys

from transformers import AutoTokenizer

src, out, heldout_out = sys.argv[1], sys.argv[2], sys.argv[3]
owner_prompts = sys.argv[4] if len(sys.argv) > 4 else None
tok = AutoTokenizer.from_pretrained(src)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative path"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the repository root and return stdout+stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff to the working tree",
            "parameters": {
                "type": "object",
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"],
            },
        },
    },
]

# --- the four banked goldens prompts: RAW continuation shape (no template) ---
RAW = [
    "Write a Python function to merge two sorted linked lists.",
    "The capital of Australia is",
    "def fib(n):\n    ",
    "Translate to French: the weather is nice today.",
]

# --- chat turns, per class. (user_text, chat_template_kwargs, tools) ---
CODE = [
    "Write a Rust function that streams a large CSV file, parses each row into a struct with serde, and returns a summary of the numeric columns. Keep memory constant and handle malformed rows without aborting the whole file.",
    "Refactor this into idiomatic Python with type hints and proper error handling:\n\ndef load(p):\n    f = open(p)\n    d = json.load(f)\n    return d['items']\n\nExplain each change in one line.",
    "Implement an LRU cache in TypeScript with O(1) get and set, backed by a Map, and write the three unit tests that would catch an eviction-order bug.",
    "Here is a failing function. Find the bug and give me the corrected version:\n\nfn median(xs: &mut Vec<f64>) -> f64 {\n    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());\n    xs[xs.len() / 2]\n}",
    "Write a SQL query that, for each customer, returns their most recent order total and the number of orders in the last 90 days. Postgres, and explain the window function you pick.",
    "Add graceful shutdown to this Go HTTP server so in-flight requests finish within 15 seconds, then exit non-zero if any are still running.\n\nfunc main() {\n    http.HandleFunc(\"/\", handler)\n    http.ListenAndServe(\":8080\", nil)\n}",
    "Convert this callback-based Node function to async/await, preserving the error semantics exactly, and say what changes for a caller that used to pass a callback.",
    "Write a Python generator that reads a JSONL file lazily, validates each record against a pydantic model, and yields (line_number, error) for the invalid ones instead of raising.",
]
REFACTOR = [
    "This module has grown to 900 lines and three responsibilities: HTTP handling, retry logic, and response caching. Propose a split into files with the exact function signatures at each boundary, and name what becomes testable that was not before.",
    "Review this diff for correctness and style:\n\n-    let total = items.iter().map(|i| i.price).sum::<f64>();\n+    let total: f64 = items.iter().filter(|i| i.active).map(|i| i.price).sum();\n\nWhat behavior changed, and what test would have caught it?",
    "Our retry wrapper retries on every error including 400s. Rewrite it to retry only on 429 and 5xx with exponential backoff and jitter, and explain why retrying a 400 is a bug and not just waste.",
]
AGENTIC_TOOLS = [
    "The test suite fails with `AssertionError: expected 200, got 404` in tests/api/test_users.py::test_get_user. Diagnose it. Start by reading the test.",
    "Bump the project to the latest minor version of the http client dependency, run the tests, and fix whatever breaks. Work one step at a time.",
    "Find every place in this repo that reads the DATABASE_URL environment variable directly and route them all through the config module instead.",
    "CI is red on main but green on my branch. Figure out why, then tell me whether to revert or forward-fix.",
]
AGENTIC_PLAN = [
    "Plan a zero-downtime migration that renames a column on a 50M-row Postgres table. Give me a numbered runbook with a rollback point after every step and the exact SQL.",
    "You are a coding agent with shell access to a production incident. The API returns 530 on one hostname and 200 on another that points at the same origin. List the checks you would run in order, and what each result would rule out.",
    "Our nightly job silently stopped writing half its output three weeks ago and nothing alerted. Design the check that would have caught it, and explain why a liveness probe would not have.",
    "Write the deploy plan for swapping an inference server to a new build behind a blue/green port switch: pre-flight gates, the flip, the post-flip probe that proves the new slot is actually serving, and the rollback.",
]
REASONING = [
    "A train leaves city A at 09:00 at 80 km/h. Another leaves city B, 240 km away, at 09:30 heading toward A at 100 km/h. When do they meet? Show the algebra.",
    "Three friends split a bill of 87.50 with a 15% tip on the pre-tip amount. One ordered a dish 12.00 more expensive than each of the other two equal dishes. What does each pay if they split fairly?",
    "A cache has a 92% hit rate. A hit costs 0.2 ms, a miss costs 40 ms. What is the mean latency, and how much does raising the hit rate to 96% save in percentage terms?",
    "I have two servers. One does 120 tokens/s and costs 2.10/hour; the other does 79 tokens/s and costs 1.30/hour. Which is cheaper per million tokens, and by how much?",
    "If a speculative decoder drafts 5 tokens per round and accepts 84% of them, what is the mean number of committed tokens per round, assuming acceptance stops at the first rejection and one bonus token always commits?",
]
CHAT = [
    "Explain the difference between TCP and UDP to someone who knows basic networking, with one concrete example where each is the right choice.",
    "What is a KV cache in transformer inference, and why does it make generation faster but prefill no faster?",
    "I have 500g of chicken thighs, rice, onions, and soy sauce. Suggest a dinner I can cook in 30 minutes, with steps.",
    "Explain what speculative decoding is to a backend engineer who has never trained a model, and be clear about what it does NOT change.",
    "My team argues about whether to cache at the CDN or in the application. Give me the two or three questions that actually decide it.",
]
JSON_OUT = [
    'Extract the fields from this log line into JSON with keys ts, level, service, message, request_id. Return only the JSON.\n\n2026-08-29T11:04:22.118Z WARN api-gateway upstream timeout after 40100ms req=7f3a9c21',
    'Return a JSON array of objects with keys name, version, and license for these dependencies, and use null where a value is missing:\n\nserde 1.0.219 (MIT/Apache-2.0)\ntokio 1.47 (MIT)\nmystery-lib (unknown)',
    'Given this API error response, produce a JSON object with keys retryable (boolean), wait_seconds (number or null), and reason (string). Only the JSON.\n\nHTTP/1.1 429 Too Many Requests\nRetry-After: 12\n{"error":{"type":"rate_limit","message":"too many requests"}}',
]
SHELL = [
    "Write a bash script that finds the 10 largest files under a directory and prints them with human-readable sizes, skipping anything under .git.",
    "One-liner: for every .py file changed in the last commit, run ruff on it and stop at the first failure.",
    "Give me the exact commands to find which process is holding port 8080, see its open files, and kill it only if it is not the systemd unit I care about.",
]
LOGS = [
    "These are the last lines before an engine crash. Tell me what failed and what input shape triggered it:\n\nthread 'gpu-worker' panicked at crates/memra-engine/src/spec.rs:3624:\nassertion failed: suffix.len() > 0\nnote: run with `RUST_BACKTRACE=1`\nfatal: worker thread exited, code 70",
    "Our p99 latency is fine but a small share of requests return 502 at almost exactly 40.1 seconds, byte-uniform. What class of cause does that pattern point at, and what would you check first?",
    "Read this nvidia-smi output and tell me whether this card is thermally throttled or power-capped:\n\nGPU 0: 600W / 600W, 87C, sm 100%, clocks: graphics 1410 MHz (max 2610 MHz), reasons: SwPowerCap",
]
TRANSLATE = [
    "Translate to French, keeping the technical terms in English: \"The verify chunk is bit-identical to the decode program, so the speculative path commits the same tokens.\"",
    "Translate this commit message to Hebrew and keep the identifiers untouched: \"fix(spec): guard empty resume suffix so decode_step cannot panic\"",
]
WRITING = [
    "Write a 150-word release note for a CLI tool that cut inference latency 40%, aimed at developers, with one code example.",
    "Draft a four-sentence status update to a customer whose API had a 12-minute outage caused by a DNS record pointing at a decommissioned box. Own it, no jargon, say what changed.",
]

# --- long multi-turn agentic sessions (tool calls + tool results in history) ---
LONG_SESSIONS = [
    [
        {"role": "user", "content": "The users endpoint 404s in tests. Fix it."},
        {"role": "assistant", "content": "I'll read the failing test first."},
        {
            "role": "user",
            "content": "tests/api/test_users.py:\n\ndef test_get_user(client, seeded_user):\n    r = client.get(f\"/api/v1/users/{seeded_user.id}\")\n    assert r.status_code == 200\n    assert r.json()[\"email\"] == seeded_user.email\n",
        },
        {
            "role": "assistant",
            "content": "The test hits /api/v1/users/<id>. Let me see how the router mounts the users blueprint.",
        },
        {
            "role": "user",
            "content": "app/routes/__init__.py:\n\nfrom .users import bp as users_bp\n\ndef register(app):\n    app.register_blueprint(users_bp, url_prefix=\"/api/users\")\n",
        },
        {
            "role": "user",
            "content": "So the prefix is /api/users but the test asks for /api/v1/users. Which side is wrong, and what is the fix that does not break existing clients?",
        },
    ],
    [
        {
            "role": "user",
            "content": "Our inference server drops every session when one request has an empty suffix. Walk me through hardening it.",
        },
        {
            "role": "assistant",
            "content": "An assert in the GPU worker thread takes the whole process down, so every concurrent session dies with it. Show me the resume path.",
        },
        {
            "role": "user",
            "content": "crates/engine/src/spec.rs:\n\nlet suffix = &tokens[committed..];\nassert!(!suffix.is_empty(), \"resume suffix must be non-empty\");\nlet head = suffix[0];\n",
        },
        {
            "role": "user",
            "content": "Give me the patch that turns this into a request-level error, plus the test that reproduces the old crash.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Review the design of a speculative decoding loop for exactness. K drafts per round, one batched verify of K+1 columns, greedy accept walk, partial rewind of recurrent state.",
        },
        {
            "role": "assistant",
            "content": "The exactness question is whether every verify row is the same program as a single-token decode step. Which sections differ between the two shapes?",
        },
        {
            "role": "user",
            "content": "Dense matvecs read weights once for all K+1 columns instead of once per token. The recurrent scan steps per column and snapshots state. Attention masks the speculative tail. The MoE merges every column's expert slots into one launch.",
        },
        {
            "role": "user",
            "content": "For each of those four, tell me whether it can be bit-identical to the one-token path and what oracle would prove it.",
        },
    ],
]

rows = []
heldout = []

# Held OUT of the counted corpus (in-class held-out cell): the last code prompt, the last
# two agentic-tools prompts, and the last reasoning prompt. Named by text so a reordering
# of the pools cannot silently pull them back into the ranks.
HELDOUT_TEXTS = {
    CODE[-1],
    CODE[-2],
    AGENTIC_TOOLS[-1],
    AGENTIC_TOOLS[-2],
    REASONING[-1],
    REASONING[-2],
}


def add(cls, ids):
    rows.append((len(rows), cls, ids))


def render(cls, messages, ck=None, tools=None):
    kw = {}
    if tools:
        kw["tools"] = tools
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, **kw, **(ck or {})
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    # A single-user-turn prompt whose text is reserved goes to the held-out file instead.
    if len(messages) == 1 and messages[0].get("content") in HELDOUT_TEXTS:
        heldout.append((len(heldout), cls, list(ids)))
        return
    add(cls, list(ids))


# The four banked goldens prompts are the PERF/GATE prompts and are deliberately absent
# from the counted corpus (see the module docstring's held-out split).

# Default (thinking on, xhigh) renders across the serving classes.
for cls, pool in [
    ("code", CODE),
    ("refactor", REFACTOR),
    ("reasoning", REASONING),
    ("chat", CHAT),
    ("json", JSON_OUT),
    ("shell", SHELL),
    ("logs", LOGS),
    ("translate", TRANSLATE),
    ("writing", WRITING),
    ("agentic_plan", AGENTIC_PLAN),
]:
    for text in pool:
        render(cls, [{"role": "user", "content": text}])

# The tools render (agentic tool-calling shape) — the class the trim must cover for
# the personal-agent workload.
for text in AGENTIC_TOOLS:
    render("agentic_tools", [{"role": "user", "content": text}], tools=AGENT_TOOLS)
render("agentic_tools", [{"role": "user", "content": "What is the weather in Tel Aviv right now?"}],
       tools=[WEATHER_TOOL])

# The thinking-kwargs matrix (banked template-goldens cases) on real tasks: the
# no-think and low-effort shapes emit a different distribution from xhigh.
for text in CODE[:3] + REASONING[:2] + CHAT[:2]:
    render("think_off", [{"role": "user", "content": text}],
           ck={"enable_thinking": False})
for text in CODE[3:6] + REASONING[2:4] + CHAT[2:4]:
    render("effort_low", [{"role": "user", "content": text}],
           ck={"reasoning_effort": "low"})

# Long multi-turn agentic sessions, with and without tools declared.
for session in LONG_SESSIONS:
    render("longctx", session)
render("longctx", LONG_SESSIONS[0], tools=AGENT_TOOLS)
render("longctx", LONG_SESSIONS[1], tools=AGENT_TOOLS)

# Owner pools LAST, so composed indices stay stable across a corpus extension and the
# own-gen resume ledger keeps every generation already banked.
if owner_prompts:
    for line in open(owner_prompts):
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        pool, text = line.split("\t", 1)
        render(f"sxc_{pool}", [{"role": "user", "content": text}])

with open(out, "w") as f:
    f.write(
        "# qwen4_exp own-gen rank corpus prompt pack (mtp9). Prompts are INPUT only;\n"
        "# the counted distribution is the engine's own emitted tokens. Composed on the\n"
        "# box from real-shaped chat-template renders (tools + thinking-kwargs matrix) —\n"
        "# the owner SXC pools are not on this box. HELD OUT of this file: the four banked\n"
        "# goldens prompts (they are the perf/gate prompts) and six chat-shaped prompts.\n"
    )
    for i, cls, ids in rows:
        f.write(f"{i}\t{cls}\t{','.join(str(t) for t in ids)}\n")

with open(heldout_out, "w") as f:
    for i, cls, ids in heldout:
        csv_ids = ",".join(str(t) for t in ids)
        # real-gate prompts.tsv shape; column 3 is a placeholder (the spec instruments
        # compare plain against spec, never against a golden).
        f.write(f"{i}\t{csv_ids}\t{csv_ids}\n")

by_class = {}
for _, cls, ids in rows:
    by_class.setdefault(cls, []).append(len(ids))
print(f"{len(rows)} counted prompts -> {out}")
for cls, lens in sorted(by_class.items()):
    print(f"  {cls:14s} n={len(lens):2d}  prompt_tokens min={min(lens)} max={max(lens)}")
print("total prompt tokens:", sum(sum(v) for v in by_class.values()))
print(f"{len(heldout)} HELD-OUT prompts -> {heldout_out}")
for i, cls, ids in heldout:
    print(f"  heldout[{i}] {cls:14s} prompt_tokens={len(ids)}")
print(json.dumps({"classes": len(by_class), "counted": len(rows), "heldout": len(heldout)}))
