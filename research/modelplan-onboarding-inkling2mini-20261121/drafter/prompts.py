#!/usr/bin/env python3
"""Lane S: the acceptance prompt set (written for this lane, not taken from a benchmark), rendered
with the checkpoint's own chat template and tokenizer. Writes JSON lines {id, kind, ids, text_sha256}.

usage: prompts.py CKPT_DIR OUT.jsonl
"""
import hashlib
import json
import sys

from transformers import AutoTokenizer

F = "`" * 3
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return stdout, stderr and the exit code.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "search_issues", "description": "Search the issue tracker.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "state": {"type": "string", "enum": ["open", "closed", "all"]}}, "required": ["query"]}}},
]

RUST = "fn longest(words: &Vec<String>) -> &str {\n    let mut best = String::new();\n    for w in words {\n        if w.len() > best.len() { best = w.clone(); }\n    }\n    &best\n}"
TS = "async function fetchAll(urls: string[]) {\n  const out = [];\n  urls.forEach(async (u) => {\n    const r = await fetch(u);\n    out.push(await r.json());\n  });\n  return out;\n}"
DEPLOY = "#!/usr/bin/env bash\nset -euo pipefail\nENV=$1\nrsync -az build/ deploy@$ENV:/srv/app/\nssh deploy@$ENV 'systemctl restart app'\necho deployed to $ENV\n"

PROMPTS = [
    ("chat-trip", "chat", None, [{"role": "user", "content": "I have four days in Lisbon at the end of March with my partner and a seven year old. We like walking, food markets and one day trip. Can you sketch a plan, day by day, and say what to book ahead?"}]),
    ("chat-email", "chat", None, [{"role": "user", "content": "Write a short, polite email to my landlord. The kitchen tap has been dripping for two weeks, I reported it on the 3rd and nobody came. I want a repair date this week. Keep it firm but friendly."}]),
    ("chat-explain", "chat", None, [{"role": "user", "content": "Why does bread go stale faster in the fridge than on the counter? Explain it so a curious teenager would get it, then give two practical tips."}]),
    ("chat-multi", "chat", None, [
        {"role": "user", "content": "I'm trying to decide between a used 2019 hatchback with 90k km and a new small crossover. I drive about 12k km a year, mostly city."},
        {"role": "assistant", "content": "Happy to help. Two questions first: what is your budget range, and do you plan to keep the car more than five years?"},
        {"role": "user", "content": "Budget is flexible up to about 28k, and yes, probably eight years. Parking in my street is tight."}]),
    ("code-py-lru", "code", None, [{"role": "user", "content": "Write a Python class LRUCache with get(key) and put(key, value), O(1) each, capacity given at construction. Use only the standard library and include a few doctest examples."}]),
    ("code-rust-fix", "code", None, [{"role": "user", "content": "This Rust function does not compile. Fix it and explain the borrow problem in two sentences.\n\n" + F + "rust\n" + RUST + "\n" + F}]),
    ("code-sql", "code", None, [{"role": "user", "content": "Tables: orders(id, customer_id, created_at, total_cents) and customers(id, name, country). Write a PostgreSQL query that returns, per country, the number of customers who placed at least two orders in 2025 and their average order total in euros. Then explain any index you would add."}]),
    ("code-ts-review", "code", None, [{"role": "user", "content": "Review this TypeScript and rewrite it so it handles errors and does not fire requests in parallel without a limit:\n\n" + F + "ts\n" + TS + "\n" + F}]),
    ("agent-ci", "agentic", TOOLS, [
        {"role": "system", "content": "You are a coding agent working in a repository. Use the tools to inspect before you change anything. Think briefly, then call one tool at a time."},
        {"role": "user", "content": "The CI job 'unit' started failing after yesterday's merge. Find out why and propose a fix."}]),
    ("agent-bug", "agentic", TOOLS, [
        {"role": "system", "content": "You are a coding agent working in a repository. Use the tools to inspect before you change anything."},
        {"role": "user", "content": "Users report that exporting a report to CSV drops rows whose name contains a comma. Check the export module and the related open issues, then tell me what you would change."}]),
    ("agent-ops", "agentic", TOOLS, [
        {"role": "system", "content": "You are an operations assistant with shell access to a Linux server. Be careful: prefer read-only commands first."},
        {"role": "user", "content": "The disk on the build server is at 95%. Figure out what is using the space and suggest what can be cleaned safely."}]),
    ("agent-followup", "agentic", TOOLS, [
        {"role": "system", "content": "You are a coding agent working in a repository."},
        {"role": "user", "content": "Add a --dry-run flag to the deploy script."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "scripts/deploy.sh"}}}]},
        {"role": "tool", "content": DEPLOY},
    ]),
]


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(ckpt)
    with open(out, "w") as f:
        for pid, kind, tools, msgs in PROMPTS:
            kw = {"tools": tools} if tools else {}
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            f.write(json.dumps({"id": pid, "kind": kind, "ids": ids,
                                "text_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
            print(pid, kind, len(ids))


if __name__ == "__main__":
    main()
