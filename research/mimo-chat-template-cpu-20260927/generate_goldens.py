#!/usr/bin/env python3
"""Render bounded MiMo text chats with the pinned HF tokenizer and Transformers."""

import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
EXPECTED = {
    "tokenizer.json": "ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e",
    "tokenizer_config.json": "413a7845f52943ccf4de0e5c838414507d16c44dbf573da9e20bc8902b384d06",
}

SYSTEM_USER = [
    {"role": "system", "content": "Keep punctuation exact."},
    {"role": "user", "content": "  Café and 你好?  "},
]
ANSWER = [{"role": "user", "content": "What is 17 plus 25? Answer with the number only."}]
TOOL = [
    {"role": "user", "content": "Weather for 東京, please."},
    {
        "role": "assistant",
        "content": "Checking now.",
        "reasoning_content": "Call lookup.",
        "tool_calls": [
            {
                "id": "call_weather",
                "type": "function",
                "function": {"name": "lookup", "arguments": {"city": "東京", "days": 2}},
            }
        ],
    },
    {"role": "tool", "name": "lookup", "content": '{"temp_c": 21}'},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]
CASES = [
    ("system_user_gen", SYSTEM_USER, [], True, None),
    ("system_user_closed", SYSTEM_USER, [], True, False),
    ("system_user_final", SYSTEM_USER, [], False, None),
    ("tool_history_gen", TOOL, TOOLS, True, None),
    ("tool_history_closed", TOOL, TOOLS, True, False),
    ("tool_history_final", TOOL, TOOLS, False, None),
    ("answer_17_plus_25", ANSWER, [], True, None),
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <pinned-hf-tokenizer-dir>")
    source = Path(sys.argv[1])
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"{name}: SHA-256 mismatch: {actual}")

    config = json.loads((source / "tokenizer_config.json").read_text())
    template = config["chat_template"]
    (HERE / "source-template.jinja").write_bytes(template.encode("utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    assert tokenizer.chat_template == template

    lines = []
    for name, messages, tools, generation_prompt, thinking in CASES:
        options = {"tools": tools, "add_generation_prompt": generation_prompt}
        if thinking is not None:
            options["enable_thinking"] = thinking
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, **options)
        assert ids == tokenizer.encode(rendered, add_special_tokens=False), name
        lines.append(
            f"{name}\t{rendered.encode('utf-8').hex()}\t{','.join(map(str, ids))}\n"
        )
        print(f"{name}: {len(rendered.encode('utf-8'))} bytes, {len(ids)} ids")
    (HERE / "goldens.tsv").write_text("".join(lines))
    print(f"wrote {len(lines)} source goldens")


if __name__ == "__main__":
    main()
