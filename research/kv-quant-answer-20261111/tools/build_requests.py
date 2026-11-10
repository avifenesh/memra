#!/usr/bin/env python3
"""Build the chat requests of the answer-position cells (memra#1177).

Each request is one user turn: a short preamble, the start of a document as tool output (whole
lines, cut so that the rendered prompt plus a 128-token answer is about the target depth), and a
question whose answer is a value in the document. The questions and their known answers are
fixed here, before any cell runs (`questions.json`).

The rendered prompt is sized with the model's tokenizer and the Qwen chat template with
thinking off: "<|im_start|>user\\n" + content + "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n
</think>\\n\\n". The server's `/v1/tokenize` gives the exact ids at run time; the cell records both.

usage: build_requests.py TOKENIZER_JSON DOCS_DIR OUT_DIR
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

PREAMBLE = "I ran a tool and it printed the output below.\n\n<tool_output>\n"
CLOSE = "</tool_output>\n\n"
TEMPLATE_HEAD = "<|im_start|>user\n"
TEMPLATE_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_TOKENS = 128
# forward_last's embedding gather launches one grid row per token, capped at 65,535 (memra#1200):
# the f32 reference cannot run past 65,535 tokens, so prompt + answer stays at or under it.
TARGETS = [32768, 65535]

DOCS = {
    "jsonl": {
        "file": "diskstats.jsonl",
        "crlf": False,
        "question": (
            'Find the "tick" record whose "seq" is 20. Report three values from that record exactly '
            'as printed: meminfo "MemAvailable_kB", meminfo "MemFree_kB", and the cpu "idle" '
            "counter. Answer in two or three sentences."
        ),
    },
    "csv": {
        "file": "owid-co2-data.csv",
        "crlf": False,
        "question": (
            "Find the row for Afghanistan in the year 1960. Report its population, gdp, co2 and "
            "co2_per_capita values exactly as printed. Answer in two or three sentences."
        ),
    },
    "prose": {
        "file": "pg2701.txt",
        "crlf": True,
        "question": (
            "Answer from the text above, in three sentences: who is the landlord of the "
            "Spouter-Inn, what is the name of the harpooneer Ishmael shares a bed with there, "
            "and what is the name of the preacher at the Whaleman's Chapel?"
        ),
    },
}


def known_values(docs: Path) -> dict:
    """The answers, read from the documents themselves."""
    out = {}
    for line in open(docs / "diskstats.jsonl"):
        d = json.loads(line)
        if d.get("seq") == 20:
            out["jsonl"] = {
                "MemAvailable_kB": str(d["meminfo"]["MemAvailable_kB"]),
                "MemFree_kB": str(d["meminfo"]["MemFree_kB"]),
                "idle": str(d["cpu"]["idle"]),
            }
            break
    for row in csv.DictReader(open(docs / "owid-co2-data.csv")):
        if row["country"] == "Afghanistan" and row["year"] == "1960":
            out["csv"] = {k: row[k] for k in ("population", "gdp", "co2", "co2_per_capita")}
            break
    # Moby-Dick, chapters 3 and 7 to 9: the landlord is Peter Coffin, the harpooneer Queequeg,
    # the preacher Father Mapple. Each name is checked to occur in every cut below.
    out["prose"] = {"landlord": "Peter Coffin", "harpooneer": "Queequeg", "preacher": "Father Mapple"}
    return out


def main() -> None:
    tok = Tokenizer.from_file(sys.argv[1])
    docs = Path(sys.argv[2])
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    known = known_values(docs)

    def ntok(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False).ids)

    manifest = []
    for name, spec in DOCS.items():
        raw = (docs / spec["file"]).read_bytes()
        text = raw.decode("utf-8")
        if spec["crlf"]:
            text = text.replace("\r\n", "\n")
        lines = text.splitlines(keepends=True)
        for target in TARGETS:
            budget = target - MAX_TOKENS - 1
            fixed = PREAMBLE + CLOSE + spec["question"]

            def render(n: int) -> str:
                return TEMPLATE_HEAD + PREAMBLE + "".join(lines[:n]) + CLOSE + spec["question"] + TEMPLATE_TAIL

            # Largest whole-line count whose rendered prompt fits the budget (binary search).
            lo, hi = 1, len(lines)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if ntok(render(mid)) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            n = lo
            body = "".join(lines[:n])
            content = PREAMBLE + body + CLOSE + spec["question"]
            for v in known[name].values():
                if v not in body:
                    raise SystemExit(f"{name}-{target}: known value {v!r} not in the cut")
            prompt_tokens = ntok(render(n))
            req = {
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            cell = f"{name}-{target}"
            (out / f"{cell}.request.json").write_text(json.dumps(req, ensure_ascii=False))
            manifest.append(
                {
                    "cell": cell,
                    "doc": name,
                    "target_total": target,
                    "source_file": spec["file"],
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "crlf_to_lf": spec["crlf"],
                    "doc_lines": n,
                    "doc_chars": len(body),
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "prompt_tokens_local": prompt_tokens,
                    "fixed_tokens_local": ntok(TEMPLATE_HEAD + fixed + TEMPLATE_TAIL),
                }
            )
            print(cell, n, "lines", prompt_tokens, "prompt tokens (local tokenizer)")
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    q = {name: {"question": spec["question"], "known": known[name]} for name, spec in DOCS.items()}
    (out / "questions.json").write_text(json.dumps(q, indent=1) + "\n")


if __name__ == "__main__":
    main()
