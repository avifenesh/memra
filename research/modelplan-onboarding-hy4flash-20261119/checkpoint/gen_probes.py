"""The CheckpointParity requests of the hy_v4_flash pack, encoded in the artifact's own dialect.

Renders each conversation with the artifact's `chat_template.jinja` through transformers'
`apply_chat_template` (reasoning_effort "no_think") and tokenizes it with the artifact's
`tokenizer.json`, as the step 2 oracle does. The assistant turn is teacher-forced: the prefill
ends after the generation prompt (`<think:opensource></think:opensource>`) or earlier, and every
later token is one decode step.

- `chat`: "What is the capital of France? Answer in one word." answered "Paris." (39 tokens, prefill
  36, 3 decode steps).
- `long` and `long-split`: one request of 2,112 tokens past the indexer budget (`index_topk` 2048):
  the user turn asks about `probe-doc.txt` (the first 7,884 characters of memra's
  docs/ONBOARDING.md at a81f506d08, the longest prefix that keeps the request at 2,112 tokens),
  the assistant answer is forced. `long` prefills 2,104 tokens (the indexer runs
  in the prefill) and decodes 8; `long-split` prefills 2,040 densely and decodes 72, so the
  indexer switches on in the decode, at the step whose cache passes 2,048 keys.

usage: venv/bin/python -I gen_probes.py CKPT_DIR OUT_DIR
"""

import json
import pathlib
import sys

from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
ckpt, out = sys.argv[1], pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(ckpt)


def encode(user, answer):
    prompt = [{"role": "user", "content": user}]
    head = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, reasoning_effort="no_think")
    whole = tok.apply_chat_template(prompt + [{"role": "assistant", "content": answer}], tokenize=False,
                                    reasoning_effort="no_think")
    assert whole.startswith(head)
    head_ids = tok(head, add_special_tokens=False)["input_ids"]
    ids = tok(whole, add_special_tokens=False)["input_ids"]
    assert ids[: len(head_ids)] == head_ids
    return ids, len(head_ids)


chat_ids, chat_prompt = encode("What is the capital of France? Answer in one word.", "Paris.")
assert (len(chat_ids), chat_prompt) == (39, 36), (len(chat_ids), chat_prompt)

TOTAL, ANSWER_TOKENS = 2112, 8
doc = (HERE / "probe-doc.txt").read_text()
question = "Read the text below, then say in one sentence what it is about.\n\n"
answer = "It describes how a new model is onboarded to memra."
answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
lo, hi = 0, len(doc)
while lo < hi:  # the longest prefix of the document that keeps the request within TOTAL tokens
    mid = (lo + hi + 1) // 2
    ids, prompt = encode(question + doc[:mid], answer)
    lo, hi = (mid, hi) if len(ids) <= TOTAL else (lo, mid - 1)
ids, prompt = encode(question + doc[:lo], answer)
while len(ids) != TOTAL:
    lo -= 1
    ids, prompt = encode(question + doc[:lo], answer)
assert prompt + ANSWER_TOKENS <= TOTAL
requests = [
    {"name": "chat", "token_ids": chat_ids, "decode_from": chat_prompt},
    {"name": "long", "token_ids": ids, "decode_from": TOTAL - ANSWER_TOKENS},
    {"name": "long-split", "token_ids": ids, "decode_from": 2040},
]
for r in requests:
    (out / f"{r['name']}.json").write_text(json.dumps(r) + "\n")
    print(r["name"], len(r["token_ids"]), r["decode_from"], "prompt_end", prompt if r["name"] != "chat" else chat_prompt)
print("doc_chars", lo, "answer_tokens_total", len(ids) - prompt)
