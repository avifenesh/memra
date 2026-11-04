# Gemma 4 unopened-thought census (2026-11-04)

Question: how long are the thoughts gemma4 models write after the template's closed channel
(an unopened `<channel|>`), and does the 256-byte stream hold-back of the parser rule
(docs/decisions/GEMMA-UNOPENED-THOUGHT.md) cover them?

Rig: RTX 5090 Laptop, memra-server at the gemma-thought lane tip (27c227fc34, the 256-byte cap).
Models: Gemma-4 E4B, 26B-A4B and 31B QAT Q4_0 GGUFs. Twenty-four prompts (`census-client.py`),
each thinking off and at `reasoning_effort` low and medium, greedy, `max_tokens` 400, sent once
as a blocking request and once as a stream: 72 cells a model. Single runs.

| model | cells with reasoning | unopened thoughts, thinking off | longest thought | stream equal to blocking | stream deltas with a tag |
|---|---|---|---|---|---|
| E4B | 61 | 12 of 24 (81 to 1,027 bytes) | 1,718 bytes | 60 of 72 | 12 (`<channel|>`) |
| 26B-A4B | 48 | 0 of 24 | 1,818 bytes | 72 of 72 | 0 |
| 31B | 48 | 0 of 24 | 1,552 bytes | 72 of 72 | 0 |

No blocking response carried a tag in content. No server log has a panic.

Verdict: only E4B writes unopened thoughts, and the 256-byte cap sends every one of them to
content with its tag on a stream. E4B's stream parsers now hold the opening with no cap
(`ModelCaps::gemma_unopened_thought`); the other models keep the cap. Six E4B thinking-off
prompts wrote a thinking process into content with no tag at all, which no parser rule can
split. Every model thinks at low effort on prompt 1 ("What is 17 times 23? Show your work
briefly."), the prompt serve-smoke's check (c) now uses; the 26B-A4B and the 31B think past 400
tokens on it.

Raw: `raw/<model>/census.json` (one row per cell), `census.log`, `status`, `server.log`,
`nonstream/*.json` (every blocking response) and `stream-sse.tar.xz` (every stream). Driver:
`census.sh` (GTAG and GMODEL pick the model).
