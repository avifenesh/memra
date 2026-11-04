# Gemma 4: text before an unopened `<channel|>` is reasoning (2026-11-04)

**Chosen (owner, 2026-11-04: "Parser rule, all gemma4"):** one parser rule for every gemma4
model, in the thought dialect and the tooluse dialect alike (`ToolStreamParser` in
`crates/memra-server/src/toolcall.rs`, state `GemmaOpening`). At the start of the turn the
parser holds every byte until the opening resolves, at the earliest of:

- an unopened `<channel|>`: the text before it is reasoning, and content follows the tag. One
  `\n` right before the tag is syntax and goes, as inside an opened channel;
- a `<|channel>` open or a `<|tool_call>` open: the turn opened with content, and the held
  bytes take the path they took before this rule;
- the end of the turn: the held bytes are content.

A non-stream response holds the whole turn and splits after the fact. A stream holds at most
256 bytes (`GEMMA_STREAM_OPENING_CAP`) and then hands the opening to content. The cap is
positional: on a stream a close tag that starts at byte 256 or later stays content, however
the text is chunked. The four SSE renderers (chat and completions, Anthropic messages,
Responses, n>1 choices) mark their parser with `streaming()`; every other path is a blocking
shape.

**Why:** on a thinking-off chat the gemma4 template ends the prompt with a closed thought
channel (`<|channel>thought\n<channel|>`). E4B writes a thought anyway and closes it with a
stray `<channel|>`, so the response was `content: 'The user is asking for a specific response:
"ok". I must reply with exactly that.<channel|>ok'` with no reasoning. A raw `/v1/completions`
of the same prompt returns the same text, so the model writes it; the chat path does not.

**Rejected:**

- Content verbatim, as before: the thought and a control tag reach `content` (serve-smoke's
  gemma4 arm FAILs on E4B).
- Dropping only the stray tag: the thought stays in `content`.
- An unbounded stream hold-back: a turn that never writes a tag, which is every dense card's
  thinking-off answer, would stream as one delta at the end of the turn.
- Holding back again at every later stray close: it adds a hold-back in the middle of every
  gemma4 stream, and the only leak on record is at the start of the turn.

**Cost of the bound:** a turn that opens with content holds its first delta for at most the
cap. The E4B thought is 81 bytes in 19 tokens, 4.3 bytes per token. Decode rates: E4B 201.8
tok/s (RTX 5090 Laptop, `research/modelplan-onboarding-e4b-20261104/serve-09348940b8/summary.txt`),
31B plain 40.0 tok/s short (`research/gemma4-bringup/rig5090-gemma4.jsonl`), 31B MTP spec chat
124.3 tok/s (`docs/PERFORMANCE.md`, Gemma-4 table).

| cap (bytes) | tokens | E4B | 31B plain | 31B MTP chat |
|---:|---:|---:|---:|---:|
| 128 | 30 | 0.15 s | 0.75 s | 0.24 s |
| **256 (chosen)** | 60 | 0.30 s | 1.5 s | 0.48 s |
| 512 | 120 | 0.60 s | 3.0 s | 0.97 s |

256 is the smallest power of two that covers both thoughts on record for this prompt: the E4B
unopened thought (81 bytes) and the 12B's opened thought on the same prompt (151 bytes,
`research/gemma4-serve-20260807/raw/gap2-on-20260807T193310Z.json`). 128 would already miss the
in-family length.

**Measurement that settled it:** `research/modelplan-onboarding-e4b-20261104/serve-09348940b8/`:
`probe.log` (rendered prompt tail and tag ids), `probe/probe-chat-off.json` (the leak),
`probe/probe-completion-raw.json` (the raw completion writes it too) and `serve-smoke-gemma.log`
(the FAIL). The 12B on the same prompt writes no unopened close: thinking-off content `ok`
(`gap2-off-20260807T193310Z.json`), and thinking-on opens its channel before any content (the
first delta of `gap2-stream-20260807T193310Z.sse` is reasoning).

**Scope and what is open:**

- One unopened thought is on record, on a trivial prompt. On a stream, a longer one still
  reaches content; the non-stream split catches it at any length. Retune the cap from a census
  of E4B unopened-thought lengths on a real prompt set when the RTX 5090 serve cells run.
- A stream that ends in a worker error or a deadline inside the hold-back sends the error
  without the held bytes, as it already does for a held partial tag or stop-string tail.
- serve-smoke's gemma4 check (c) needs the model to think: at `reasoning_effort: low` E4B
  answered `ok` with no thought (`probe/probe-chat-low.json`, 2 completion tokens), and no
  parser rule can change that.
- GPU cells owed: serve-smoke's gemma4 arm on E4B, the 31B and the 12B default.
