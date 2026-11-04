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
cap. The E4B thought is 81 bytes in 19 tokens, 4.3 bytes per token. Every rate below is an
engine decode cell on the RTX 5090 Laptop (gen-only `run-gen` plain or `run-spec`), not a
served rate. A served stream also pays HTTP and detokenizer time, so its hold-back is at least
the time in the table.

- E4B plain 201.8 tok/s (`research/modelplan-onboarding-e4b-20261104/serve-09348940b8/summary.txt`).
- 12B plain 92.5 tok/s, tg128 at depth 512 (`research/gemma4-bringup/g12tg-cellstamp.log`). The
  12B is serve-smoke's default gemma4 card.
- 26B-A4B plain 193.9 tok/s short (`research/gemma4-bringup/rig5090-gemma4.jsonl`, 2026-07-11).
- 31B plain 40.0 tok/s short (the same log, 2026-07-12).
- 31B MTP spec, chat prompt, 124.3 tok/s (`docs/PERFORMANCE.md`, Gemma-4 table): a `run-spec`
  engine cell at K=5.

| cap (bytes) | tokens | E4B plain | 12B plain | 26B-A4B plain | 31B plain | 31B MTP spec (engine cell) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 30 | 0.15 s | 0.32 s | 0.15 s | 0.75 s | 0.24 s |
| **256 (chosen)** | 60 | 0.30 s | 0.65 s | 0.31 s | 1.5 s | 0.48 s |
| 512 | 120 | 0.59 s | 1.30 s | 0.62 s | 3.0 s | 0.97 s |

256 is the smallest power of two that covers both thoughts on record for this prompt: the E4B
unopened thought (81 bytes) and the 12B's opened thought on the same prompt (151 bytes,
`research/gemma4-serve-20260807/raw/gap2-on-20260807T193310Z.json`). 128 would already miss the
in-family length. Both thoughts come from one trivial prompt, so the cap rests on two samples.

**Stream and non-stream differ above the cap.** A non-stream response routes an unopened
thought to reasoning at any length. On a stream, an unopened thought of 256 bytes or more
already went out as content when the cap filled, so the thought and the literal `<channel|>`
reach content, the defect this rule fixes. A greedy chat with such a thought therefore gives
different outputs on the two shapes. `gemma_stream_opening_holdback_is_bounded_and_chunking_independent`
in `toolcall.rs` pins both sides.

**Open owner call:** keep 256 until the census of E4B unopened-thought lengths on a real prompt
set runs on the RTX 5090 (below), or set another bound now. Every other part of the rule stands
either way.

**Measurement that settled it:** `research/modelplan-onboarding-e4b-20261104/serve-09348940b8/`:
`probe.log` (rendered prompt tail and tag ids), `probe/probe-chat-off.json` (the leak),
`probe/probe-completion-raw.json` (the raw completion writes it too) and `serve-smoke-gemma.log`
(the FAIL). The 12B on the same prompt writes no unopened close: thinking-off content `ok`
(`gap2-off-20260807T193310Z.json`), and thinking-on opens its channel before any content (the
first delta of `gap2-stream-20260807T193310Z.sse` is reasoning).

**Scope and what is open:**

- One unopened thought is on record, on a trivial prompt. Retune the cap from a census of E4B
  unopened-thought lengths on a real prompt set when the RTX 5090 serve cells run (the open
  owner call above).
- A stream that ends in a worker error or a deadline inside the hold-back sends the error
  without the held bytes, as it already does for a held partial tag or stop-string tail.
- serve-smoke's gemma4 arm has check (e) for the stream half. Two greedy chats run as stream
  and non-stream on the same binary, and the deltas must reassemble to the non-stream message
  with no tag in any delta: the (a) chat, which resolves inside the hold-back, and a
  thinking-off chat whose content passes 256 bytes, which crosses the cap on a card that
  writes no unopened close. It does not compare against the binary before the rule; the
  `toolcall.rs` unit tests pin that.
- serve-smoke's gemma4 check (c) needs the model to think: at `reasoning_effort: low` E4B
  answered `ok` with no thought (`probe/probe-chat-low.json`, 2 completion tokens), and no
  parser rule can change that. Open owner call: give (c) a prompt E4B thinks on, or record
  (c) as an expected E4B FAIL.
- GPU cells owed, on the RTX 5090: serve-smoke's gemma4 arm on E4B, the 31B and the 12B
  default (`GEMMA_MODEL=<gguf> tools/serve-smoke.sh`, with the E4B cell run as in its
  onboarding receipt). Expected: (a), both (e) chats and (d) PASS on all three, and (c)
  PASSes on the 12B and FAILs on E4B as above. On E4B the long (e) chat FAILs with a tag in
  a delta if E4B writes an unopened thought of 256 bytes or more there, which is a census
  data point for the cap, not a regression. The 31B has no receipt on this arm yet, so its
  (c) result is open.
