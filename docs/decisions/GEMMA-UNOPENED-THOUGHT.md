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

A non-stream response holds the whole turn and splits after the fact. A stream on E4B, the one
model that writes unopened thoughts (`ModelCaps::gemma_unopened_thought`), holds the opening
the same way, with no byte cap, so both shapes split a turn alike. A stream on any other gemma4
model holds at most 256 bytes (`GEMMA_STREAM_OPENING_CAP`) and then hands the opening to
content. That cap is positional: a close tag that starts at byte 256 or later stays content,
however the text is chunked. The four SSE renderers (chat and completions, Anthropic messages,
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

**Stream and non-stream differ above the cap, on a capped model only.** On a model with the
256-byte cap, an unopened thought of 256 bytes or more already went out as content when the
cap filled, so a greedy chat with such a thought would split differently on the two shapes.
The census found no such thought outside E4B, and E4B has no cap.
`gemma_stream_opening_holdback_is_bounded_and_chunking_independent` and
`an_unopened_thought_model_streams_a_long_thought_as_its_blocking_twin` in `toolcall.rs` pin
both arms.

**Census, 2026-11-04 (settles the cap).** Twenty-four prompts, thinking off and at
`reasoning_effort` low and medium, greedy, `max_tokens` 400, each sent as a stream and as a
blocking request, on the RTX 5090 at the lane tip (`research/gemma-unopened-thought-census-20261104/`):

| model | unopened thoughts, thinking off | their length | stream equal to blocking, 256-byte cap |
|---|---|---|---|
| E4B | 12 of 24 | 81 to 1,027 bytes | 60 of 72; all 12 misses carry `<channel|>` in a delta |
| 26B-A4B | 0 of 24 | none | 72 of 72 |
| 31B | 0 of 24 | none | 72 of 72 |

The cap fails every long E4B thought, and no other model writes one. So E4B's streams hold
the opening with no cap, and the other models keep the 256-byte cap, which on them only bounds
the first content delta. Rejected: a larger cap for every model, which would delay every
26B-A4B and 31B stream with no thought to catch. The cost of the uncapped hold lands on E4B
alone: a thinking-off answer with no tag at all streams only at the turn end. Six E4B prompts
wrote a thinking process into content with no tag ("Here's a thinking process to arrive
at..."), which no parser rule can split.

**Measurement that settled it:** `research/modelplan-onboarding-e4b-20261104/serve-09348940b8/`:
`probe.log` (rendered prompt tail and tag ids), `probe/probe-chat-off.json` (the leak),
`probe/probe-completion-raw.json` (the raw completion writes it too) and `serve-smoke-gemma.log`
(the FAIL). The 12B on the same prompt writes no unopened close: thinking-off content `ok`
(`gap2-off-20260807T193310Z.json`), and thinking-on opens its channel before any content (the
first delta of `gap2-stream-20260807T193310Z.sse` is reasoning).

**Scope and what is open:**

- The census is one card, greedy, 400 tokens a request. A model whose pack starts writing
  unopened thoughts needs the same census before it joins `gemma_unopened_thought`.
- A stream that ends in a worker error or a deadline inside the hold-back sends the error
  without the held bytes, as it already does for a held partial tag or stop-string tail.
- serve-smoke's gemma4 arm has check (e) for the stream half. Two greedy chats run as stream
  and non-stream on the same binary, and the deltas must reassemble to the non-stream message
  with no tag in any delta: the (a) chat, which resolves inside the hold-back, and a
  thinking-off chat whose content passes 256 bytes, which crosses the cap on a card that
  writes no unopened close. It does not compare against the binary before the rule; the
  `toolcall.rs` unit tests pin that.
- serve-smoke's gemma4 check (c) needs the model to think: at `reasoning_effort: low` E4B
  answered `ok` with no thought (`probe/probe-chat-low.json`, 2 completion tokens). Check (c)
  now asks "What is 17 times 23? Show your work briefly." with 1,200 tokens: the census read
  reasoning on it at low effort for E4B, the 26B-A4B and the 31B (the last two think past 400
  tokens on it).
- GPU cells, on the RTX 5090: serve-smoke's gemma4 arm on E4B, the 26B-A4B and the 31B
  (`GEMMA_MODEL=<gguf> tools/serve-smoke.sh`, the E4B cell run as in its onboarding receipt),
  and on the 12B default when its model is on the rig. Expected: (a), (c), both (e) chats and
  (d) PASS on each.
