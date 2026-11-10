# Answer positions after a chat turn that holds repetitive tool output (memra#1177)

Qwen3.8-27B NVFP4+Q5_K MTP GGUF on one DGX Spark (GB10, 48 SMs), 2026-11-11. main at
3de59bbf80 (v0.141.0) for the server; the research probe of memra#1197 rebuilt on the same
commit. Research only; nothing here changes a default.

## Outcome

Six served chat answers: a JSONL log, a CSV table and a prose control, each inside a user turn
as tool output, at about 32,768 and 65,535 tokens, each with a question whose answer is a value
in the document. 378 answer positions in all.

1. **No answer position differs.** At every one of the 378 rows, the f32-KV prefill (a), the
   prime over the quantized cache (b) and the served batch decode over a prime-written cache (c)
   all have the served token as top-1. Every served answer gives every known value right (3 of
   3, 4 of 4, 3 of 3 at both depths). The server's default arm (the MTP draft) and the plain arm
   wrote the same text in all six cells.
2. **The spread at answer rows is close to raw prose, not to the in-turn rows.** a against c at
   answer rows: median 0.19 to 0.40, p90 0.38 to 1.01, max 0.60 to 2.35 (memra#1197's raw prose
   at 32,768: p90 0.50, max 2.31). On the in-turn document rows of the same requests: p90 up to
   15.97 and max up to 36.83, with 40 to 724 rows above 3 per cell. The rows that copy a value
   (digits, names) have a top-2 margin of at least 4.07 in every program on the JSONL and CSV
   answers. The 21 answer rows where any program's margin is under 1 (smallest 0.09) are word
   choices: "with" against "where", "In" against "For", " Que" (of Queequeg) against " named".
   At each of them a's and c's runner-up is another wording, not another value.
3. **The large in-turn spread is not specific to repetitive content.** memra#1197 saw it with
   the JSONL log in a user turn. Here the prose control in a user turn also reaches an a/c max of
   25.51 and 25.92 on its document rows (32,768 and 65,535), with 63 and 62 rows above 3;
   memra#1197's raw prose at 32,768 tokens had an a/c max of 2.31 and no row above 3. Most
   document rows in a user turn are rows where the model would end the turn (`<|im_end|>` is a's
   top-1 at 1,482 to 1,978 of the 1,978 to 1,993 document rows per cell), but the large spreads
   are not confined to those rows (`SUMMARY.md`, "Document rows by a's top-1").

What this does not show:

- **It does not show that answer positions are safer than document positions.** 378 rows is
  small against the rate memra#1197 measured. On the raw JSONL log at 32,768 tokens, a and the
  served decode flipped with a's margin above 1 at 9 of 2,048 rows. At that rate 378 rows would
  hold about 1.7 such flips, and zero flips would still turn up about 19% of the time
  (e^-1.66). On the 150 JSONL answer rows alone the chance is about 52%. Zero here is a result
  for these six answers, not a bound on the rate.
- One model, one device, one question per document, greedy, thinking off, short answers (46 to
  74 tokens; the prose answers came in under the 64 planned). Answers that reason at length,
  sample, or quote long spans from the tool output are not measured.
- The cache format and the kernel that reads it are still not separated (no f32-cache control,
  memra#1197 item 3), and 131,072 tokens still cannot run against the f32 reference (memra#1200).
- The default arm is compared by its text only; its logits were not probed.

For memra#1177: this answers the open item "the answer positions after a chat turn holding such
content" within this scope: no position differs and no answer is wrong. The issue stays open for
the f32-cache control and the 131,072-token depth.

## Question

memra#1197 measured the quantized KV cache (q8_0 K, q5_1 V) on raw repetitive documents, and in
one cell with a JSONL log inside a user turn. Inside that turn the spread between programs was an
order larger than on the raw log, and the served decode flipped one row at an f32 margin of 15.8.
Those rows are inside the user turn: the model never generates from them. What a user receives
is the answer after the turn. Do the programs disagree there?

## Cells

Six cells: three documents at two depths. Each is one chat request, greedy, thinking off
(`chat_template_kwargs.enable_thinking: false`), `max_tokens` 128. The user turn holds a short
preamble, the start of the document as tool output (whole lines, inside `<tool_output>` tags) and
a question whose answer is a value in the document:

| document | question | known answer |
|---|---|---|
| jsonl (the host sampler log of memra#1197) | the `tick` record with `seq` 20: its meminfo `MemAvailable_kB`, meminfo `MemFree_kB` and cpu `idle` | 105263308, 18168220, 1247423019 |
| csv (Our World in Data's CO2 table, memra#1197) | Afghanistan in 1960: population, gdp, co2, co2_per_capita | 9035048, 13033254000, 0.396, 0.044 |
| prose (Moby-Dick, the control) | the landlord of the Spouter-Inn, the harpooneer Ishmael shares a bed with there, the preacher at the Whaleman's Chapel | Peter Coffin, Queequeg, Father Mapple |

The questions and their answers were committed (`receipts/questions.json`, commit 310802551b)
before any cell ran. All three JSONL values differ from the neighboring records' (seq 19:
105243156, 18148108, 1247420264; seq 21: 105263096, 18168008, 1247425794), so a value read from
a neighboring record is wrong. The source documents and their sha256 are the ones in
`research/kv-quant-repetitive-20261110/README.md`. `tools/build_requests.py` writes the requests;
`receipts/requests-manifest.jsonl` has each cut's line count, character count and content sha256.

Depth is the prompt plus the answer: about 32,768 and at most 65,535 tokens. The f32 reference
stops past 65,535 rows (memra#1200), so 131,072 does not run. Records are long in the JSONL log
(about 650 tokens), so its cuts end shorter than the others.

## Method

Per cell, on one GB10, under its own lease (`tools/answer_cell.py`):

1. **Serve.** main's `memra-server` at 3de59bbf80 (v0.141.0) serves the request twice, one boot
   and a cold prefix cache each: with the server's defaults (on this machine a single stream
   runs the MTP draft; the usage block reports the spec rounds) and with `MEMRA_SPEC_K=0`
   (plain decode, the served batch program). The prompt ids come from the server's
   `/v1/tokenize` with the request's messages; the count equals the response's
   `usage.prompt_tokens` in every cell.
2. **Answer ids.** The plain arm's answer text, tokenized by `/v1/tokenize`. `/v1/detokenize` of
   prompt ids plus answer ids gives back the rendered prompt followed by the answer text in every
   cell. `<|im_end|>` is appended when the answer stopped on it. The probe confirms the ids: the
   served batch decode's top-1 is the answer token at every answer row (below).
3. **Teacher-force.** `kvq_rows_probe --ids` reads prompt plus answer as ids. The window is the
   2,048 rows before the first answer position plus every answer position. Row i's logits choose
   token i + 1, so the first answer row is the last prompt row. Programs:
   - **a**, prefill over f32 K/V (`forward_last`'s trunk once over the whole sequence, then the
     LM head on each window row): the reference.
   - **b**, the prime over the quantized cache (`prime_cache` once, then `prime_row_logits`).
   - **c**, the served batch decode at B=1 over a cache the prime wrote up to the window, then
     stepped one token at a time through the window: the program that produced the served answer.

"Answer rows" are the rows that chose an answer token, the closing `<|im_end|>` included; the
row after the last id is left out. "In-turn rows" are the 2,048 rows before them: the end of
the document, the question and the template, inside the user turn as in memra#1197's user-turn
cell. A row differs when a program's top-1 is not the served token. The spread at a row is the
gate's `delta@ids`: the largest |x - y| over both programs' top-2 ids. Margins are top-1 minus
top-2 logits.

## Results

`SUMMARY.md` has every cell in full: the served answer, the spread tables for every program pair
and row class, and every answer row with a spread above 1 or a margin under 1, with the text
before it and both programs' runner-up. The tables below come from `SUMMARY.jsonl` through
`tools/tables.py`.

Served answers. "a differs" counts the answer rows where a's top-1 is not the served token;
likewise b and c. The local tokenizer's prompt count (`receipts/requests-manifest.jsonl`) equals
the server's in every cell.

| cell | prompt | answer tokens | answer rows | a differs | b differs | c differs | known values right | default text = plain |
|---|---|---|---|---|---|---|---|---|
| jsonl-32768 | 32325 | 74 | 75 | 0 | 0 | 0 | 3 of 3 | yes |
| jsonl-65535 | 65038 | 74 | 75 | 0 | 0 | 0 | 3 of 3 | yes |
| csv-32768 | 32317 | 66 | 67 | 0 | 0 | 0 | 4 of 4 | yes |
| csv-65535 | 65313 | 66 | 67 | 0 | 0 | 0 | 4 of 4 | yes |
| prose-32768 | 32625 | 46 | 47 | 0 | 0 | 0 | 3 of 3 | yes |
| prose-65535 | 65394 | 46 | 47 | 0 | 0 | 0 | 3 of 3 | yes |

a against c (the served program) by row class. "answer": the rows that chose an answer token;
"question": in-turn rows after the document (the closing tag, the question, the template);
"document": in-turn rows inside the document.

| cell | rows | a/c spread median | p90 | max | rows > 1 | rows > 3 | a/c flips |
|---|---|---|---|---|---|---|---|
| jsonl-32768 | answer 75 | 0.23 | 0.49 | 1.11 | 1 | 0 | 0 |
| jsonl-32768 | question 70 | 0.31 | 0.55 | 2.70 | 4 | 0 | 1 |
| jsonl-32768 | document 1978 | 1.22 | 15.80 | 36.83 | 1110 | 616 | 6 |
| jsonl-65535 | answer 75 | 0.40 | 1.01 | 2.35 | 8 | 0 | 0 |
| jsonl-65535 | question 70 | 0.19 | 0.58 | 1.81 | 2 | 0 | 1 |
| jsonl-65535 | document 1978 | 1.55 | 15.97 | 28.47 | 1137 | 724 | 0 |
| csv-32768 | answer 67 | 0.19 | 0.49 | 0.94 | 0 | 0 | 0 |
| csv-32768 | question 55 | 0.35 | 0.75 | 3.01 | 4 | 1 | 0 |
| csv-32768 | document 1993 | 0.58 | 1.42 | 24.82 | 340 | 95 | 151 |
| csv-65535 | answer 67 | 0.20 | 0.45 | 0.60 | 0 | 0 | 0 |
| csv-65535 | question 55 | 0.18 | 0.63 | 4.20 | 4 | 1 | 0 |
| csv-65535 | document 1993 | 0.22 | 0.73 | 17.90 | 147 | 40 | 53 |
| prose-32768 | answer 47 | 0.20 | 0.38 | 0.90 | 0 | 0 | 0 |
| prose-32768 | question 69 | 0.21 | 0.54 | 1.48 | 1 | 0 | 1 |
| prose-32768 | document 1979 | 0.30 | 0.82 | 25.51 | 150 | 63 | 92 |
| prose-65535 | answer 47 | 0.38 | 0.84 | 1.30 | 4 | 0 | 0 |
| prose-65535 | question 69 | 0.21 | 0.55 | 10.37 | 4 | 1 | 1 |
| prose-65535 | document 1979 | 0.22 | 0.72 | 25.92 | 147 | 62 | 53 |

a/b and b/c, answer rows against document rows:

| cell | a/b answer: median, p90, max | b/c answer: median, p90, max | a/b document: p90, max | b/c document: p90, max |
|---|---|---|---|---|
| jsonl-32768 | 0.21, 0.49, 3.00 | 0.22, 0.43, 1.89 | 14.65, 36.72 | 14.14, 32.01 |
| jsonl-65535 | 0.27, 0.90, 1.83 | 0.33, 0.81, 1.11 | 15.37, 27.90 | 15.60, 27.88 |
| csv-32768 | 0.32, 0.67, 1.19 | 0.24, 0.41, 0.89 | 1.29, 22.64 | 0.94, 23.59 |
| csv-65535 | 0.20, 0.42, 0.82 | 0.17, 0.34, 0.67 | 0.75, 16.12 | 0.60, 17.38 |
| prose-32768 | 0.23, 0.70, 1.29 | 0.22, 0.52, 0.96 | 0.72, 31.78 | 0.60, 30.18 |
| prose-65535 | 0.31, 0.73, 1.29 | 0.22, 0.37, 0.50 | 0.73, 31.02 | 0.70, 25.23 |

Margins. A value row chooses a token of a known value as it appears in the answer (a digit, a
piece of a name).

| cell | smallest answer-row margin a / b / c | answer rows with a margin under 1 | of them inside a known value | value rows | smallest margin at a value row a / b / c | largest a/c spread at a value row |
|---|---|---|---|---|---|---|
| jsonl-32768 | 0.48 / 0.52 / 0.34 | 5 | 0 | 27 | 5.32 / 5.35 / 5.40 | 0.57 |
| jsonl-65535 | 0.44 / 0.44 / 0.20 | 4 | 0 | 27 | 4.66 / 5.03 / 4.96 | 1.46 |
| csv-32768 | 0.31 / 0.38 / 0.40 | 4 | 0 | 28 | 4.47 / 4.39 / 4.44 | 0.94 |
| csv-65535 | 0.43 / 0.49 / 0.39 | 4 | 0 | 28 | 4.10 / 4.07 / 4.12 | 0.58 |
| prose-32768 | 0.28 / 0.29 / 0.11 | 2 | 1 | 10 | 0.28 / 0.29 / 0.11 | 0.73 |
| prose-65535 | 0.31 / 0.09 / 0.25 | 2 | 1 | 10 | 0.98 / 0.94 / 0.90 | 0.65 |

The prose value rows with small margins are row 28 of both prose answers: " Que" (the start of
Queequeg) against " named", margins 0.28, 0.29, 0.11 at 32,768 and 0.98, 0.94, 0.90 at 65,535.
In the JSONL answer at 65,535, `<|im_end|>` is a's runner-up at two value digits (rows 65,064
and 65,084), at a margins of 10.51 and 10.06.

Probe self-checks, every cell: `PRIME-LAST` bit-identical (the prime's hidden stack through
`prime_row_logits` against the prime's own last-row logits), and the served batch step's
device-sampled token equal to the host argmax at every window row (`stdout.log`). No cell came
near the 5 GiB memory floor (`result.json`).

## Rebuild

The probe is research code and stays off main. It is `kvq_rows_probe` and the `forward_last`
split from the archive ref `bank/kv-quant-repetitive-20261110/101c0b669c` (memra#1197; main's
`hybrid_forward.rs` has not changed since that ref's base), plus `receipts/probe-ids.patch`, which
adds `--ids`. On a GB10, from a checkout of main at 3de59bbf80:

```
git fetch origin bank/kv-quant-repetitive-20261110/101c0b669c
git checkout 101c0b669c -- crates/memra-engine/src/bin/kvq_rows_probe.rs crates/memra-engine/src/hybrid_forward.rs
git apply research/kv-quant-answer-20261111/receipts/probe-ids.patch
MEMRA_CUDA_ARCH=121a cargo build --release --locked -p memra-engine --bin kvq_rows_probe
```

`memra-server` was built from the unmodified tree first, in the same target directory
(`MEMRA_CUDA_ARCH=121a cargo build --release --locked -p memra-server --bin memra-server`). The
two binaries' sha256 are in `receipts/bins.sha256`. One cell:

```
python3 tools/answer_cell.py ROOT jsonl-32768 Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
```

with `ROOT/bins/` holding the two binaries and `ROOT/requests/` the output of
`tools/build_requests.py tokenizer.json DOCS_DIR ROOT/requests`. Then
`tools/summarize.py RUNS_DIR REQUESTS_DIR tokenizer.json OUT CELL...` and
`tools/tables.py OUT/SUMMARY.jsonl`.

## Files

- `tools/build_requests.py`: the six chat requests, the questions and their known answers.
- `tools/answer_cell.py`: one cell on the device (serve twice, ids, probe).
- `tools/summarize.py`: `SUMMARY.md` and `SUMMARY.jsonl`. One percentile everywhere:
  sorted[round((n - 1) * p)].
- `tools/tables.py`: the tables above, from `SUMMARY.jsonl`.
- `receipts/questions.json`: questions and known answers, committed before the runs.
- `receipts/requests-manifest.jsonl`: each request's cut (lines, characters, content sha256,
  local prompt token count).
- `receipts/probe-ids.patch`: the `--ids` change on top of the archived probe.
- `receipts/bins.sha256`, `receipts/model.txt`: the two binaries' hashes, and the model file's
  Hub record (revision, etag, size; the etag of an LFS file is its sha256, not recomputed here).
- `runs/<cell>/`: `cell.log` (steps and checks), `meta.json` (prompt and answer lengths, window),
  `served-default.json.gz` and `served-plain.json.gz` (the two responses with prompt and answer
  ids and the round-trip checks), `server-*.log` (the two boots, allocator growth lines
  removed), `ids.json.gz` (the teacher-forced ids), `rows.jsonl.gz` (one line per window row:
  each program's top-5 ids and logits, the next token's logit and rank, per pair the spread, the
  largest vocabulary difference and the smaller margin; its `explained` field is the gate's
  rule, which holds for every flip by construction, memra#1199), `stdout.log`/`stderr.log` (the
  probe), `result.json` and `mem.tsv` (exit, lowest MemAvailable), `pre.txt`/`post.txt` (memory
  and GPU state). Home paths are written as `~`.

The documents are not committed. `tools/build_requests.py` rebuilds the requests from the three
source files of memra#1197 (sources and sha256 in its README); the manifest's content sha256
checks the result.
