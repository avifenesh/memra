# Does the quantized KV cache move the argmax on real repetitive context? (memra#1177)

Qwen3.8-27B NVFP4+Q5_K MTP GGUF on one DGX Spark (GB10, 48 SMs), 2026-11-10. Main's engine at
031f146690, before the GB10 deep split rung lands. Research only; nothing here changes a default.

## Outcome

On raw documents, at 32,768 and 65,535 tokens, the numbers support four statements.

1. The f32 reference is not systematically closer to the document's next token than the
   programs over the quantized cache. On the JSONL log, at the 21 rows where the f32-KV prefill
   (a) and the prime over the quantized cache (b) disagree, the log's next token is a's top-1 at
   7 and b's at 8. Over all 2,047 rows per cell, top-1 accuracy differs from a's by at most 4 rows
   (32,768: a 1,956, b 1,959, c batch 1,954, c eager 1,952; 65,535: 1,978, 1,976, 1,976).
2. On machine-generated records a few rows per thousand sit where any difference between
   programs moves a wrong candidate by 10 to 20 logits. On the JSONL log, a and b disagree at 14
   rows at 32,768 and 7 at 65,535. At 7 and 3 of them a's top-2 margin is above 1 logit (up to
   11.3 and 8.5); b's margin is above 1 at 8 and 1. At one row both sides are confident the other
   way (a 11.3, b 5.2). The prime and the served decode, both over the quantized cache, disagree
   at the same kind of rows by as much (at 32,768: spread p90 1.69 against 1.77 for a/b, max 19.7
   against 19.6). The f32 program is not the odd one out. On the CSV table and the prose control
   no flip has a margin above 1.6 on either side, and no spread at a flip exceeds 2.6.
3. No control separates the cache format from the kernel that reads it. a and b differ in both,
   and there is no pair of f32-KV programs to measure the program-to-program spread without the
   quantized cache. So these numbers do not show that the cache format is harmless; they show
   that its effect is not larger than the difference between two programs over the same cache.
4. The answer positions of a served chat turn are not measured. One cell put the JSONL log
   inside a user turn. Those rows are inside the turn, positions the model never generates from,
   and the model's top-1 there is `<|im_end|>` at 2,047 or 2,048 of 2,048 rows. But the spreads
   there are an order larger than on the raw log (a/b p90 13.1, max 28.7, 606 rows above 3,
   against 1.77, 19.6 and 91), and the served decode flips one row at an f32 margin of 15.8
   (spread 24.0), the largest real-margin flip in these receipts. Whether that reaches the
   logits after the turn, where the model answers, needs a cell that measures those positions.

Still open: 131,072 tokens (the f32 reference does not run there, see below), the answer
positions after a chat turn, and an f32-cache control.

`argmax-margin-probe` would call every flip here "margin-explained" (spread above the smaller
top-2 margin). That rule cannot fail: for any flip, the spread over both programs' top-2 ids is at
least the mean of the two margins, so it always exceeds the smaller one. Over all 1,278 distinct
flips in these receipts the smallest spread to smaller-margin ratio is 1.036. The count is not
evidence and is not reported; memra#1199 has the proof. The measure here is the margin itself:
flips where either side's top-2 margin is above 1 and above 2 logits.

## Question

memra#1175 found that on a synthetic repetitive document the f32-KV prefill and the programs
over the quantized cache (q8_0 K, q5_1 V) disagree by 5 to 14 logits at several rows, while real
prose at the same depth stays at 0.05 to 1.06. Repetitive context is a real agent workload: tool
output, logs, tables, JSON records. Does the quantized cache change the argmax there, and at what
margins?

## Documents

Two real repetitive documents and one prose control. None was written for this test; each prompt
is the start of the document, cut on a token boundary.

| name | what it is | source | sha256 of the source file |
|---|---|---|---|
| jsonl | a host sampler's JSONL log: one record per second with disk, memory and CPU counters (about 1,100 characters a record, the same keys on every line) | `research/spill-f-20260919/box27/background-diskstats.jsonl` in this repo at 031f146690 | `bd69820e7ad8391617a7686136738689099eddca38a05636209b40fa954d006d` |
| csv | Our World in Data's CO2 table: one row per country and year, 79 columns, nearly all numeric and many of them empty | `owid-co2-data.csv`, github.com/owid/co2-data at commit e830f0992c29d9d587330ac6c076a149315bc5a1 | `300fb68fe6b0d514ed7f5cf773df77d7e24a495438d84b76e2f39eba1584a0f7` |
| prose | Herman Melville, Moby-Dick, Project Gutenberg eBook #2701 (CRLF line ends turned into LF) | gutenberg.org/cache/epub/2701/pg2701.txt, fetched 2026-11-10 | `b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d` |

`tools/cut_prompts.py` cuts each document to an exact token count with the model's tokenizer
(sha256 in `receipts/tokenizer.sha256`): the first N ids are decoded, and the text must encode
back to exactly those ids. Every cut round-tripped at its target. `prompts-manifest.jsonl` lists
every prompt that ran, with its token count, its sha256 and the sha256 of its ids. The probe's
own tokenizer (the GGUF's) produced the same ids at every probed row (summarize.py checks it).

Most prompts are raw documents. Two are user turns (`--head user`: `<|im_start|>user\n` + the
document): `jsonl_user-32768` (the cell `jsonl_user-32768-batch`) and `prose_user-2048` (the
reproduction runs). Both ran before the raw prompts existed, from the paths the raw prompts took
later: `result.json` of those runs names `prompts/jsonl-32768.txt` and `prompts/prose-2048.txt`.
The files were renamed after the runs. The probe prints an FNV-64 hash of the ids it encoded:
`8b20e675016e7d33` in `jsonl_user-32768-batch/stdout.log` and `b13bd0794ed5ea23` in the
`repro-*` logs, which are the hashes of the user-turn prompts. The raw prompts hash to
`00bcd540a106b376` (JSONL, in `jsonl-32768-batch/stdout.log`) and `cc0f5d6ea9b276fa` (prose,
2,048 tokens, not run). The manifest's user-turn rows carry `path_when_run` and `ids_fnv64`.

## Method

Model: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, loaded without the MTP head. `receipts/model.txt` is the
Hub's download record for the file (revision bf4ce72174a7f1936849d23efec421a75f212f40, etag
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`, which for an LFS file is its
sha256; not recomputed here) and the file size. Greedy, teacher-forced on the prompt: row i's
logits predict token i + 1.

Three programs at every row of the last 2,048 rows of each prompt:

- **a, prefill over f32 K/V**: `forward_last`'s trunk once over the whole prompt, then the LM
  head on each window row.
- **b, the prefill program over the quantized cache**: `prime_cache` once over the whole prompt
  (4,096-row chunks; every chunk, the first included, attends over the q8_0/q5_1 cache), then
  `prime_row_logits` on each window row.
- **c, decode over the quantized cache**: the prompt up to the window primed with `prime_cache`,
  then each window token stepped one at a time. This is the served shape: a prompt is primed and
  generation decodes over the cache the prime wrote. Two decode programs: `batch`, the served
  batched program at B=1 (the call `argmax-margin-probe --decode batch` makes, which serves this
  model's plain decode), and `eager`, `decode_step`.

a against b changes the cache format and the kernel that reads it: same prefill trunk, f32 K/V
attended directly against the quantized cache attended through a bf16 dequant workspace by the
chunked prime. b against c is two programs over the same quantized prefix.

The probe is `kvq_rows_probe` (`crates/memra-engine/src/bin/kvq_rows_probe.rs`). It is not on
main: the probe and its engine change live on the archive ref
`bank/kv-quant-repetitive-20261110/101c0b669c` (commit 101c0b669c, on main 031f146690).
It needs one engine change, on that ref only: `forward_last` is split into its trunk
(`forward_last_normed`) and its head, and `forward_normed_rows` returns the trunk's normed rows
of the window. `forward_last` runs the same calls in the same order.

A row is a flip when two programs' top-1 ids differ. The spread at a row is the gate's
`delta@ids`: the largest |x - y| over both programs' top-2 ids. The tables count flips where each
side's top-2 margin is above 1 and above 2 logits, and say which side holds the prompt's next
token.

### The probe reproduces main's probe

On one 2,048-token prompt (`prose_user-2048`, the first 2,048 tokens of the prose document in a
user turn; provenance above), main's `argmax-margin-probe` (built at 6cffb35360, sha256
`9469da6bf67d1085a31e73c09126d82b4765f301e84e6c319a3c8aa71bad2be1`) and `kvq_rows_probe` (sha256
`cc17085cfcca0bb4741b9b46a3eb6c2240883375761dd120d61210626b04db68`, built from 101c0b669c)
agree bit for bit (`runs/repro-*`, hashes in `runs/repro-bins.sha256`):

- a: the prefill margins of the four window rows equal the probe's `margin_p` (1.260871887e-1,
  4.767938614e0, 8.367450714e0, 4.276718140e0), and the top ids match.
- c, stepped tokenwise from position 0 as the probe does: the eager and batch decode margins
  equal the probe's `margin_d` at all four rows, `delta@ids` too, including the probe's one flip
  (row 2,044: prefill 0.126 against decode 0.180 eager and 0.090 batch).
- b at the last row equals the probe's `PRIME-GATE` `bp_margin` (4.493068695e0).

At depth, every cell prints `PRIME-LAST` (row T - 1 of the prime's hidden stack through
`prime_row_logits` against the prime's own logits) and some print `CHECK-A`/`CHECK-B`
(`forward_last` and a fresh `prime_cache` over the prefix ending at one window row, against that
row of the whole-prompt pass). All were bit-identical (`differing=0`) at rows 2,045, 31,743 (all
three documents) and 64,511 (JSONL).

## Results

Depths: 32,768 and 65,535 tokens. 2,048 rows per cell, the last 2,048 positions of the prompt.

a is the f32-KV prefill, b the prefill over the quantized cache, c the decode over the quantized
cache. x and y are the pair's two programs in order. "x margin > 1" counts the flips where x's
top-2 margin is above 1 logit. "x right" and "y right" count the flips where that program's top-1
is the prompt's next token. Spread is the gate's `delta@ids`; margins are top-2 margins in
logits. `SUMMARY.md` has every column per cell, the largest vocabulary difference, a breakdown by
row class (record boundary, field boundary, inside a value) and every flip.

| document | depth | decode | rows | pair | flips | x margin > 1 | > 2 | y margin > 1 | > 2 | x right | y right | spread median | p90 | max | rows spread > 3 | top-1 = next token (x / y) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| csv | 32768 | batch | 2048 | ab | 51 | 0 | 0 | 1 | 0 | 8 | 9 | 0.18 | 0.44 | 2.30 | 0 | 1698 / 1699 |
| csv | 32768 | batch | 2048 | ac | 48 | 0 | 0 | 0 | 0 | 5 | 11 | 0.18 | 0.45 | 3.29 | 1 | 1698 / 1704 |
| csv | 32768 | batch | 2048 | bc | 44 | 0 | 0 | 0 | 0 | 3 | 8 | 0.14 | 0.34 | 5.12 | 1 | 1699 / 1704 |
| csv | 32768 | eager | 2048 | ac | 56 | 0 | 0 | 0 | 0 | 7 | 10 | 0.18 | 0.44 | 2.49 | 0 | 1698 / 1701 |
| csv | 32768 | eager | 2048 | bc | 48 | 1 | 0 | 0 | 0 | 7 | 9 | 0.14 | 0.33 | 2.26 | 0 | 1699 / 1701 |
| csv | 65535 | batch | 2048 | ab | 33 | 0 | 0 | 0 | 0 | 3 | 10 | 0.16 | 0.37 | 3.77 | 2 | 1774 / 1781 |
| csv | 65535 | batch | 2048 | ac | 34 | 0 | 0 | 1 | 0 | 5 | 8 | 0.16 | 0.37 | 6.69 | 3 | 1774 / 1777 |
| csv | 65535 | batch | 2048 | bc | 30 | 0 | 0 | 0 | 0 | 6 | 2 | 0.13 | 0.31 | 8.02 | 3 | 1781 / 1777 |
| jsonl | 32768 | batch | 2048 | ab | 14 | 7 | 4 | 8 | 7 | 4 | 7 | 0.29 | 1.77 | 19.59 | 91 | 1956 / 1959 |
| jsonl | 32768 | batch | 2048 | ac | 19 | 9 | 5 | 7 | 7 | 8 | 6 | 0.33 | 2.00 | 19.56 | 113 | 1956 / 1954 |
| jsonl | 32768 | batch | 2048 | bc | 8 | 5 | 5 | 3 | 3 | 6 | 1 | 0.17 | 1.69 | 19.68 | 107 | 1959 / 1954 |
| jsonl | 32768 | eager | 2048 | ac | 18 | 11 | 7 | 9 | 8 | 8 | 4 | 0.33 | 1.77 | 20.59 | 100 | 1956 / 1952 |
| jsonl | 32768 | eager | 2048 | bc | 17 | 12 | 11 | 8 | 7 | 11 | 4 | 0.18 | 1.78 | 19.05 | 102 | 1959 / 1952 |
| jsonl | 65535 | batch | 2048 | ab | 7 | 3 | 2 | 1 | 0 | 3 | 1 | 0.24 | 1.22 | 11.28 | 54 | 1978 / 1976 |
| jsonl | 65535 | batch | 2048 | ac | 9 | 3 | 2 | 3 | 2 | 4 | 2 | 0.29 | 1.28 | 22.41 | 59 | 1978 / 1976 |
| jsonl | 65535 | batch | 2048 | bc | 7 | 2 | 1 | 5 | 4 | 3 | 3 | 0.17 | 1.10 | 21.03 | 57 | 1976 / 1976 |
| jsonl_user | 32768 | batch | 2048 | ab | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.35 | 13.13 | 28.65 | 606 | 0 / 0 |
| jsonl_user | 32768 | batch | 2048 | ac | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1.45 | 14.23 | 28.31 | 648 | 0 / 0 |
| jsonl_user | 32768 | batch | 2048 | bc | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.80 | 12.93 | 31.37 | 548 | 0 / 0 |
| prose | 32768 | batch | 2048 | ab | 133 | 1 | 0 | 1 | 0 | 19 | 31 | 0.21 | 0.51 | 2.05 | 0 | 1042 / 1054 |
| prose | 32768 | batch | 2048 | ac | 120 | 0 | 0 | 3 | 0 | 20 | 29 | 0.22 | 0.50 | 2.31 | 0 | 1042 / 1051 |
| prose | 32768 | batch | 2048 | bc | 106 | 1 | 0 | 2 | 0 | 19 | 16 | 0.22 | 0.51 | 1.86 | 0 | 1054 / 1051 |
| prose | 32768 | eager | 2048 | ac | 128 | 0 | 0 | 1 | 0 | 23 | 29 | 0.21 | 0.50 | 1.90 | 0 | 1042 / 1048 |
| prose | 32768 | eager | 2048 | bc | 112 | 1 | 0 | 1 | 0 | 22 | 16 | 0.22 | 0.50 | 1.70 | 0 | 1054 / 1048 |
| prose | 65535 | batch | 2048 | ab | 73 | 0 | 0 | 1 | 0 | 13 | 16 | 0.14 | 0.35 | 5.20 | 2 | 988 / 991 |
| prose | 65535 | batch | 2048 | ac | 82 | 0 | 0 | 0 | 0 | 16 | 12 | 0.14 | 0.36 | 14.68 | 4 | 988 / 984 |
| prose | 65535 | batch | 2048 | bc | 79 | 1 | 0 | 1 | 0 | 17 | 10 | 0.14 | 0.36 | 12.40 | 2 | 991 / 984 |

| document | depth | margin a p10 | median | margin b p10 | median | margin c p10 | median |
|---|---|---|---|---|---|---|---|
| csv | 32768 | 0.08 | 8.90 | 0.08 | 8.86 | 0.08 | 8.93 |
| csv | 65535 | 0.26 | 8.51 | 0.24 | 8.59 | 0.24 | 8.57 |
| jsonl | 32768 | 3.98 | 10.62 | 3.96 | 10.43 | 3.80 | 10.43 |
| jsonl | 65535 | 5.80 | 10.95 | 5.76 | 10.92 | 5.73 | 10.94 |
| jsonl_user | 32768 | 5.11 | 13.72 | 5.22 | 13.98 | 5.23 | 13.95 |
| prose | 32768 | 0.16 | 1.13 | 0.16 | 1.14 | 0.17 | 1.17 |
| prose | 65535 | 0.18 | 1.15 | 0.17 | 1.15 | 0.17 | 1.17 |

The batch and eager cells run the same a and b; their rows are identical (2,048 of 2,048 on the
JSONL log at 32,768), so the eager rows above show only ac and bc. `jsonl_user` is the user-turn
cell (Outcome, item 4); its next token is never top-1 because `<|im_end|>` is.

Prefill against decode on the controls: a against the served decode reaches 3.29 (CSV) and 2.31
(prose) at 32,768, and 6.69 and 14.68 at 65,535 (prose row 65,440). None of those rows flips: the
smaller top-2 margin at every control row with an a/c spread above 3 is 8.7 or more.

### Where the large spreads are

On the JSONL log the large spreads mostly involve `\n\n` (id 271), a blank line in the middle
of a record. The log contains no blank line, so it is never the right answer, and each program
ranks it first at some rows. `\n\n` is among the four contending ids at most rows, though: 1,417
of 2,048 (69%) at 32,768 and 1,438 (70%) at 65,535. Among the rows with an a/b spread above 3 the
share is 78 of 91 (86%) and 45 of 54 (83%), a modest enrichment. Over the rows where `\n\n` does
not contend (631 and 610), the JSONL spread comes close to the other documents' (p90 0.71 to 0.74
against 0.35 to 0.51).

| document | depth | rows ab > 3 | with \n\n contending | all rows with \n\n contending | rows bc > 3 | both | rows without \n\n | ab median there | p90 |
|---|---|---|---|---|---|---|---|---|---|
| jsonl | 32768 | 91 | 78 | 1417 | 107 | 44 | 631 | 0.15 | 0.74 |
| jsonl | 65535 | 54 | 45 | 1438 | 57 | 22 | 610 | 0.16 | 0.71 |
| csv | 32768 | 0 | 0 | 0 | 1 | 0 | 2048 | 0.18 | 0.44 |
| csv | 65535 | 2 | 0 | 0 | 3 | 1 | 2048 | 0.16 | 0.37 |
| prose | 32768 | 0 | 0 | 37 | 0 | 0 | 2011 | 0.22 | 0.51 |
| prose | 65535 | 2 | 0 | 38 | 2 | 0 | 2010 | 0.14 | 0.35 |

"rows bc > 3" counts rows where the two quantized programs (prime and served decode) differ by
more than 3. On the JSONL log they are as many as the a/b rows, and about half are the same rows.
Across all 21 a/b flips on the JSONL log the spread runs from 0.01 to 18.1; at the 10 where a's
margin is above 1 it runs from 6.4 to 18.1.

### Rows where a flip has a real f32 margin

Every JSONL row where a flip against the f32 reference (a against b, a against c batch, a against
c eager) comes with an f32 margin above 1. Each program column is its top token and its top-2
margin. No such row exists on the CSV table; on the prose control one row qualifies (row 32,598 at
32,768: f32 ` hair` at 1.02, prime ` sentences` at 1.43, spread 1.51, true next ` arms`).

| depth | row | text before | true next | a | b | c batch | c eager | spread ab | spread bc |
|---|---|---|---|---|---|---|---|---|---|
| 32768 | 31193 | `...6903238, \"reads_merged\": 12` | `5` | `5` 11.00 | `5` 10.46 | `5` 1.24 | `4` 1.75 | 0.3 | 16.8 |
| 32768 | 31214 | `...read_sectors\": 2079206497` | `8` | `\n\n` 2.03 | `8` 9.94 | `8` 9.77 | `8` 10.58 | 8.5 | 0.5 |
| 32768 | 31787 | `...3903002, \"in_flight\": 0,` | ` \"` | ` \"` 5.74 | ` \"` 2.64 | space and backtick 0.99 | ` \"` 6.52 | 13.5 | 9.8 |
| 32768 | 31824 | `...4666837}, \"nvme0n1p` | `3` | `\n\n` 1.18 | `3` 7.98 | `3` 7.96 | `3` 7.23 | 12.3 | 0.4 |
| 32768 | 31842 | `...reads\": 156903238, \"reads` | `_merged` | `_merged` 11.34 | `\n\n` 5.24 | `_merged` 7.55 | `_merged` 6.89 | 18.1 | 13.2 |
| 32768 | 31847 | `...6903238, \"reads_merged\": 12` | `5` | `1` 1.61 | `5` 11.88 | `2` 5.20 | `2` 6.38 | 10.5 | 9.0 |
| 32768 | 31861 | `...25052, \"read_sectors\": 207` | `9` | `\n\n` 1.37 | `9` 5.45 | `9` 8.07 | `9` 10.39 | 6.4 | 3.2 |
| 32768 | 31936 | `...163866, \"write_ms\": 340` | `8` | `8` 9.94 | `8` 10.16 | `8` 3.30 | `\n\n` 3.41 | 0.4 | 8.0 |
| 32768 | 31952 | `...778295, \"in_flight\": 0, \"` | `io` | `io` 10.19 | `io` 10.00 | `io` 10.13 | `in` 0.47 | 0.3 | 0.2 |
| 32768 | 31967 | `...io_ms\": 3177486, \"weighted_io` | `_ms` | `_ms` 1.93 | `_ms` 2.67 | `\n\n` 3.64 | `\n\n` 4.28 | 1.9 | 8.3 |
| 32768 | 31986 | `...350463334}}, \"missing_devices\": [],` | ` \"` | ` \"` 8.91 | ` \"` 9.43 | ` and` 0.33 | `\n\n` 2.45 | 2.2 | 14.5 |
| 32768 | 32337 | `...962134, \"reads_merged\": 150` | `8` | `8` 8.48 | `8` 7.47 | `8` 6.60 | `\n\n` 4.21 | 1.6 | 0.7 |
| 32768 | 32491 | `...1p3\": {\"reads\": 1569032` | `3` | `3` 4.74 | `\n\n` 3.04 | `\n\n` 2.39 | `3` 0.91 | 7.6 | 10.0 |
| 32768 | 32572 | `... 939089, \"write_sectors\": 3` | `0` | `0` 10.46 | `0` 8.58 | `0` 4.73 | `\n\n` 0.65 | 7.6 | 3.4 |
| 32768 | 32625 | `...3177486, \"weighted_io_ms\": 2` | `3` | `3` 4.39 | `\n\n` 1.51 | `\n\n` 2.44 | `3` 2.75 | 12.6 | 6.0 |
| 65535 | 63897 | `...p3\": {\"reads\": 15690323` | `8` | `8` 5.80 | `8` 0.32 | `2` 2.70 | not run | 3.1 | 2.5 |
| 65535 | 63909 | `...3238, \"reads_merged\": 12505` | `2` | `2` 4.17 | `2` 2.65 | `\n\n` 1.22 | not run | 0.8 | 21.0 |
| 65535 | 63914 | `... \"reads_merged\": 125052, \"read_se` | `ctors` | `ctors` 8.46 | `\n\n` 0.41 | `ctors` 6.01 | not run | 9.0 | 7.7 |
| 65535 | 64012 | `...78302, \"in_flight\": 0, \"io` | `_ms` | `_ms` 5.53 | `\n\n` 1.29 | `_ms` 2.50 | not run | 8.3 | 5.7 |
| 65535 | 64026 | `...io_ms\": 3177486, \"weighted_io` | `_ms` | `\n\n` 1.81 | `伴随` 0.65 | `_ms` 2.95 | not run | 11.3 | 7.8 |

These 20 rows are picked by the f32 side's margin, so they favor the f32 reference. Even so it has
the wrong token at 5 of them (4 times `\n\n`), the prime at 6 (5), the served decode at 8 (4) and
the eager decode at 8 of 15 (5). At row 31,193 the f32 prefill and the prime agree (spread 0.3)
and the two decode programs move from the prime by 16.8 (batch) and 15.2 (eager).

The user-turn cell's real-margin flip, for comparison: row 31,157 (`"weighted_io_ms": 30`, true
next `946`): a `<|im_end|>` at 15.80, b `<|im_end|>` at 8.45, served decode `Me` at 0.78; spread
a/c 24.0, b/c 8.2.

## What did not run

- **131,072 tokens.** The f32 reference cannot run there. `forward_last` at 131,072 tokens stops
  in phase a, before it prints anything, with `Error: DriverError(CUDA_ERROR_INVALID_VALUE,
  "invalid argument")` (`runs/fail-jsonl-131072-batch`, and the same error again in
  `runs/diag-131072`). Reading the code, the embedding gather launches one grid row per token
  (`embed_gather_device_td`, `grid_dim.y = t`), and CUDA caps grid y at 65,535; the failing launch
  was not traced. Past that, the dense FFN's `T x n_ff` activation (n_ff 17,408) passes 2^31
  elements from 123,362 tokens on (not tested). MemAvailable was 97.6 GiB when it failed (the
  65,535-token cells bottomed at 70.2 GiB). 65,535 is the deepest prompt `forward_last` ran, so
  the deep cells ran there. The prime and decode run at 131k and beyond (they chunk), so a b/c
  cell at 131,072 would extend item 2 of the outcome; it was not run, and it would still lack the
  f32 side. The issue's 131k depth stays open.
- **Decode stepped from position 0** (memra#1175's tokenwise program over a cache it wrote
  itself) ran only on the 2,048-token reproduction. At depth, c decodes the window over a cache
  the prime wrote, which is how a served request reaches decode.
- **The answer positions after a chat turn** and **an f32-cache control** (Outcome, items 3 and 4).

## Limits

- One model, one device, one sample of each document type, greedy argmax only. Teacher-forced
  next-token rows inside a document stand in for generation over that document.
- b differs from a in the cache format and in the attention kernel that reads it (the prime
  attends over a bf16 dequant of the quantized cache; the f32 path attends over the batch's f32
  K/V), and the prime is a chunked program. There is no f32-cache variant of the prime to split
  them.
- The window is the last 2,048 rows of each prompt. Rows are not independent: one region of a
  document can hold several sensitive rows.
- Numbers are main's engine before the GB10 deep split rung lands (memra#1175). The split rung
  changes decode attention rounding at 16,384 keys and past; the c columns may move with it.

## Files

- `tools/cut_prompts.py`: cuts the documents to exact token counts.
- `tools/summarize.py`: per-cell statistics, `SUMMARY.md` and `SUMMARY.jsonl`.
- `tools/tables.py`: the condensed tables above, from `SUMMARY.jsonl`.
- `tools/rows_table.py`: the two row tables (where the large spreads are, and the real-margin
  rows). It and summarize.py use one percentile: sorted[round((n - 1) * p)].
- `prompts-manifest.jsonl`: every prompt that ran, with its token count and hashes.
- `receipts/model.txt`, `receipts/tokenizer.sha256`: the model's Hub record and the tokenizer hash.
- `runs/<cell>/`: `stdout.log` (phases, timings, the prompt's id hash, `PRIME-LAST`, `CHECK-*`),
  `stderr.log` (the engine's allocator growth lines removed), `rows.jsonl.gz` (one line per row:
  each program's top-5 ids and logits, the next token's logit and rank, per pair the spread, the
  largest vocabulary difference and the smaller margin; its `explained` field is the gate's rule
  and holds for every flip, see memra#1199), `result.json` (command, exit, wall time, lowest
  MemAvailable), `mem.tsv` (MemAvailable every second), `pre.txt`/`post.txt` (memory and GPU
  state). Home paths are written as `~`. `runs/diag-131072` has only its logs.
- On the archive ref `bank/kv-quant-repetitive-20261110/101c0b669c`, not on main:
  `crates/memra-engine/src/bin/kvq_rows_probe.rs` and the `forward_last` split in
  `crates/memra-engine/src/hybrid_forward.rs`.

Reproduce one cell on a GB10 (about 6 minutes at 32,768 tokens, 9 at 65,535), from a checkout of
that ref:

```
cargo build --release -p memra-engine --bin kvq_rows_probe
target/release/kvq_rows_probe Qwen3.8-27B-NVFP4-Q5K-mtp.gguf jsonl-32768.txt \
  --rows 2048 --decode batch --decode-from prime --check-row 31743 --out rows.jsonl
```
