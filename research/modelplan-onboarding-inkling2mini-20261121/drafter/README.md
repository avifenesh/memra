# Lane S: the DSpark drafter of Inkling 2 Mini

Status: open (draft PR #1398, stacked on #1397). The drafter of `mtp.safetensors` (54 BF16 tensors) runs
on CPU (f64 reference) and on the GPU route; speculative decode verified by the trunk equals greedy decode
without drafting, token for token. No native qualification is claimed; nothing here is timed.

## Program and sources

The plan's `Inkling2DrafterPlan` (`crates/memra-gguf/src/model_plan/inkling2.rs`) carries the whole program,
read from the vLLM image's drafter (lane note `NOTES.md` section 1): block of 8 (anchor = the trunk's normed
embedding of the last committed token, 7 learned mask rows, proposals at rows 1..7), fuse softmax per draft
layer plus one row for the head over the 4 taps (inputs of slots 5, 29, 47 and the final norm), context K/V
with per-head K norm and RoPE 1e6, bidirectional block attention (local layers 0 and 2: the block plus 1024
prior positions), dense SwiGLU 9216, and the rnn_hidden head (state 256; inputs [state | markov_w1[prev] |
row hidden | feature]; chunks [gate | candidate | out]; latent = tanh(out); folded through the shared output
head; softcap 30 after base + bias).

## Oracle and reference

[`dspark_oracle.py`](dspark_oracle.py) is a float64 torch port of the image's drafter. Its docstring lists
what is ported line by line (the layer, MLP, decoder, Markov head and speculator loop), what is replaced by
its math (the triton kernels, the paged cache, the block attention kernel, the RoPE cache) and what is not
ported (TP, graphs, Gumbel sampling). It generates the tiny fixture (hidden 64, window 5, vocab 40) and pins
its outputs: [`drafter-tiny-run.json`](drafter-tiny-run.json) has the seed and both sha256.

- CPU (`memra-reference::inkling2_drafter`, f64): context K/V, block rows, feature, greedy and teacher-forced
  logits within 1.7e-14 relative of the oracle on 1, 4 and 13 committed positions. Red arms fail: no local
  window, swapped fuse rows, a window one wider.
- GPU (`ink2-spec --tiny-gate`, f32 activations): all three scenarios PASS, logits within 5.4e-6, proposals equal,
  commits split in threes ([`fixed-head/tiny.log`](fixed-head/tiny.log)).
- Checkpoint crosscheck (`--crosscheck`): GPU drafter vs the f64 reference on the first block of a 64-token
  prompt, taps from the route's capture: 7 of 7 proposals equal, logits within 6.0e-6.

## Verify: one numeric program

A verify call runs the anchor plus K proposals as one route call; the route computes each row from that row
alone. On a partial accept the route goes back to the committed convolution buffers (the #1397 parity
double-buffer: nothing is copied) and the position, and runs the accepted rows again. Rejected rows' K/V sit
past the committed positions and are rewritten before any read (sliding rings hold window - 1 + chunk rows).

run-spec self-consistency on the fixed #1397 head (`85992185e9`, binary sha256 `4db63d73...`), 4 prompts,
64 tokens, K = 1..7 proposals per verify (2..8 rows): 28 of 28 runs equal plain greedy decode
([`fixed-head/gate.jsonl`](fixed-head/gate.jsonl)). The 12 greedy acceptance runs below also equal plain
decode (12 of 12). Red arm `--red-no-restore` (keep the rejected rows' conv history): chat-email diverges,
the gate exits 1 ([`fixed-head/red.log`](fixed-head/red.log)). The same gate on base `0c88a56ca9` (before the
review fixes) gave the same counts ([`runspec-gate.jsonl`](runspec-gate.jsonl)).

## Acceptance

[`prompts.py`](prompts.py) renders 12 prompts written for this lane (4 chat, 4 code, 4 agentic with tool
definitions; no public benchmark) with the checkpoint's chat template. 128 tokens or end of sampling, K = 7,
declared numeric class, the drafter's f32 program.

| Sampling | kind | verify steps | accepted | acceptance length | tokens per verify |
|---|---|---:|---:|---:|---:|
| temperature 0 | chat | 236 | 275 | 1.165 | 2.165 |
| temperature 0 | code | 144 | 367 | 2.549 | 3.549 |
| temperature 0 | agentic | 54 | 251 | 4.648 | 5.648 |
| temperature 0 | all | 434 | 893 | 2.058 | 3.058 |
| probabilistic, T 1.0, seed 1 | all | 477 | 890 | 1.866 | 2.866 |
| probabilistic, T 1.0, seed 2 | all | 421 | 878 | 2.086 | 3.086 |

Per kind at T 1.0 (seed 1 / seed 2): chat 1.346 / 1.377, code 1.661 / 2.566, agentic 4.103 / 3.413.
Accepted-count histogram over verify calls (0..7): greedy [140, 107, 54, 34, 22, 21, 9, 47]; T 1.0 seed 1
[159, 108, 84, 46, 15, 13, 13, 39]; seed 2 [141, 86, 56, 35, 32, 17, 16, 38]. Acceptance length is accepted
proposals per verify call; tokens per verify adds the trunk's own token. Raw lines: `fixed-head/*.jsonl`.

Scope: one checkpoint, one card, 12 prompts, two seeds; chat outputs open with the model's thinking channel.

## Open

- Cost of a partial accept: one extra route call over the accepted rows (replay). Committing the verify's own
  conv history at row n would remove it; that needs a per-row history from the route (coordinate with lane P).
- The image's serving class rounds the drafter to bf16 at many points; this lane runs f32 (acceptance only).
  An A/B of the two classes on acceptance is not run.
- No timed numbers: decode speed belongs with the perf lane's kernels.
