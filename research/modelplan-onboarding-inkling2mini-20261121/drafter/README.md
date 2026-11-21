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

## Review round (#1398) and the acceptance checks, on the #1397 head `d789cb6f1c`

Branch head `70802efd76`, binary sha256 in [`review-head/binary.sha256`](review-head/binary.sha256); raw lines
in [`review-head/`](review-head/). The earlier sections are on the older bases they name.

### Gates

| Gate | Result |
|---|---|
| tiny gate, commits of 16 and of 3 rows (local rings of 8 wrap at 13 positions) | 6/6 PASS |
| run-spec K = 1..7, 4 prompts, 64 tokens, `--check-plain` | 28/28 equal plain greedy |
| `--red-no-restore` | diverges, exit 1 |
| a 3273-token prompt (route context 4096, past every 2048-row ring), 64 tokens | equal plain greedy |
| context end: 103-token prompt, 40 tokens, context 143 (verifies shrink, then plain steps) | equal plain greedy |
| 12-prompt greedy acceptance, declared and float classes | 12/12 and 12/12 equal plain greedy |
| CPU: ratio-test Monte Carlo (400k chains) and its red arm, `draw`, plan-field refusals, six drafter red arms | pass |
| a clean `git archive` checkout: `cargo test -p memra-reference --lib inkling2_drafter`, `cargo check -p memra-engine --bin ink2-spec` | pass |

### Which class, and whether the implementation loses acceptance

The earlier acceptance runs used the declared class (the default; the binary prints
`inkling2-gpu-f32-stream/DeclaredActivationQuant`). Both classes now, K = 7, 12 prompts, temperature 0:

| Target class | chat | code | agentic | all: accepted / verify | tokens / verify |
|---|---:|---:|---:|---:|---:|
| declared (act-quant, default) | 1.184 | 2.469 | 4.673 | 2.057 | 3.057 |
| float (bring-up) | 1.302 | 2.446 | 4.509 | 2.126 | 3.126 |

The oracle pair: `ink2-spec --per-position` dumps the target's greedy sequence, its drafter taps and the GPU
drafter's 7 proposals at every generated position. [`oracle_taps.py`](oracle_taps.py) runs lane E's oracle trunk
(PR #51050 code; `exact` for float, the `actq` simulation for declared; f32) teacher-forced along the same
sequence with its own hooks for the taps (slot 5, 29, 47 inputs and the final norm) and its own argmax.
[`accept_eval.py`](accept_eval.py) runs the torch-port drafter (f64, fold cached; it reproduces the pinned
tiny oracle exactly) on either set of taps. [`accept_summary.py`](accept_summary.py) simulates a K = 7 greedy
run from the per-position proposals; on the GPU drafter the simulation reproduces the real runs (float 908
accepted over 427 verifies in both; declared 896 / 436 against 897 / 436).

| Arm (drafter / taps / target) | float: all | float: chat | declared: all | declared: chat |
|---|---:|---:|---:|---:|
| GPU drafter / route taps / route | 2.126 | 1.302 | 2.055 | 1.184 |
| torch f64 / route taps / route | 2.126 | 1.302 | 2.055 | 1.184 |
| torch f64 / oracle taps / oracle argmax (the oracle pair) | 2.126 | 1.302 | 1.927 | 1.044 |
| torch bf16 points / route taps / route | 2.119 | | 2.041 | 1.156 |
| torch bf16 points / oracle taps / oracle argmax | 2.112 | 1.291 | 1.915 | 1.036 |

- Float: the oracle argmax equals the route's greedy on 1340 of 1340 generated positions, and the oracle pair
  accepts exactly what the GPU accepts (908 / 427). The implementation loses no acceptance.
- Declared: the oracle is lane E's CPU simulation of the act-quant class, not the bit-exact program; its argmax
  equals the route's on 1277 of 1337 positions, and the oracle pair accepts less (1.927) than the GPU (2.055).
  On the route's own sequence the torch-port drafter on the oracle's taps gives 2.027.
- Chat acceptance is the drafter's, not ours: the oracle pair gives the same 1.30 (float).

### Taps, checked against an independent trunk

[`compare_taps.py`](compare_taps.py), the route's capture against the oracle trunk's own hooks, relative L2 over every position:

| Tap | float class | declared class | declared, red arm `--red-tap-slot 0:6` |
|---|---:|---:|---:|
| slot 5 input | 2.7e-7 | 2.7e-2 | 8.3e-1 |
| slot 29 input | 7.0e-7 | 1.6e-1 | 1.6e-1 |
| slot 47 input | 5.1e-4 | 2.1e-1 | 2.1e-1 |
| final norm | 6.1e-4 | 2.2e-1 | 2.2e-1 |

The tap points are correct (float: the stream on entry to each slot to 1e-6, the later taps to 6e-4, which is
the float route against the f32 oracle). The red arm is caught by the tap check (a 30x error on its tap). On
acceptance alone it costs much less, 2.055 to 1.938 (declared), so acceptance alone would be a weak gate. The
declared-class differences are the act-quant noise between the route and the simulation, which grows with
depth.

### bf16 rounding points (the image's serving numerics)

`accept_eval.py --arms bf16pts` rounds the drafter where the image does (taps, every linear output, each norm,
q/k norm then RoPE, silu*up, the residual stream, base logits, the bf16 fold and bias, the softcap of
bf16(base + bias), the gate's per-op rounding). It accepts slightly less in every comparison: float 2.119
against 2.126, declared 2.041 against 2.055 (route taps). Nothing shows the bf16 points to be part of the
program the drafter was trained to. The default stays the f32 program; the arm stays in the research script
only.

### Chat and the template boundary

[`boundary_probe.py`](boundary_probe.py): no boundary defect. The first positions after the prompt's
`<|end_message|>` accept normally: `<|message_model|>` then `<|content_thinking|>` are proposed and accepted,
and mean a(P) over the first 4 positions is 1.88 (chat), 1.69 (code), 2.00 (agentic). Chat accepts least in the
thinking text past 16 tokens, at 1.20. There the target writes terse notes ("Need produce final email only
likely"), and the drafter proposes fluent paraphrases ("The user asks" for "The user wants"). Copied spans
from the prompt accept at 5 to 7.

## Open

- Cost of a partial accept: one extra route call over the accepted rows (replay). Committing the verify's own
  conv history at row n would remove it; that needs a per-row history from the route (coordinate with lane P).
- The bf16-points arm is measured above (slightly lower acceptance); it is not ported to the GPU.
- No timed numbers: decode speed belongs with the perf lane's kernels.
