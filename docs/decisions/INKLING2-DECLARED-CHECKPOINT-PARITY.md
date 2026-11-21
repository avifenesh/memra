# Inkling 2 Mini: the declared class's checkpoint-parity band (2026-11-21)

**Status:** adopted 2026-11-21 for the `inkling2` pack, class `inkling2-declared-activation-quant`
(the checkpoint's shipped program: NVFP4 W4A4 four-over-six on layers 0..19, MXFP8 W8A8 on 20..23,
f32 outside the experts). Receipts: [`research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity/`](../../research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity/README.md).
Band: `crates/memra-gguf/src/model_packs/inkling2/mod.rs` (`checkpoint_parity_classes`, `checkpoint_probe`).

## Question

`memra model verify checkpoint` needs a band for the declared class. The only executable source of
that program is the serving image's ops, and no inference engine runs here. The onboarding rule
(`docs/ONBOARDING.md`): set a class band from the source program's own spread between two backends
on the probe, never from the native runner's result. Which backends, which probe, which band, and
what holds the route beyond one probe row?

## The method

Three layers, each against a source that is not the route:

1. **Per MoE, bit for bit.** Every slot's MoE output, with the oracle's MoE input forced, equals lane
   D's reference (`memra-reference` `inkling2_actquant`, the sm_120 accumulator model) computed on the
   host from the route's resident expert bytes (`ink2-gpu --moe-ref`). The red arm (the reference's
   index-order accumulator) must move outputs.
2. **The float trunk, bit-close.** The float class against lane E's f64 oracle tapes: residual
   streams and logits at f32 rounding distance.
3. **End to end, inside the source program's spread.** Two independent backends simulate the
   declared program: lane E's CPU oracle (`reference/ink2_oracle.py`, arm `actq`, numpy and torch on
   CPU) and lane F's harness (`klsens/klh.py`, arm `actq`, the transformers PR #51050 model on torch),
   run on CPU and on the GPU. The probe band is twice their largest spread on the probe; on longer
   text the route's per-token KL to each source backend must sit inside the KL the source backends
   have to each other.

## Measurement

### Per MoE (forced inputs, every slot)

| prompt | rows checked (positions x 48 slots) | bit-equal to the reference |
|---|---|---|
| p1_chat (57 positions, every one) | 2736 | 2736 |
| p2_window (1531 positions, every 16th) | 4608 | 4608 |

The red arm (the reference's index-order accumulator) moves 42 of the 2736 p1 rows (2169 bf16
outputs), so the check sees the accumulator. After the merge of lane P's decode (memra#1399) p1 holds
2736 of 2736 again (`../serve/s3/p1-moe-ref.json`).

### `memra model verify`

`verify checkpoint` against lane E's tape on the probe: passed, max abs 1.737 under the 3.7 band,
argmax 200001 on both sides (`verify/checkpoint-parity.tsv`). `verify serve` on that receipt:
passed (`verify/serve-gate.tsv`). Gates: Config, TokenizerTemplate, TensorCensus, CheckpointParity
and Serve passed; TinyParity is carried by CI (`ci:verify-tiny`), RewriteParity is pending (no tuned
rewrite is eligible).

### Float trunk against the f64 oracle

| tape | positions | worst slot rel L2 (attn / ffn stream) | logits max abs | KL mean | argmax |
|---|---|---|---|---|---|
| p2 f64, chunk 512 | 1500 | 2.7e-6 / 2.8e-6 | 2.3e-4 | 1.9e-12 | 288 of 288 |
| p3 f64, chunk 1024 | 8260 | 2.2e-6 / 2.1e-6 | 1.1e-4 | 8.2e-13 | 148 of 148 |

### The probe (26 tokens, the last token's logits)

| pair | max abs | KL | argmax |
|---|---|---|---|
| lane E CPU vs harness GPU | 1.808 | 9.6e-10 | same |
| lane E CPU vs harness CPU | 1.756 | 3.4e-10 | same |
| harness GPU vs harness CPU | 1.844 | 1.0e-9 | same |
| harness actq vs actq_rev (either device) | 0 | 0 | same |
| lane E CPU (the oracle) vs the route | 1.737 | 5.7e-10 | same |
| harness GPU vs the route | 1.843 | 3.7e-10 | same |

### End to end, teacher-forced (per-token KL over the full vocabulary, f64)

`route` is KL(harness GPU actq || route), `peer` KL(harness GPU actq || harness CPU actq), the same
program on two devices; `self` KL(actq || actq_rev), the reversed expert sum order on one device.
The 95% interval is of mean(route) - mean(peer) over documents, paired.

| set | tokens | route mean / p99 | peer mean / p99 | self mean / p99 | route - peer, 95% |
|---|---|---|---|---|---|
| rawdoc (15 documents) | 15360 | 0.0141 / 0.141 | 0.0142 / 0.141 | 0.0010 / 0.022 | -0.0007 .. 0.0006 |
| chatwrap (15) | 15360 | 0.0189 / 0.216 | 0.0181 / 0.204 | 0.0006 / 0.019 | -0.0001 .. 0.0019 |
| oracle prompts p1, p2 | 1588 | 0.431 / 8.07 | 0.466 / 8.71 | 0.227 / 5.64 | -0.035 .. -0.028 |
| pooled | 32308 | 0.0369 / 0.339 | 0.0383 / 0.357 | 0.0119 / 0.037 | |

Lane E's CPU tape against both at its own positions (p2_window, 1531): lane E vs harness GPU 0.533,
vs harness CPU 0.474, harness actq vs actq_rev 0.235, lane E vs the route 0.496, harness GPU vs the
route 0.446. On p1 (57): 0.053, 0.039, 0, 0.051, 0.044.

Top-1 agreement: route 0.951 pooled, peer 0.951, self 0.995.

The sum-order twin is exact on most documents and on the probe (rows of 8 bf16 terms add exactly in
fp32 unless their exponents spread), and where it is not, one flipped activation-quant code
compounds through the stream (p2: 0.235). It measures one perturbation, not the spread of the
program between implementations. Every independent backend sits at the cross-backend distance from
every other: lane E, the harness on either device and the route are each other's peers.


## Decision

- The declared class's band is 3.7 logits max abs, no relative term, argmax required:
  twice the largest probe spread between independent backends of the source program (1.844,
  harness GPU vs CPU), the factor the DSv4.1 pack uses. `verify checkpoint` holds the route against
  lane E's CPU tape (`probe/lane-e-cpu-actq.tsv`).
- The probe is a real chat turn (lane E's p1_chat prompt, 26 tokens) with a next-token margin above
  20 logits on every backend.
- The end-to-end gate on longer text is two-sided against the harness's cross-device spread: the
  route's mean KL to the harness may not exceed the harness's own GPU-vs-CPU mean KL by more than
  the documents' 95% interval allows, on the oracle prompts, rawdoc and chatwrap. It passes on all
  three. That gate is a research receipt (`kl-harness.json`, `kl-peer.json`), not a field of the
  pack: `CheckpointParityGate` holds one probe row.
- The same method serves any class whose source program has no inference-engine capture: lane M's
  `nvfp4-rig` takes its band from the harness on two devices with its arm, and its corpus gate from
  the same pair.

### The onboarding rule

The rule (set a class band from the source program's own spread between two backends on the probe,
never from the native runner's result) covers this: the backends are lane E's CPU oracle and lane
F's harness on CPU and GPU, none of them memra. No rule change is needed for the band. One addition
is proposed for `docs/ONBOARDING.md`, as the smallest one that keeps the next class from the
mistake measured here: "For a class that quantizes activations, the two backends are two
implementations or two devices of the source program; a reduction-order twin on one backend does
not count, since it is exact on short sums and understates the spread."


## Rejected

- **The route inside the sum-order self-noise** (actq against actq_rev). It is not: rawdoc 0.0141
  against 0.0010, chatwrap 0.0189 against 0.0006. Neither is the harness on its other device
  (0.0142 and 0.0181), so the test would refuse every independent implementation of the program,
  the source's own included.
- **A band from the route's own spread to the oracle.** Forbidden by the onboarding rule; the route
  sits at 1.74, inside the band set without it.
- **The default probe 1,2,3,4.** Not a sequence the model sees; a chat turn exercises the template's
  tokens, the controllers and the loop on real routing.
- **A KL field in `CheckpointParityGate`.** One probe row's KL is under 1e-9 for every pair, so it
  separates nothing there; the distribution gate needs a corpus and lives in its receipt.
