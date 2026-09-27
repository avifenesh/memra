# Draft-only K/C/D development controls

K is the MTP **draft sampler top-k**; target top-k stayed at 20.
D is the offered draft length. All rows use C=0, temperature 1.0,
top-p 0.95, the same Qwen3.8 full-head model and one nonproduction
RTX 5090. Each row pools six distinct, eight-turn native code
conversations, with native KV reuse on later turns. The score is
returned output tokens divided by complete request seconds. Exact
loops are excluded from performance aggregates; none occurred in
these controls.

| Draft K | D | Output tok/s | Parseable code / 48 | Bounded function cases / 24 |
|---:|---:|---:|---:|---:|
| 3 | 1 | 116.08 | 48 | 16 |
| 3 | 2 | 133.80 | 47 | 23 |
| 3 | 3 | 139.54 | 48 | 24 |
| 3 | 4 | 133.69 | 47 | 23 |
| 10 | 1 | 115.50 | 48 | 24 |
| 10 | 2 | 134.97 | 48 | 24 |
| 10 | 3 | 139.84 | 47 | 23 |
| 10 | 4 | 134.70 | 47 | 23 |
| 20 | 1 | 115.74 | 48 | 24 |
| 20 | 2 | 134.59 | 48 | 24 |
| **20** | **3** | **141.46** | **48** | **24** |
| 20 | 4 | 135.43 | 47 | 23 |

The bounded function check covers the last three development
topics, one valid-domain case for each requested function. A
parseable response can still fail it. K=10/D=3 and K=10/D=4
reached the shared 8,192-token limit before finishing one requested
function. The table is a fixed-control development result, not
fresh heldout performance or a general code-quality estimate.

At draft K=20/D=3, a second six-topic grid compared fixed
after-offer C cutoffs derived from empirical development
proposal-probability quartiles and medians:

| Fixed C for offer positions 1, 2 | Output tok/s | Parseable code / 48 | Function cases / 24 |
|---|---:|---:|---:|
| (0, 0) | 141.46 | 48 | 24 |
| (0.534, 0.529) | 141.75 | 48 | 24 |
| (0.534, 0) | 139.43 | 48 | 24 |
| **(0, 0.529)** | **143.46** | **48** | **24** |
| (0.957, 0.962) | 138.02 | 48 | 24 |

No fixed-C candidate looped. The selected fixed C/K/D
control is draft K=20, D=3, C=(0, 0.529), up 1.41%
in-sample versus C=0. These values are **hard-coded control
settings selected on development**, not a trained confidence
policy or a fresh-performance result.

The six development conversations also supplied randomized D
offers for training. At draft K=3, D=4 accepted 2.21 draft tokens
per eligible round versus D=3's 1.87, but its round utility was
0.356 versus 0.376. At draft K=20, D=4 accepted 2.32 versus
1.94 and had slightly higher round utility, 0.471 versus 0.444.
The fixed native runs above resolve the first question: D=4 did
not improve complete-request tok/s at either K and missed a code
case. Acceptance and round utility are mechanism diagnostics.

Separate tiny C/D models were fitted at draft K=3 and 20 from
10,413 and 10,030 eligible randomized D rounds respectively.
K=10 was excluded from model actions after its fixed D=3
development code-format miss. The K models used 15 randomized
K=3 turns and 16 randomized K=20 turns. First-16 and first-32
user-token-prefix fits chose K=20 on all 48 development prefixes;
the bounded-prefix plus previous-turn-acceptance fit chose K=3
on 9 of 48. Those are training-set actions, not a learned-policy
throughput verdict.

On three further **old** selection topics, the prior-acceptance K
model scored 138.248 complete-request tok/s versus fixed draft
K=20's 138.046 and its no-op twin's 138.146. It applied K=3
on just 1/24 turns, using K=20 on the other 23. The first-16
and first-32 models applied K=20 on all 24 selection turns.
All three learned K candidates and the K=20 fixed control
passed 24/24 bounded function checks with no loops; fixed
K=3 passed 23/24. The selection rule chose the prior model
on its point estimate, but this small development difference
does not establish a fresh K-routing improvement.

The model artifact is
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` SHA-256
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
The native Rust source archive is SHA-256
`a391ff6337434e45acac4def721b816ccb4c6027ea93aad83cb90d40390c729a`;
the measured binary is SHA-256
`07033ec249485f9da202f266d07d5432554a1f8b65ea6d14f55bee0387327c15`.
`development-summary.json`, `model-training-summary.json`,
`topk-training-summary.json`, `depth-grid-analysis.json`,
`fixed-confidence-candidates.json`, `selected-fixed.json`,
`selection-summary.json`, and `selected-topk.json`
retain the underlying rows for the sealed archive.
