# Reconciling this lane's 0/6 with the affinity lane's 5/5

Two lanes reported opposite answers to what looked like one question. Read against each other's
RAW rows, they are answering two different questions, and the discriminating variable is exactly
the one this lane named: **the `m` at which the REUSED PREFIX was primed, versus the `m` the cold
twin primes at.** Neither result is wrong. Neither generalises to the other's case.

## What the affinity lane actually measured (`578598fb4d`, cell2 LEG A)

Their harness, from `cell2-drive.py` verbatim:

```python
for i in range(1, 6):
    P = [{"role": "user", "content": "[pairA %s %02d] " % (name, i) + U1}]
    sid = "pairA-%s-%02d" % (name, i)
    c = send("A%d-cold" % i, P, sid)      # send 1: cold
    w = send("A%d-reused" % i, P, sid)    # send 2: SAME prompt, SAME session
```

and their rows:

```
A1-cold      ttft=1.976 chars=568 sha=3c2ddef37f911d35 rewound=False reused=0/0    suffix=0
A1-reused    ttft=0.240 chars=568 sha=3c2ddef37f911d35 rewound=True  reused=1472/1488 suffix=16
```

The labels are honest: `rewound=False` on the cold row and `rewound=True` on the reused row, so
this is NOT one of the mislabelled cold-vs-cold legs they themselves caught elsewhere. Their
5/5 MATCH is genuine.

**But it is guaranteed by construction.** The prompt is 1488 tokens and the worker splits it at
1472 (the same split shape this lane's receipts show directly: a 1520-token prompt primed as
`ENGAGED t=1504 base=0` + a 16-token chunk at `base=1504`). So:

* send 1 (cold) runs `prime(t=1472, base=0)` then `prime(t=16, base=1472)`;
* send 2 (reused) rewinds to 1472 — the rows send 1 itself just wrote, same session, same cache
  — and runs `prime(t=16, base=1472)`.

Send 2's suffix call is the **identical call over identical cache bytes** that send 1 made for
its own tail. Same `m`, same base, same `seq_end`, same inputs. Byte identity there is not a
property of the prime being chunk-invariant; it is the arithmetic being literally repeated.

## What this lane measured (LEG S)

A GROWING conversation, which is the agentic traffic shape this work is for:

```
s-est  (turn 1):        ENGAGED t=1440 base=0    seq_end=1487
s-cold (turn 2, fresh): ENGAGED t=1696 base=0    seq_end=1729
s-warm (turn 2, reuse): ENGAGED t=256  base=1440 seq_end=1729
                        ENGAGED t=33   base=1696 seq_end=1729   <- identical in both
```

The reused prefix rows 0..1439 were primed by TURN 1, inside a chunk of **m = 1440**. The cold
twin's rows 0..1439 are primed inside a chunk of **m = 1696**. Different `m`, so under an
m-dependent prime they differ before a single suffix token exists — and every later row
inherits it. 0/6.

## The reconciliation

| | affinity LEG A | this lane's LEG S |
|---|---|---|
| what is resent | the SAME prompt, twice | turn 2 of a GROWING conversation |
| reused prefix primed at | m = 1472, by the cold send itself | m = 1440, by turn 1 |
| cold twin primes at | m = 1472 | m = 1696 |
| m equal? | **yes, identical call** | **no** |
| identity | 5/5 MATCH | 0/6 DIFFER |

Both are correct. The affinity lane proved that **session restore is faithful**: replaying the
identical computation over restored KV reproduces the bytes, which is exactly what their
TP-mirror fix needed to show. This lane shows that **byte identity does not survive a change of
chunk shape**, which is what every growing conversation does by construction.

## Narrowing this lane's claim, as asked

WRONG, too broad: "a rewound session answers the same bytes differently from a cold session."

RIGHT: **for a GROWING conversation — where the reused prefix was primed as part of a shorter
prompt than the cold twin's — the reused answer differs from the cold answer.** Measured 0/6
across suffixes of 80 to 4440 tokens, in BOTH the walk-suffix incumbent and the batched-suffix
arm. Where the identical prompt is resent, bytes match, 5/5, receipted by the affinity lane.

The growing-conversation case is the one production serves for multi-turn agentic traffic, so
the product question stands — but it is a narrower and more precisely located claim than the one
I first stated, and the correction is the affinity lane's.

## Checking the coordinator's fourth candidate: is the 0/6 an artefact of my filler weakness?

No, and the direction matters. The filler caveat says prompts that differ ONLY in filler words
can produce identical completions, which makes CROSS-ROW sha equality uninformative. It cannot
manufacture a DIFFERENCE within a pair. Within each LEG S pair, warm and cold send the identical
message list — verified independently by the engine's own receipt, since `seq_end` is the prompt
length and it agrees pair-by-pair (1520, 1729, 1929, 2180, 2680, 5880), with the trailing chunk
literally identical (`ENGAGED t=33 base=1696 seq_end=1729` on both sides of s0250).

Determinism is also not the explanation: `s-cold/s0030` returned sha `11a610757e5b91ce` in all
three arms — three separate server boots, and in the off arm its tail even ran a different
implementation — so a fresh prime of a fixed prompt is reproducible.

If anything the filler property strengthens the finding: appending hundreds of meaningless
tokens does NOT change the answer, while re-priming the same prompt at a different chunk shape
DOES, with different output lengths.

## What both lanes should change in their receipts

The coordinator is right that `m` per prime call is the discriminating variable and belongs in
every identity row. This lane's engagement line already carries it — `[gemm-prime] ENGAGED t=<m>
base=<base> seq_end=<seq_end>` — which is the only reason this reconciliation was possible from
banked output. The affinity lane's rows carry `reused=X/Y suffix=Z`, from which the split can be
inferred but not read; adding the per-call `m` would make their identity rows self-describing.

## What this does NOT settle

Whether the differing answers differ in QUALITY. This lane measured bytes, not correctness or
preference, and says nothing about which of the two answers is better. That is the owner's
product call, and it should not be inferred from a sha.
