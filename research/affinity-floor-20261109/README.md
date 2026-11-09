# The plain-affinity prime floor (memra#1093 item 3, 2026-11-09)

Verdict: the 1..15-row suffix past an affinity checkpoint is reachable at the RTX 5090 device
default, and it breaks warm == cold. The prime floor fixes it and makes the turn faster. On the
plain route, an edited resend rewound to the checkpoint 160 of a 175-token prompt and primed the
last 15 rows token by token. It left the cold text in 3 of 3 runs (821 against 904 characters).
With the floor, the turn declines and primes cold: byte-equal in 3 of 3, and its TTFT drops from
0.327 s to 0.217 s (median of 3 each).

Model and rig: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, no DFlash drafter, RTX 5090 Laptop,
context 8,192, two sessions, prefix cache at its derived size, every switch at the device default.
Base: PR #1133's head bb4f8c5b7d (binary sha256 891c478fe0188685). Fix: this branch, stacked on it
(9f86ca39c9, 2c453c9995b3e501). `probe-main/` is the same cell on main 65fb297a59, non-streamed
(82645099dc93c91d): the same failure, 821 against 904.

## The shape

`floor_pair_cells.py`, an "edit the last message and resend" client:

1. Turn 1 sends a system line and a question with a `user` id. The session parks with its rewind
   checkpoint at the stable boundary `b`: the last turn marker rounded down to the 32-token grid,
   at least `PRIME_MIN_T` (16) rows under the prompt end (`plain_checkpoint_boundary`). The cell
   computes `b` from `/v1/tokenize`: the prompt without the generation header ends at the header's
   turn marker.
2. Turn 2 resends the conversation with the question's last word cut. The new prompt keeps turn 1's
   tokens through `b` and ends 15 rows past it (`b` = 160, turn 1 = 176 tokens, turn 2 = 175).
3. One cold boot serves turn 2 alone. Turn 2 is streamed on both boots (TTFT = first chunk) and
   must be byte-equal.

Route `plain` is greedy with `presence_penalty` 0.5. The spec route refuses a greedy-penalized
request, so it serves plain. Route `spec` is greedy with no penalty, the MTP spec route.
`max_tokens` is 160.

## Results (`cells/`, cases in the order run)

| case | binary | turn 2 warm path | bytes | warm TTFT (s) | cold TTFT (s) |
|---|---|---|---|---:|---:|
| plain-base-1 | #1133 head | plain-affinity rewound to 160 of 175 | FAIL (821 vs 904) | 0.327 | 0.210 |
| plain-fix-1 | floor | declined (grid law: fewer than the prime floor) | PASS | 0.217 | 0.205 |
| plain-fix-2 | floor | declined | PASS | 0.215 | 0.206 |
| plain-base-2 | #1133 head | rewound to 160 of 175 | FAIL (821 vs 904) | 0.329 | 0.211 |
| plain-base-3 | #1133 head | rewound to 160 of 175 | FAIL (821 vs 904) | 0.327 | 0.206 |
| plain-fix-3 | floor | declined | PASS | 0.218 | 0.206 |
| spec-base-1 | #1133 head | spec-affinity declined (grid law) | PASS | 0.266 | 0.238 |
| spec-fix-1 | floor | spec-affinity declined (grid law) | PASS | 0.274 | 0.231 |

On main the spec route took the same 15-row rewind (`probe-main/`, `spec-affinity: rewound to 160
of 175`) and stayed byte-equal on that answer. PR #1133 already declines it, by the same floor.

Why the floor is cheaper here: the 15 rows primed token by token cost 15 sequential decode steps.
The cold prime runs them inside its batched last chunk. The declined turn is a cold prime of 175
tokens, served in 0.217 s against 0.327 s.

## The change

`affinity_resume_target` refuses only an empty suffix. The plain-affinity tier now also asks a
recurrent cache for the prime floor the exact-extension rewind applies (`plain_pool_resume_at`): at
least `PRIME_MIN_T` rows past the checkpoint. Otherwise it declines (`plain-affinity: declined
(grid law: the prompt extends checkpoint N by fewer than the prime floor ...)`), and the turn
primes cold or from the prefix cache. Attention-only caches keep every non-empty suffix. The
earlier lane left this documented as open because a batched prime under 16 rows is another
program. The floor needs no such prime: it gives the turn to the cold path.

## Commands

```
floor_pair_cells.py OUT "plain-base-1=<#1133 binary>|plain|" "plain-fix-1=<floor binary>|plain|" ...
```

The eight cases ran in one lease in the order above (`chain.log`).
