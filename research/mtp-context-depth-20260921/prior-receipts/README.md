# Initial contextual-controller development receipts

This bundle preserves the completed first-iteration calibration, correctness and
development work. It is separate from the paired-evidence revision's held-out
score.

Native source: `e1cd38a47c8c36aeef75860f4cc623d5203188fc`.
Runtime archive SHA-256:
`d313e757edbb3b469890526e75522fd7f9893a74d990e9c5c5345bb50b9cd7f0`.
Manifest SHA-256:
`4c89b2a243cea44be43ba19f3826e57d3df06b6c2629cbfbd1c4019c14e9ff62`.

Two complete development sets per model produced these returned-token rates:

| Model | Calibrated fixed | Native adaptive | Frozen context | Online estimates |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 118.976 | 113.096 | 119.342 | 116.681 |
| Gemma 4 12B | 185.371 | 195.114 | 194.056 | 192.290 |

The metric is pooled returned output tokens divided by complete native request
seconds. These are development results from 160 timed turns. They motivated a
change to how the online policy updates a context's preferred depth. No
held-out outcome from this iteration was used to choose that change, and the
unused partial held-out output is not included in this public bundle.

Each run used an eight-turn synthetic conversation, an initial input of about
16K tokens, a 2,048-token output cap per turn, and at least ten seconds of
unscored warmup. Sampling was temperature 0.7, top-k 20 and top-p 0.95.
Prompt checkpoints were reused across turns. Startup, warmup and receipt I/O
were outside the request clock. Qwen used NVFP4+Q5_K with embedded MTP; Gemma
used a QAT Q4_0 target and QAT Q8_0 assistant.

One original Qwen forward calibration set was excluded because fixed K=4
repeated a numeric cycle. All seven arms of that set remain in the archive under
`experiment/qwen-excluded-calibration-0`. The replacement seed was recorded before
running the replacement. Both accepted calibration sets and their resulting
priors are preserved.

The source and receipts include the 352 native-session correctness turns and
the ordinary kernel, prefill/decode, Qwen K=1..8 plain/spec, and Gemma
session/plain runner checks. Hardware was one RTX 5090 32GB, driver 595.84,
CUDA 13.1 / sm_120a, with full vocabulary heads and the pinned model locks.

From the repository root:

```sh
python3 research/mtp-context-depth-20260921/publication/reproduce_development.py \
  --receipts research/mtp-context-depth-20260921/prior-receipts \
  --manifest-sha256 4c89b2a243cea44be43ba19f3826e57d3df06b6c2629cbfbd1c4019c14e9ff62

python3 research/mtp-context-depth-20260921/publication/verify_boundary.py \
  --receipts research/mtp-context-depth-20260921/prior-receipts \
  --review research/mtp-context-depth-20260921/prior-receipts/boundary-review.json \
  --prefix research/mtp-context-depth-20260921/prior-receipts \
  --check
```

The review covers all four archives and every expanded regular file. Compressed
byte matches have exact archive/rule pins. The one record-level match is token
40328's hexadecimal representation of seven hyphens; it is tokenizer data.
Source matches are exact blobs covered by their archived source approvals.
No archive bytes were changed to avoid a match.

Reproduction checks the recorded report without overwriting it. Extraction
rejects path traversal, path aliases, duplicate members and incomplete manifests
before creating its destination.
