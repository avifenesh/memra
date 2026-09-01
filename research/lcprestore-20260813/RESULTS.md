# Immediate partial-node prefix reuse — results

Date: 2026-08-13
Verdict: **NO-GO**

The transformer-only candidate copied the selected prefix state exactly, but it did **not** produce
byte-identical output to a genuinely cold request at every split boundary. The exactness gate failed
at 512 and 2048 tokens, so the campaign stopped before scored performance or the standard GPU
battery. This branch must not merge or ship in its current form.

## Frozen cell

- Source: `0ab3c23658b4949b4ea33a492ef5601ce53c185b` on `lane/cx-lcprestore`, based on
  `v0.81.3` (`7cf5fd842ebc76f6e8a82910a8e6d4b864b6b42d`).
- Rig: box1 physical card 1, NVIDIA RTX PRO 6000 Blackwell Server Edition,
  `GPU-2b4cf166-fd33-f161-8536-ca04bc72280c`, driver 595.71.05.
- Lock: one uninterrupted `/tmp/memra-gpu-1.lock` hold; preflight reported
  `compute_apps=none`. Both server PIDs recorded in the next preflight were lane-owned.
- Positive model: `gemma-4-12b-it-qat-q4_0.gguf`, SHA-256
  `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`, from
  `google/gemma-4-12B-it-qat-q4_0-gguf` revision
  `29d097773436b69ff9feafd636ab4cf873786537`.
- Prompt: 4,860 tokens; 60 generated tokens; one exactness repetition at each split. This was a
  trace-enabled correctness cell, not a scored latency cell.
- Thermal trace: the card started at 27 C / 0 MiB and ranged up to 56 C during the cell. Cleanup
  recorded 0 MiB. The 250 ms and 1 s telemetry streams are retained raw.

## HIRADIX-EXACT-ISO split receipt

The source hash covers only the first `l` rows selected from the longer cache entry plus the
position/length state. The restored hash independently covers the fresh session at exactly `l`.
Their equality proves the bounded transfer, but the verdict comes from the separate cold-oracle
output comparison; it does not borrow correctness from a whole-entry hit.

| Split | Source/restored state SHA-256 | Candidate request 2/3 output SHA-256 | Cold request 2 output SHA-256 | Result |
|---:|---|---|---|---|
| 64 | `3f207a8cc567953a53cb8f9ca9d94bf5935073ace7365832e35b3bce57bee0a3` | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | PASS |
| 512 | `014d7a82fb958a411fd35b97ff32db6cbb05590827e2cf0bcca322951d1799bb` | `bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525` | `719a43f41b407364130580b2f12a8c09e78da460dc25ada2f1781dd436780079` | **FAIL** |
| 2048 | `8ff99f2a76b962b8a29293a1af258851c54113b753b64ce0459cf576cb299d36` | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | `223618bfd84e4f30bb454fb7383f139753011e918926af620cf047dda7c136c2` | **FAIL** |
| 4374 | `0c03aafac4537c83917caf0dd922679c22738ce5ce62e70338163a81e7c7acd4` | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df` | PASS |

The two exactness-refuting reducer messages, verbatim:

```text
BYTE MISMATCH split=512 rep=1: {"candidate_request2": "bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525", "candidate_request3": "bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525", "genuinely_cold_request2": "719a43f41b407364130580b2f12a8c09e78da460dc25ada2f1781dd436780079"}
BYTE MISMATCH split=2048 rep=1: {"candidate_request2": "eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df", "candidate_request3": "eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df", "genuinely_cold_request2": "223618bfd84e4f30bb454fb7383f139753011e918926af620cf047dda7c136c2"}
```

The control server also returned a 4,860-token whole-entry hit for request 3 rather than the
split-sized boundary expected by the harness. Its additional reducer messages were:

```text
split=64 rep=1 control: request3 cached_tokens 4860 != 64
split=512 rep=1 control: request3 cached_tokens 4860 != 512
split=2048 rep=1 control: request3 cached_tokens 4860 != 2048
split=4374 rep=1 control: request3 cached_tokens 4860 != 4374
```

That control-shape mismatch is separate from, and does not explain away, the two candidate-versus-
cold byte mismatches. The divergence cause is not established by this run; a future lane must
reproduce and isolate it before changing the eligibility gate or suffix-prime path.

## Timing status

No scored request-2 latency result exists. The required trace-free N>=5 alternating cell did not
run after exactness failed. For transparency only, the trace-enabled N=1 diagnostic at the 4,374
split observed request-2 TTFT 955.642 ms control versus 4,741.861 ms candidate (+3,786.219 ms,
+396.2%). It is explicitly not a median claim and must not be compared with the Q27 receipts.

The mixed c=4/knee replay and c=64 stress were not run, so this lane makes no no-regression claim.

## Gate ledger

| Gate | Result |
|---|---|
| Host and remote `cargo test --workspace` | PASS; all runnable tests passed, including `memra-server` 224/224; GPU-only tests remained explicitly ignored |
| Split source -> restored state hashes | PASS at 64/512/2048/4374 |
| Split request bytes vs genuinely cold | **FAIL at 512 and 2048** |
| Namespace isolation probe | PASS: cached-token sequence `0, 4860, 0, 0` |
| `kernel-check` | NOT RUN — exactness STOP |
| `run-gen` argmax | NOT RUN — exactness STOP |
| `run-spec` K=1..8 | NOT RUN — exactness STOP |
| `serve-smoke`, including Q35 c=4 | NOT RUN — exactness STOP |
| c=64 stress | NOT RUN — exactness STOP |
| Q27 mixed c=4 and knee | NOT RUN — exactness STOP |
| Q27 hybrid / Q35 routed-MoE live refusal cells | NOT RUN — exactness STOP; fail-closed host fixtures passed |

The first device attempt used an official Qwen3-4B GGUF as the dense control, but the pinned
server could not load that non-hybrid architecture. It stopped before requests with these captured
lines:

```text
not a hybrid arch
[server] FATAL: worker init failed: worker died during init
[worker] PANIC in the GPU worker thread: not a hybrid arch
```

## Raw evidence and cleanup

- `raw/attempt1-qwen3-unsupported/`: first attempt, stopped before requests.
- `raw/attempt2-gemma-exactness-fail/`: committed source/model manifests, full cargo/build logs,
  both server logs, request JSONL with raw output bytes, orchestration log, GPU telemetry, and
  cleanup receipt.

The scored attempt cleanup at `2026-08-12T23:55:23Z` recorded physical GPU 1 at 0 MiB. A separate
post-run probe at `2026-08-12T23:59:57Z` found no compute app on the assigned UUID, no lane port
listener, and successfully acquired then released `/tmp/memra-gpu-1.lock`. No live serve host was
touched.
