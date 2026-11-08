# The MTP draft fill and the prime's tail on GB10 (round 4 design, 2026-11-08)

Design note before code. Ledger rows 46 and on.

## What the request window holds

`parts_last.py` ends its window at the last prime attention launch. That kernel is not the end of
a served prefill on the MTP route: after the trunk's chunks the walker runs the trunk's boundary
tail chunk and then the MTP draft fill, and only then emits the first token. A window that ends at
the last attention launch therefore starts before the request arrived, and its "2.68 s idle gap at
the start" (`profile-r4-main-131k/gaps-api.txt`) is the server waiting for the request.
`harness/r4/parts_req.py` anchors the window at the prime's first kernel instead, and the timestamped
server log (`ts-probe-main-unit1/`, every line stamped by `harness/r4/tsrun.py`) gives the phases:

| phase | 32,784 tokens | 131,051 tokens |
|---|---:|---:|
| request to the first prime kernel (host) | 0.124 s | 0.446 s |
| trunk, 4,096-row chunks | 22.144 s (8) | 119.340 s (31 + 4,032 rows) |
| trunk boundary tail (prefix-cache stable boundary at a 128-row multiple) | 0.257 s (16 rows) | 0.460 s (43 rows) |
| MTP draft fill, 4,096-row chunks | 0.583 s (8 + 16 rows) | 2.342 s (32) |
| TTFT | 23.109 s | 122.597 s |

The draft fill costs 72-74 ms per 4,096-row chunk at every depth: it computes only the MTP
block's K/V (eh_proj, wk, wv, k-norm, RoPE and the quantized append; no attention). In the nsys
trace each fill chunk opens with a 35-38 ms GPU-idle gap (`profile-r4-main-131k/gaps-api-req.txt`:
31 such gaps in the request window, 1.110 s, with 0.946 s of `cuMemcpyDtoDAsync` and 0.104 s of
`cuLaunchKernel` API time inside them; `memcpy.txt`: 254,490 device-to-device copies), and the
chunk then runs 4,096 single-row `append_quantize_kv_q8_0_q5_1` launches. The source,
`mtp_kv_fill_at` in `crates/memra-engine/src/spec.rs`:

- op 3 builds the `[e_norm ; h_norm]` concat with two `copy_view_into` calls per row;
- op 6 appends each row's K/V with one `append_kv_quantized_view` launch.

## The change

- The concat is two `copy_2d_dtod_async` calls (row pitch 2 x n_embd in the destination, n_embd
  in the sources): the same bytes in the same places.
- The append is one `append_kv_quantized_rows` launch over the T rows. That kernel's per-(block,
  token) warp program is the per-token append verbatim, and kernel-check pins it against the
  per-row loop ("kv append rows-vs-loop bit-identity", now also at the fill's shape: 4,096 rows of
  kv_dim 1,024 at slot 255,855; and "mtp fill concat strided-vs-rows bit-identity" for the copies).
- The per-row slices used to fail on a K/V width that did not match the plane; the batched call
  checks the widths, the device (the PP guard the per-row append ran) and a T above the kernel's
  grid height (65,535 rows) instead, and returns an error. Prime fill chunks default to 4,096 rows
  (`MEMRA_PRIME_CHUNK` and the SWA ring's chunk size can raise them to 65,535).

Exactness class: identical bytes in the draft KV and the concat, so the drafts, the acceptance and
every served token are unchanged by construction; the served fingerprint must match.

Expected: per fill chunk the 36 ms gap and most of the 9 ms of single-row appends go, so about 45
ms per chunk: 0.36 s at 32k (1.6% of TTFT), 1.4 s at 131k (1.2%), 2.9 s at 260k (0.9%).

The same function also runs in every speculative round: the verify refresh over the round's rows,
the one-row boundary fills, the restore suffix fill and `replay_acceptance`. Those fills are a
handful of rows, so decode should not move; the served cells check it (40 decoded tokens after the
first: total minus TTFT). It is a mechanism change (fewer host calls, the same bytes) and applies on
every target; the sm_120a perf cell (plain `run-gen` decode) does not reach the MTP fill, so the
GB10 TTFT numbers are the measurement and no per-device key is needed.

## Left for later, from the same timeline

- The trunk's boundary tail chunk (16 / 43 rows) runs all 64 layers at a tiny T, with the db
  attention twin over the whole prefix (16 x 10.2 ms at 131k): 0.26-0.46 s. The split exists so
  the prefix cache can capture the recurrent state at a 128-row boundary.
- The 0.45 s between the request and the first prime kernel at 131k (0.12 s at 32k) is host work
  that grows with the prompt: the admission, spec-affinity and spec-k steps each take 100-185 ms.
