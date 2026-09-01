# P2P on the Vast serve home — results

Date: 2026-08-10

## Verdict

**HOLD. Do not build `MEMRA_PP_COPY_OVERLAP`.** The serving configuration selects memra's
cross-device `cudaMemcpyPeerAsync` path and emits the expected peer/mempool-grant line, but the
path does not preserve bytes on this host. `cuDeviceCanAccessPeer=1` is only a capability report;
the effective usable peer path is **NO**.

That correctness failure precedes the 3% performance gate. Forced-anatomy timings report an
apparent 0.09-0.10% decode boundary and the prime scheduler proves its existing stage overlap is
live, but both computations consume a corrupt boundary. They are diagnostic observations, not
scored performance evidence and not grounds for an overlap implementation.

The production launcher and soak were restored after both bounded blocks. At final verification,
`GET /v1/models` answered and the SSE request completed with valid framing, but generated content
was sixteen repeated BOS tokens. The endpoint is operational; model output is not semantically
healthy on this PP placement.

## Rig and controls

- Vast serve home: 2x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB each.
- Topology: `NODE`; BAR1 is 256 MiB per GPU.
- Kernel driver: 580.105.08. Build toolchain: CUDA 13.1 (`nvcc` 13.1.115).
- Production library order: CUDA 13.1 compatibility library first
  (`libcuda.so.590.48.01`), then CUDA 13.1/12.8 runtime libraries.
- Model: Step-3.7-Flash IQ4_XS, `n_embd=4096`; the PP boundary is `[T,4096]` f32.
- Block 1 source: `d3f1199fb6e6336972067cda4b69529c686ca8f0`.
- Block 2 source: `f67e8f02736f30fb9096ae888a176e53eb437a54`.
- Server binary SHA-256 in both blocks:
  `cbfa77425df967f66d88a9fdb08691ba26719f29ac8e4a9b653382d489313567`.
- Serving and soak were stopped once per bounded block without `flock`, and each block's driver
  log records its stop and successful restore interval.

NVIDIA's current compatibility material lists CUDA 13.x minor-version compatibility at driver
580 or newer, while the toolkit-paired CUDA 13.1 Update 1 driver is 590.48.01. That distinction is
why the byte checks were repeated both with the production compatibility library and directly
against the host 580 library; neither path worked. See NVIDIA's
[minor-version compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and [forward-compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

## 1. Peer-path receipt

### Dispatch is active; transport is corrupt

The production-shaped boot emitted:

```text
[pp] cross-device transport: stage0=dev0 stage1=dev1 (cudaMemcpyPeerAsync per cross boundary; peer + default-pool access granted all pairs over [0, 1]; weight home: per-stage (sharded loader))
```

Both directions returned `cuDeviceCanAccessPeer=1`. Those facts prove selection and grant setup,
not data movement. Three independent correctness checks failed:

| Check | Production compat library | Host driver library | Verdict |
|---|---:|---:|---|
| Custom synchronized `cudaMemcpyPeerAsync` sweep, 16 KiB verification window | 16,320 / 16,384 bytes wrong | 16,320 / 16,384 bytes wrong | FAIL |
| memra boundary roundtrip, four alternating slots | 4 / 4 failed; 5,119 then 5,120 f32 elements differ | same | FAIL |
| NVIDIA `simpleP2P` verification | `Test failed!` | `Test failed!` | FAIL |

The wrong-byte signature is almost entirely zero destination data. CUDA calls return success;
this is not a captured API error. NVIDIA `simpleP2P` came from cuda-samples commit
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`; its pinned executable hash is in the raw receipt.

### Apparent timing — invalid as bandwidth

The production library path printed the following timings. `batch_us` is CUDA-event time over a
batch; `serialized_us` includes a stream synchronization after every copy and is the closest shape
to the anatomy diagnostic. Because the destination fails verification, `GB/s` is an apparent API
timing only and **must not be cited as transport bandwidth**.

| Boundary bytes | Rows at `n_embd=4096` | Batch us | Serialized us | Apparent GB/s |
|---:|---:|---:|---:|---:|
| 16 KiB | 1 | 1.806 | 5.100 | 9.074 |
| 2 MiB | 128 | 19.730 | 24.156 | 106.294 |
| 4 MiB | 256 | 38.781 | 43.265 | 108.155 |
| 8 MiB | 512 | 76.921 | 81.831 | 109.055 |
| 16 MiB | 1,024 | 153.669 | 158.583 | 109.177 |

The official bandwidth-only sample likewise printed 112.91-113.89 GB/s unidirectional and about
225 GB/s bidirectional while `simpleP2P` failed verification. This directly demonstrates why the
prior bandwidth-only receipt on this host cannot establish an active, correct peer path.

## 2. Live boundary anatomy

`MEMRA_SPEC_PP_ANATOMY=1`, `MEMRA_SPEC_GATE=0`, and `MEMRA_SPEC_K=1` forced natural PP-2 verify
boundaries to synchronize for component timing. One c=1 short request generated 63 rounds; one
fresh 4k request had 4,107 prompt tokens and generated seven rounds. The table excludes each
request's initial `T=1` round and reports medians over its `T=2` rounds. `Boundary share` is the
median per-round `(tx + rx) / total`; TX is the peer copy and RX is the receiving-device local
copy.

| Request | T=2 rounds | Reverse ms | Stage 0 ms | TX ms | RX ms | Stage 1 + head ms | Total ms | Boundary share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c=1 short, 23 prompt tokens | 62 | 0.006 | 10.326 | 0.010 | 0.009 | 11.7365 | 22.0415 | 0.0873% |
| c=1 4k, 4,107 prompt tokens | 6 | 0.008 | 9.354 | 0.010 | 0.010 | 10.609 | 19.9965 | 0.0998% |

Thermal regime was one bounded, unclocked Max-Q run: both GPUs were 32 C at entry and 36 C after
the final prime diagnostic, with 2,272 MHz reported at both snapshots. These are within-request
round medians, not repeated-request performance samples. The forced synchronizations also make
them anatomy numbers rather than normal serving throughput.

The 4k request's TTFT trace recorded 6,850.309 ms in prime. Both requests returned repeated BOS
tokens with 0% speculative acceptance, consistent with the independently proven bad boundary;
their timings therefore cannot cross the research correctness gate.

### Prime path

The pp512-class diagnostic encoded `T=402` and selected four dynamic chunks
`[64, 120, 113, 105]`, corresponding to 1.000, 1.875, 1.766, and 1.641 MiB boundary payloads.
It was a single diagnostic sample, not an N=5 comparison:

| Schedule | Wall | Throughput | Split chunks | Live stage overlaps |
|---|---:|---:|---:|---:|
| Serial stage walk | 1,457.7 ms | 275.8 tok/s | 4 | 0 |
| Existing prime pipeline | 819.1 ms | 490.8 tok/s | 4 | 3 / 3 |

The liveness counter proves stage 0 of chunk N+1 and stage 1 of chunk N overlap on this box. Source
inspection narrows the copy relationship: `tx_pipelined()` alternates the two boundary slots, but
`tx_slot()` issues `cudaMemcpyPeerAsync` on stage 0's compute stream. The copy is ordered after its
producer computation and before that stream's next computation; in steady state it can sit under
stage 1's preceding-chunk work, but it is not on a dedicated copy stream.

The nearest direct shape, 2 MiB, reported 24.156 us serialized. Four such apparent copies would
be about 0.097 ms, or 0.012% of the 819.1 ms pipelined wall. This is only an upper-envelope
diagnostic: the copies did not preserve bytes, so it is not a valid boundary-cost claim.

## 3. Build-or-hold decision

The requested implementation condition was a **correct, non-overlapped copy consuming at least
3%** of decode-round or prime time. It is not met:

1. Peer-transfer correctness is red before performance is considered.
2. The decode diagnostic's apparent combined boundary is about 0.1%, far below 3%.
3. The prime pipeline already overlaps all three adjacent stage-walk pairs. Its producer stream
   does serialize its own copy, but the corrupt-copy timing envelope is far below 3%.

Therefore this lane adds no `MEMRA_PP_COPY_OVERLAP`, runs no byte-identity gate for such a change,
and runs no N=5 A/B. No runtime defaults, published boards, releases, tags, or origin refs changed.

Before any P2P performance work resumes, the host/hypervisor/driver path must make the pinned
NVIDIA `simpleP2P` verification pass and then make memra's `pp-transport-smoke` pass with
`MEMRA_PP_DEVICES=0,1`. The present evidence does not isolate whether the defect is in the kernel
driver, virtualization/IOMMU plumbing, firmware, or PCIe mapping, so it would be incorrect to name
one as the cause.

## Raw evidence index

- `raw/block1/`: capability, topology, driver/library inventory, source/binary hashes, custom
  shape probe, NVIDIA samples, memra transport smoke, and block restore receipt.
- `raw/block2/`: server boot, short/4k request and response bodies, per-round anatomy, TTFT trace,
  pp512-class pipeline liveness/timing, GPU snapshots, and block restore receipt.
- `raw/final-service/`: restored process list, `/v1/models`, SSE headers/body, curl status, and
  final server tail.

Final service receipt at 11:39:55Z: `memra-server` PID 184136 and `/root/soak.py` PID 184456 were
running; `/v1/models` returned HTTP success; the streamed completion produced 17 JSON frames,
`[DONE]`, and `finish_reason=length`, but its 16 token deltas were all BOS.
