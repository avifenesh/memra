# Qwen2 tokenizer — native integration checks, 2026-09-20

The freshly built candidate passed the complete native release battery and the pinned tokenizer oracle in both loader paths. This closes the native integration gate for the literal Qwen2 pre-tokenizer correction in #540 / PR #558. Declared NFC normalization remains separately tracked by #554.

| Check | Result |
| --- | --- |
| Pinned real vocabulary, independently normalized inputs, HF loader | 551/551 identical in both encode modes |
| Same inputs and vocabulary in metadata-only GGUF | 551/551 identical in both encode modes |
| Kernel release coverage | 103 executed, all 10 required passed; 7 named skips within the fixed limit of 11 |
| Ornith 1.5 calibrated argmax | PASS, 1 near-tie flip, 0 bad flips |
| Qwen3.8 calibrated argmax | PASS, 0 flips, 0 bad flips |
| Greedy speculation on each roster model | K=1..8 each exactly once, 32 tokens, PASS |
| Physical-card lease | Wrapper and child exit 0; no timeout, interruption or lingering compute |

## Exact source and artifacts

The native build source is `fc5df430686f31ad34b8ec9b4e772a8f76b53f08`. Merging main `058259ccbbb76a8f8362f31d27efe515fa70f3a0` produced `ce7ce6c22adbc344e3b1bdcf491af3102f1b724f` with the exact same Git tree; `integration.json` records that equality. This later research publication adds evidence only and does not relabel a rebuilt executable.

`build-candidate.py`, `build.log` and `build.json` record the fresh native Rust 1.97.1 / CUDA 13.1 / sm120a build and hashes of all six ELF executables. The build ran with GPUs hidden. The complete executable archive has been preserved off the rented host and all six hashes were independently rechecked before publication.

`tokenizer-manifest.json` binds the actual `tok-parity` binary and every oracle input/output artifact. The pinned oracle is `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`, captured using Hugging Face `tokenizers==0.22.2`. The native CPU executable ran against 39 adversarial and 512 seeded inputs, independently NFC-normalized to isolate the splitter/BPE stage. The metadata-only GGUF is not a model-weight artifact. The original raw-input mismatch and #554 remain valid; this is not full raw-input normalization support.

The GPU gate ran `bash -x tools/release-battery.sh` using the normal roster and the actual named, hash-verified 9B and 35B kernel oracles recorded in `artifacts.json`. `lease.json` identifies the exclusively held RTX PRO 6000 Blackwell Server Edition card on a non-serving host. No missing-model waiver or coverage/skip override was supplied.

The seven skips are `dtype5-Q3_K`, `f16g-kq-direct-ornith`, `iq4xs-mmq-real`, `nvfp4-27b-shape`, `q4_0-mmq`, `q4_0-sk-arm`, and `sigrouter-served-replay`. Their missing artifact/tensor/capture reasons remain in the complete raw trace. None is required by the manifests; the required DUAL-BATCHED-AUX cell ran. Skipped cells remain unqualified, and this run makes no throughput claim.

The gzipped Bash trace retains complete successful command-substitution outputs, including xtrace repetitions. Count each captured producer output once, not every repeated string in the trace. The tokenizer logs likewise retain every case result. `SHA256SUMS` covers the public payloads.
