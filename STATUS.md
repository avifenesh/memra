# ASR lane status

Updated 2026-09-09 (Whisper clip parity on the HF subset, whole RNNT streaming path, 71-clip sweep in flight).
Worktree `~/projects/memra/wt-asr-modality`, branch `lane/asr-modality-20260909`.
Draft PR https://github.com/avifenesh/memra/pull/416, issue #414 remains claimed.

## Committed and gated

Whisper: native log-mel (padded and clip programs), both convolutions, all 32 encoder blocks,
the cached decoder, the beam-1 transcription policy and the clip window program all execute in
the CPU reference path. `whisper-stage transcribe` takes PCM and returns token ids.

Nemotron RNNT: the whole streaming path executes. `rnnt-stage stream` takes PCM and returns
token ids: native log-mel, cache-aware FastConformer over the `[56, 0]` arm, prompt
conditioning, the two-layer LSTM predictor, the joint and greedy decoding, with encoder caches
and the predictor hypothesis carried across chunks.

| Stage | Comparison | Result | Bound |
| --- | --- | ---: | ---: |
| Log-mel, synthetic | Native vs HF FP32 | 0.0000567436 | 0.001 |
| Log-mel, 364 real windows | Native vs CT2 | 0.00000011920929 | 0.001 |
| Encoder FP32 | Native vs HF FP32 | 0.0002231598 | 0.001 |
| Encoder FP16 | Native vs HF FP16 | 0.25 | 0.2771682739 |
| Decoder FP32 | Native vs HF FP32 | 0.0000143051 | 0.001 |
| Decoder FP16 | Native vs HF FP16 | 0.015625 | 0.0345668793 |
| Window program | Native rule vs 364 pinned boundaries | 364 of 364 | exact |
| Clip parity, 8 HF clips | Native ids vs CT2 beam-1 | 49 of 61 windows | see below |
| Clip text, 8 HF clips | Native vs CT2, canonized | 6 of 8 clips | see below |
| RNNT census | Contract vs `clean-step-21959.nemo` | 657 of 657 tensors | exact |
| RNNT frontend | Native log-mel vs NeMo CPU FP32 | 0.0000534058 | 0.001 |
| RNNT encoder, [56,0] | 26 chunks vs NeMo CPU FP32 | 0.0000002533 | 0.001 |
| RNNT encoder on native mel | 26 chunks, frontend composed | 0.0000002086 | 0.001 |
| RNNT head | Prompt, predictor, joint | 0.0000915527 | 0.001 |
| RNNT greedy | Token ids | 9 of 9 identical | exact |
| RNNT session | 26 chunk partials plus final | 26 of 26 identical | exact |

Whisper text parity, 71-clip sweep checkpoint at 17 clips and 125 windows:
**106/125 windows token-exact, 15/17 clips text-exact**, `d1` 0.0000 pt and
`whatsapp` 0.2089 pt against CT2. The stage gate wanted every clip equal or under
0.05 pt per domain, so it is **still not met on `whatsapp`**, but the delta is falling as the
corpus grows (0.5656 at 4 clips, 0.3390 at 7, 0.2089 at 10) toward the 0.2262 pt the
native text sits from the oracle's FP32 backend. The two oracle backends differ from each other
by 0.2114 pt on `d1` and 0.3394 pt on `whatsapp`, so 0.05 pt is below the disagreement between
the references. No divergence is a native defect. The classes are boundary_cascade 4,
fp16_tie 13, fp16_ulp 1 and real 1; the single `real` is `whatsapp-003` window 4 step 86, two
adjacent timestamps two fp16 steps apart with the clip's text unchanged, and the one that
looked like a defect (`whatsapp-001` window 3) turned out to be a differently padded reference
window. The engine's own
detokenizer agrees with the checker's tokenizer on 17/17 clips. Full analysis:
`research/asr-modality-20260909/TRANSCRIBE-PARITY.md` and `WHATSAPP-001-W3-DIAGNOSIS.json`.

The speech matrix product is cache-blocked and optionally threaded; both are bit-identical by
construction and re-proved on the real checkpoint. One encoder window fell 122.950 s to
60.477 s single-threaded, and one clip end to end fell 117.2 s to 60.9 s at 16 threads.

## In flight

The 71-clip end-to-end sweep is running on the rig, one process, `nice -n 15`, pinned to the
efficiency cores, ~82 s per window, 364 windows. It is ordered so the eight HF-oracle clips
come first and then `d1` and `whatsapp` interleave, so both domains grow together and a stop
at any point still has both. It is stopped at 22:30 UTC by an armed timer if it has not
finished:

```
tools/whisper_transcribe_sweep.sh ~/hebrew-asr-data/oracle/whisper-large-v3-ivrit \
  ~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9 \
  "$PWD/.lane-asr-stage2/target/release/whisper-stage" "$PWD/.lane-asr-stage2/sweep" f32 CLIPS...
```

Clip order puts the eight HF-oracle clips first. Each clip banks its own directory and the
sweep skips clips that already have `windows.tsv`, so an interrupt resumes. Do not rebuild the
release binary while it runs: the script invokes the binary per clip and a rebuild would put
two binaries behind one receipt. The banked copy is `.lane-asr-stage2/sweep-binary/`,
sha256 `baf38cb6ada267058cb1f8b776ab72303c4fb0c284417ecbfa3d6e25a7eb590c`.

Score it when it finishes:

```
~/hebrew-asr-data/venv-nemo/bin/python tools/check_whisper_transcribe.py \
  --oracle ~/hebrew-asr-data/oracle/whisper-large-v3-ivrit \
  --native .lane-asr-stage2/sweep \
  --checkpoint ~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9 \
  --binary .lane-asr-stage2/sweep-binary/whisper-stage --threads 16 \
  --receipt research/asr-modality-20260909/stage5-transcribe-parity.json
```

The checker scores only clips that have finished, so it is safe to run mid-sweep.

## Next executable stage

1. Finish or stop the sweep at 22:30 UTC, bank `stage5-transcribe-parity.json`, and read the
   per-domain delta against the 0.21 to 0.34 pt band the two oracle backends differ by. Any
   window that flips gets a matched-program FP32 reference before it gets a verdict: the banked
   `hf-fp32` windows pad in waveform space and the CT2 program zero-fills in feature space, so
   on any padded window the two references are not the same program.
3. Native detokenizer and tokenizer binding, for both paths. Until then `NativeReference`
   cannot be claimed: every text number depends on offline tooling.
4. RNNT beyond one clip and one arm: more audio, the `[56,3]`/`[56,6]`/`[56,13]` arms if they
   are ever wanted, session revision semantics, cancellation and reset.

The measured ladder, including everything `NativeReference` and `NativeQualified` still need,
is the "Measured status ladder" section of `ASR-MODALITY-PLAN.md`.

## Rules this lane runs under

CPU only on the rig, one process, `nice -n 15`, efficiency cores, no GPU, no local CI, no
battery, no smoke server. Pushes use `MEMRA_SKIP_PERF_CI=1` and say so in the PR body; hosted
CI still gates the merge. Engine licence is FSL-1.1-ALv2. External implementations only ever
produce offline oracle files.

Do not resume the removed mixed-precision or matrix variants, and do not tighten FP16 below its
measured reference floor.

## Artifact locations

Checkpoint: `~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9/`, both shard hashes verified.
RNNT checkpoint: `~/hebrew-asr-data/models/campaign-20260908/clean-step-21959.nemo`,
sha256 `585c40f214a53e8b5ba565c6176aa6cb548c7c8ab2f7ec5dc6dd45dbdff46ee4`.
CPU oracle root: `~/hebrew-asr-data/oracle-cpu/asr-modality-20260909/` (`final-enc-f32/`,
`final-enc-f16/`, `final-dec-f32/`, `final-dec-f16/`; references `hf-f32/`, `hf-f16/`,
`decoder-hf-f32/`, `decoder-hf-f16-encf32/`).
Rental oracle: `~/hebrew-asr-data/oracle/whisper-large-v3-ivrit/` (MANIFEST.json; 71 CT2 clips
with log-mel, encoder output, beam-1 tokens and post-suppression logits; 8 HF FP32 clips with
raw pre-processor logits).
Receipts: `research/asr-modality-20260909/`.
Tiny fixture directory: `crates/memra-reference/src/speech/fixtures/`.
Synthetic `tiny-encoder.safetensors` needs `git add -f`; generic weight files are ignored.

## Commands

Python: `~/hebrew-asr-data/venv-nemo/bin/python` (numpy, torch, transformers, ct2, nemo, jiwer).
Build: `PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=.lane-asr-stage2/target nice -n 15 \
cargo build -p memra-reference --bin whisper-stage --release -j 2`.
Mel: `whisper-stage mel|mel-clip CHECKPOINT PCM.f32 OUT.f32`.
Encoder: `whisper-stage encoder CHECKPOINT MEL.f32 NEW_OUT f32|f16` (output dir must not exist).
Decoder: `whisper-stage decoder CHECKPOINT ENCODER.f32 TOKENS.txt NEW_OUT f32|f16`.
Transcribe: `whisper-stage transcribe CHECKPOINT PCM.f32 NEW_OUT f32|f16`.
RNNT: `rnnt-stage frontend|encoder|head|stream ARCHIVE ... ` built into
`.lane-asr-stage3/target` so the sweep binary is never rebuilt under it.
Gates: `tools/check_whisper_stages.py`, `tools/check_whisper_decoder.py`,
`tools/check_whisper_real_audio.py`, `tools/check_whisper_transcribe.py`,
`tools/check_rnnt_frontend.py`, `tools/check_rnnt_encoder.py`, `tools/check_rnnt_head.py`,
`tools/check_rnnt_stream.py`.
Captures: `tools/nemo_encoder_oracle.py`, `tools/nemo_rnnt_oracle.py`,
`tools/nemo_stream_oracle.py`.
Run every gate from the worktree root; the receipts hash source paths relative to it.

Clean `.lane-asr-stage2/` and `.lane-asr-stage3/` scratch at handoff after preserving the
binary and receipts.
Both root checkouts stay on main. Keep this worktree/branch for the open PR.
