# Real-audio parity against the rental oracle

Oracle: `~/hebrew-asr-data/oracle/whisper-large-v3-ivrit/`, cell lock
`876f0a7f74268c6b914623bce2799bc61d7c4d901a98cd6f886600362da6bbf2`.
71 CT2 clips (364 windows of 30 seconds) plus 8 HF FP32 clips, recorded Hebrew speech.
Runner and gate: `tools/check_whisper_real_audio.py`. This is frontend and encoder
numerics only. No WER, no transcription policy, no GPU, no serving qualification.

## Frontend: 364 of 364 windows, 1.1920929e-7

Every window in the oracle matches at **1.1920928955078125e-7**, one FP32 ULP at these
magnitudes, against a 1e-3 bound. That includes all 71 partial trailing windows.
Receipt: `stage4-real-audio-mel.json`. Total cost 65 s wall, 61 s user, one niced core.

Getting there required a frontend the reference contract actually has. The first sweep
passed 21 of 364, and the 21 were exactly the single-window clips. Three separate facts
were behind that, none of them numerical:

1. **The dynamic-range clamp is clip-wide, not window-wide.** faster-whisper builds one
   feature array per utterance and clamps at `global_max - 8`. Computing a window in
   isolation clamps at that window's own maximum. On `d1-000` window 0 the oracle floor is
   -0.5534494 and the isolated-window floor is -0.5959742; the predicted clip-global floor
   matched the oracle to the printed digit.
2. **A window edge needs the audio that follows it.** The centered STFT at frame 3000 reads
   past the window. Isolated, it reflects instead, and that frame alone moved by 0.2581367.
3. **Past the last sample the extractor reads zeros, it does not reflect.** Reflecting there
   left 13 windows failing, every one of them the last window of its clip, worst 0.2506757.

`WhisperFrontend::compute_clip` implements the clip program: frames are
`samples / 160 + 1`, the reflection applies only at the start of the utterance, and the
clamp floor is clip-wide. Native frame counts match the oracle exactly on every clip.
A window is then the oracle's own construction: content frames copied into a zero-filled
`[128,3000]`. The gate also asserts that the oracle's own pad region is zero, so the
zero-fill convention cannot be assumed silently.

`compute` is unchanged and still serves the padded single-window path. The two are
different programs on purpose, and `clip_features_use_a_clip_wide_clamp_and_read_zeros_past_the_last_sample`
pins that they disagree, so a later refactor cannot quietly collapse them.

Red arms: this gate failed 343 of 364 before the clip program, and 13 of 364 with the
wrong clip-end convention. Both failures were real oracle disagreement, not injected.

## Cost of full parity, measured

The frontend sweep is cheap. The encoder is not: one 30-second window costs about
230 to 270 s of CPU in the reference path. Full oracle parity is therefore roughly
25 CPU-hours for the 364 encoder windows, and the beam-1 token sweep costs more again,
since a window decodes on the order of 150 steps at about 3.1 s per step. That is a
rented-box job, not a rig job. The encoder rows here are a bounded subset.
