# MiMo PCM-to-mel same-host comparison, 2026-09-27

Decision: **the bounded PCM-to-mel component passes a same-host source
comparison on the tested AMD EPYC 9655; full audio remains unqualified**.

The model source was
`XiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
Memra's raw-modal source run used
`f9f322d99656945b28279057d945794ba438be81`. The same-host mel
dump example used
`05367f3e33da52998afc7136eeb0d4115185f3f8`, which adds only the
diagnostic output program above that source head. The comparison ran on
one Vast development box with 2×RTX PRO 6000 Blackwell Server Edition
and an AMD EPYC 9655 host CPU.

The pinned tokenizer utility's `mel_spectrogram` function came from
`XiaomiMiMo/MiMo-Audio-Tokenizer@b62b59922979bf9f389b373169298a251587653f`.
It ran with the V2.6 audio-tokenizer config, CPU Torch and torchaudio
`2.6.0+cpu`, and one Torch thread. Memra used standalone oneMKL 2024.2
with `MKL_THREADING_LAYER=SEQUENTIAL`. Ten 24 kHz F32 PCM inputs were
paired: nine fixed fixtures and the 2,048-sample waveform from the
raw-modal source probe.

| Comparison | Largest absolute post-log-mel difference |
| --- | ---: |
| Memra versus Torch on the same EPYC host, all ten cases | 9.536743e-7 |
| Memra versus Torch on the raw-modal probe waveform | 4.768372e-7 |
| Torch on EPYC versus the pinned fixture made on the local host | 0.380252838 |

The raw-modal source run's mel SHA-256 matched the standalone Memra
mel dump for that same waveform. The fixed-fixture `verify-mkl` test
failed in both debug and release on EPYC: its first case had
`2.1457672e-6` absolute difference against a `2e-6` gate. Torch 2.6
on EPYC also differed from that fixed fixture, and sparse pure-tone
differences reached `0.380252838`. These red test receipts are
preserved. The fixed fixture is a local-backend regression check;
source fidelity on another CPU needs a same-host publisher comparison.
No tolerance was raised or failed test recast as passing.

This proves only the bounded, decoded, mono 24 kHz PCM-to-mel step on
this CPU and source version. The public older tokenizer implementation
does not carry the pinned V2.6 hybrid attention schedule, so it cannot
serve as an independent oracle for the 24-layer codec or its 20 RVQ
code IDs. Full audio quality, longer clips, native library packaging,
request intake, and serving remain open.

The private raw archive is
`~/.local/state/mimo-memra-vast-raw-modal/20260927/qualification-receipts.tar.gz`
with SHA-256
`c8802f4ea87312207e469fbbbd84e82c1bc09922b1d5d510d1171fd2069d4e92`.
It includes both failed fixed-fixture attempts, the first publisher
vision wrapper failure, the later successful source comparisons, and
the scripts and output arrays. The task-owned VM and timer were closed;
no customer route changed.
