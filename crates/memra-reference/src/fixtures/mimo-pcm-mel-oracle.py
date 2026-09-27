"""Regenerate the bounded PCM/log-mel pair from the pinned publisher function.

Example:
  python mimo-pcm-mel-oracle.py /tmp/mimo-audio-tokenizer-github-20260927

Requires CPU torch==2.6.0 and torchaudio==2.6.0. This loads only the
publisher's config.py and utils.py, avoiding its model-only flash_attn import.
"""

import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType

import torch
import torchaudio


SOURCE_COMMIT = "b62b59922979bf9f389b373169298a251587653f"
CONFIG_SHA256 = "e0702adae37947e0c980c38bae58ffa0d48bd492d523afe814aa4d73f008c7d1"
SAMPLE_COUNTS = (1680, 721)


def load_publisher_module(root: Path, name: str):
    path = root / "mimo_audio_tokenizer" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mimo_audio_tokenizer.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_f32(path: Path, values: torch.Tensor):
    path.write_bytes(struct.pack(f"<{values.numel()}f", *values.reshape(-1).tolist()))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_pcm(sample_count: int):
    state = 0x9E3779B9
    samples = []
    for i in range(sample_count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        noise = (state / 2**32 - 0.5) * 0.2
        samples.append(
            0.57 * math.sin(2 * math.pi * 440 * i / 24000)
            + 0.19 * math.sin(2 * math.pi * 1270 * i / 24000)
            + 0.08 * math.cos(2 * math.pi * (310 * i / 24000 + 1500 * (i / 24000) ** 2))
            + noise
        )
    pcm = torch.tensor(samples, dtype=torch.float32)
    pcm[0] += 0.31
    pcm[350] -= 0.24
    pcm[-1] += 0.27
    return pcm


def seeded_noise(sample_count: int, seed: int):
    state = seed
    values = []
    for _ in range(sample_count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        values.append(2 * (state / 2**32) - 1)
    return values


def corpus():
    for sample_count in SAMPLE_COUNTS:
        yield f"{sample_count}", make_pcm(sample_count)

    yield "silence-960", torch.zeros(960, dtype=torch.float32)

    n = 1680
    sparse = torch.tensor([
        0.57 * math.sin(2 * math.pi * 440 * i / 24000)
        + 0.19 * math.sin(2 * math.pi * 1270 * i / 24000)
        + 0.08 * math.cos(2 * math.pi * (310 * i / 24000 + 1500 * (i / 24000) ** 2))
        for i in range(n)
    ], dtype=torch.float32)
    sparse[0] += 0.31
    sparse[350] -= 0.24
    sparse[-1] += 0.27
    yield "sparse-1680", sparse

    n = 1680
    yield "tone-1680", torch.tensor(
        [0.72 * math.sin(2 * math.pi * 440 * i / 24000) for i in range(n)],
        dtype=torch.float32,
    )

    n = 1537
    white = seeded_noise(n, 0x31415926)
    yield "noise-1537", torch.tensor([0.3 * value for value in white], dtype=torch.float32)

    n = 2048
    white = seeded_noise(n, 0x27182818)
    voiced = [
        (0.45 + 0.35 * math.sin(2 * math.pi * 3 * i / 24000))
        * sum(math.sin(2 * math.pi * 215 * harmonic * i / 24000) / harmonic for harmonic in range(1, 8))
        * 0.45
        + 0.02 * white[i]
        for i in range(n)
    ]
    yield "voiced-2048", torch.tensor(voiced, dtype=torch.float32)

    n = 1201
    white = seeded_noise(n, 0xDEADBEEF)
    chirp = [
        0.5 * math.sin(2 * math.pi * (220 * i / 24000 + 3500 * (i / 24000) ** 2))
        + 0.015 * white[i]
        for i in range(n)
    ]
    yield "chirp-1201", torch.tensor(chirp, dtype=torch.float32)

    n = 1680
    yield "quiet-tone-1680", torch.tensor(
        [1e-4 * math.sin(2 * math.pi * 440 * i / 24000) for i in range(n)],
        dtype=torch.float32,
    )


def print_precision_diagnosis(utils, config):
    sparse = next(pcm for name, pcm in corpus() if name == "sparse-1680")
    transform = utils.MEL_TRANSFORM
    window = transform.spectrogram.window
    analytic_window = torch.tensor(
        [0.5 - 0.5 * math.cos(2 * math.pi * i / 960) for i in range(960)],
        dtype=torch.float32,
    )
    f32_cosine_window = 0.5 - 0.5 * torch.cos(
        (2 * math.pi / 960) * torch.arange(960, dtype=torch.float32)
    )
    assert torch.equal(window, f32_cosine_window)
    frames = torch.nn.functional.pad(sparse[None, None, :], (480, 480), mode="reflect")
    frames = frames[0, 0].unfold(0, 960, 240)
    filters = transform.mel_scale.fb

    def mel_outputs(hann, fft_dtype):
        windowed_f32 = frames * hann
        magnitudes = torch.fft.rfft(windowed_f32.to(fft_dtype), dim=-1).abs().float()
        raw = (magnitudes @ filters).T
        return raw, torch.log(torch.clamp(raw, min=1e-7))

    source = utils.mel_spectrogram(sparse, config)
    source_raw = transform(sparse[None, :])[0]
    manual_raw_f32, manual_f32 = mel_outputs(window, torch.float32)
    assert torch.equal(source_raw, manual_raw_f32)
    assert torch.equal(source, manual_f32)
    exact_window_raw_f64_fft, exact_window_f64_fft = mel_outputs(window, torch.float64)
    analytic_window_raw_f64_fft, analytic_window_f64_fft = mel_outputs(analytic_window, torch.float64)
    mel, frame = 127, 4
    print(f"torch_mkl_available={torch.backends.mkl.is_available()}")
    print(f"publisher_vs_manual_f32_max_abs={(source - manual_f32).abs().max().item()}")
    print(f"torch_hann_vs_analytic_f32_max_abs={(window - analytic_window).abs().max().item()}")
    print(
        "sparse_mel127_frame4_log "
        f"source={source[mel, frame].item()} "
        f"same_window_f64_fft={exact_window_f64_fft[mel, frame].item()} "
        f"analytic_window_f64_fft={analytic_window_f64_fft[mel, frame].item()}"
    )
    print(
        "sparse_mel127_frame4_raw "
        f"source={source_raw[mel, frame].item()} "
        f"same_window_f64_fft={exact_window_raw_f64_fft[mel, frame].item()} "
        f"analytic_window_f64_fft={analytic_window_raw_f64_fft[mel, frame].item()}"
    )


def main():
    root = Path(sys.argv[1]).resolve()
    found = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert found == SOURCE_COMMIT, found
    assert torch.__version__.startswith("2.6.0"), torch.__version__
    assert torchaudio.__version__.startswith("2.6.0"), torchaudio.__version__
    torch.set_num_threads(1)

    repo = Path(__file__).resolve().parents[4]
    config_path = repo / "crates/memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-config.json"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == CONFIG_SHA256
    sys.dont_write_bytecode = True
    package = ModuleType("mimo_audio_tokenizer")
    package.__path__ = [str(root / "mimo_audio_tokenizer")]
    sys.modules[package.__name__] = package
    config_module = load_publisher_module(root, "config")
    utils = load_publisher_module(root, "utils")
    config = config_module.MiMoAudioTokenizerConfig.from_pretrained(str(config_path))
    assert (config.sampling_rate, config.nfft, config.hop_length, config.window_size,
            config.n_mels, config.fmin, config.fmax) == (24000, 960, 240, 960, 128, 0, None)

    here = Path(__file__).resolve().parent
    print(f"publisher={found} torch={torch.__version__} torchaudio={torchaudio.__version__}")
    print(f"config_sha256={CONFIG_SHA256}")
    for name, pcm in corpus():
        mel = utils.mel_spectrogram(pcm, config)
        assert tuple(mel.shape) == (128, pcm.numel() // 240 + 1)
        raw_mel = utils.MEL_TRANSFORM(pcm[None, :])
        peak = pcm.abs().max().item()
        minimum_ratio = raw_mel.min().item() / peak if peak else 0.0
        pcm_sha = write_f32(here / f"mimo-pcm-mel-{name}.pcm.f32", pcm)
        mel_sha = write_f32(here / f"mimo-pcm-mel-{name}.logmel.f32", mel)
        print(f"case={name} shape={list(mel.shape)} min_mel_per_pcm_peak={minimum_ratio} pcm_sha256={pcm_sha}")
        print(f"mel_sha256={mel_sha} min={mel.min().item()} max={mel.max().item()}")
    window = utils.MEL_TRANSFORM.spectrogram.window
    print(f"hann960_sha256={write_f32(here / 'mimo-pcm-mel-hann960.f32', window)}")
    filters = utils.MEL_TRANSFORM.mel_scale.fb.T.contiguous()
    print(f"htk128x481_sha256={write_f32(here / 'mimo-pcm-mel-htk128x481.f32', filters)}")
    print_precision_diagnosis(utils, config)


if __name__ == "__main__":
    main()
