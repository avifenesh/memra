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
    for sample_count in SAMPLE_COUNTS:
        pcm = make_pcm(sample_count)
        mel = utils.mel_spectrogram(pcm, config)
        assert tuple(mel.shape) == (128, sample_count // 240 + 1)
        pcm_sha = write_f32(here / f"mimo-pcm-mel-{sample_count}.pcm.f32", pcm)
        mel_sha = write_f32(here / f"mimo-pcm-mel-{sample_count}.logmel.f32", mel)
        print(f"samples={sample_count} shape={list(mel.shape)} pcm_sha256={pcm_sha}")
        print(f"mel_sha256={mel_sha} min={mel.min().item()} max={mel.max().item()}")


if __name__ == "__main__":
    main()
