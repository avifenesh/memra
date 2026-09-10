#!/usr/bin/env python3
"""Offline HF oracle for a fixed 2-second synthetic clip. Never a serving backend.

Run with nice, one CPU thread/process. Weights and outputs live outside the repository.
The native runner consumes only the resulting PCM/tensor files, never this implementation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

# Set before numpy/torch imports. No device discovery or CUDA API is called.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("mel", "encoder"), default="mel")
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import WhisperConfig, WhisperFeatureExtractor

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(416)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "manifest.json").exists():
        raise SystemExit("refusing to overwrite a completed oracle manifest")
    started = time.monotonic()
    files = {}

    def save(name, array):
        if isinstance(array, torch.Tensor):
            array = array.detach().float().cpu().numpy()
        array = np.asarray(array, dtype="<f4")
        raw = array.tobytes(order="C")
        path = args.out / (name + ".f32")
        path.write_bytes(raw)
        files[name] = {"path": path.name, "shape": list(array.shape), "dtype": "F32", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    # Fixed waveform with ramps, multiple bands, a chirp and low-amplitude noise.
    # This is a correctness fixture, not speech quality evidence or evaluation audio.
    t = np.arange(32000, dtype=np.float64) / 16000.0
    ramp = np.minimum(t / 0.05, 1.0) * np.minimum((2.0-t) / 0.05, 1.0)
    pcm = (ramp * (0.16*np.sin(2*np.pi*233*t) + 0.07*np.sin(2*np.pi*997*t)
                  + 0.04*np.sin(2*np.pi*(430*t+600*t*t)))
           + np.random.default_rng(416).normal(0, 0.002, len(t))).astype(np.float32)
    save("pcm", pcm)
    processor = WhisperFeatureExtractor.from_pretrained(args.checkpoint, local_files_only=True)
    mel = processor(pcm, sampling_rate=16000, return_tensors="pt", device="cpu").input_features
    save("log-mel", mel[0])
    print("oracle mel", list(mel.shape), flush=True)
    if args.stage == "encoder":
        from safetensors import safe_open
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        config = WhisperConfig.from_pretrained(args.checkpoint, local_files_only=True)
        config._attn_implementation = "eager"
        with torch.device("meta"):
            encoder = WhisperEncoder(config)
        weights = {}
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
        for name in sorted(set(index["weight_map"].values())):
            with safe_open(args.checkpoint / name, framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key.startswith("model.encoder."):
                        weights[key.removeprefix("model.encoder.")] = shard.get_tensor(key)
        encoder.load_state_dict(weights, strict=True, assign=True)
        del weights
        encoder.eval()
        if args.dtype == "f16":
            encoder.half()
            mel = mel.half()
        handles = []
        for i, conv in enumerate((encoder.conv1, encoder.conv2), 1):
            handles.append(conv.register_forward_hook(lambda _m, _a, out, i=i: save(f"conv{i}", torch.nn.functional.gelu(out)[0].transpose(0,1))))
        for i, layer in enumerate(encoder.layers):
            def hook(_m, _a, out, i=i):
                save(f"layer-{i:02d}", out[0])
                print("oracle layer", i, round(time.monotonic()-started, 2), flush=True)
            handles.append(layer.register_forward_hook(hook))
        with torch.inference_mode():
            out = encoder(mel).last_hidden_state[0]
        save("encoder", out)
        for handle in handles:
            handle.remove()
    import transformers
    root = Path(transformers.__file__).parent
    source_hashes = {}
    for file in ("models/whisper/modeling_whisper.py", "models/whisper/feature_extraction_whisper.py", "audio_utils.py"):
        source_hashes[file] = hashlib.sha256((root / file).read_bytes()).hexdigest()
    manifest = {
        "schema": "memra-whisper-cpu-oracle-v1", "source_repository": "ivrit-ai/whisper-large-v3",
        "source_revision": "766847c9795b3b5cc0d42f8476199c711d5cee21",
        "input": "deterministic synthetic 2-second clip; not a WER sample",
        "input_pcm_sha256": files["pcm"]["sha256"], "sample_rate": 16000,
        "padded_samples": 480000, "dtype": args.dtype, "stage": args.stage,
        "attention": "eager", "device": "cpu", "threads": 1,
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "safetensors")},
        "source_sha256": source_hashes,
        "config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": files, "elapsed_seconds": time.monotonic()-started,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": "captured", "stage": args.stage, "dtype": args.dtype, "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
