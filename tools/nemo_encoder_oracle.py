#!/usr/bin/env python3
"""Capture a pinned CPU FP32 reference for the FastConformer cache-aware encoder.

Offline capture tooling. It builds the preprocessor and `ConformerEncoder` straight from the
archive's own `model_config.yaml` and loads the matching subset of the checkpoint's tensors, so
no custom model class and no prompt kernel are involved: this pins the encoder alone.

Determinism: dither is forced to zero, dropout is off in eval, and the run is single threaded.
The manifest records both, because a capture with dither on is a different program.

Nothing here runs inside Memra. The native encoder consumes these files and is gated on them.
"""
import argparse, hashlib, io, json, os, tarfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import torch
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True)
    p.add_argument("--pcm", required=True, help="int16 little-endian mono 16 kHz")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--left-context", type=int, default=56)
    p.add_argument("--right-context", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    from nemo.collections.asr.modules import (
        AudioToMelSpectrogramPreprocessor,
        ConformerEncoder,
    )

    archive = tarfile.open(args.nemo)
    config = yaml.safe_load(
        archive.extractfile(
            next(m for m in archive.getmembers() if m.name.endswith("model_config.yaml"))
        ).read()
    )
    weights = io.BytesIO(
        archive.extractfile(
            next(m for m in archive.getmembers() if m.name.endswith("model_weights.ckpt"))
        ).read()
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)

    front = dict(config["preprocessor"])
    front.pop("_target_", None)
    # Dither is a random perturbation. A reference cannot carry one.
    front["dither"] = 0.0
    preprocessor = AudioToMelSpectrogramPreprocessor(**front).eval()

    encoder_config = dict(config["encoder"])
    encoder_config.pop("_target_", None)
    encoder = ConformerEncoder(**encoder_config).eval()
    missing, unexpected = encoder.load_state_dict(
        {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")},
        strict=True,
    )
    assert not missing and not unexpected, (missing, unexpected)
    for name, buffer in preprocessor.featurizer.named_buffers():
        key = f"preprocessor.featurizer.{name}"
        if key in state:
            buffer.copy_(state[key])

    pcm = np.fromfile(args.pcm, dtype="<i2").astype(np.float32) / 32768.0
    samples = int(args.seconds * front["sample_rate"])
    pcm = pcm[:samples]
    signal = torch.from_numpy(pcm).unsqueeze(0)
    length = torch.tensor([pcm.size], dtype=torch.int64)
    mel, mel_length = preprocessor(input_signal=signal, length=length)

    out = args.out
    os.makedirs(out, exist_ok=False)
    entries = {}

    def bank(name, tensor):
        array = tensor.detach().contiguous().float().numpy()
        path = os.path.join(out, f"{name}.f32.bin")
        array.tofile(path)
        entries[name] = {
            "path": f"{name}.f32.bin",
            "shape": list(array.shape),
            "dtype": "float32",
            "sha256": sha(path),
        }
        return array

    bank("pcm", torch.from_numpy(pcm))
    bank("mel", mel[0])

    context = [args.left_context, args.right_context]
    encoder.set_default_att_context_size(context)
    encoder.setup_streaming_params(att_context_size=context)
    params = encoder.streaming_cfg
    cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(batch_size=1)
    bank("initial-cache-last-channel", cache_channel)
    bank("initial-cache-last-time", cache_time)

    # Offline full-context pass over the same features, as the control the streaming run has to
    # approach but is not required to equal: limited context is a different program.
    offline, offline_length = encoder(audio_signal=mel, length=mel_length)
    bank("offline-encoder", offline[0])

    # Layer-level captures. A 1e-3 gate on a 24-block stack is undebuggable without the
    # intermediates, so every chunk banks its pre-encode row, its positional embedding and
    # every block output.
    traces = {}

    def record(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            traces.setdefault(name, []).append(value.detach().clone())

        return hook

    handles = [encoder.pre_encode.register_forward_hook(record("pre-encode"))]

    def record_pos(_module, _inputs, output):
        traces.setdefault("pos-emb", []).append(output[1].detach().clone())

    handles.append(encoder.pos_enc.register_forward_hook(record_pos))
    for layer_index, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_hook(record(f"layer-{layer_index:02d}")))

    def pick(value, first):
        if isinstance(value, list):
            return value[0] if first else value[1]
        return value

    frames = int(mel_length[0])
    chunks = []
    index = 0
    buffer_idx = 0
    while buffer_idx < frames:
        first = buffer_idx == 0
        chunk_size = pick(params.chunk_size, first)
        shift_size = pick(params.shift_size, first)
        sampling_frames = pick(getattr(params, "sampling_frames", 0), first)
        audio_chunk = mel[:, :, buffer_idx : buffer_idx + chunk_size]
        if sampling_frames and audio_chunk.shape[-1] < sampling_frames:
            break
        zeros_pads = None
        if first and isinstance(params.pre_encode_cache_size, list):
            cache_pre_encode = torch.zeros(
                (1, mel.shape[1], params.pre_encode_cache_size[0]), dtype=mel.dtype
            )
        else:
            size = pick(params.pre_encode_cache_size, False)
            start = max(0, buffer_idx - size)
            cache_pre_encode = mel[:, :, start:buffer_idx]
            if cache_pre_encode.shape[-1] < size:
                zeros_pads = torch.zeros(
                    (1, mel.shape[1], size - cache_pre_encode.shape[-1]), dtype=mel.dtype
                )
        added = cache_pre_encode.shape[-1]
        window = torch.cat((cache_pre_encode, audio_chunk), dim=-1)
        if zeros_pads is not None:
            window = torch.cat((zeros_pads, window), dim=-1)
            added += zeros_pads.shape[-1]
        window_length = torch.tensor(
            [min(frames - buffer_idx + added, window.shape[-1])], dtype=torch.int64
        )
        last = buffer_idx + shift_size >= frames
        (
            step_out,
            step_len,
            cache_channel,
            cache_time,
            cache_len,
        ) = encoder.cache_aware_stream_step(
            processed_signal=window,
            processed_signal_length=window_length,
            cache_last_channel=cache_channel,
            cache_last_time=cache_time,
            cache_last_channel_len=cache_len,
            keep_all_outputs=last,
            drop_extra_pre_encoded=0 if first else params.drop_extra_pre_encoded,
        )
        bank(f"chunk-{index:03d}-mel", window[0])
        bank(f"chunk-{index:03d}-encoder", step_out[0])
        for name, values in traces.items():
            if name == "pos-emb":
                bank(f"chunk-{index:03d}-pos-emb", values[-1][0])
                # The module returns (x, pos_emb); the hook keeps the first element, so grab
                # the embedding from the second by re-running nothing: bank what it produced.
                continue
            bank(f"chunk-{index:03d}-{name}", values[-1][0])
        traces.clear()
        bank(f"chunk-{index:03d}-cache-last-channel", cache_channel)
        bank(f"chunk-{index:03d}-cache-last-time", cache_time)
        chunks.append(
            {
                "index": index,
                "buffer_idx": buffer_idx,
                "mel_frames_in": int(window.shape[-1]),
                "declared_length": int(window_length[0]),
                "pre_encode_frames": added,
                "keep_all_outputs": bool(last),
                "drop_extra_pre_encoded": 0 if first else int(params.drop_extra_pre_encoded),
                "output_frames": int(step_out.shape[-1]),
                "valid_output_frames": int(step_len[0]),
                "cache_last_channel_len": int(cache_len[0]),
            }
        )
        buffer_idx += shift_size
        index += 1

    manifest = {
        "schema": "memra-nemo-encoder-oracle-v1",
        "archive": os.path.abspath(args.nemo),
        "archive_sha256": sha(args.nemo),
        "pcm_source": os.path.abspath(args.pcm),
        "seconds": args.seconds,
        "samples": int(pcm.size),
        "torch": __import__("torch").__version__,
        "nemo": __import__("nemo").__version__,
        "threads": 1,
        "dither": 0.0,
        "eval_mode": True,
        "att_context_size": context,
        "streaming": {
            key: (value if not hasattr(value, "tolist") else value.tolist())
            for key, value in vars(params).items()
            if not key.startswith("_")
        },
        "preprocessor": front,
        "encoder": {
            k: v for k, v in encoder_config.items() if not isinstance(v, (dict,))
        },
        "chunks": chunks,
        "tensors": entries,
        "scope": "CPU FP32 encoder reference for the [56,0] cache-aware arm. Encoder only: no "
        "prompt kernel, no predictor, no joint, no decoding, no GPU.",
    }
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"banked {len(chunks)} chunks, mel {list(mel.shape)}, offline encoder "
        f"{list(offline.shape)}, streaming {manifest['streaming']}"
    )


if __name__ == "__main__":
    main()
