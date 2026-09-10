#!/usr/bin/env python3
"""Capture a pinned CPU FP32 reference for the whole streaming session: audio in, tokens out.

This is the reference's own `conformer_stream_step`, which carries the encoder caches and the
predictor hypothesis across chunks, so it pins the thing a session does rather than the pieces
a stage does. Per chunk it banks the emitted token ids and the running transcript.

Determinism: eval mode, one thread, dither off. Nothing here runs inside Memra.
"""
import argparse, hashlib, json, os

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

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True)
    p.add_argument("--pcm", required=True, help="int16 little-endian mono 16 kHz")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--language", default="he-IL")
    args = p.parse_args()

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
        EncDecRNNTBPEModelWithPrompt,
    )

    model = EncDecRNNTBPEModelWithPrompt.restore_from(args.nemo, map_location="cpu").eval()
    model.preprocessor.featurizer.dither = 0.0
    model.set_inference_prompt(args.language)
    model.encoder.set_default_att_context_size([56, 0])
    model.encoder.setup_streaming_params(att_context_size=[56, 0])
    params = model.encoder.streaming_cfg

    pcm = np.fromfile(args.pcm, dtype="<i2").astype(np.float32) / 32768.0
    pcm = pcm[: int(args.seconds * model.cfg.sample_rate)]
    signal = torch.from_numpy(pcm).unsqueeze(0)
    length = torch.tensor([pcm.size], dtype=torch.int64)
    mel, mel_length = model.preprocessor(input_signal=signal, length=length)

    out = args.out
    os.makedirs(out, exist_ok=False)

    def pick(value, first):
        return (value[0] if first else value[1]) if isinstance(value, list) else value

    cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
    previous = None
    previous_pred = None
    frames = int(mel_length[0])
    chunks = []
    index = 0
    buffer_idx = 0
    while buffer_idx < frames:
        first = buffer_idx == 0
        chunk_size = pick(params.chunk_size, first)
        shift_size = pick(params.shift_size, first)
        audio_chunk = mel[:, :, buffer_idx : buffer_idx + chunk_size]
        zeros_pads = None
        if first and isinstance(params.pre_encode_cache_size, list):
            cache_pre_encode = torch.zeros((1, mel.shape[1], params.pre_encode_cache_size[0]))
        else:
            size = pick(params.pre_encode_cache_size, False)
            start = max(0, buffer_idx - size)
            cache_pre_encode = mel[:, :, start:buffer_idx]
            if cache_pre_encode.shape[-1] < size:
                zeros_pads = torch.zeros(
                    (1, mel.shape[1], size - cache_pre_encode.shape[-1])
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
            pred_out,
            transcribed,
            cache_channel,
            cache_time,
            cache_len,
            previous,
        ) = model.conformer_stream_step(
            processed_signal=window,
            processed_signal_length=window_length,
            cache_last_channel=cache_channel,
            cache_last_time=cache_time,
            cache_last_channel_len=cache_len,
            keep_all_outputs=last,
            previous_hypotheses=previous,
            previous_pred_out=previous_pred,
            drop_extra_pre_encoded=0 if first else params.drop_extra_pre_encoded,
            return_transcription=True,
        )
        previous_pred = pred_out
        emitted = [int(t) for t in previous[0].y_sequence.tolist()]
        text = transcribed[0] if transcribed else ""
        if hasattr(text, "text"):
            text = text.text
        chunks.append(
            {
                "index": index,
                "buffer_idx": buffer_idx,
                "tokens_so_far": emitted,
                "transcript_so_far": text,
            }
        )
        buffer_idx += shift_size
        index += 1

    result = {
        "schema": "memra-nemo-stream-oracle-v1",
        "archive": os.path.abspath(args.nemo),
        "archive_sha256": sha(args.nemo),
        "pcm_source": os.path.abspath(args.pcm),
        "samples": int(pcm.size),
        "language": args.language,
        "att_context_size": [56, 0],
        "torch": torch.__version__,
        "nemo": __import__("nemo").__version__,
        "threads": 1,
        "dither": 0.0,
        "chunks": chunks,
        "final_tokens": chunks[-1]["tokens_so_far"] if chunks else [],
        "final_transcript": chunks[-1]["transcript_so_far"] if chunks else "",
        "scope": "CPU FP32 streaming session reference: audio to tokens over the [56,0] arm, "
        "carrying encoder caches and the predictor hypothesis across chunks. No GPU.",
    }
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(
        f"banked {len(chunks)} streaming chunks, final {len(result['final_tokens'])} tokens: "
        f"{result['final_transcript']!r}"
    )


if __name__ == "__main__":
    main()
