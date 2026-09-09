#!/usr/bin/env python3
"""Capture a pinned CPU FP32 reference for the RNNT head: prompt, predictor, joint, greedy.

The encoder is already gated separately. This restores the checkpoint's own model class so the
prompt wiring is the model's, not a reconstruction, and banks:

  prompted encoder   prompt_kernel(concat(encoder frames, one-hot he-IL)), per frame
  predictor          embedding and two LSTM layers over a fixed token sequence, with states
  joint              logits for a grid of encoder frames against predictor rows
  greedy             the token ids NeMo's own greedy RNNT emits for the clip

Determinism: eval mode, one thread, no dither. Nothing here runs inside Memra.
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
    p.add_argument("--encoder-oracle", required=True, help="directory from nemo_encoder_oracle")
    p.add_argument("--out", required=True)
    p.add_argument("--language", default="he-IL")
    args = p.parse_args()

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
        EncDecRNNTBPEModelWithPrompt,
    )

    model = EncDecRNNTBPEModelWithPrompt.restore_from(args.nemo, map_location="cpu").eval()
    prompt_index = model.cfg.model_defaults.prompt_dictionary[args.language]

    manifest = json.loads(open(os.path.join(args.encoder_oracle, "MANIFEST.json")).read())
    frames = []
    for chunk in manifest["chunks"]:
        entry = manifest["tensors"][f"chunk-{chunk['index']:03d}-encoder"]
        path = os.path.join(args.encoder_oracle, entry["path"])
        if sha(path) != entry["sha256"]:
            raise SystemExit("encoder oracle hash mismatch")
        frames.append(np.fromfile(path, dtype="<f4").reshape(entry["shape"]))
    # Each chunk banks [width, frames]; the head consumes [time, width].
    encoded = torch.from_numpy(np.concatenate(frames, axis=1).T.copy()).unsqueeze(0)

    out = args.out
    os.makedirs(out, exist_ok=False)
    entries = {}

    def bank(name, tensor):
        array = np.ascontiguousarray(tensor.detach().float().numpy())
        path = os.path.join(out, f"{name}.f32.bin")
        array.tofile(path)
        entries[name] = {
            "path": f"{name}.f32.bin",
            "shape": list(array.shape),
            "dtype": "float32",
            "sha256": sha(path),
        }

    bank("encoder", encoded[0])

    # Prompt conditioning, exactly as the model class wires it: a one-hot of the language over
    # every time step, concatenated onto the encoder row, through a two-layer kernel.
    time_steps = encoded.shape[1]
    prompt = torch.zeros(1, time_steps, model.num_prompts)
    prompt[:, :, prompt_index] = 1.0
    prompted = model.prompt_kernel(torch.cat([encoded, prompt], dim=-1))
    bank("prompted-encoder", prompted[0])

    # Predictor: a fixed token walk, banked step by step with its state.
    tokens = [0, 5, 61, 137, 900, 4321, 13086]
    state = None
    predictor_rows = []
    for step, token in enumerate(tokens):
        y = torch.tensor([[token]], dtype=torch.int32)
        g, state = model.decoder.predict(y=y, state=state, add_sos=False, batch_size=1)
        predictor_rows.append(g[:, 0])
        bank(f"predictor-{step:02d}-out", g[0, 0])
        for layer, piece in enumerate(state):
            bank(f"predictor-{step:02d}-state-{layer}", piece)
    # The start of a sequence is a null token, which the reference produces with y=None.
    start, start_state = model.decoder.predict(y=None, state=None, add_sos=False, batch_size=1)
    bank("predictor-start-out", start[0, 0])
    for layer, piece in enumerate(start_state):
        bank(f"predictor-start-state-{layer}", piece)

    # Joint: every encoder frame against the start row and a mid-walk row.
    encoder_side = prompted.transpose(1, 2)  # B x D x T, which is what joint() expects
    for label, row in (("start", start), ("walk", predictor_rows[-1].unsqueeze(1))):
        logits = model.joint.joint(encoder_side.transpose(1, 2), row)
        bank(f"joint-{label}", logits[0, :, 0])

    # Greedy: the reference's own decoding over the same frames.
    lengths = torch.tensor([time_steps], dtype=torch.int64)
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoder_side, encoded_lengths=lengths, return_hypotheses=True
    )
    best = hypotheses[0] if isinstance(hypotheses, list) else hypotheses
    if isinstance(best, (list, tuple)):
        best = best[0]
    emitted = [int(t) for t in best.y_sequence.tolist()]
    text = model.tokenizer.ids_to_text(emitted)

    result = {
        "schema": "memra-nemo-rnnt-head-oracle-v1",
        "archive": os.path.abspath(args.nemo),
        "archive_sha256": sha(args.nemo),
        "encoder_oracle_manifest_sha256": sha(
            os.path.join(args.encoder_oracle, "MANIFEST.json")
        ),
        "language": args.language,
        "prompt_index": int(prompt_index),
        "num_prompts": int(model.num_prompts),
        "blank_index": int(model.decoder.blank_idx),
        "vocabulary": int(model.tokenizer.vocab_size),
        "max_symbols": int(model.cfg.decoding.greedy.max_symbols),
        "torch": torch.__version__,
        "nemo": __import__("nemo").__version__,
        "threads": 1,
        "frames": int(time_steps),
        "predictor_tokens": tokens,
        "greedy_tokens": emitted,
        "greedy_text": text,
        "tensors": entries,
        "scope": "CPU FP32 reference for the RNNT head on already-gated encoder frames. No "
        "audio, no encoder, no GPU, no serving surface.",
    }
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(
        f"banked {time_steps} frames, prompt {args.language}={prompt_index}, "
        f"greedy {len(emitted)} tokens: {text!r}"
    )


if __name__ == "__main__":
    main()
