#!/usr/bin/env python3
"""Generate small independent HF Whisper encoder fixtures for hosted CPU tests."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "tiny-encoder.safetensors").exists():
        raise SystemExit("refusing to overwrite pinned tiny fixture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(416)
    config = WhisperConfig(d_model=16, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, num_mel_bins=128, max_source_positions=8,
        max_target_positions=16, vocab_size=64, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, decoder_start_token_id=1, dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    encoder = WhisperEncoder(config).eval()
    state = {"model.encoder."+k: v.detach().contiguous() for k,v in encoder.state_dict().items()}
    save_file(state, args.out / "tiny-encoder.safetensors")
    mel = torch.randn(1,128,16)
    mel[0,:,0] = -1.5
    mel[0,:,-1] = 0.25
    files = {}
    def save(name, array):
        data = array.detach().float().numpy().astype("<f4")
        raw = data.tobytes()
        path = args.out / (name+".f32")
        path.write_bytes(raw)
        files[path.name] = {"shape": list(data.shape), "sha256": hashlib.sha256(raw).hexdigest()}
    save("tiny-mel", mel[0])
    for numeric in ("f32", "f16"):
        if numeric == "f16":
            encoder.half()
            mel = mel.half()
        handles = []
        for i,conv in enumerate((encoder.conv1,encoder.conv2),1):
            handles.append(conv.register_forward_hook(lambda _m,_a,out,i=i: save(f"tiny-{numeric}-conv{i}",torch.nn.functional.gelu(out)[0].transpose(0,1))))
        for i,layer in enumerate(encoder.layers):
            handles.append(layer.register_forward_hook(lambda _m,_a,out,i=i: save(f"tiny-{numeric}-layer-{i:02d}",out[0])))
        with torch.inference_mode():
            save(f"tiny-{numeric}-encoder", encoder(mel).last_hidden_state[0])
        for handle in handles:
            handle.remove()
    import transformers
    manifest = {"seed":416,"config":config.to_dict(),"versions":{"torch":torch.__version__,"transformers":transformers.__version__},"files":files,
        "checkpoint_sha256":hashlib.sha256((args.out/'tiny-encoder.safetensors').read_bytes()).hexdigest(),
        "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"synthetic topology only, no native support or real-checkpoint qualification"}
    (args.out/'tiny-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('captured', len(files), 'tensor fixtures')


if __name__ == '__main__':
    main()
