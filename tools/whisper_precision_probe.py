#!/usr/bin/env python3
"""Offline HF layer probe: isolate attention/FFN weight rounding on one saved input."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--layer',type=int,default=20)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit('refusing to overwrite a precision receipt')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = WhisperConfig.from_pretrained(args.checkpoint,local_files_only=True)
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        layer = WhisperEncoderLayer(config)
    state = {}
    prefix = f'model.encoder.layers.{args.layer}.'
    index = json.loads((args.checkpoint/'model.safetensors.index.json').read_text())
    for shard_name in sorted(set(index['weight_map'].values())):
        with safe_open(args.checkpoint/shard_name,framework='pt',device='cpu') as shard:
            for name in shard.keys():
                if name.startswith(prefix):
                    state[name[len(prefix):]] = shard.get_tensor(name).clone()
    layer.load_state_dict(state,strict=True,assign=True)
    layer.eval()
    source = args.oracle/f'layer-{args.layer-1:02d}.f32'
    raw = source.read_bytes()
    x = torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(1,1500,1280))
    base = {}
    result = []
    names = ['self_attn_layer_norm','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.out_proj','final_layer_norm','fc1','fc2']
    modules = dict(layer.named_modules())
    with torch.inference_mode():
        for profile in ['f32','all_matrix_f16','attention_f16','ffn_f16']:
            rounded = []
            weights = {}
            for name,value in state.items():
                cast = value.ndim==2 and (profile=='all_matrix_f16' or (profile=='attention_f16' and name.startswith('self_attn.')) or (profile=='ffn_f16' and name.startswith(('fc1.','fc2.'))))
                weights[name] = value.half().float() if cast else value
                if cast: rounded.append(name)
            layer.load_state_dict(weights,strict=True,assign=True)
            captures = {}
            handles = [modules[name].register_forward_hook(lambda _m,_a,out,name=name: captures.__setitem__(name,out.detach().clone())) for name in names]
            captures['output'] = layer(x,None)
            for h in handles: h.remove()
            if profile=='f32': base = captures
            rows=[]
            for name,value in captures.items():
                delta = (value-base[name]).abs()
                rows.append({'stage':name,'max_abs':delta.max().item(),'mean_abs':delta.mean().item()})
            result.append({'profile':profile,'rounded_tensors':rounded,'stages':rows})
            print(profile,rows[-1],flush=True)
    receipt={'layer':args.layer,'input_sha256':hashlib.sha256(raw).hexdigest(),'input_shape':[1,1500,1280],
        'scope':'isolated HF block on fixed synthetic two-second clip, FP32 activations in every arm',
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profiles':result}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
