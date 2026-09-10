#!/usr/bin/env python3
"""Offline cached Whisper decoder oracle on an already captured two-second encoder fixture."""
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
    from transformers.models.whisper.modeling_whisper import WhisperDecoder
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--encoder-oracle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dtype',choices=('f32','f16'),required=True)
    p.add_argument('--tokens-from',type=Path,help='reuse the F32 input-token sequence for the F16 numerical comparison')
    args = p.parse_args()
    if args.out.exists():raise SystemExit('refusing to overwrite decoder capture')
    args.out.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=WhisperConfig.from_pretrained(args.checkpoint,local_files_only=True)
    config._attn_implementation='eager'
    with torch.device('meta'):decoder=WhisperDecoder(config)
    state={}
    index=json.loads((args.checkpoint/'model.safetensors.index.json').read_text())
    for shard_name in sorted(set(index['weight_map'].values())):
        with safe_open(args.checkpoint/shard_name,framework='pt',device='cpu') as shard:
            for name in shard.keys():
                if name.startswith('model.decoder.'):
                    state[name.removeprefix('model.decoder.')]=shard.get_tensor(name)
    decoder.load_state_dict(state,strict=True,assign=True)
    del state
    decoder.eval()
    encoder_manifest=json.loads((args.encoder_oracle/'manifest.json').read_text())
    entry=encoder_manifest['files']['encoder']
    raw=(args.encoder_oracle/entry['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise SystemExit('encoder fixture hash mismatch')
    encoded=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(1,*entry['shape']))
    if args.dtype=='f16':decoder.half();encoded=encoded.half()
    files={}
    def save(name,value):
        data=value.detach().float().numpy().astype('<f4')
        raw=data.tobytes()
        path=args.out/(name+'.f32');path.write_bytes(raw)
        files[name]={'path':path.name,'shape':list(data.shape),'sha256':hashlib.sha256(raw).hexdigest(),'dtype':'F32'}
    save('encoder-input',encoded[0])
    fixed=json.loads(args.tokens_from.read_text())['input_tokens'] if args.tokens_from else None
    prefix=[50258,50279,50360,50364]
    tokens=[];raw_argmax=[];next_ids=[];cache=None
    policy=json.loads((args.checkpoint/'generation_config.json').read_text())
    suppression=policy.get('suppress_tokens') or []
    with torch.inference_mode():
        for step in range(len(fixed) if fixed else 12):
            token=fixed[step] if fixed else (prefix[step] if step<len(prefix) else next_ids[-1])
            tokens.append(token)
            out=decoder(input_ids=torch.tensor([[token]],dtype=torch.long),encoder_hidden_states=encoded,past_key_values=cache,use_cache=True)
            cache=out.past_key_values
            hidden=out.last_hidden_state[0,-1]
            logits=torch.nn.functional.linear(hidden,decoder.embed_tokens.weight)
            save(f'step-{step:02d}-hidden',hidden);save(f'step-{step:02d}-logits',logits)
            raw_argmax.append(int(logits.argmax()))
            masked=logits.float().clone()
            if suppression:masked[suppression]=-float('inf')
            # Text-only ASR arm. Prefix and suppression are separate from raw-logit parity.
            masked[50364:]=-float('inf')
            if step==len(prefix)-1:
                masked[policy.get('begin_suppress_tokens') or []]=-float('inf')
            nxt=int(masked.argmax());next_ids.append(nxt)
            print('decoder',args.dtype,'step',step,'input',token,'raw_argmax',raw_argmax[-1],'next',nxt,flush=True)
            if fixed is None and step>=len(prefix)-1 and nxt==50257:break
    import transformers
    manifest={'schema':'memra-whisper-decoder-oracle-v1','dtype':args.dtype,'device':'cpu','threads':1,
        'source_revision':encoder_manifest['source_revision'],'input_pcm_sha256':encoder_manifest['input_pcm_sha256'],
        'encoder_manifest_sha256':hashlib.sha256((args.encoder_oracle/'manifest.json').read_bytes()).hexdigest(),
        'input_tokens':tokens,'raw_argmax':raw_argmax,'selected_next_tokens':next_ids,
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__},
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':files,
        'scope':'cached decoder math on fixed synthetic two-second clip; this bounded token loop is not the full Whisper transcription policy'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.out/'tokens.txt').write_text('\n'.join(map(str,tokens))+'\n')


if __name__=='__main__':main()
