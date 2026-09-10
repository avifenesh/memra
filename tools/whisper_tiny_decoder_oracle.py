#!/usr/bin/env python3
"""Capture independent tiny HF cached-decoder outputs and KV for native CPU tests."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    import argparse,hashlib,json
    from pathlib import Path
    import numpy as np,torch
    from safetensors.torch import save_file
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperDecoder
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if (args.out/'tiny-decoder.safetensors').exists():raise SystemExit('refusing to overwrite decoder fixture')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(417)
    config=WhisperConfig(d_model=16,encoder_layers=2,decoder_layers=2,encoder_attention_heads=2,decoder_attention_heads=2,
        encoder_ffn_dim=32,decoder_ffn_dim=32,num_mel_bins=128,max_source_positions=8,max_target_positions=16,
        vocab_size=64,pad_token_id=0,bos_token_id=1,eos_token_id=2,decoder_start_token_id=1,dropout=0.0,attention_dropout=0.0)
    config._attn_implementation='eager'
    decoder=WhisperDecoder(config).eval()
    save_file({'model.decoder.'+k:v.detach().contiguous() for k,v in decoder.state_dict().items()},args.out/'tiny-decoder.safetensors')
    encoded=torch.from_numpy(np.fromfile(args.out/'tiny-f32-encoder.f32',dtype='<f4').reshape(1,8,16))
    tokens=[1,7,12,3,2];files={}
    def save(name,value):
        arr=value.detach().float().numpy().astype('<f4');raw=arr.tobytes();path=args.out/(name+'.f32');path.write_bytes(raw)
        files[path.name]={'shape':list(arr.shape),'sha256':hashlib.sha256(raw).hexdigest()}
    with torch.inference_mode():
        for dtype in ['f32','f16']:
            if dtype=='f16':decoder.half();encoded=encoded.half()
            cache=None
            for step,token in enumerate(tokens):
                out=decoder(input_ids=torch.tensor([[token]]),encoder_hidden_states=encoded,past_key_values=cache,use_cache=True)
                cache=out.past_key_values;hidden=out.last_hidden_state[0,-1];logits=torch.nn.functional.linear(hidden,decoder.embed_tokens.weight)
                save(f'tiny-decoder-{dtype}-{step}-hidden',hidden);save(f'tiny-decoder-{dtype}-{step}-logits',logits)
                full=decoder(input_ids=torch.tensor([tokens[:step+1]]),encoder_hidden_states=encoded,use_cache=False).last_hidden_state[0,-1]
                save(f'tiny-decoder-{dtype}-{step}-full-hidden',full)
                for layer in range(2):
                    for name,kv in [('self',cache.self_attention_cache),('cross',cache.cross_attention_cache)]:
                        if name=='cross' and step!=0:continue
                        for plane in ['keys','values']:
                            value=getattr(kv.layers[layer],plane)[0].transpose(0,1).reshape(-1,16)
                            save(f'tiny-decoder-{dtype}-{step}-{layer}-{name}-{plane}',value)
    import transformers
    manifest={'seed':417,'tokens':tokens,'config':config.to_dict(),'versions':{'torch':torch.__version__,'transformers':transformers.__version__},
              'checkpoint_sha256':hashlib.sha256((args.out/'tiny-decoder.safetensors').read_bytes()).hexdigest(),
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':files,'scope':'tiny synthetic decoder, no full-model qualification'}
    (args.out/'tiny-decoder-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('captured tiny cached decoder',len(files),'arrays')


if __name__=='__main__':main()
