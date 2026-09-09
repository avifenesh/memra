"""Offline source-FP32 oracle. Never a serving runtime.
FP8 codes and their BF16 scalar weight multipliers are expanded directly to FP32.
Static FP8 activation rounding is deliberately absent in this FP32 numeric class.
The pinned source functions own YaRN, rotary application, and query scaling.
"""
import ast,hashlib,json,math,struct,sys,types,typing
from pathlib import Path
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
root=Path(sys.argv[1]);out=Path(sys.argv[2]);refs=Path(sys.argv[3])
torch.set_num_threads(8)
config=json.loads((root/'config.json').read_text())['text_config']
cfg=types.SimpleNamespace(**config);cfg.standardize_rope_params=lambda:None
ns={'torch':torch,'math':math,'Optional':typing.Optional,'nn':torch.nn,'Unpack':typing.Unpack if hasattr(typing,'Unpack') else object,'TransformersKwargs':dict}
def load_functions(path,names):
 tree=ast.parse(path.read_text());nodes=[]
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in names:n.decorator_list=[];nodes.append(n)
 assert {n.name for n in nodes}==set(names)
 # Future annotations keep unrelated type imports out of an offline numeric oracle.
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
load_functions(refs/'modeling_rope_utils.py',['_compute_yarn_parameters'])
load_functions(refs/'modeling_ministral3.py',['rotate_half','apply_rotary_pos_emb','get_llama_4_attn_scale','repeat_kv','eager_attention_forward'])
inv,amplitude=ns['_compute_yarn_parameters'](cfg,torch.device('cpu'))
assert amplitude==1.0
ids=[1]+Tokenizer.from_file(str(root/'tokenizer.json')).encode('שלום עולם',add_special_tokens=False).ids
out.parent.mkdir(parents=True,exist_ok=True)
(out.parent/'source-tokens.txt').write_text(' '.join(map(str,ids)))
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
opened={name:safe_open(str(root/name),framework='pt',device='cpu') for name in sorted(set(index.values()))}
def weight(name):
 name='language_model.'+name
 value=opened[index[name]].get_tensor(name)
 if value.dtype==torch.float8_e4m3fn:
  scale=name.removesuffix('.weight')+'.weight_scale_inv'
  return value.float()*opened[index[scale]].get_tensor(scale).float()
 return value.float()
def linear(x,name):return torch.nn.functional.linear(x,weight(name+'.weight'))
def norm(x,name):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*weight(name+'.weight')
positions=torch.arange(len(ids)).unsqueeze(0)
freq=positions.float().unsqueeze(-1)*inv
angles=torch.cat([freq,freq],dim=-1);cos=angles.cos();sin=angles.sin()
mask=torch.full((len(ids),len(ids)),float('-inf')).triu(1)
with torch.inference_mode():
 x=weight('model.embed_tokens.weight')[ids].unsqueeze(0)
 for layer in range(34):
  stem=f'model.layers.{layer}';h=norm(x,stem+'.input_layernorm')
  q=linear(h,stem+'.self_attn.q_proj').view(1,-1,32,128).transpose(1,2)
  k=linear(h,stem+'.self_attn.k_proj').view(1,-1,8,128).transpose(1,2)
  v=linear(h,stem+'.self_attn.v_proj').view(1,-1,8,128).transpose(1,2)
  q,k=ns['apply_rotary_pos_emb'](q,k,cos,sin)
  q=q*ns['get_llama_4_attn_scale'](positions,0.1,16384)
  a,_=ns['eager_attention_forward'](types.SimpleNamespace(num_key_value_groups=4,training=False),q,k,v,mask,128**-0.5)
  x=x+linear(a.reshape(1,-1,4096),stem+'.self_attn.o_proj')
  h=norm(x,stem+'.post_attention_layernorm')
  x=x+linear(torch.nn.functional.silu(linear(h,stem+'.mlp.gate_proj'))*linear(h,stem+'.mlp.up_proj'),stem+'.mlp.down_proj')
  print('source-FP32 layer',layer,flush=True)
 logits=linear(norm(x,'model.norm'),'lm_head')[0,-1].contiguous()
with out.open('w') as f:
 f.write('format\tmemra-checkpoint-oracle-v1\nengine\ttransformers-pinned-source-fp32\nnumeric_class\tsource-weights-float32-accumulation\n')
 f.write('tokens\t'+','.join(map(str,ids))+'\nvocab\t131072\n')
 for i,v in enumerate(logits.tolist()):f.write(f'logit\t{i}\t{struct.unpack("<I",struct.pack("<f",v))[0]:08x}\n')
(out.parent/'source-fp32-metadata.json').write_text(json.dumps({'tokens':ids,'weight_program':'source FP8 codes * weight_scale_inv in FP32','activation_program':'FP32; static FP8 activation quantization disabled for this oracle class','reference_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in refs.glob('*.py')},'torch':torch.__version__,'argmax':int(logits.argmax())},indent=2))
print('SOURCE-FP32-DONE',int(logits.argmax()),flush=True)
