"""Pinned-source rotated Q/K oracle at ordinary and query-scaling boundary positions."""
import ast,math,json,types,struct,sys
from pathlib import Path
import torch
root=Path(sys.argv[1]);out=Path(sys.argv[2])
ns={'torch':torch,'math':math}
for file,names in [('modeling_rope_utils.py',['_compute_yarn_parameters']),('modeling_ministral3.py',['rotate_half','apply_rotary_pos_emb','get_llama_4_attn_scale'])]:
 nodes=[n for n in ast.parse((root/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in nodes:n.decorator_list=[]
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 exec(compile(ast.fix_missing_locations(tree),file,'exec'),ns)
cfg=types.SimpleNamespace(**json.loads((root/'config.json').read_text())['text_config']);cfg.standardize_rope_params=lambda:None
freq,amp=ns['_compute_yarn_parameters'](cfg,torch.device('cpu'));assert amp==1.0
pos=torch.tensor([[0,16383,16384,32768]])
q=((torch.arange(4*2*128).float()%37-18)/19).reshape(1,4,2,128).transpose(1,2)
k=((torch.arange(4*128).float()%29-14)/17).reshape(1,4,1,128).transpose(1,2)
a=pos.float().unsqueeze(-1)*freq;angles=torch.cat([a,a],dim=-1)
q,k=ns['apply_rotary_pos_emb'](q,k,angles.cos(),angles.sin());q*=ns['get_llama_4_attn_scale'](pos,0.1,16384)
with out.open('w') as f:
 for name,t in [('q',q),('k',k)]:
  for i,v in enumerate(t.transpose(1,2).contiguous().flatten().tolist()):f.write(f'{name}\t{i}\t{struct.unpack("<I",struct.pack("<f",v))[0]:08x}\n')
print('PINNED-ROPE-ORACLE-DONE')
