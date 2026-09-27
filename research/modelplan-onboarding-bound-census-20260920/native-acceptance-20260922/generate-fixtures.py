"""Small real loader containers; deterministic diagnostics, never model qualification."""
import argparse, hashlib, json, pathlib, struct, copy
D,F,V=256,512,64
p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def f32(n,x):return struct.pack('<f',x)*n
def tensor(shape,x):return {'dtype':'F32','shape':shape,'raw':f32(__import__('math').prod(shape),x)}
names={'token_embd.weight':'model.embed_tokens.weight','output.weight':'lm_head.weight','output_norm.weight':'model.norm.weight','blk.0.attn_norm.weight':'model.layers.0.input_layernorm.weight','blk.0.ffn_norm.weight':'model.layers.0.post_attention_layernorm.weight','blk.0.attn_q.weight':'model.layers.0.self_attn.q_proj.weight','blk.0.attn_k.weight':'model.layers.0.self_attn.k_proj.weight','blk.0.attn_v.weight':'model.layers.0.self_attn.v_proj.weight','blk.0.attn_output.weight':'model.layers.0.self_attn.o_proj.weight','blk.0.ffn_gate.weight':'model.layers.0.mlp.gate_proj.weight','blk.0.ffn_up.weight':'model.layers.0.mlp.up_proj.weight','blk.0.ffn_down.weight':'model.layers.0.mlp.down_proj.weight'}
def base():
 t={'token_embd.weight':tensor([V,D],.25),'output.weight':tensor([V,D],.5),'output_norm.weight':tensor([D],1),'blk.0.attn_norm.weight':tensor([D],1),'blk.0.ffn_norm.weight':tensor([D],1)}
 for i,n in enumerate(['q','k','v','output']):t['blk.0.attn_'+n+'.weight']=tensor([D,D],(i+1)/16)
 for i,n in enumerate(['gate','up','down']):t['blk.0.ffn_'+n+'.weight']=tensor([D,F] if n=='down' else [F,D],(i+1)/32)
 return t
def string(x):b=x.encode();return struct.pack('<Q',len(b))+b
def gguf(path,t,tied,duplicate=False):
 meta={'general.architecture':'llama','llama.block_count':1,'llama.embedding_length':D,'llama.feed_forward_length':F,'llama.attention.head_count':8,'llama.attention.head_count_kv':8,'llama.attention.key_length':32,'llama.attention.value_length':32,'llama.context_length':64,'llama.rope.dimension_count':32,'llama.rope.freq_base':10000.,'llama.attention.layer_norm_rms_epsilon':1e-6,'llama.tie_word_embeddings':tied}
 rows=list(t.items());rows+=rows[:1] if duplicate else []
 h=b'GGUF'+struct.pack('<IQQ',3,len(rows),len(meta))
 for k,v in meta.items():
  ty=8 if isinstance(v,str) else 7 if isinstance(v,bool) else 6 if isinstance(v,float) else 4
  h+=string(k)+struct.pack('<I',ty)+(string(v) if ty==8 else struct.pack('<B',v) if ty==7 else struct.pack('<f',v) if ty==6 else struct.pack('<I',v))
 raw=bytearray()
 for name,x in rows:
  raw.extend(b'\0'*((-len(raw))%32));shape=x['shape'][::-1];qt={'F32':0,'Q8_0':8,'Q5_K':13,'I64':27}[x['dtype']]
  h+=string(name)+struct.pack('<I',len(shape))+b''.join(struct.pack('<Q',s) for s in shape)+struct.pack('<IQ',qt,len(raw));raw.extend(x['raw'])
 h+=b'\0'*((-len(h))%32);path.write_bytes(h+raw)
def hf(path,t,tied,alias=False,missing_aux=False,nvfp4_query=False):
 path.mkdir();cfg={'model_type':'llama','num_hidden_layers':1,'hidden_size':D,'num_attention_heads':8,'num_key_value_heads':8,'head_dim':32,'intermediate_size':F,'vocab_size':V,'max_position_embeddings':64,'rms_norm_eps':1e-6,'rope_theta':10000.,'tie_word_embeddings':tied}
 (path/'config.json').write_text(json.dumps(cfg));ts={names.get(n,n):x for n,x in t.items()}
 if alias:ts['model.language_model.layers.0.self_attn.q_proj.weight']=copy.deepcopy(ts['model.layers.0.self_attn.q_proj.weight'])
 if missing_aux or nvfp4_query:
  stem='model.layers.0.self_attn.q_proj';ts[stem+'.weight']={'dtype':'U8','shape':[D,D//2],'raw':bytes([0x22])*(D*D//2)}
  ts[stem+'.weight_scale_2']=tensor([1],2.)
  if nvfp4_query:ts[stem+'.weight_scale']={'dtype':'F8_E4M3','shape':[D,D//16],'raw':bytes([0x38])*(D*D//16)}
 header={};raw=bytearray()
 for n,x in sorted(ts.items()):start=len(raw);raw.extend(x['raw']);header[n]={'dtype':x['dtype'],'shape':x['shape'],'data_offsets':[start,len(raw)]}
 h=json.dumps(header,separators=(',',':')).encode();(path/'model.safetensors').write_bytes(struct.pack('<Q',len(h))+h+raw)
cases=[]
for fmt in ['gguf','hf']:
 for variant in ['separate','tied','missing','unexpected','wrong_shape','wrong_quant','untied_missing']+(['duplicate','q5_head'] if fmt=='gguf' else ['ambiguous','missing_aux','bf16_head','nvfp4_query']):
  t=base();tied=variant=='tied';expected_error=variant not in {'separate','tied','q5_head','bf16_head','nvfp4_query'}
  if tied or variant=='untied_missing':t.pop('output.weight')
  if variant=='missing':t.pop('blk.0.attn_q.weight')
  if variant=='unexpected':t['unclaimed.weight']=tensor([1],.125)
  if variant=='wrong_shape':t['blk.0.attn_q.weight']=tensor([D,D+1],.125)
  if variant=='wrong_quant':t['blk.0.attn_norm.weight']={'dtype':'Q8_0' if fmt=='gguf' else 'I64','shape':[D],'raw':(b'\x00\x3c'+bytes([1])*32)*(D//32) if fmt=='gguf' else bytes(D*8)}
  if variant=='q5_head':t['output.weight']={'dtype':'Q5_K','shape':[V,D],'raw':(struct.pack('<ee',.5,.125)+bytes([1])*12+bytes([0x55])*32+bytes([0x22])*128)*V}
  if variant=='bf16_head':t['output.weight']={'dtype':'BF16','shape':[V,D],'raw':struct.pack('<H',0x3f00)*(V*D)}
  ident=fmt+'-'+variant;path=a.output/(ident+'.gguf' if fmt=='gguf' else ident)
  if fmt=='gguf':gguf(path,t,tied,variant=='duplicate')
  else:hf(path,t,tied,variant=='ambiguous',variant=='missing_aux',variant=='nvfp4_query')
  expected=t['token_embd.weight' if tied else 'output.weight']['raw'] if not expected_error else b''
  # Small BF16 tensors follow the existing loader's documented F32 expansion.
  if variant=='bf16_head':expected=f32(V*D,.5)
  cases.append({'id':ident,'format':fmt,'path':path.name,'expect_error':expected_error,'expected_owner':'token_embd.weight' if tied else 'output.weight','expected_output_sha256':hashlib.sha256(expected).hexdigest() if expected else None,'expected_output_bytes':len(expected),'expected_output_kind':'Quant' if variant=='q5_head' else 'Float','native_shape':[D,V],'nvfp4_query':variant=='nvfp4_query','expected_error_fragment':{'missing':'missing tensor','unexpected':'extra checkpoint tensors','wrong_shape':'shape mismatch','wrong_quant':'storage mismatch','untied_missing':'missing tensor OutputProjection','duplicate':'duplicate census tensor','ambiguous':'multiple physical tensors normalize','missing_aux':'NVFP4 is missing weight_scale'}.get(variant,'')})
manifest={'scope':'real tiny container loader diagnostics only; no qualification or numerical oracle','cases':cases,'files':[]}
for f in sorted(a.output.rglob('*')):
 if f.is_file():manifest['files'].append({'path':str(f.relative_to(a.output)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(a.output/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'cases':len(cases),'positive':sum(not x['expect_error'] for x in cases),'negative':sum(x['expect_error'] for x in cases)},indent=2))
