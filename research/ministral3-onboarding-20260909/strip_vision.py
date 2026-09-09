"""Copy exact text FP8 tensors and scalars to a text-only inspection artifact."""
import hashlib,json,shutil,sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
result={};bytes_total=0;dropped=[]
for shard in sorted(set(index.values())):
 with safe_open(str(source/shard),framework='pt',device='cpu') as f:
  state={}
  for name in f.keys():
   if name.startswith('language_model.'):
    tensor=f.get_tensor(name);state[name]=tensor;result[name]=shard;bytes_total+=tensor.numel()*tensor.element_size()
   else:dropped.append(name)
  save_file(state,str(out/shard))
 print('text FP8 shard',shard,flush=True)
config=json.loads((source/'config.json').read_text());config.pop('vision_config',None)
(out/'config.json').write_text(json.dumps(config,indent=2))
(out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':bytes_total},'weight_map':result},indent=2))
for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json','tekken.json']:
 if (source/name).exists():shutil.copyfile(source/name,out/name)
(out/'text-only-manifest.json').write_text(json.dumps({'source_repo':'mistralai/Ministral-3-8B-Instruct-2512','source_revision':'5b26027e7b19eeb4b7352e1fed3926375dd2cb4d','transform':'drop vision and projector; preserve every text tensor byte, dtype and name','kept_tensors':len(result),'kept_bytes':bytes_total,'dropped':dropped},indent=2))
assert len(result)==785 and len(dropped)==222 and bytes_total==9563579320
print('TEXT-FP8-COPY-DONE',len(result),bytes_total,flush=True)
