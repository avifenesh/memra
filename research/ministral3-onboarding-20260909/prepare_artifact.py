"""REMOTE ONLY: pin FP8 input and expand text weights for an explicitly new NVFP4 mint.
The expanded BF16 directory is calibration input, never the serving deliverable.
Static activation_scale tensors remain in the source and are recorded in the lock.
"""
import os,json,hashlib,shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch
def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
REV='5b26027e7b19eeb4b7352e1fed3926375dd2cb4d'
ROOT=Path('/workspace/ministral');SOURCE=ROOT/'source';EXPANDED=ROOT/'expanded-bf16'
SOURCE.mkdir(parents=True,exist_ok=True);EXPANDED.mkdir(exist_ok=True)
snapshot_download('mistralai/Ministral-3-8B-Instruct-2512',revision=REV,token=os.environ['HF_TOKEN'],local_dir=str(SOURCE),allow_patterns=['model-*.safetensors','*.json','chat_template.jinja','README.md'])
print('SOURCE-DOWNLOAD-DONE',flush=True)
config=json.loads((SOURCE/'config.json').read_text());text=dict(config['text_config']);text.update(architectures=['Ministral3ForCausalLM'],tie_word_embeddings=False,dtype='bfloat16')
(EXPANDED/'config.json').write_text(json.dumps(text,indent=2))
index=json.loads((SOURCE/'model.safetensors.index.json').read_text())['weight_map']
manifest={'source_repo':'mistralai/Ministral-3-8B-Instruct-2512','revision':REV,'source_precision':'FP8 E4M3 with static activation_scale','calibration_precision':'BF16 expansion of FP8 codes times weight_scale_inv; activation quantization intentionally removed for new NVFP4 artifact','dropped':[],'source':{},'outputs':{}}
weight_map={};total=0
for shard in sorted(set(index.values())):
 path=SOURCE/shard
 digest=file_hash(path);manifest['source'][shard]=digest
 state={}
 with safe_open(str(path),framework='pt',device='cpu') as sf:
  for key in sf.keys():
   if not key.startswith('language_model.'):
    manifest['dropped'].append(key);continue
   if key.endswith(('.activation_scale','.weight_scale_inv')):continue
   value=sf.get_tensor(key)
   if value.dtype==torch.float8_e4m3fn:
    scale_key=key.removesuffix('.weight')+'.weight_scale_inv'
    # Both scale planes belong to the same source shard by the pinned index.
    assert index[scale_key]==shard
    value=(value.float()*sf.get_tensor(scale_key).float()).bfloat16()
   key=key.removeprefix('language_model.')
   state[key]=value.contiguous();weight_map[key]=shard;total+=value.numel()*value.element_size()
 save_file(state,str(EXPANDED/shard));del state
 manifest['outputs'][shard]=file_hash(EXPANDED/shard)
 print('EXPANDED',shard,flush=True)
(EXPANDED/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':weight_map},indent=2))
for name in ['tokenizer.json','tokenizer_config.json','tekken.json','chat_template.jinja','generation_config.json']:
 if (SOURCE/name).exists():shutil.copyfile(SOURCE/name,EXPANDED/name)
(ROOT/'expansion-lock.json').write_text(json.dumps(manifest,indent=2))
print('TEXT-EXPANSION-DONE',len(weight_map),len(manifest['dropped']),flush=True)
