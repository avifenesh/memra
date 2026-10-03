import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from native_choices import Native,save,sha
from replay_native_choices import cell
from choice_verifier import check_completed,check_greedy,require,normalized

p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
model=Path('/data/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf')
settings=SimpleNamespace(binary=args.binary,binary_sha=sha(args.binary),model=model,model_sha='0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c',out=args.out,context=4096,sessions=8,output_budget=1000000,reserve_mb=None,port=18290,phase='candidate')
native=Native(settings);records=[];status='failed'
try:
    native.start()
    payload={'model':'fixture','prompt':('A repeated prefix for the cache accounting probe. '*32)+'Count upward from one, with one integer per line.','session_id':'cache-choice-fixture','temperature':0,'max_tokens':32,'seed':73}
    for name,n in [('warm',1),('hit',1),('group',4),('after',1)]:
        records.append(native.request(name,'/v1/completions',{**payload,'n':n}))
    status='complete'
finally:
    if hasattr(native,'process'):native.stop()
    manifest={'status':status,'phase':'cache','source':'4ebe099459cfd2c1b05da6bdfb9c077e025abdd9','binary_sha256':settings.binary_sha,'model_sha256':settings.model_sha,'hardware':getattr(native,'hardware',None),'support_promotion':False,'results':records,'files':{str(f.relative_to(args.out)):sha(f) for f in args.out.rglob('*') if f.is_file()}}
    save(args.out/'manifest.json',manifest)
require(status=='complete' and all(r['status']==200 for r in records),'cache','all real requests complete')
warm,hit,group,after=records
body=hit['body'];cached=body.get('cached_tokens',body.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',0))
require(cached>0,'cache','real native nonzero cached prompt required')
check_greedy(body,group['body'])
require(group['body']['usage']['prompt_tokens_details']['cached_tokens']==cached,'cache','parent cached count equals singleton')
c=cell(args.out,group)
if c['leader_prime_segments']==0:
    require(c['body']['usage']['prompt_tokens']==cached,'cache','zero prime requires a whole-prompt cached leader')
check_completed(c)
save(args.out/'cache-proof.json',{'pass':True,'cached_tokens':cached,'group_output':group['body']['usage']['completion_tokens'],'actual_leader_prime_segments':len(group['prime_events']),'follower_prime_segments':c['follower_prime_segments'],'native_qualification':False})
print(json.dumps({'pass':True,'cached_tokens':cached,'native_qualification':False}))
