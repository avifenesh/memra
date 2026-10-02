import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd(); out=root/'research/qwen9b-greedy-identity-20261002/controls';out.mkdir(parents=True,exist_ok=True)
os.fstat(9)
env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_') or k in ('MEMRA_GPU_LOCK','MEMRA_CI_LOCK','MEMRA_CI_LOCK_HELD','MEMRA_RIG_LOCK_FD')}
env.update(MEMRA_GATE_PORT='18112',MEMRA_NGEN='32')
models=[('q9','/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf'),('q35','/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf')]
results=[]
for label,model in models:
 for gate,cmd,overrides in [
  ('run-gen',['target/release/run-gen',model],{}),
  ('run-spec-short',['target/release/run-spec',model],{'MEMRA_PROMPT':'Count from one to twenty in words, separated by commas.'}),
  ('run-spec-long',['target/release/run-spec',model],{'MEMRA_PROMPT':'Explain how prefix caching works for a repeated request. Describe how a restored prefix and the newly computed suffix compose, and give a short concrete example.'}),
  ('spec-on-cache-hit',['bash','tools/spec-on-cache-hit-gate.sh','--external-lock','9','qwen',model,'target/release/memra-server',str(out/(label+'-cache'))],{}),
 ]:
  runenv=env|overrides
  with open(out/(label+'-'+gate+'.log'),'w') as log:
   try:r=subprocess.run(cmd,env=runenv,stdout=log,stderr=subprocess.STDOUT,timeout=300,pass_fds=(9,));code=r.returncode
   except subprocess.TimeoutExpired:code=124
  results.append({'model':label,'gate':gate,'exit':code})
  (out/'results.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(results[-1]),flush=True)
print(json.dumps(results,indent=2))
raise SystemExit(0 if all(r['exit']==0 for r in results) else 1)
