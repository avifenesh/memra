from pathlib import Path
import json,hashlib,sys,re,subprocess,difflib
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import llvm_compare as c
O=R/'instrument-witnesses';O.mkdir(exist_ok=True)
if len(sys.argv)>1:
 d=json.loads(Path(sys.argv[1]).read_text());a=d['original'];b=d['candidate'];attrs=d['attributes'];meta=d['metadata'];types=set(d['types'])
 left=c.canonical(a,attrs,meta,types);right=c.canonical(b,attrs,meta,types)
 Path(sys.argv[1]).with_suffix('.original.canonical.ll').write_text(left+'\n');Path(sys.argv[1]).with_suffix('.candidate.canonical.ll').write_text(right+'\n')
 print(json.dumps({'equal':left==right,'original_sha256':hashlib.sha256(left.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(right.encode()).hexdigest()}));sys.exit(0 if left==right else 1)
f,a,m,t=c.read_functions(R/'emit-before-tier-kv/memra_kv-b149b237936aa196.ll')
symbol=next(s for s in f if 'QwenMaterializer' in s and c.name(s).endswith('::new'));body=f[symbol]
cas=next(line for line in body.splitlines() if 'cmpxchg' in line);assert 'monotonic monotonic' in cas
overflow=next(line for line in body.splitlines() if 'icmp eq i64' in line and '-1' in line)
mutations={'CAS-order':body.replace(cas,cas.replace('monotonic monotonic','acq_rel acquire'),1),'overflow-constant':body.replace(overflow,overflow.replace('-1','-2'),1),'same-restored':body,'comment':body.replace('{','{\n; owned nonoperative comment',1)}
records=[]
for label,candidate in mutations.items():
 p=O/(label+'.json');p.write_text(json.dumps({'symbol':symbol,'original':body,'candidate':candidate,'attributes':a,'metadata':m,'types':sorted(t)},indent=2)+'\n')
 (O/(label+'.original.raw.ll')).write_text(body);(O/(label+'.candidate.raw.ll')).write_text(candidate)
 command=[sys.executable,str(Path(__file__).resolve()),str(p)];result=subprocess.run(command,capture_output=True,text=True,timeout=60)
 (O/(label+'.command.json')).write_text(json.dumps(command)+'\n');(O/(label+'.output.log')).write_text(result.stdout+result.stderr)
 expected=1 if label in ['CAS-order','overflow-constant'] else 0;assert result.returncode==expected,(label,result.returncode,result.stdout,result.stderr)
 records.append({'label':label,'actual_exit':result.returncode,'expected_exit':expected,'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'output':label+'.output.log','source_symbol':symbol})
(O/'RESULTS.json').write_text(json.dumps({'rows':records,'normalizer_sha256':hashlib.sha256((R/'llvm_compare.py').read_bytes()).hexdigest(),'actual_raw_witnesses':True,'qualification':False},indent=2)+'\n');print('Actual separate raw CAS-order/overflow mutations refused; restored/comment controls equal.')
