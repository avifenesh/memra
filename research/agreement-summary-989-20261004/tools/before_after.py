from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,tempfile
if not __debug__:raise RuntimeError('summary proof requires assertions')
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
original=Path(sys.argv[3]).resolve();current=root/'tools/summarize_hy3_plan_agreement.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='8f77b291cc59c1a5877c0c780ef72fc091d570b7db9185a3a86a1c514f989ce4'
def funcs(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name!='self_test'}
assert funcs(original.read_text())==funcs(current.read_text())
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
rows=[]
def run(name,path,optimized=False):
    p=subprocess.run([sys.executable,*(['-O'] if optimized else []),str(path),'--self-test'],env=env,capture_output=True,timeout=20)
    raw=p.stdout+p.stderr;(out/(name+'.log')).write_bytes(raw)
    rows.append({'name':name,'exit':p.returncode,'optimized':optimized,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256(raw).hexdigest()})
    return p,raw
mutations=(('model','result["model"]["expert_count"] = 0'),('stable','result["consensus"]["stable_pruned_experts"] = 1'),('variable','result["consensus"]["variable_prune_experts"] = 0'),('layer','result["layers"]["2"]["all_state_agreement_fraction"] = 0'),('mass','result["traffic_overlay"]["stable_pruned_router_weight_mass_fraction"] = 1'),*[(name,'result["traffic_overlay"]["plans"]["'+name+'"]["state_router_weight_mass_fraction"] = {}') for name in ('a','b','c')],('pairs','result["pairwise"] = []'),('csv',None))
with tempfile.TemporaryDirectory(prefix='agreement-proof-') as owned:
    folder=Path(owned)
    for version,source in (('original4ba',original.read_text()),('current',current.read_text())):
        path=folder/(version+'.py');path.write_text(source)
        p,raw=run(version+'-before',path);assert p.returncode==0 and b'self-test: PASS' in raw
        for name,line in mutations:
            changed=source.replace('        writer.writeheader()', '        handle.write("wrong-header\\n")') if name=='csv' else source.replace('    return result\n','    '+line+'\n    return result\n',1)
            assert changed!=source,name;compile(changed,str(path),'exec');path.write_text(changed)
            p,raw=run(version+'-wrong-'+name,path);assert p.returncode==1 and b'AssertionError' in raw and b'self-test: PASS' not in raw,(name,raw.decode())
            p,raw=run(version+'-wrong-'+name+'-optimized',path,True)
            assert p.returncode==(0 if version=='original4ba' else 1)
            if version=='current':assert b'requires enabled assertions' in raw and b'self-test: PASS' not in raw
        path.write_text(source);p,raw=run(version+'-restored',path);assert p.returncode==0 and b'self-test: PASS' in raw
(out/'actual-original-current-before-wrong-restore.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'actual_runs':len(rows),'wrong_output_targets':10,'ordinary_nonselftest_AST_equal':True,'owned_roots_removed':True,'native_qualification':False}))
