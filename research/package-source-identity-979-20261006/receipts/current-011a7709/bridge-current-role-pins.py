#!/usr/bin/python3 -I
import sys
sys.dont_write_bytecode=True
import importlib.util,json,hashlib
from pathlib import Path
R=Path(__file__).absolute().parent;OLD=Path('/home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/current-main-335e-context-standard-1800');WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
spec=importlib.util.spec_from_file_location('m',WT/'tools/package_source_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.owned_json(OLD/'prepared-current-source/metadata.stdout');b=m.owned_json(R/'prepared-current-source/metadata.stdout');names={'proc-macro2','anyhow','num-traits','autocfg','unicode-ident'}
def selected(v):
 nodes={x['id']:x for x in v['resolve']['nodes']}
 return {p['name']+'-'+p['version']:(p,nodes[p['id']]['features']) for p in v['packages'] if p['name'] in names}
x,y=selected(a),selected(b);m.require(set(x)==set(y),'current finite role package membership differs')
for name in x:
 oldpkg,oldfeatures=x[name];newpkg,newfeatures=y[name];m.require(oldfeatures==newfeatures and m.inventory(Path(oldpkg['manifest_path']).parent)==m.inventory(Path(newpkg['manifest_path']).parent),'finite role source/mode/features byte bridge differs')
role=m.owned_json(OLD/'current-rust-role-paths/RESULT.json');result=json.loads(json.dumps(role));result['source']=m.owned_json(R/'APPROVED-SOURCE.json')['head'];oldout=Path(OLD/'prepared-current-source/final-target');newout=R/'prepared-current-source/final-target'
for group in ['source_roles','stdin_roles']:
 for row in result[group]:
  argv=row['argv'];index=argv.index('--out-dir')+1;m.require(Path(argv[index]).is_relative_to(oldout),'original role path ownership differs');argv[index]=str(newout/Path(argv[index]).relative_to(oldout))
result['actual_OUT_DIRs']={k:str(newout/Path(v).relative_to(oldout)) for k,v in role['actual_OUT_DIRs'].items()};result['actual_produced_files']={};result['final_receiver_restored']=True;result['role_proof_replayed']=False;result['current_out_relative_source_tool_feature_bridge']=True;result['new_absolute_paths_are_planned_owned_prefix']=True;result['limits']='Historical actual6role observations stay at their original record paths. This is a current source/mode/features bridge with planned owned-prefix relocation only; helper bdbfe568 and receiver ef5f6eb2 are unchanged from 335e. No compiler, probe, artifact or control replay is claimed. Final actual path/profile mismatch must refuse; no historical output fallback.'
(R/'current-rust-role-paths').mkdir();m.immutable_json(R/'current-rust-role-paths/RESULT.json',result);(R/'prepared-current-source/final-target').mkdir();print(json.dumps({'current_source_bridge':True,'mode_controls_replayed':False,'qualified':False}))
