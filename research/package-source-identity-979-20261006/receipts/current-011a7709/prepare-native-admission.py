#!/usr/bin/python3 -I -B
"""Current-source native1800 admission recipe; no build or artifact reuse."""
import copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).absolute().parent
NS=Path('/home/evidence-user/.local/state/evidence-campaign-20261002')
OLD=NS/'receipts/manager/package-identity-979/final-joint-after-self-stem-refusal'
WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
source=json.loads((R/'APPROVED-SOURCE.json').read_bytes());head=source['head']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==head and subprocess.check_output(['git','status','--porcelain'],cwd=WT)==b''
subprocess.run(['git','merge-base','--is-ancestor','5d5fade93d8807bfc1dd4cd118535b8286511de7',head],cwd=WT,check=True)
spec=importlib.util.spec_from_file_location('source_io',WT/'tools/package_source_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for name,row in source['files'].items():assert m.regular(WT,name)==row,name
assert source['files']['Cargo.lock']['sha256']=='40639da422ba31c4bbf72ef5e038a158c9174998e4a63b7aeeb9ebf33b1348ca'
result=m.owned_json(R/'FINAL-CAPSULE-PREPARATION-RESULT.json');assert result['source']==head and result['native_executed'] is False and result['qualified'] is False
assert result['registry_members']==14 and result['registry_inputs']==748 and result['engine_generated_outputs']==46
entry=R/'prepared-current-source/package';output=R/'prepared-current-source/final-target';expectations=R/'prepared-current-source/final-expectations';records=R/'current-native-records-1800'
assert result['entry']==str(entry) and result['output']==str(output) and result['expectations']==str(expectations) and not os.path.lexists(records)
# Empty owned directory is valid here; regular inventory deliberately refuses
# empty source trees. Keep the corrected descriptor check and close its FD.
fd,_=m.directory(output)
try:assert os.listdir(fd)==[],'current native target is not empty'
finally:os.close(fd)
cap=m.package_capsule(entry);assert cap['source_seal']==result['source_seal'] and m.regular(entry,m.RESERVED)==result['capsule']
assert m.regular(entry,'build-support/package_source_identity.py')==source['files']['tools/package_source_identity.py'];assert m.regular(entry,'build-support/package_source_rustc.py')==source['files']['tools/package_source_rustc.py']
assert not any(row['name']=='yoke-derive' and row['version']=='0.8.3' for row in cap['snapshot']['payload']['packages'].values());assert any(row['name']=='yoke-derive' and row['version']=='0.8.4' for row in cap['snapshot']['payload']['packages'].values())
budget=m.owned_json(R/'ROOT-REVIEWED-FRESH-NATIVE-BUDGET.json')
assert set(budget)=={'source','source_seal','native_wall_seconds','native_CPU_seconds','native_policy','Root_reviewed_current_integrity_recipe'}
assert budget['source']==head and budget['source_seal']==cap['source_seal'] and budget['Root_reviewed_current_integrity_recipe'] is True
assert budget['native_policy']=='standard43-no-cache' and budget['native_wall_seconds']==1800 and budget['native_CPU_seconds']==3600
old=m.owned_json(OLD/'APPROVED-NATIVE.json')
for name,row in old['tools'].items():assert m.regular(Path(row['path']).parent,Path(row['path']).name)==row['file'],name
assert cap['compiler']==old['tools']['rustc'] and cap['recipe']['target']=='x86_64-unknown-linux-gnu'
engine=next(key for key,row in cap['snapshot']['payload']['packages'].items() if row['name']=='memra-engine');transport=m.registry_inputs(entry/cap['roots'][engine]/'Cargo.toml');assert transport['source_seal']==cap['source_seal'] and transport['corpus_sha256']==result['corpus_sha256'] and len(transport['workspace_members'])==14 and len(transport['inputs'])==748
assert len(cap['recipe']['generated'][engine])==46
native=copy.deepcopy(old);native.update({'source':head,'entry':str(entry),'output':str(output),'expectations':str(expectations),'records':str(records),'source_pins':source['files'],'admission':'one1800wall3600CPU-current-build-only'});native['profile']['CARGO_TARGET_DIR']=str(output)
assert {key:value for key,value in native['profile'].items() if key!='CARGO_TARGET_DIR'}=={key:value for key,value in old['profile'].items() if key!='CARGO_TARGET_DIR'}
m.immutable_json(R/'APPROVED-NATIVE.json',native)
m.immutable_json(R/'CURRENT-NATIVE-1800-ADMISSION-RESULT.json',{'source':head,'source_seal':cap['source_seal'],'Cargo_lock':source['files']['Cargo.lock'],'resolved_source_packages_actual':len(cap['roots']),'currentyoke':'0.8.4','registry_members':14,'registry_inputs':748,'generated_outputs':46,'six_tools_and_profile_unchanged':True,'receiver_sha256':source['files']['tools/package_source_rustc.py']['sha256'],'wall_seconds':1800,'CPU_ceiling':3600,'originallane_CPU_ceiling':36000,'justification':'Root separately reviewed this actual current-source capsule/profile and explicit fresh native budget. Helper and receiver remain unchanged; current archive/corpus bytes and producer paths require fresh issuance under the preserved standard43-no-cache policy. Earlier native and control receipts remain historical; no replay, artifact-cache/default change or receipt retag is claimed.','oldRust_receipts_or_nativecache_reused':False,'native_execution_launched':False,'qualified':False})
print(json.dumps({'current_native_admission_prepared':True,'source':head,'wall':1800,'CPU':3600,'actualsourcepackages':len(cap['roots']),'native_executed':False,'qualified':False}))
