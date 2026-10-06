#!/usr/bin/python3 -I
import sys
sys.dont_write_bytecode=True
"""Current source/corpus and finite selected metadata recipe; no native execution."""
import hashlib,importlib.util,json,os,shutil,tarfile
from pathlib import Path
R=Path(__file__).absolute().parent;WT=Path('/home/evidence-user/projects/evidence-source-worktree-979');entry=R/'prepared-current-source/package';stage=entry.parent
spec=importlib.util.spec_from_file_location('receiver',WT/'tools/package_source_rustc.py');receiver=importlib.util.module_from_spec(spec);spec.loader.exec_module(receiver);m=receiver.m
source=m.owned_json(R/'APPROVED-SOURCE.json');m.require(source['head']==m.owned_json(R/'CONTROL-AND-SOURCE-BRIDGE.json')['reviewed_source'],'final source differs')
recipe=m.owned_json(R/'OBSERVED-FINAL-RECIPE-NOMINATION.json');roles=m.owned_json(R/'current-rust-role-paths/RESULT.json');m.require(roles['source']==source['head'] and roles['final_receiver_restored'] and not roles['native_or_engine_compiled'],'current role preparation differs')
provenance=m.owned_json(R/'independent-registry-corpus/PRODUCER-PROVENANCE.json');corpus_sha=provenance['corpus_sha256'];corpus=m.owned_json(R/'independent-registry-corpus'/(corpus_sha+'.registry-corpus.json'));m.require(m.digest(corpus)==corpus_sha and provenance['head']==source['head'],'independent corpus commitment differs')
supp=entry/'supplementary';m.require(not os.path.lexists(supp),'supplementary attachment alreadyexists');supp.mkdir();archives={}
for owner,row in provenance['supplementary_inputs'].items():
 origin=provenance['owners'][owner]
 if origin['role']=='original producer Git':raw=m.regular(WT,origin['path'],contents=True)
 else:
  archive=R/'source-archives'/origin['archive'];m.require(m.regular(archive.parent,archive.name)==origin['archive_file'],'current corpus archive differs')
  if archive.name not in archives:
   with tarfile.open(archive,'r:gz') as tar:archives[archive.name]={Path(*Path(x.name).parts[1:]).as_posix():tar.extractfile(x).read() for x in tar.getmembers() if x.isfile()}
  raw=archives[archive.name][origin['entry']]
 m.require(hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes'],'independent corpus payload differs')
 path=supp/m.relative_name(owner);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o755 if row['mode']=='100755' else 0o644)
m.require(m.inventory(supp,package=False)==provenance['supplementary_inputs'],'attached supplementary corpus membership differs')
output=stage/'final-target';m.require(output.is_dir(),'current role target missing')
# Unaccepted role-output bytes are retained in the preparation receipt. Clear
# this exclusively owned target so the final build cannot reuse unissued units.
for name,row in roles['actual_produced_files'].items():m.require(m.regular(output,name)==row,'unaccepted role artifact differs before retirement')
shutil.rmtree(output);output.mkdir();expected=stage/'final-expectations';expected.mkdir()
saved=dict(os.environ)
try:
 os.environ['RUSTUP_TOOLCHAIN']='1.97.1'
 if os.environ.get('RUSTC_WRAPPER')=='':os.environ.pop('RUSTC_WRAPPER',None)
 graph=m.cargo_graph(entry/'Cargo.toml','x86_64-unknown-linux-gnu',[]);packages=graph['payload']['packages'];engine=next(k for k,v in packages.items() if v['name']=='memra-engine')
 plan={'env':{},'generated':{},'cfgs':{},'codegen':recipe['codegen_choices'],'targets':['memra_server'],'supplementary':{'supplementary:registry':{'owner':engine,'role':'env-registry-source-v1','root':'supplementary','files':m.inventory(supp,package=False)}}}
 for key,row in packages.items():
  owner=row['name']+'@'+row['version'];plan['env'][key]=recipe['env_names'].get(owner,[]);plan['generated'][key]=recipe['generated_names'].get(owner,[]);plan['cfgs'][key]=recipe['cfg_choices'].get(owner,[])
 declaration=stage/'final-declarations.json';m.immutable_json(declaration,plan)
 cap=m.prepare_package(entry/'Cargo.toml','x86_64-unknown-linux-gnu',[],declaration,output,expected)
finally:os.environ.clear();os.environ.update(saved)
m.immutable_json(expected/(corpus_sha+'.registry-corpus.json'),corpus)
m.immutable_json(expected/('registry-corpus-'+cap['source_seal']+'.json'),{'schema':'memra-registry-corpus-reference-v1','owner':engine,'corpus_sha256':corpus_sha})
for actual in roles['source_roles']:
 spec=receiver.FINITE_METADATA_PROBES[actual['role']];key=next(k for k,v in packages.items() if v['name']+'-'+v['version']==spec['package']);row=packages[key]
 authority={'schema':'memra-owned-metadata-probe-v1','role':actual['role'],'package':key,'source_seal':m.digest(row),'compiler':cap['compiler'],'argv':actual['argv'],'out_relative':actual['out_relative'],'source':spec['source'],'source_file':spec['source_file'],'authority_record_sha256':spec['authority_record_sha256'],'package_archive_sha256':spec['package_archive_sha256']};m.immutable_json(expected/('metadata-probe-'+actual['role']+'.json'),authority)
key=next(k for k,v in packages.items() if v['name']=='num-traits');refs={k:m.digest(v) for k,v in packages.items() if v['name'] in ('num-traits','autocfg')};stdin=roles['stdin_roles'];m.require(len({x['out_relative'] for x in stdin})==1,'current stdin output roles differ')
m.immutable_json(expected/'stdin-probe-num-traits.json',{'schema':'memra-owned-num-traits-probe-v1','package':key,'sources':refs,'compiler':cap['compiler'],'out_relative':stdin[0]['out_relative'],'profile':{'target':'x86_64-unknown-linux-gnu','features':['default','std'],'runtime_cfg':'has_total_cmp','encoded_rustflags_sha256':hashlib.sha256(b'').hexdigest()},'roles':receiver.NUM_TRAITS_CAPTURE})
m.require(m.package_capsule(entry)==cap,'final capsule notfresh')
response=m.registry_inputs(entry/cap['roots'][engine]/'Cargo.toml');m.require(response['schema']=='memra-registry-source-inputs-v2' and response['corpus_sha256']==corpus_sha and response['source_seal']==cap['source_seal'],'current registry source transport differs')
m.immutable_json(R/'FINAL-CAPSULE-PREPARATION-RESULT.json',{'source':source['head'],'entry':str(entry),'output':str(output),'expectations':str(expected),'capsule':m.regular(entry,m.RESERVED),'declarations':m.regular(declaration.parent,declaration.name),'source_seal':cap['source_seal'],'corpus_sha256':corpus_sha,'owner':engine,'registry_members':len(response['workspace_members']),'registry_inputs':len(response['inputs']),'engine_generated_outputs':len(plan['generated'][engine]),'unaccepted_role_artifacts_retired':True,'native_executed':False,'qualified':False,'limitations':'Currentselectedthree-package normalCargo OUT_DIR paths are exactauthority candidates. Finalfullgraph changedpath/profile mustrefuse; no historicalfallback. This prepares sourceandselectedmetadata, notbinary/sysroot/modelqualification.'})
print(json.dumps({'source':source['head'],'registry_inputs':len(response['inputs']),'engine_outputs':len(plan['generated'][engine]),'capsule_prepared':True,'native_executed':False,'qualified':False}))
