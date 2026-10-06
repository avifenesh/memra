#!/usr/bin/python3 -I
"""Current normal Cargo three-package role preparation, not engine/native build."""
import hashlib,importlib.util,json,os,shutil,subprocess
from pathlib import Path
R=Path(__file__).absolute().parent;PREP=R.parent;WT=Path('/home/evidence-user/projects/evidence-source-worktree-979');entry=PREP/'prepared-current-source/package'
spec=importlib.util.spec_from_file_location('receiver',WT/'tools/package_source_rustc.py');receiver=importlib.util.module_from_spec(spec);spec.loader.exec_module(receiver);m=receiver.m
pins=m.owned_json(R/'INPUT-PINS.json')
def fresh():
 for owner,row in pins['source_roots'].items():m.require(m.inventory(Path(row['root']))==row['files'],'current Rust role source membership differs')
 for name,row in pins['files'].items():
  path=Path(name);m.require(m.regular(path.parent,path.name)==row,'current role input differs')
fresh();wrapper=entry/'build-support/package_source_rustc.py';original=m.regular(wrapper.parent,wrapper.name,contents=True);mode=wrapper.stat().st_mode&0o777;m.require(hashlib.sha256(original).hexdigest()=='041972d47b707a72a60f707b9b85087aa3c1b194ee22be56da937c36c3e02b42','final receiver source differs')
(R/'observations').mkdir(exist_ok=True);m.require(not list((R/'observations').iterdir()),'prior compiler observations exist');output=PREP/'prepared-current-source/final-target';m.require(not os.path.lexists(output),'final target alreadyexists')
# The observer is installed at the SAME selected effective wrapper path. It is
# restored before final capsule preparation; this cell produces no checked ID.
env=dict(os.environ)
# Canonical broker sets exactly empty RUSTC_WRAPPER to disable sccache.
# Remove only that known empty child default so the pinned Cargo config is used.
if 'RUSTC_WRAPPER' in env:
 m.require(env['RUSTC_WRAPPER']=='' and 'MEMRA_RIG_JOB' in env,'unexpected effective compiler wrapper');env.pop('RUSTC_WRAPPER')
reject=['RUSTC_BOOTSTRAP','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CARGO_BUILD_RUSTC','CARGO_BUILD_RUSTC_WRAPPER','CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER']
m.require(not any(name in env for name in reject),'unknown role-preparation compiler override')
for name in ('LIBRARY_PATH','LD_LIBRARY_PATH'):env.pop(name,None)
env.update({'RUSTC':pins['compiler']['path'],'RUSTUP_TOOLCHAIN':'1.97.1','CARGO_BUILD_JOBS':'2','CARGO_TARGET_DIR':str(output)})
command=[pins['cargo']['path'],'build','--release','--manifest-path',str(entry/'Cargo.toml'),'-p','proc-macro2','-p','anyhow','-p','num-traits','--offline','--locked','--jobs','2','--message-format=json-render-diagnostics','--target-dir',str(output)]
try:
 shutil.copyfile(R/'observe-rustc.py',wrapper);wrapper.chmod(0o755)
 result=subprocess.run(command,cwd=entry,env=env,capture_output=True,timeout=110);(R/'cargo.jsonl').write_bytes(result.stdout);(R/'cargo.stderr').write_bytes(result.stderr);m.require(result.returncode==0,'current Rust role preparation failed')
finally:wrapper.write_bytes(original);wrapper.chmod(mode)
fresh()
events=[m.json_bytes(line) for line in result.stdout.splitlines()];outs={}
for event in events:
 if event.get('reason')=='compiler-artifact':m.require(event['target']['kind']!=['custom-build'] or not event['package_id'].endswith('memra-engine@0.138.0'),'engine role mustnotcompile')
 if event.get('reason')=='build-script-executed':
  name=event['package_id'].rsplit('#',1)[-1];m.require(name in ('proc-macro2@1.0.106','anyhow@1.0.104','num-traits@0.2.19'),'unexpected build-script producer');outs[name]=event['out_dir']
m.require(len(outs)==3,'current three producer outputs missing')
source_roles=[];stdin_roles=[]
for path in sorted((R/'observations').glob('unit*.json')):
 record=m.owned_json(path);args=record['argv'];m.require(record['package_name'] not in ('memra-engine','memra-server'),'root/engine compilation is outside role preparation')
 if '--emit=dep-info,metadata' in args:
  candidate=list(args);out=candidate[candidate.index('--out-dir')+1];candidate[candidate.index('--out-dir')+1]='<owned-output-role>'
  matches=[role for role,spec in receiver.FINITE_METADATA_PROBES.items() if candidate==spec['argv_template'] and record['package_name']+'-'+record['package_version']==spec['package']];m.require(len(matches)==1,'unadmitted current source probe argv');role=matches[0]
  spec=receiver.FINITE_METADATA_PROBES[role];manifest=Path(record['cwd']);m.require(m.regular(manifest,spec['source'])==spec['source_file'],'source probe body/mode differs');source_roles.append({'role':role,'record':str(path),'file':m.regular(path.parent,path.name),'argv':args,'exit':record['exit'],'out_relative':Path(out).relative_to(output).as_posix()})
 if '--emit=llvm-ir' in args:
  m.require(record['package_name']=='num-traits','unknown stdinproducer');role=args[1].rsplit('_',1)[1];m.require(role in receiver.NUM_TRAITS_STDIN and record['stdin']['utf8'].encode()==receiver.NUM_TRAITS_STDIN[role],'current stdin body differs');stdin_roles.append({'role':role,'record':str(path),'file':m.regular(path.parent,path.name),'argv':args,'exit':record['exit'],'out_relative':Path(record['out_dir']).relative_to(output).as_posix()})
m.require(sorted(x['role'] for x in source_roles)==['0','1','2','3'] and sorted(x['role'] for x in stdin_roles)==['0','1'],'current finite roles incomplete')
produced={p.relative_to(output).as_posix():m.regular(output,p.relative_to(output).as_posix()) for p in output.rglob('*') if p.is_file()}
report={'source':pins['head'],'command':command,'actual_graph_source':pins['actual_graph_source'],'source_roles':source_roles,'stdin_roles':stdin_roles,'actual_OUT_DIRs':outs,'actual_produced_files':produced,'wrapper_effective_path':str(wrapper),'wrapper_observer_sha256':hashlib.sha256((R/'observe-rustc.py').read_bytes()).hexdigest(),'final_receiver_restored':True,'native_or_engine_compiled':False,'artifact_marker':False,'qualified':False,'limits':'Current Cargo selectedthreepackages role paths, originalflag/profile outcomes; thisobserverpass doesnotpublish checkedcompilerreceipts. Actual final fullgraph may change rolepath: final receiver exactauthority mustrefuse mismatch, no historical fallback.'}
m.immutable_json(R/'RESULT.json',report);print(json.dumps({'source':pins['head'],'source_probe_roles':len(source_roles),'stdin_probe_roles':len(stdin_roles),'producer_OUT_DIRs':outs,'qualified':False}))
