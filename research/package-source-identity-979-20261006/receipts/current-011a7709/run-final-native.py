#!/usr/bin/python3 -I
import sys
sys.dont_write_bytecode=True
"""One reviewed stock Cargo build; refuses absent final role/source admission."""
import importlib.util,json,os,subprocess
from pathlib import Path
R=Path(__file__).absolute().parent;WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
spec=importlib.util.spec_from_file_location('receiver',WT/'tools/package_source_rustc.py');receiver=importlib.util.module_from_spec(spec);spec.loader.exec_module(receiver);m=receiver.m
approved=m.owned_json(R/'APPROVED-NATIVE.json')
fields={'source','entry','output','expectations','tools','profile','reject_names','records','source_pins','admission'}
m.require(type(approved) is dict and set(approved)==fields and approved['admission']=='one1800wall3600CPU-current-build-only','unknown native admission')
m.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==approved['source'] and not subprocess.check_output(['git','status','--porcelain'],cwd=WT),'final build source differs')
for name,row in approved['source_pins'].items():m.require(m.regular(WT,name)==row,'final source bytes/mode differ')
entry=Path(approved['entry']);output=Path(approved['output']);records=Path(approved['records'])
cap=m.package_capsule(entry);m.require(Path(cap['output'])==output and cap['expectations']==approved['expectations'],'final prepared roles differ')
m.require(cap['recipe']['target']=='x86_64-unknown-linux-gnu' and cap['compiler']==approved['tools']['rustc'],'final prepared compiler/target differs')
for name,tool in approved['tools'].items():
 path=Path(tool['path']);m.require(m.regular(path.parent,path.name)==tool['file'],'final tool bytes/mode differ: '+name)
m.require(set(approved['tools'])=={'cargo','rustc','nvcc','cxx','ar','strace'},'unknown final tool roles')
m.require(type(approved['profile']) is dict and set(approved['profile'])=={'MEMRA_CUDA_ARCH','MEMRA_NVCC','CARGO_BUILD_JOBS','NUM_JOBS','MEMRA_RELEASE_QUALIFICATION_MODE','CARGO_TARGET_DIR','RUSTC'},'unknown final child profile')
m.require(approved['profile']['MEMRA_CUDA_ARCH']=='120a' and approved['profile']['MEMRA_NVCC']==approved['tools']['nvcc']['path'] and approved['profile']['CARGO_BUILD_JOBS']=='2' and approved['profile']['NUM_JOBS']=='2' and approved['profile']['MEMRA_RELEASE_QUALIFICATION_MODE']=='development' and approved['profile']['CARGO_TARGET_DIR']==str(output) and approved['profile']['RUSTC']==approved['tools']['rustc']['path'],'final child profile differs')
m.require(type(approved['reject_names']) is list and all(type(x) is str for x in approved['reject_names']),'unknown override name list')
required_reject={'DOCS_RS','RUSTC_BOOTSTRAP','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CARGO_BUILD_RUSTC','CARGO_BUILD_RUSTC_WRAPPER','CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER','RUSTC'}
env=dict(os.environ)
if 'RUSTC_WRAPPER' in env:
 m.require(env['RUSTC_WRAPPER']=='' and 'MEMRA_RIG_JOB' in env,'unexpected effective compiler wrapper');env.pop('RUSTC_WRAPPER')
m.require(required_reject<=set(approved['reject_names']) and not any(x in env for x in approved['reject_names']),'unadmitted compiler/build profile presence')
m.require(not os.path.lexists(records),'immutable native records destination exists');records.mkdir()
omitted=[x for x in ('LIBRARY_PATH','LD_LIBRARY_PATH') if x in env]
for name in ('LIBRARY_PATH','LD_LIBRARY_PATH'):env.pop(name,None)
env.update(approved['profile'])
m.immutable_json(records/'CHILD-ENVIRONMENT-NAMES.json',{'names':sorted(env),'ambient_loader_names_omitted':omitted,'values_recorded':False,'HOME_forwarded_unchanged':env.get('HOME')==os.environ.get('HOME')})
cargo=approved['tools']['cargo']['path'];command=[cargo,'build','--manifest-path',str(entry/'Cargo.toml'),'--release','--bin','memra-server','--locked','--offline','--jobs','2','--message-format=json-render-diagnostics','--target-dir',str(output)]
m.immutable_json(records/'ACTUAL-COMMAND.json',{'argv':command,'cwd':str(entry),'source':approved['source'],'qualification':False})
with (records/'cargo.jsonl').open('xb') as stdout,(records/'cargo.stderr').open('xb') as stderr:
 trace=[approved['tools']['strace']['path'],'--seccomp-bpf','-ff','-yy','-s','65536','-e','trace=process','-o',str(records/'owned-child'),*command]
 result=subprocess.run(trace,cwd=entry,env=env,stdout=stdout,stderr=stderr,timeout=1775)
m.immutable_json(records/'BUILD-EXIT.json',{'exit':result.returncode,'source':approved['source'],'qualification':False})
m.require(result.returncode==0,'final actual Cargo build failed')
m.require(m.package_capsule(entry)==cap,'final capsule/source changed during build')
version=subprocess.run([str(output/'release/memra-server'),'--version'],cwd=entry,env=env,capture_output=True,timeout=10)
(records/'version.stdout').write_bytes(version.stdout);(records/'version.stderr').write_bytes(version.stderr)
m.require(version.returncode==0,'actual version consumer failed')
# Marker/identity/declared outputs and restored-path consumer acceptance require
# the separately selected final records. Success here never implies them.
print(json.dumps({'actual_build_exit':0,'actual_version_exit':0,'source':approved['source'],'native_model_qualification':False,'restore_checked':False}))
