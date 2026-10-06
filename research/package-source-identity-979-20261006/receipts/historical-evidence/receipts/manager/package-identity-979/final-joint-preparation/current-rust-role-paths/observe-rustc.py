#!/usr/bin/python3 -I
"""Owned marker-free Rust-only path preparation; no receiver qualification."""
import hashlib,importlib.util,json,os,re,stat,subprocess,sys,time
from pathlib import Path
R=Path('/home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths');WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
spec=importlib.util.spec_from_file_location('m',WT/'tools/package_source_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
pins=m.owned_json(R/'INPUT-PINS.json');compiler=Path(sys.argv[1]);args=sys.argv[2:]
m.require(str(compiler)==pins['compiler']['path'] and m.regular(compiler.parent,compiler.name)==pins['compiler']['file'],'actual role compiler differs')
fds=[]
for a,b in re.findall(r'--jobserver-(?:auth|fds)=(\d+),(\d+)',os.environ.get('CARGO_MAKEFLAGS','')):
 for value in (int(a),int(b)):
  m.require(value>=3 and stat.S_ISFIFO(os.fstat(value).st_mode),'unknown jobserver descriptor');fds.append(value)
package=os.environ.get('CARGO_PKG_NAME')
query=args==['-vV'] or '--version' in args or ('-' in args and any(x.startswith('--print') for x in args))
if not query:
 matches=[row for row in pins['source_roots'].values() if Path(row['root']).name==str(package)+'-'+str(os.environ.get('CARGO_PKG_VERSION'))]
 m.require(package not in ('memra-engine','memra-server') and len(matches)==1 and os.environ.get('CARGO_MANIFEST_DIR')==matches[0]['root'],'unadmitted root/engine/compiler owner before execution')
body=None
if '--emit=llvm-ir' in args and '-' in args:
 body=sys.stdin.buffer.read(4097);m.require(len(body)<=4096,'role stdin exceeds finitecapture budget')
record={'schema':'memra979-current-rust-role-path-observation-v1','argv':args,'compiler':pins['compiler'],'cwd':os.getcwd(),'environment_names':sorted(os.environ),'out_dir':os.environ.get('OUT_DIR'),'package_name':os.environ.get('CARGO_PKG_NAME'),'package_version':os.environ.get('CARGO_PKG_VERSION'),'started':time.time(),'identity_injected':False,'qualified':False}
if body is not None:record['stdin']={'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'utf8':body.decode()}
result=subprocess.run([str(compiler),*args],input=body,capture_output=True,pass_fds=tuple(sorted(set(fds))))
record.update({'exit':result.returncode,'finished':time.time(),'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()})
m.immutable_json(R/'observations'/('unit-'+str(os.getpid())+'.json'),record)
sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr);sys.exit(result.returncode)
