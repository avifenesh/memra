import argparse,hashlib,json,os,subprocess,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--head',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
repo=a.repo.resolve();out=a.out.resolve()
def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
assert git('rev-parse','HEAD')==a.head
assert not git('status','--porcelain')
out.mkdir(parents=True,exist_ok=False)
assert not (repo/'target').exists(), 'Target path already exists; fresh clone required'
env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_') and k not in ['DOCS_RS','CARGO_TARGET_DIR','CARGO_BUILD_TARGET','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS']}
rustbin=Path('/opt/memra/rust-bin').read_text().strip()
env.update(CUDA_VISIBLE_DEVICES='',RUSTUP_TOOLCHAIN='1.97.1',CUDA_HOME='/usr/local/cuda-13.1',MEMRA_NVCC='/usr/local/cuda-13.1/bin/nvcc',MEMRA_CUDA_ARCH='120a',CARGO_TARGET_DIR=str(out/'target'),PATH=rustbin+':/usr/local/cuda-13.1/bin:'+env['PATH'],LD_LIBRARY_PATH='/usr/local/cuda-13.1/lib64:'+env.get('LD_LIBRARY_PATH',''))
command=['cargo','build','--release','--locked','--jobs','8','-p','memra-engine','--bin','kernel-check','--bin','run-gen','--bin','run-spec','--bin','argmax-margin-probe','-p','memra-server','--bin','memra-server','-p','memra-tokenizer','--bin','tok-parity']
record={'source_sha':a.head,'source_tree':git('rev-parse','HEAD^{tree}'),'repo':str(repo),'out':str(out),'command':command,'started_unix':time.time(),'cuda_visible_devices':'','cuda_arch':'120a','docs_rs':False}
record['rustc']=subprocess.check_output(['rustc','-Vv'],env=env,text=True);record['nvcc']=subprocess.check_output([env['MEMRA_NVCC'],'--version'],env=env,text=True)
(out/'build.pending.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'build.log').open('w') as f:r=subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
record['exit_code']=r.returncode;record['finished_unix']=time.time()
assert r.returncode==0, 'Native build failed; retain build.log'
assert git('rev-parse','HEAD')==a.head and not git('status','--porcelain')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
record['binaries']={}
for name in ['kernel-check','run-gen','run-spec','argmax-margin-probe','memra-server','tok-parity']:
 path=out/'target/release'/name
 with path.open('rb') as f:header=f.read(20)
 assert header[:4]==b'\x7fELF' and int.from_bytes(header[18:20],'little')==62,name
 record['binaries'][name]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
(repo/'target/release').mkdir(parents=True,exist_ok=False)
for name,info in record['binaries'].items():
 staged=repo/'target/release'/name;shutil.copy2(info['path'],staged);assert sha(staged)==info['sha256'];info['staged_path']=str(staged)
assert not git('status','--porcelain')
record['builder_sha256']=sha(Path(__file__))
record['log_sha256']=sha(out/'build.log');(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
print('BUILD_PASS',a.head,flush=True)
