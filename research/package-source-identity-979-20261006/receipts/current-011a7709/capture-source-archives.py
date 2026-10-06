#!/usr/bin/python3 -I
import sys
sys.dont_write_bytecode=True
"""Reviewed-composition source archives only; no native verification claim."""
import hashlib,importlib.util,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
R=Path(__file__).absolute().parent;WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
spec=importlib.util.spec_from_file_location('source_io',WT/'tools/package_source_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.owned_json(R/'APPROVED-SOURCE.json')
m.require(type(a) is dict and set(a)=={'head','files','cargo','archive_count','admission'},'unknown source admission shape')
m.require(a['admission']=='source-archive-capture-only' and type(a['head']) is str and len(a['head'])==40,'composed immutable source admission missing')
m.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==a['head'] and not subprocess.check_output(['git','status','--porcelain'],cwd=WT),'source head/worktree differs')
bridge=m.owned_json(R/'CURRENT-CAPTURE-BRIDGE.json')
for name,pin in a['files'].items():
 m.require(m.regular(WT,name)==pin,'composed source bytes/mode differ')
for name,pin in bridge['A5files'].items():
 m.require(name in a['files'] and a['files'][name]['sha256']==pin['A_sha256'] and a['files'][name]['mode']==pin['A_git_mode'],'A reviewed source bridge differs')
for name,expected in [('tools/package_source_identity.py',bridge['helper_sha256']),('tools/package_source_rustc.py',bridge['receiver_sha256'])]:m.require(a['files'][name]['sha256']==expected,'receiver/helper bridge differs')
cargo=Path(a['cargo']['path']);m.require(m.regular(cargo.parent,cargo.name)==a['cargo']['file'],'actual toolchain Cargo differs')
dest=R/'source-archives';m.require(not os.path.lexists(dest),'immutable archive destination exists');dest.mkdir()
env=dict(os.environ);env['CARGO_BUILD_JOBS']='2'
with tempfile.TemporaryDirectory(prefix='memra979-final-package-',dir=Path.home()/'.cache') as scratch:
 result=subprocess.run([str(cargo),'package','--offline','--locked','--no-verify','--workspace','--exclude','memra-probe','--target-dir',scratch],cwd=WT,env=env,capture_output=True)
 (dest/'capture.stdout').write_bytes(result.stdout);(dest/'capture.stderr').write_bytes(result.stderr);m.require(result.returncode==0,'source archive capture failed')
 inventory={}
 for archive in sorted((Path(scratch)/'package').glob('*.crate')):
  target=dest/archive.name;shutil.copy2(archive,target);members={}
  with tarfile.open(target,'r:gz') as tar:
   for member in tar.getmembers():
    m.require(member.isfile(),'source archive nonregular member');parts=Path(member.name).parts;m.require(len(parts)>=2 and not Path(member.name).is_absolute(),'source archive member path differs');relative=m.relative_name(Path(*parts[1:]).as_posix());m.require(relative not in members,'duplicate archive member');raw=tar.extractfile(member).read();members[relative]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mode':'100755' if member.mode&0o100 else '100644'}
  m.require({'Cargo.toml','Cargo.toml.orig','Cargo.lock'}<=set(members),'normalized source manifest/lock missing')
  inventory[archive.name]={'file':m.regular(dest,archive.name),'members':members}
 m.require(len(inventory)==a['archive_count'],'actual archive count differs')
 m.immutable_json(dest/'SOURCE-ARCHIVE-RESULT.json',{'source':a['head'],'archives':inventory,'native_verified':False,'qualified':False,'scope':'NormalCargo --no-verify onlycaptures normalizedsourcearchives; no native/package-marker verification.'})
m.require(not subprocess.check_output(['git','status','--porcelain'],cwd=WT),'source changed during capture')
print(json.dumps({'source':a['head'],'archives':len(inventory),'native_verified':False}))
