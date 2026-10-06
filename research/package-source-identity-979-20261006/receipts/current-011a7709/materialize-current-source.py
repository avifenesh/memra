#!/usr/bin/python3 -I
import sys
sys.dont_write_bytecode=True
"""Normal current archive materialization plus real metadata; no capsule/native claim."""
import hashlib,importlib.util,json,os,shutil,subprocess,tarfile
from pathlib import Path
R=Path(__file__).absolute().parent;WT=Path('/home/evidence-user/projects/evidence-source-worktree-979');BANK=R.parent
spec=importlib.util.spec_from_file_location('source_io',WT/'tools/package_source_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.owned_json(R/'APPROVED-SOURCE.json');head=a['head'];m.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==head and not subprocess.check_output(['git','status','--porcelain'],cwd=WT),'source materialization head differs')
archives=R/'source-archives';captured=m.owned_json(archives/'SOURCE-ARCHIVE-RESULT.json');m.require(captured['source']==head,'current archive source differs')
stage=R/'prepared-current-source';m.require(not os.path.lexists(stage),'immutable source stage exists');stage.mkdir();entry=stage/'package';entry.mkdir();vendor=entry/'vendor';vendor.mkdir()
def unpack(archive,dest):
 files={}
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar.getmembers():
   m.require(member.isfile(),'archive nonregular member');name=m.relative_name(Path(*Path(member.name).parts[1:]).as_posix());m.require(name not in files,'duplicate archive member');raw=tar.extractfile(member).read();path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o755 if member.mode&0o100 else 0o644);files[name]=hashlib.sha256(raw).hexdigest()
 return files
firstparty={}
for name,row in captured['archives'].items():
 m.require(m.regular(archives,name)==row['file'],'archive custody differs');firstparty[name.removesuffix('.crate')]=row
 m.require(name.endswith('.crate'),'unknown archive filename')
 if name.startswith('memra-server-'):unpack(archives/name,entry)
 else:
  root=vendor/name.removesuffix('.crate');root.mkdir();files=unpack(archives/name,root);(root/'.cargo-checksum.json').write_bytes(m.canonical({'files':files,'package':row['file']['sha256']}))
# Reuse only exact custodied third-party source bodies, never oldfirstparty.
# Every checksum/member/type/mode is validated, then copied and revalidated.
original=BANK/'native-current-4951-observation/package/vendor';thirdparty={}
original_pins=m.owned_json(BANK/'native-current-4951-observation/SOURCE-STAGE-PINS.json')
m.require(original_pins['source']=='4951db56acd22a196a6e9484e3a2bdcf8da59d76','historical source custody pin differs')
for root in sorted(original.iterdir()):
 m.require(root.is_dir() and not root.is_symlink(),'thirdparty source root type differs')
 if root.name.startswith('memra-'):continue
 if root.name=='yoke-derive-0.8.3':continue
 rows=m.inventory(root,package=False);m.require(rows==original_pins['vendor_exact_source_membership'][root.name],'historicalthirdparty source custody differs');checksum=m.owned_json(root/'.cargo-checksum.json');m.require(type(checksum) is dict and set(checksum)=={'files','package'} and type(checksum['package']) is str,'thirdparty checksum shape differs');m.require(set(rows)==set(checksum['files'])|{'.cargo-checksum.json'} and all(rows[n]['sha256']==h for n,h in checksum['files'].items()),'thirdparty checksum/membership differs')
 target=vendor/root.name;m.require(not os.path.lexists(target),'source namespace collision');shutil.copytree(root,target);m.require(m.inventory(target,package=False)==rows,'thirdparty raw/mode copy differs');thirdparty[root.name]={'checksum_sha256':rows['.cargo-checksum.json']['sha256'],'archive_sha256':checksum['package'],'files':rows}
# Current locked yoke source is newly acquired source custody, never an old
# compiler artifact or old source-seal retag. The exact cache archive is pinned.
yoke_archive=Path('/home/evidence-user/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/yoke-derive-0.8.4.crate')
m.require(m.regular(yoke_archive.parent,yoke_archive.name)['sha256']=='ec8ebde2db3681e8c9980cc27822030e68752690ddfa9473e739aeb4dbde6d71','current yoke archive checksum differs')
yoke_root=vendor/'yoke-derive-0.8.4';m.require(not os.path.lexists(yoke_root),'current yoke source namespace exists');yoke_root.mkdir()
yoke_files=unpack(yoke_archive,yoke_root)
(yoke_root/'.cargo-checksum.json').write_bytes(m.canonical({'files':yoke_files,'package':'ec8ebde2db3681e8c9980cc27822030e68752690ddfa9473e739aeb4dbde6d71'}))
thirdparty[yoke_root.name]={'checksum_sha256':m.regular(yoke_root,'.cargo-checksum.json')['sha256'],'archive_sha256':'ec8ebde2db3681e8c9980cc27822030e68752690ddfa9473e739aeb4dbde6d71','files':m.inventory(yoke_root,package=False)}
support=entry/'build-support';support.mkdir(exist_ok=True)
for name in ['package_source_identity.py','package_source_rustc.py']:
 m.require(m.regular(WT,'tools/'+name)==a['files']['tools/'+name],'helper/receiver source pin differs');shutil.copy2(WT/'tools'/name,support/name)
config=entry/'.cargo/config.toml';config.parent.mkdir(exist_ok=True);config.write_text('[source.crates-io]\nreplace-with="memra-package-source"\n[source.memra-package-source]\ndirectory="vendor"\n[build]\nrustc-wrapper="build-support/package_source_rustc.py"\n')
env=dict(os.environ);env['CARGO_BUILD_JOBS']='2';command=[a['cargo']['path'],'metadata','--offline','--locked','--format-version','1','--filter-platform','x86_64-unknown-linux-gnu','--manifest-path',str(entry/'Cargo.toml')]
result=subprocess.run(command,cwd=entry,env=env,capture_output=True);(stage/'metadata.stdout').write_bytes(result.stdout);(stage/'metadata.stderr').write_bytes(result.stderr);m.require(result.returncode==0,'actual normal metadata resolution failed')
graph=m.json_bytes(result.stdout);m.require(any(row['name']=='yoke-derive' and row['version']=='0.8.4' for row in graph['packages']) and not any(row['name']=='yoke-derive' and row['version']=='0.8.3' for row in graph['packages']),'current resolved yoke binding differs');m.require(Path(graph['workspace_root'])==entry and all(Path(x['manifest_path']).is_relative_to(entry) for x in graph['packages']),'normal package borrowed external/parent source')
m.immutable_json(stage/'MATERIALIZATION-RESULT.json',{'head':head,'firstparty_archives':captured['archives'],'thirdparty_sources':thirdparty,'actual_Cargo_argv':command,'resolved_packages':len(graph['packages']),'workspace_root':str(entry),'prepared_root':str(entry),'current_helpers':{n:m.regular(support,n) for n in ['package_source_identity.py','package_source_rustc.py']},'candidate_capsule_prepared':False,'native_compiled':False,'qualification':False})
print(json.dumps({'source':head,'resolved_packages':len(graph['packages']),'normal_source_metadata':True,'capsule_ready':False,'qualified':False}))
