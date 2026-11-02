#!/usr/bin/python3 -I
import sys
sys.dont_write_bytecode=True
"""Derive external corpus before candidate attachment, from actual current archives/Git."""
import hashlib,importlib.util,json,os,subprocess,tarfile,tomllib
from pathlib import Path
R=Path(__file__).absolute().parent;WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
spec=importlib.util.spec_from_file_location('source_io',WT/'tools/package_source_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
approved=m.owned_json(R/'APPROVED-SOURCE.json');head=approved['head']
m.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==head and not subprocess.check_output(['git','status','--porcelain'],cwd=WT),'corpus source differs')
archive_root=R/'source-archives';result=m.owned_json(archive_root/'SOURCE-ARCHIVE-RESULT.json');m.require(result['source']==head,'archive capture source differs')
manifest=m.regular(WT,'Cargo.toml',contents=True);members=tomllib.loads(manifest.decode())['workspace']['members'];member_map={};versions={}
for relative in members:
 m.relative_name(relative);raw=m.regular(WT,relative+'/Cargo.toml',contents=True);package=tomllib.loads(raw.decode())['package'];name=package['name'];m.require(name not in member_map,'duplicate original member');member_map[name]=relative
 version=package['version'];versions[name]=version if type(version) is str else tomllib.loads(manifest.decode())['workspace']['package']['version']
archives={};archive_pins={}
for name,row in result['archives'].items():
 m.require(m.regular(archive_root,name)==row['file'],'current archive custody differs');archive_pins[name]=row['file'];content={}
 with tarfile.open(archive_root/name,'r:gz') as tar:
  for member in tar.getmembers():
   m.require(member.isfile(),'nonregular archive member');relative=m.relative_name(Path(*Path(member.name).parts[1:]).as_posix());raw=tar.extractfile(member).read();entry={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mode':'100755' if member.mode&0o100 else '100644'};m.require(row['members'][relative]==entry,'archive/member mismatch');content[relative]=raw
 archives[name]=content
# Independently derive required role membership from the pinned producer Git tree.
# This selection is not taken from an attachment or a proposed corpus.
tree=subprocess.check_output(['git','ls-tree','-r','-z',head],cwd=WT);tracked={}
for item in tree.split(b'\0'):
 if not item:continue
 meta,path=item.split(b'\t',1);mode,kind,oid=meta.decode().split();tracked[path.decode()]=(mode,kind,oid)
required={'Cargo.toml','docs/FLAGS.md'}
for name,root in member_map.items():
 if name!='memra-engine':required.add(root+'/Cargo.toml')
 for path in tracked:
  if path.startswith(root+'/src/') and (name!='memra-engine' or path.endswith('.rs')):required.add(path)
# The semantic original corpus includes every physical src leaf. Refuse
# an untracked addition rather than silently deriving a partial Git-only view.
for name,root in member_map.items():
 physical=m.inventory(WT/root/'src',package=False)
 expected={path[len(root+'/src/'):]:mode for path,(mode,kind,oid) in tracked.items() if path.startswith(root+'/src/')}
 m.require(set(physical)==set(expected) and all(physical[path]['mode']==mode for path,mode in expected.items()),'producer physical/Git source membership differs')
inputs={};origins={};supp={};engine=member_map['memra-engine'];engine_archive='memra-engine-'+versions['memra-engine']+'.crate'
for owner in sorted(required):
 mode,kind,oid=tracked[owner];m.require(kind=='blob' and mode in ('100644','100755'),'unsupported producer source role')
 raw=m.regular(WT,owner,contents=True);git_raw=subprocess.check_output(['git','cat-file','blob',oid],cwd=WT);m.require(raw==git_raw and m.regular(WT,owner)['mode']==mode,'producer Git/raw/mode binding differs')
 if owner in ('Cargo.toml','docs/FLAGS.md') or owner.startswith(member_map.get('memra-probe','__none__')+'/'):
  origin={'role':'original producer Git','head':head,'path':owner,'blob':oid}
 else:
  name=next(name for name,root in member_map.items() if owner.startswith(root+'/'));root=member_map[name];relative=owner[len(root)+1:];archive=name+'-'+versions[name]+'.crate';entry='Cargo.toml.orig' if relative=='Cargo.toml' else relative
  m.require(archive in archives and entry in archives[archive] and archives[archive][entry]==raw,'actual normalized archive/producer source bridge differs')
  m.require(result['archives'][archive]['members'][entry]['mode']==mode,'actual archive/producer source mode differs');origin={'role':'actual current Cargo archive','archive':archive,'entry':entry,'archive_file':archive_pins[archive]}
 inputs[owner]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mode':mode};origins[owner]=origin
 if not owner.startswith(engine+'/src/'):supp[owner]=inputs[owner]
engine_manifest=result['archives'][engine_archive]['members']['Cargo.toml']
corpus={'schema':'memra-registry-original-corpus-v1','workspace_members':member_map,'inputs':inputs,'engine_manifest':engine_manifest};digest=m.digest(corpus)
external=R/'independent-registry-corpus';m.require(not os.path.lexists(external),'immutable producer corpus destination exists');external.mkdir()
m.immutable_json(external/(digest+'.registry-corpus.json'),corpus)
m.immutable_json(external/'PRODUCER-PROVENANCE.json',{'head':head,'archives':archive_pins,'owners':origins,'supplementary_inputs':supp,'required_git_roles':sorted(required),'corpus_sha256':digest,'derived_before_attachment':True,'qualification':False})
print(json.dumps({'source':head,'members':len(member_map),'input_owners':len(inputs),'rust_source_owners':sum(x.endswith('.rs') for x in inputs),'corpus_sha256':digest,'candidate_attached':False,'qualified':False}))
