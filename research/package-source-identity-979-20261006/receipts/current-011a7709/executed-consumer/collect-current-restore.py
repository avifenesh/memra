#!/usr/bin/python3 -I
"""Final selected output/control roles; distinct from historical812 observations."""
import argparse,fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys
sys.dont_write_bytecode=True
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--preparation',type=Path,required=True);parser.add_argument('--admission',type=Path,required=True);args=parser.parse_args()
R=args.preparation.absolute();WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
# Refuse a failed/incomplete current build before source imports or restore writes.
native=json.loads((R/'APPROVED-NATIVE.json').read_bytes());terminal=json.loads((Path(native['records'])/'BUILD-EXIT.json').read_bytes())
assert terminal['exit']==0 and terminal['source']==native['source']
entry=Path(native['entry']);support=entry/'build-support'
assert hashlib.sha256((support/'package_source_identity.py').read_bytes()).hexdigest()=='bdbfe568d72654c220a7b20df0c678568c7395ed192df329fb8ba61980666a39' and hashlib.sha256((support/'package_source_rustc.py').read_bytes()).hexdigest()==native['source_pins']['tools/package_source_rustc.py']['sha256']=='ef5f6eb2922a08979f420fee24f136bc79eee27c20daac1f73cc74eb6befedf7'
spec=importlib.util.spec_from_file_location('receiver',support/'package_source_rustc.py');receiver=importlib.util.module_from_spec(spec);spec.loader.exec_module(receiver);m=receiver.m
admission=m.owned_json(args.admission)
m.require(type(admission) is dict and set(admission)=={'head','entry','output','expectations','cargo_events','issued','identity','aliases','restore','evidence'},'unknown final restore admission')
m.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==admission['head'] and not subprocess.check_output(['git','status','--porcelain'],cwd=WT),'restore source differs')
entry=Path(admission['entry']);cap=m.package_capsule(entry);output=Path(cap['output']);expectations=Path(cap['expectations'])
m.require(output==Path(admission['output']) and expectations==Path(admission['expectations']),'restore role roots differ')
m.require(admission['head']==native['source'] and admission['entry']==native['entry'] and admission['output']==native['output'] and admission['expectations']==native['expectations'],'restore/native tuple differs')
m.require(m.regular(WT,'tools/package_source_identity.py')['sha256']=='bdbfe568d72654c220a7b20df0c678568c7395ed192df329fb8ba61980666a39','current helper pin differs')
# This admission is independently retained before mutations. It selects actual
# issued producer paths; candidate JSON discovery cannot add authority.
required=set();externals=set();records={};issued_members=set()
for name in admission['issued']:
 path=Path(name);m.require(path.is_absolute() and path.is_relative_to(output),'unowned issued producer path')
 record=receiver.completed_artifact_record(path,cap);records[str(path)]=record;required.add(path);required.add(receiver.record_path(path));externals.add(expectations/(record['custody_key']+'.json'))
 for role in record['artifact_bundle']:
  member=path if role=='binary' else path.with_suffix('.'+role);required.add(member);issued_members.add(member)
 for name in record['declared_outputs']:required.add(Path(name))
cargo_members=set();cargo_finished=False;actual_engine_out=None
for line in m.regular(Path(admission['cargo_events']).parent,Path(admission['cargo_events']).name,contents=True).decode().splitlines():
 event=m.json_bytes(line.encode())
 if event.get('reason')=='build-finished':cargo_finished=event.get('success') is True
 if event.get('reason')=='compiler-artifact':
  for name in [*event['filenames'],*([event['executable']] if event.get('executable') else [])]:
   path=Path(name);m.require(path.is_absolute() and path.is_relative_to(output),'unowned Cargo member');required.add(path);cargo_members.add(path)
 elif event.get('reason')=='build-script-executed':
  owners=[key for key,binding in cap['snapshot']['bindings'].items() if binding['cargo_package_id']==event['package_id']];m.require(len(owners)==1,'ambiguous generated output owner');out=Path(event['out_dir']);m.require(out.is_relative_to(output),'unowned generated output')
  for name in cap['recipe']['generated'][owners[0]]:required.add(out/m.relative_name(name))
  if cap['snapshot']['payload']['packages'][owners[0]]['name']=='memra-engine':
   m.require(actual_engine_out is None or actual_engine_out==out,'ambiguous actual engine output');actual_engine_out=out
m.require(cargo_finished,'Cargo build-finished success missing')
m.require(actual_engine_out is not None,'actual engine generated output event missing')
engine=next(key for key,row in cap['snapshot']['payload']['packages'].items() if row['name']=='memra-engine')
m.require(len(cap['recipe']['generated'][engine])==46 and set(m.inventory(actual_engine_out,package=False))==set(cap['recipe']['generated'][engine]),'actual engine46 output membership differs')
identity=admission['identity'];m.require(type(identity) is str and len(identity)==12,'selected entry identity differs')
identity_path=output/('identity-'+identity+'.json');value=m.owned_json(identity_path);expected=expectations/identity_path.name
m.require(m.owned_json(expected)=={'sha256':m.digest(value)} and m.identity('memra-package-compiled-input-v1',value['tuple'])==identity and value['source_seal']==cap['source_seal'],'selected identity custody differs')
required.add(identity_path);externals.add(expected)
for binding in value['bindings']:
 path=Path(binding['path']);m.require(path.is_absolute() and path.is_relative_to(output) and m.regular(path.parent,path.name)==binding['file'],'selected compiled input binding differs');required.add(path)
for alias,producer in admission['aliases'].items():
 alias=Path(alias);producer=Path(producer);m.require(alias.is_relative_to(output) and str(producer) in records,'unknown producer/Cargo alias');m.require(m.regular(alias.parent,alias.name)==m.regular(producer.parent,producer.name),'producer/Cargo alias bytes/mode differ');required.add(alias);required.add(producer)
aliases={Path(name) for name in admission['aliases']}
m.require(cargo_members <= issued_members|aliases,'Cargo artifact lacks validated issued member or alias')
# Full handoff transport comes from the already validated issued units, not
# candidate file discovery. Hold only completed producer roles while copying.
handoff_handles=[];handoff_paths=set();handoff_bindings={}
try:
 for artifact_name in admission['issued']:
  artifact=Path(artifact_name);role=receiver.producer_role(artifact,cap)
  if role in handoff_bindings:continue
  handle=receiver.producer_descriptor(artifact,cap);handoff_handles.append(handle)
  try:fcntl.flock(handle[0],fcntl.LOCK_SH|fcntl.LOCK_NB)
  except BlockingIOError:m.require(False,'selected producer became active during transport')
  registration=receiver.read_producer_registration(handle,cap,artifact)
  completed=receiver.completed_artifact_record(artifact,cap)
  m.require(completed==records[artifact_name],'selected producer changed before transport')
  lock=expectations/handle[3];terminal=expectations/('producer-terminal-'+registration['attempt']+'.json')
  value=m.owned_json(terminal);m.require(value['registration_seal']==registration['seal'] and value['status']=='complete','transport terminal differs')
  externals.update([lock,terminal]);handoff_paths.update([lock,terminal]);handoff_bindings[role]={'artifact':str(artifact),'registration':registration,'handle_index':len(handoff_handles)-1}
 reference=expectations/('registry-corpus-'+cap['source_seal']+'.json');ref=m.owned_json(reference)
 m.require(ref['schema']=='memra-registry-corpus-reference-v1' and ref['owner']==engine,'current registry transport reference differs')
 corpus_file=expectations/(ref['corpus_sha256']+'.registry-corpus.json');m.require(m.digest(m.owned_json(corpus_file))==ref['corpus_sha256'],'current registry transport corpus differs');externals.update([reference,corpus_file])
 restore=Path(admission['restore']);evidence=Path(admission['evidence']);m.require(not os.path.lexists(restore) and not os.path.lexists(evidence),'immutable restore destination exists');
 for dest in (restore,evidence):
  m.require(dest.is_absolute() and not any(dest.is_relative_to(root) or root.is_relative_to(dest) for root in (entry,output,expectations)),'restore role overlaps source/producer custody')
 m.require(not restore.is_relative_to(evidence) and not evidence.is_relative_to(restore),'restore/evidence roles overlap');restore.mkdir();evidence.mkdir()
 rows={};control_rows={}
 for path in sorted(required):
  name=path.relative_to(output).as_posix();row=m.regular(output,name);dest=restore/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);m.require(m.regular(restore,name)==row,'restored bytes/mode differ');rows[name]=row
 for path in sorted(externals):
  name=path.relative_to(expectations).as_posix();row=m.regular(expectations,name);dest=evidence/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);m.require(m.regular(evidence,name)==row,'external custody copy differs');control_rows[name]=row
 m.require({p.relative_to(output).as_posix() for p in required}<=set(rows),'required output/control member omitted')
 for binding in handoff_bindings.values():
  handle=handoff_handles[binding['handle_index']];m.require(receiver.read_producer_registration(handle,cap,Path(binding['artifact']))==binding['registration'],'selected producer changed during transport')
 m.immutable_json(R/'FINAL-RESTORE-SELECTION-RESULT.json',{'source':admission['head'],'source_output':str(output),'restored_output':str(restore),'separate_external_expectations':str(evidence),'outputs':rows,'external_expectations':control_rows,'identity':identity,'byte_mode_restore':True,'full_completed_custody_from_consumer_cwd':True,'producer_completion_controls_transported':True,'producer_role_count':len(handoff_bindings),'consumer_rederive_restored_paths_executed':False,'qualification':False,'limits':'This collector verifies selected transport only. Absolute path-custody remains original; subsequent original-path restoration or separately reviewed relocation consumer/rederive required, never claim restored validation at old paths.'})
 print(json.dumps({'selected_output_and_control_files':len(rows),'external_expectations':len(control_rows),'restored_path_consumer_checked':False,'qualified':False}))
finally:
 for handle in handoff_handles:
  os.close(handle[0]);os.close(handle[1])
