#!/usr/bin/python3 -I -B
"""Proposed current postbuild phases; Root review/admission required; no build/GPU."""
import argparse,hashlib,importlib.util,json,os,re,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
NS=Path('/home/evidence-user/.local/state/evidence-campaign-20261002')
WT=Path('/home/evidence-user/projects/evidence-source-worktree-979')
PINS=json.loads((HERE/'INPUT-PINS.json').read_bytes())
PREP=Path(PINS['preparation'])
HEAD=PINS['current_source']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def gate(approval):
 native=json.loads((PREP/'APPROVED-NATIVE.json').read_bytes());records=Path(native['records']);exit_file=records/'BUILD-EXIT.json'
 assert native['source']==HEAD and native['source_pins']['tools/package_source_rustc.py']['sha256']==PINS['receiver_sha256']
 assert json.loads(exit_file.read_bytes())=={'exit':0,'qualification':False,'source':HEAD}
 reviewed=json.loads(approval.read_bytes())
 assert set(reviewed)=={'source','Root_inspected_native_success','files','CPU_guard_file','CPU_guard_sha256'} and reviewed['source']==HEAD and reviewed['Root_inspected_native_success'] is True
 for name in ['BUILD-EXIT.json','cargo.jsonl','cargo.stderr','version.stdout','version.stderr']:
  assert reviewed['files'][name]==sha(records/name)
 guard=Path(reviewed['CPU_guard_file']);assert guard==PREP/'resource-guard-records/CPU-GUARD-RESULT.json' and sha(guard)==reviewed['CPU_guard_sha256']
 guard_result=json.loads(guard.read_bytes());assert guard_result['native_success'] is True and guard_result['child_exit']==0 and guard_result['CPU_from_before_preflight_through_teardown_seconds']<=3600 and guard_result['failure'] is None
 assert guard_result['CPU_ceiling_seconds']==3600 and guard_result['wall_deadline_seconds']==1775
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==HEAD and subprocess.check_output(['git','status','--porcelain'],cwd=WT)==b''
 for name,row in PINS['inputs'].items():assert sha(Path(name))==row['sha256'],name
 entry=Path(native['entry']);receiver=load(entry/'build-support/package_source_rustc.py','post_receiver');m=receiver.m
 assert sha(entry/'build-support/package_source_rustc.py')==PINS['receiver_sha256'] and sha(entry/'build-support/package_source_identity.py')==PINS['helper_sha256']
 cap=m.package_capsule(entry);assert cap['source_seal']==PINS['source_seal'] and native['source']!='dc31381046bf7293fc03bf45bc7d68cb0cc48d97' and PREP.name!='current-main-5d-standard-1800'
 for name,row in native['tools'].items():assert m.regular(Path(row['path']).parent,Path(row['path']).name)==row['file'],name
 return native,records,entry,receiver,m,cap

def traces(records):
 result=[]
 for path in sorted(records.glob('owned-child.*')):
  lines=path.read_text().splitlines()
  for line in lines:
   if not line.startswith('execve(') or not line.endswith(' = 0'):continue
   executable,offset=json.JSONDecoder().raw_decode(line[len('execve('):]);argv=json.JSONDecoder().raw_decode(line[line.index(', [')+2:])[0]
   result.append({'trace':str(path),'trace_sha256':sha(path),'executable':executable,'argv':argv,'exited_zero':any(re.fullmatch(r'\+\+\+ exited with 0 \+\+\+',x) for x in lines)})
 return result

def current_unit_from_event(argv,event,entry,r,m,cap):
 compiler,args_=argv[0],argv[1:]
 m.require(compiler==cap['compiler']['path'],'unit compiler differs from current capsule')
 owners=[key for key,binding in cap['snapshot']['bindings'].items() if binding['cargo_package_id']==event['package_id']]
 m.require(len(owners)==1,'current Cargo source owner is ambiguous')
 owner=owners[0];manifest=entry/cap['roots'][owner];source_row=cap['snapshot']['payload']['packages'][owner]
 m.require(Path(event['manifest_path'])==manifest/'Cargo.toml','Cargo package manifest/context differs')
 m.require(r.crate_types(args_)==event['target']['crate_types'],'Cargo/compiler crate role differs')
 source=[arg for arg in args_ if arg.endswith('.rs') and not arg.startswith('-')]
 m.require(len(source)==1,'current compiler primary source is ambiguous')
 raw=Path(source[0]);lexical=raw if raw.is_absolute() else manifest/raw
 normalized=r.checked_lexical_source(lexical,manifest,source_row['files'])
 m.require(normalized==Path(event['target']['src_path']),'compiler source/current Cargo target differs')
 products=r.compiler_products(args_,source_row,manifest)
 # Product paths are derived from executed argv, never sidecar discovery.
 # Full issued/external custody fixes the owner before a relative name can
 # select a lookalike package, even for ubiquitous build.rs/src/lib.rs.
 for product in products:
  record=r.completed_artifact_record(product,cap)
  m.require(record['package']==owner and record['manifest_root']==cap['roots'][owner] and record['argv']==argv,'executed unit custody/package/context differs')
 return {'owner':owner,'source':str(normalized),'argv':argv,'products':products,'source_argument':source[0],'manifest_context':str(manifest)}

def inspect(args,native,records,entry,r,m,cap):
 root=args.result_root.absolute();assert not root.exists();root.mkdir();output=Path(cap['output']);events=[m.json_bytes(line) for line in (records/'cargo.jsonl').read_bytes().splitlines()];assert events[-1]=={'reason':'build-finished','success':True}
 observed=traces(records);engine=next(key for key,row in cap['snapshot']['payload']['packages'].items() if row['name']=='memra-engine')
 out_events=[event for event in events if event.get('reason')=='build-script-executed' and event['package_id']==cap['snapshot']['bindings'][engine]['cargo_package_id']];assert len(out_events)==1;engine_out=Path(out_events[0]['out_dir']);assert engine_out.is_relative_to(output)
 expected=cap['recipe']['generated'][engine];actual=m.inventory(engine_out,package=False);assert len(expected)==46 and set(actual)==set(expected)
 recipe=m.owned_json(Path(PINS['recipe_path']));assert len(recipe['native_commands'])==43 and len(recipe['MMQ_members_in_order'])==33
 source_bridge=m.owned_json(Path(PINS['native_source_bridge']));native_root=entry/cap['roots'][engine]
 for row in source_bridge['native_source_rows']:
  relative=Path(row['path']).relative_to('crates/memra-engine');raw=m.regular(native_root,relative.as_posix(),contents=True);blob=subprocess.check_output(['git','cat-file','blob',row['blob']],cwd=WT);assert raw==blob and m.regular(native_root,relative.as_posix())['mode']==row['mode']
 assert m.regular(native_root,'build.rs')['sha256']==PINS['engine_build_sha256']
 engaged=[]
 for command in recipe['native_commands']:
  expected_argv=[native['tools']['nvcc']['path'],*[arg.replace('{OUT_DIR}',str(engine_out)) for arg in command['argv_template']]]
  matches=[row for row in observed if row['executable']==expected_argv[0] and row['argv']==expected_argv and row['exited_zero']]
  assert len(matches)==1,command['output_name'];engaged.append(matches[0])
 assert sum(row['kind']=='fatbin' for row in recipe['native_commands'])==10
 ar_argv=['ar',recipe['ar_options'],str(engine_out/'libmemra_mmq.a'),*[str(engine_out/name) for name in recipe['MMQ_members_in_order']]]
 ar_matches=[row for row in observed if row['argv']==ar_argv and Path(row['executable']).resolve()==Path(native['tools']['ar']['path']).resolve() and row['exited_zero']];assert len(ar_matches)==1
 listed=subprocess.run([native['tools']['ar']['path'],'t',str(engine_out/'libmemra_mmq.a')],capture_output=True,timeout=10,check=True);assert listed.stdout.decode().splitlines()==recipe['MMQ_members_in_order']
 (root/'ar-members.stdout').write_bytes(listed.stdout);(root/'ar-members.stderr').write_bytes(listed.stderr)
 # Issued path candidates come only from actual successful compiler-wrapper
 # argv under this current trace. No sidecar glob may select authority.
 candidates=[]
 bindings={row['cargo_package_id']:key for key,row in cap['snapshot']['bindings'].items()}
 for row in observed:
  argv=row['argv']
  if not argv or argv[0]!=str(entry/'build-support/package_source_rustc.py') or not row['exited_zero']:continue
  compiler,args_=argv[1],argv[2:]
  if compiler!=cap['compiler']['path'] or '--crate-name' not in args_ or 'link' not in r.val(args_,'--emit','').split(','):continue
  raw_sources=[arg for arg in args_ if arg.endswith('.rs') and not arg.startswith('-')]
  if len(raw_sources)!=1:continue
  matches=[]
  for event in events:
   if event.get('reason')!='compiler-artifact' or r.crate_types(args_)!=event['target']['crate_types']:continue
   owner=bindings[event['package_id']];manifest=entry/cap['roots'][owner];raw=Path(raw_sources[0]);lexical=raw if raw.is_absolute() else manifest/raw
   # Cheap role/path filtering cannot grant custody. The helper below does
   # the actual nofollow lexical source and complete producer-owner checks.
   if Path(os.path.normpath(str(lexical)))!=Path(event['target']['src_path']):continue
   products=r.compiler_products(args_,cap['snapshot']['payload']['packages'][owner],manifest)
   try:
    record=r.completed_artifact_record(products[0],cap)
   except (m.Refused,OSError):raise
   if record['package']!=owner:continue
   unit=current_unit_from_event([compiler,*args_],event,entry,r,m,cap);unit['trace']=row;matches.append(unit)
  assert len(matches)==1,(r.val(args_,'--crate-name'),raw_sources[0],len(matches))
  candidates.append(matches[0])
 bindings={row['cargo_package_id']:key for key,row in cap['snapshot']['bindings'].items()};issued={};aliases={};mapping=[]
 def map_file(path,unit,event):
  path=Path(path).absolute();products=unit['products']
  if path in products:return path
  args_=unit['argv'][1:];crate=r.val(args_,'--crate-name');kind=event['target']['kind'];selected=[]
  if kind==['lib']:
   for product in products:
    if path==output/'release'/('lib'+crate+product.suffix):selected.append(product)
  elif kind==['bin']:
   if path==output/'release'/event['target']['name']:selected=[p for p in products if r.artifact_role(p)=='binary']
  elif kind==['custom-build'] and path.name=='build-script-build' and r.original_custom_build_unit(args_,cap['snapshot']['payload']['packages'][unit['owner']],entry/cap['roots'][unit['owner']]):selected=[p for p in products if p.parent==path.parent and r.artifact_role(p)=='binary']
  return selected[0] if len(selected)==1 else None
 for event in events:
  if event.get('reason')!='compiler-artifact':continue
  owner=bindings[event['package_id']];files=list(dict.fromkeys([*event['filenames'],*([event['executable']] if event.get('executable') else [])]));matches=[]
  for unit in candidates:
   if unit['owner']!=owner or unit['source']!=event['target']['src_path'] or r.crate_types(unit['argv'][1:])!=event['target']['crate_types']:continue
   selected=[map_file(path,unit,event) for path in files]
   if all(path is not None for path in selected):matches.append((unit,selected))
  assert len(matches)==1,(event['target']['name'],len(matches));unit,selected=matches[0]
  for path in unit['products']:
   record=r.completed_artifact_record(path,cap);assert record['package']==owner and record['argv']==unit['argv'];issued[str(path)]={'seal':record['seal'],'package':owner,'role':record['artifact_role'],'compiler_unit':record['compiler_unit'],'custody_key':record['custody_key'],'file':record['artifact']}
  for path,primary in zip(files,selected):
   path=Path(path);primary=Path(primary);assert m.regular(path.parent,path.name)==m.regular(primary.parent,primary.name)
   if path!=primary:aliases[str(path)]=str(primary)
   mapping.append({'Cargo':str(path),'primary':str(primary),'copied_alias':path!=primary,'file':m.regular(path.parent,path.name)})
 identity_paths=[];engine_libs=[];server_bins=[]
 for path,row in issued.items():
  record=r.completed_artifact_record(path,cap);name=cap['snapshot']['payload']['packages'][record['package']]['name']
  if name=='memra-engine' and row['role']=='rlib':engine_libs.append(path);assert record['declared_outputs']=={str(engine_out/name):file for name,file in actual.items()}
  if name=='memra-server' and row['role']=='binary' and r.val(record['argv'][1:],'--crate-name')=='memra_server' and next(unit['source'] for unit in candidates if path in [str(product) for product in unit['products']])==str(entry/'src/main.rs'):
   server_bins.append(path);identity=m.identity('memra-package-compiled-input-v1',record['input_tuple']);identity_paths.append(identity)
 assert len(engine_libs)==len(server_bins)==len(identity_paths)==1
 identity=identity_paths[0];value=m.owned_json(output/('identity-'+identity+'.json'));assert value['source_seal']==cap['source_seal'] and m.owned_json(Path(cap['expectations'])/('identity-'+identity+'.json'))=={'sha256':m.digest(value)}
 binary=Path(server_bins[0]);binary_file=issued[str(binary)]['file']
 assert m.regular(binary.parent,binary.name)==binary_file
 version=subprocess.run([str(binary),'--version'],cwd=entry,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},capture_output=True,timeout=10)
 assert version.returncode==0 and version.stdout==(records/'version.stdout').read_bytes() and version.stderr==(records/'version.stderr').read_bytes()
 fingerprint='memra-'+cap['snapshot']['payload']['packages'][cap['snapshot']['payload']['entry']]['version']+'-'+identity
 assert version.stdout.decode().splitlines().count('system_fingerprint '+fingerprint)==1 and 'build_id_src package-source-v1' in version.stdout.decode().splitlines()
 assert m.regular(binary.parent,binary.name)==binary_file
 (root/'inspected-version.stdout').write_bytes(version.stdout);(root/'inspected-version.stderr').write_bytes(version.stderr)
 proposal={'head':HEAD,'entry':str(entry),'output':str(output),'expectations':cap['expectations'],'cargo_events':str(records/'cargo.jsonl'),'issued':sorted(issued),'identity':identity,'aliases':aliases,'restore':str(root/'restored-output'),'evidence':str(root/'restored-external-expectations')}
 m.immutable_json(root/'PROPOSED-RESTORE.json',proposal)
 m.immutable_json(root/'NATIVE-INSPECTION.json',{'source':HEAD,'source_seal':cap['source_seal'],'native43_engagements':engaged,'ar_engagement':ar_matches[0],'ar_order':recipe['MMQ_members_in_order'],'engine46_outputs':actual,'issued_records':issued,'Cargo_primary_copy_map':mapping,'engine_RLIB':engine_libs[0],'stock_BIN':server_bins[0],'stock_BIN_file':binary_file,'system_fingerprint':fingerprint,'actual_version_checked':True,'identity':identity,'restore_executed':False,'audits_or_startup_executed':False,'serving_reach_pending':True,'qualified':False})
 m.immutable_json(root/'SERVING-REACH-PENDING.json',{'source':HEAD,'binary':server_bins[0],'binary_file':issued[server_bins[0]]['file'],'current_native_identity':identity,'runner':PINS['actual_serving_runner'],'runner_sha256':PINS['inputs'][PINS['actual_serving_runner']]['sha256'],'required':'Root separate600s serving admission after native/source/restore acceptance, current cached model/draft/vendor metadata/request pins, canonicalFD9 throughteardown, port/process proof, actualstartupready/health/model/JSON+SSE andvendor-default8turncache-off/on spec/cache engagement; all applicablemodel/nativegates preserved','actual_startup_or_serve_reach':False,'qualified':False})
 assert m.package_capsule(entry)==cap
 print(json.dumps({'inspection_passed':True,'native43':43,'archive_members':33,'engine_outputs':46,'issued_primary':len(issued),'copy_aliases':len(aliases),'restore':False,'qualified':False}))

def command(argv,timeout):
 result=subprocess.run(argv,check=False,timeout=timeout);assert result.returncode==0,argv

def approved_selection(args,native,r,m,cap):
 assert args.selection is not None and args.selection.is_absolute();selection=m.owned_json(args.selection)
 assert selection['head']==HEAD and selection['entry']==native['entry'] and selection['output']==native['output'] and selection['expectations']==native['expectations']
 inspection=m.owned_json(args.result_root/'NATIVE-INSPECTION.json');proposal=m.owned_json(args.result_root/'PROPOSED-RESTORE.json');assert selection==proposal
 return selection,inspection

def restore_consumer(args,native,records,entry,r,m,cap):
 selection,inspection=approved_selection(args,native,r,m,cap);restored=Path(selection['restore']);restored_evidence=Path(selection['evidence']);out=Path(native['output']);expectations=Path(native['expectations']);backup=args.result_root/'original-target-preserved';external_backup=args.result_root/'original-expectations-preserved';assert restored.is_dir() and restored_evidence.is_dir() and not backup.exists() and not external_backup.exists()
 transport=m.owned_json(PREP/'FINAL-RESTORE-SELECTION-RESULT.json');assert transport['source']==HEAD and transport['identity']==selection['identity'] and transport['restored_output']==str(restored) and transport['separate_external_expectations']==str(restored_evidence) and transport['producer_completion_controls_transported'] is True
 assert m.inventory(restored,package=False)==transport['outputs'] and m.inventory(restored_evidence,package=False)==transport['external_expectations']
 original=out.stat();original_expectations=expectations.stat();out.rename(backup);results=[];external_moved=False
 try:
  expectations.rename(external_backup);external_moved=True
  shutil.copytree(restored,out);shutil.copytree(restored_evidence,expectations)
  assert m.package_capsule(entry)==cap
  engine_record=r.completed_artifact_record(inspection['engine_RLIB'],cap)
  current_transport=m.registry_inputs(entry/cap['roots'][engine_record['package']]/'Cargo.toml');assert current_transport['source_seal']==cap['source_seal']
  for path in selection['issued']:r.completed_artifact_record(path,cap)
  identity=m.owned_json(out/('identity-'+selection['identity']+'.json'));assert identity['source_seal']==cap['source_seal']
  for binding in identity['bindings']:assert m.regular(Path(binding['path']).parent,Path(binding['path']).name)==binding['file']
  version=subprocess.run([inspection['stock_BIN'],'--version'],env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},capture_output=True,timeout=10);assert version.returncode==0 and version.stdout==(records/'version.stdout').read_bytes()
  (args.result_root/'restored-version.stdout').write_bytes(version.stdout);(args.result_root/'restored-version.stderr').write_bytes(version.stderr)
  # Existing helper red controls remain historical. Do not replay their fault
  # injections; all current restored-byte and completed-custody checks remain.
  for path in selection['issued']:r.completed_artifact_record(path,cap)
 finally:
  try:
   if out.exists():shutil.rmtree(out)
   backup.rename(out);assert out.stat().st_ino==original.st_ino and out.stat().st_dev==original.st_dev
  finally:
   if external_moved:
    if expectations.exists():shutil.rmtree(expectations)
    external_backup.rename(expectations)
   assert expectations.stat().st_ino==original_expectations.st_ino and expectations.stat().st_dev==original_expectations.st_dev
 for path in selection['issued']:r.completed_artifact_record(path,cap)
 m.immutable_json(args.result_root/'RESTORED-CONSUMER.json',{'source':HEAD,'original_absolute_role_reconstructed':True,'original_absolute_external_role_reconstructed':True,'copied_producer_completion_controls_consumed':True,'current_registry_corpus_from_copied_expectations_consumed':True,'current_stock_BIN_version':True,'negative_controls':results,'helper_red_controls_replayed':False,'complete_original_target_restored':True,'complete_original_expectations_restored':True,'functional9plus9_replayed':False,'qualified':False})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['inspect','collect','restore-consumer','bind','audit','startup'],required=True);parser.add_argument('--build-approval',type=Path,required=True);parser.add_argument('--result-root',type=Path,required=True);parser.add_argument('--selection',type=Path);args=parser.parse_args();assert args.result_root.is_absolute()
 native,records,entry,r,m,cap=gate(args.build_approval)
 assert not any(args.result_root.is_relative_to(path) or path.is_relative_to(args.result_root) for path in [WT,entry,Path(cap['output']),Path(cap['expectations'])])
 if args.phase=='inspect':inspect(args,native,records,entry,r,m,cap);return
 selection,inspection=approved_selection(args,native,r,m,cap)
 if args.phase in ('restore-consumer','bind','audit','startup'):
  collected=m.owned_json(PREP/'FINAL-RESTORE-SELECTION-RESULT.json');assert collected['source']==HEAD and collected['identity']==selection['identity'] and collected['producer_completion_controls_transported'] is True
 if args.phase in ('bind','audit','startup'):
  restored=m.owned_json(args.result_root/'RESTORED-CONSUMER.json');assert restored['source']==HEAD and restored['complete_original_target_restored'] is True and restored['complete_original_expectations_restored'] is True
 if args.phase in ('audit','startup'):
  binding=m.owned_json(args.result_root/'audit-binding/BINDING-RESULT.json');assert binding['source']==HEAD and binding['selected_issued_paths']==selection['issued'] and binding['selection_file']==m.regular(args.selection.parent,args.selection.name)
 if args.phase=='collect':
  command(['/usr/bin/python3','-I','-B',PINS['collector'],'--preparation',str(PREP),'--admission',str(args.selection)],90)
 elif args.phase=='restore-consumer':restore_consumer(args,native,records,entry,r,m,cap)
 elif args.phase=='bind':
  command(['/usr/bin/python3','-I','-B',PINS['binder'],'--preparation',str(PREP),'--workspace',str(WT),'--selection',str(args.selection),'--destination',str(args.result_root/'audit-binding')],30)
 elif args.phase=='audit':command(['/usr/bin/python3','-I','-B',PINS['audits'],'--inputs',str(args.result_root/'audit-binding/AUDIT-INPUTS.json')],90)
 elif args.phase=='startup':
  assert (args.result_root/'audit-binding/audit-results/RESULT.json').exists()
  command(['/usr/bin/python3','-I','-B',PINS['startup'],'--inputs',str(args.result_root/'audit-binding/STARTUP-INPUTS.json')],120)
 # Actual sampled serving/model/cache engagement still requires the separately
 # admitted rig input tuple, currentbinary, canonicalFD9 and applicablegates.
if __name__=='__main__':main()
