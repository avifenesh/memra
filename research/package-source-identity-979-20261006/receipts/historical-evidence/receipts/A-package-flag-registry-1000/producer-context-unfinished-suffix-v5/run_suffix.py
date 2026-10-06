#!/usr/bin/python3 -I -B
"""Read-only proposed-reader control on old actual artifacts, not new issuance."""
import argparse,contextlib,copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
BASE=HERE.parent
OLD=BASE/'producer-context-contract-v4'
PINS=json.loads((HERE/'INPUT-PINS.json').read_bytes())
@contextlib.contextmanager
def cwd(path):
 old=Path.cwd();os.chdir(path)
 try:yield
 finally:os.chdir(old)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--build-approval',type=Path,required=True);parser.add_argument('--records',type=Path,required=True);args=parser.parse_args()
 assert args.records.is_absolute() and not args.records.exists();args.records.mkdir(parents=True)
 v=load(BASE/'post-build-dc313-fb17-relative-v2/post_build.py','prior_private_mapper');native,records,entry,old,m,cap=v.gate(args.build_approval)
 for name,pin in PINS['failed_job_files'].items():assert sha(Path(PINS['failed_job'])/name)==pin['sha256']
 assert sha(Path(PINS['original_control']))==PINS['original_control_sha256']=='7e0572476b9b9f419cf60e1e91dda70d134a3fb3ef651bfcc65ab2c2d8f8f0e3' and sha(Path(PINS['old_preparation']))==PINS['old_preparation_sha256']
 failed=json.loads((Path(PINS['failed_job'])/'result.json').read_bytes());assert failed['exit_code']==1
 log=(Path(PINS['failed_job'])/'output.log').read_text();assert 'line 57, in main' in log and 'AssertionError: outside sealed package source accepted' in log and Path(PINS['original_empty_records']).is_dir() and not list(Path(PINS['original_empty_records']).iterdir())
 prep=json.loads((OLD/'PREPARATION.json').read_bytes());assert sha(OLD/'package_source_rustc.py')==prep['candidate_receiver_sha256']==PINS['candidate_sha256'] and sha(OLD/'package_source_identity.py')==prep['helper_sha256']
 proposed=load(OLD/'package_source_rustc.py','proposed_reader_observation')
 # Explicit module fixture binding: candidate is not installed/resealed in the
 # old package, and this observation grants no new-source acceptance.
 proposed.ROOT=entry
 refused=(m.Refused,proposed.m.Refused,OSError)
 artifacts=[m.json_bytes(line) for line in (records/'cargo.jsonl').read_bytes().splitlines() if m.json_bytes(line).get('reason')=='compiler-artifact']
 witness=json.loads((BASE/'post-build-dc313-fb17-relative-v2/FAILED-RELATIVE-SOURCE-WITNESS.json').read_bytes());observed=v.traces(records);cases=[];units=[]
 for ordinal,row in enumerate(witness['bad_rows']):
  matches=[trace for trace in observed if trace['trace']==row['trace'] and trace['argv']==row['argv'] and trace['exited_zero']];assert len(matches)==1
  argv=row['argv'][1:];args_=argv[1:];sources=[x for x in args_ if x.endswith('.rs') and not x.startswith('-')];assert len(sources)==1 and not Path(sources[0]).is_absolute()
  events=[e for e in artifacts if e['target']['name'].replace('-','_')==row['crate'] and e['target']['crate_types']==old.crate_types(args_)]
  if ordinal==0:
   # Actual failed trace reaches line57: all assertions through first owner,
   # oldwrongcwd/oldmanifest/newreader, absolute and owner controls passed in
   # that attempt. Retain that prefix; derive only its metadata for resumption.
   owner=cap['snapshot']['payload']['entry'];manifest=entry/cap['roots'][owner]
   exact=[event for event in events if event['package_id']==cap['snapshot']['bindings'][owner]['cargo_package_id'] and event['manifest_path']==str(manifest/'Cargo.toml') and event['target']['src_path']==str(manifest/'build.rs')]
   assert len(exact)==1 and sources==['build.rs'];event=exact[0]
   unit={'owner':owner,'manifest_context':str(manifest),'source':str(manifest/'build.rs'),'argv':argv,'products':proposed.compiler_products(args_,cap['snapshot']['payload']['packages'][owner],manifest)};units.append(unit)
   old_red='producer source owner differs'
  else:
   accepted=[]
   with cwd(v.WT):
    for event in events:
     try:accepted.append((event,v.current_unit_from_event(argv,event,entry,proposed,proposed.m,cap)))
     except refused:pass
   assert len(accepted)==1,(row['crate'],len(accepted));event,unit=accepted[0];units.append(unit);manifest=Path(unit['manifest_context']);product=unit['products'][0]
   with cwd(v.WT):
    try:old.completed_artifact_record(product,cap)
    except m.Refused as error:assert 'producer source owner differs' in str(error);old_red=str(error)
    else:raise AssertionError('old reader unexpectedly accepted actual wrong consumer cwd')
   with cwd(manifest):old_good=old.completed_artifact_record(product,cap)
   with cwd(v.WT):new_good=proposed.completed_artifact_record(product,cap)
   assert old_good==new_good and new_good['argv']==argv
   absolute=manifest/sources[0];assert proposed.checked_lexical_source(absolute,manifest,cap['snapshot']['payload']['packages'][unit['owner']]['files'])==Path(unit['source'])
   duplicate=copy.deepcopy(cap);duplicate['snapshot']['bindings']['ambiguous-fixture-owner']=copy.deepcopy(cap['snapshot']['bindings'][unit['owner']])
   try:v.current_unit_from_event(argv,event,entry,proposed,proposed.m,duplicate)
   except proposed.m.Refused as error:assert 'ambiguous' in str(error)
   else:raise AssertionError('ambiguous current owner accepted')
   wrong=copy.deepcopy(event);other=next(k for k,rel in cap['roots'].items() if k!=unit['owner'] and (entry/rel/sources[0]).is_file());wrong['package_id']=cap['snapshot']['bindings'][other]['cargo_package_id'];wrong['manifest_path']=str(entry/cap['roots'][other]/'Cargo.toml');wrong['target']['src_path']=str(entry/cap['roots'][other]/sources[0])
   try:v.current_unit_from_event(argv,wrong,entry,proposed,proposed.m,cap)
   except refused:pass
   else:raise AssertionError('wrong current package/source context accepted')
  source_row=cap['snapshot']['payload']['packages'][unit['owner']]
  fixture=args.records/('outside-owner-fixture-'+str(ordinal));fixture.mkdir();leaf=fixture/'leaf.rs';leaf.write_bytes(b'owned isolated source fixture\n');outside_pin=proposed.m.regular(fixture,'leaf.rs')
  assert leaf.is_absolute() and not leaf.is_relative_to(manifest) and leaf.is_file()
  try:
   try:proposed.owned_primary_source([str(leaf)],source_row,manifest)
   except proposed.m.Refused as error:assert 'outside package owner' in str(error)
   else:raise AssertionError('truly outside sealed package source accepted')
   assert proposed.m.regular(fixture,'leaf.rs')==outside_pin
  finally:leaf.unlink();fixture.rmdir()
  if row['crate']=='build_script_build':
   with cwd(v.WT):assert not old.original_custom_build_unit(args_,source_row,manifest) and proposed.original_custom_build_unit(args_,source_row,manifest)
   with cwd(manifest):assert old.original_custom_build_unit(args_,source_row,manifest)
  cases.append({'trace':row['trace'],'trace_sha256':matches[0]['trace_sha256'],'actual_argv':argv,'actual_owner':unit['owner'],'actual_source':unit['source'],'old_wrong_cwd_refusal':old_red,'old_manifest_context_passed':True,'candidate_wrong_consumer_cwd_passed_with_identical_record':True,'absolute_lexical_equivalence':True,'wrong_owner_refused':True,'ambiguous_owner_refused':True,'outside_owner_refused':True,'outside_fixture_pin':outside_pin,'retained_prefix_from_805080_before_line57':ordinal==0,'completed_prefix_controls_replayed':False if ordinal==0 else None})
 rootlib=next(unit for unit in units if old.crate_types(unit['argv'][1:])==['lib']);foreign=next(key for key,row in cap['snapshot']['payload']['packages'].items() if row['name']=='autocfg');foreign_manifest=entry/cap['roots'][foreign]
 extern=['--extern','memra_server='+str(rootlib['products'][0])]
 with cwd(foreign_manifest):
  try:old.dependencies(extern,cap,foreign)
  except m.Refused as error:assert 'producer source owner differs' in str(error)
  else:raise AssertionError('old foreign consumer dependency unexpectedly accepted')
  deps,own=proposed.dependencies(extern,cap,foreign);assert len(deps)==1 and deps[0]['package']==rootlib['owner'] and own==[]
 fixture=args.records/'nofollow-fixture';fixture.mkdir();(fixture/'leaf.rs').write_bytes(b'fixture source\n');(fixture/'hop').symlink_to(entry/'src',target_is_directory=True)
 try:proposed.owned_primary_source(['hop/../leaf.rs'],{'files':{'leaf.rs':proposed.m.regular(fixture,'leaf.rs')}},fixture)
 except refused:pass
 else:raise AssertionError('symlink canceled by dotdot bypassed source guard')
 finally:(fixture/'hop').unlink();(fixture/'leaf.rs').unlink();fixture.rmdir()
 assert m.package_capsule(entry)==cap
 report={'failed_job_805080_preserved':PINS['failed_job'],'retained_prefix_evidence':PINS['failed_job_files'],'first_completed_prefix_not_replayed':True,'old_source':v.HEAD,'old_source_seal':cap['source_seal'],'candidate_source_sha256':prep['candidate_receiver_sha256'],'cases':cases,'foreign_manifest_dependency_consumer':str(foreign_manifest),'cross_package_original_argv_and_custody_passed':True,'nofollow_symlink_dotdot_refused':True,'source_or_artifact_mutation':False,'compiler_Cargo_native_ar_GPU_invoked':False,'new_source_capsule_admission_or_qualification':False,'candidate_against_old_capsule_is_regression_observation_only':True}
 (args.records/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
