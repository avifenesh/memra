#!/usr/bin/python3 -I -B
"""Read-only proposed-reader control on old actual artifacts, not new issuance."""
import argparse,contextlib,copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
BASE=HERE.parent
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
 prep=json.loads((HERE/'PREPARATION.json').read_bytes());assert sha(HERE/'package_source_rustc.py')==prep['candidate_receiver_sha256'] and sha(HERE/'package_source_identity.py')==prep['helper_sha256']
 proposed=load(HERE/'package_source_rustc.py','proposed_reader_observation')
 # Explicit module fixture binding: candidate is not installed/resealed in the
 # old package, and this observation grants no new-source acceptance.
 proposed.ROOT=entry
 refused=(m.Refused,proposed.m.Refused,OSError)
 artifacts=[m.json_bytes(line) for line in (records/'cargo.jsonl').read_bytes().splitlines() if m.json_bytes(line).get('reason')=='compiler-artifact']
 witness=json.loads((BASE/'post-build-dc313-fb17-relative-v2/FAILED-RELATIVE-SOURCE-WITNESS.json').read_bytes());observed=v.traces(records);cases=[];units=[]
 for row in witness['bad_rows']:
  matches=[trace for trace in observed if trace['trace']==row['trace'] and trace['argv']==row['argv'] and trace['exited_zero']];assert len(matches)==1
  argv=row['argv'][1:];args_=argv[1:];sources=[x for x in args_ if x.endswith('.rs') and not x.startswith('-')];assert len(sources)==1 and not Path(sources[0]).is_absolute()
  events=[e for e in artifacts if e['target']['name'].replace('-','_')==row['crate'] and e['target']['crate_types']==old.crate_types(args_)]
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
  try:proposed.owned_primary_source([str(entry/cap['roots'][other]/sources[0])],source_row,manifest)
  except refused:pass
  else:raise AssertionError('outside sealed package source accepted')
  if row['crate']=='build_script_build':
   with cwd(v.WT):assert not old.original_custom_build_unit(args_,source_row,manifest) and proposed.original_custom_build_unit(args_,source_row,manifest)
   with cwd(manifest):assert old.original_custom_build_unit(args_,source_row,manifest)
  cases.append({'trace':row['trace'],'trace_sha256':matches[0]['trace_sha256'],'actual_argv':argv,'actual_owner':unit['owner'],'actual_source':unit['source'],'old_wrong_cwd_refusal':old_red,'old_manifest_context_passed':True,'candidate_wrong_consumer_cwd_passed_with_identical_record':True,'absolute_lexical_equivalence':True,'wrong_owner_refused':True,'ambiguous_owner_refused':True,'outside_owner_refused':True})
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
 report={'old_source':v.HEAD,'old_source_seal':cap['source_seal'],'candidate_source_sha256':prep['candidate_receiver_sha256'],'cases':cases,'foreign_manifest_dependency_consumer':str(foreign_manifest),'cross_package_original_argv_and_custody_passed':True,'nofollow_symlink_dotdot_refused':True,'source_or_artifact_mutation':False,'compiler_Cargo_native_ar_GPU_invoked':False,'new_source_capsule_admission_or_qualification':False,'candidate_against_old_capsule_is_regression_observation_only':True}
 (args.records/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
