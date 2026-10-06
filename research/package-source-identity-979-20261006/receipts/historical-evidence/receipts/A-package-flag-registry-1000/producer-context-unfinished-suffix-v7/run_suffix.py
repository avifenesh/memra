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
 for name,pin in PINS['failed_suffix_files'].items():assert sha(Path(PINS['failed_suffix_job'])/name)==pin['sha256']
 assert sha(Path(PINS['prior_suffix_script']))==PINS['prior_suffix_script_sha256']=='0ca613b326f754ee6b909c90f487d7f88493ee05edbcc154d90467fbe4545465' and sha(Path(PINS['prior_suffix_inputs']))==PINS['prior_suffix_inputs_sha256']
 suffix_failed=json.loads((Path(PINS['failed_suffix_job'])/'result.json').read_bytes());assert suffix_failed['exit_code']==1
 suffix_log=(Path(PINS['failed_suffix_job'])/'output.log').read_text();assert 'line 66, in main' in suffix_log and 'StopIteration' in suffix_log
 for name,pin in PINS['last_failed_files'].items():assert sha(Path(PINS['last_failed_job'])/name)==pin['sha256']
 assert sha(Path(PINS['last_suffix']))==PINS['last_suffix_sha256']=='159dc2c4011d8fd85a18177d2c0768b1f6bf8ee5883d315c44afc8927d7d5764' and sha(Path(PINS['last_suffix_inputs']))==PINS['last_suffix_inputs_sha256']
 last_failed=json.loads((Path(PINS['last_failed_job'])/'result.json').read_bytes());assert last_failed['exit_code']==1
 last_log=(Path(PINS['last_failed_job'])/'output.log').read_text();assert 'line 74, in main' in last_log and 'old foreign consumer dependency unexpectedly accepted' in last_log
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
  # Metadata derivation only. v5 reached line66 for src/main.rs: the first
  # two cases completed, and third oldwrongcwd/manifest/newreader/absolute/
  # ambiguity controls completed. None of those functional cases reruns here.
  owner=cap['snapshot']['payload']['entry'];manifest=entry/cap['roots'][owner]
  exact=[event for event in events if event['package_id']==cap['snapshot']['bindings'][owner]['cargo_package_id'] and event['manifest_path']==str(manifest/'Cargo.toml') and event['target']['src_path']==str(manifest/sources[0])]
  assert len(exact)==1;event=exact[0]
  unit={'owner':owner,'manifest_context':str(manifest),'source':str(manifest/sources[0]),'argv':argv,'products':proposed.compiler_products(args_,cap['snapshot']['payload']['packages'][owner],manifest)};units.append(unit)
  old_red='producer source owner differs'
  outside_pin={'retained_in_failed_suffix':'4ce5cf49b2ac' if ordinal==2 else '7ef914986fe2','completed_outside_owner_case':ordinal}
  cases.append({'trace':row['trace'],'trace_sha256':matches[0]['trace_sha256'],'actual_argv':argv,'actual_owner':unit['owner'],'actual_source':unit['source'],'old_wrong_cwd_refusal':old_red,'old_manifest_context_passed':True,'candidate_wrong_consumer_cwd_passed_with_identical_record':True,'absolute_lexical_equivalence':True,'wrong_owner_refused':True,'ambiguous_owner_refused':True,'outside_owner_refused':True,'outside_fixture_pin':outside_pin,'retained_prefix_from_805080_before_line57':ordinal==0,'retained_prefix_from_7ef914_before_line66':True,'retained_prefix_from_4ce5cf_before_line74':True,'completed_prefix_controls_replayed':False})
 rootlib=next(unit for unit in units if old.crate_types(unit['argv'][1:])==['lib']);foreign=next(key for key,row in cap['snapshot']['payload']['packages'].items() if row['name']=='autocfg');foreign_manifest=entry/cap['roots'][foreign]
 extern=['--extern','memra_server='+str(rootlib['products'][0])]
 # The old dependency call accepted in4ce5cf. Do not invent a red or replay
 # that call: bind its actual returned status to the same sealed record fields.
 old_library=old.artifact_record(rootlib['products'][0],cap)
 old_deps=[{'alias':'memra_server','package':old_library['package'],'artifact_bundle':old_library['artifact_bundle'],'compiler_unit':old_library['compiler_unit'],'input_seal':old_library['input_seal']}]
 with cwd(foreign_manifest):
  deps,own=proposed.dependencies(extern,cap,foreign);assert deps==old_deps and own==[]
 old_resolved=foreign_manifest/'src/lib.rs';assert old_resolved.is_relative_to(entry) and old_resolved.relative_to(entry).as_posix() in cap['snapshot']['payload']['packages'][rootlib['owner']]['files']
 foreign_observation={'old_status':'accepted in actual4ce5cf before incorrect negative assertion','old_relative_source_argument':'src/lib.rs','old_consumer_resolved_source':str(old_resolved),'old_prefix_containment_passes':True,'candidate_owner_resolved_source':str(entry/'src/lib.rs'),'old_expected_dependency_record_from_same_sealed_artifact':old_deps,'candidate_actual_dependency_records':deps,'records_identical':True,'old_foreign_red_claimed':False}
 # A new independent arbitrary outside-consumer context for the actual relative
 # BIN proves the source fix. The three old RootWT cases are retained, not replayed.
 rootbin=next(unit for unit in units if proposed.crate_types(unit['argv'][1:])==['bin'] and proposed.val(unit['argv'][1:],'--crate-name')=='memra_server')
 assert not args.records.is_relative_to(entry)
 with cwd(args.records):
  try:old.completed_artifact_record(rootbin['products'][0],cap)
  except m.Refused as error:assert 'producer source owner differs' in str(error);bin_old_status='typed producer source owner refusal'
  else:raise AssertionError('old relative BIN accepted outside owner context')
  bin_new_record=proposed.completed_artifact_record(rootbin['products'][0],cap);assert bin_new_record['argv']==rootbin['argv'] and bin_new_record['package']==rootbin['owner']
 outside_bin_observation={'actual_BIN':str(rootbin['products'][0]),'consumer_cwd':str(args.records),'old_status':bin_old_status,'candidate_status':'full completed custody accepted','actual_argv':rootbin['argv'],'current_record_seal':bin_new_record['seal'],'candidate_on_old_capsule_only':True}
 fixture=args.records/'nofollow-fixture';fixture.mkdir();(fixture/'leaf.rs').write_bytes(b'fixture source\n');(fixture/'hop').symlink_to(entry/'src',target_is_directory=True)
 try:proposed.owned_primary_source(['hop/../leaf.rs'],{'files':{'leaf.rs':proposed.m.regular(fixture,'leaf.rs')}},fixture)
 except refused:pass
 else:raise AssertionError('symlink canceled by dotdot bypassed source guard')
 finally:(fixture/'hop').unlink();(fixture/'leaf.rs').unlink();fixture.rmdir()
 assert m.package_capsule(entry)==cap
 report={'failed_suffix_4ce5cf_preserved':PINS['last_failed_job'],'failed_suffix_4ce5cf_evidence':PINS['last_failed_files'],'foreign_dependency_observation':foreign_observation,'outside_BIN_observation':outside_bin_observation,'failed_suffix_7ef914_preserved':PINS['failed_suffix_job'],'failed_suffix_evidence':PINS['failed_suffix_files'],'failed_job_805080_preserved':PINS['failed_job'],'retained_prefix_evidence':PINS['failed_job_files'],'first_completed_prefix_not_replayed':True,'old_source':v.HEAD,'old_source_seal':cap['source_seal'],'candidate_source_sha256':prep['candidate_receiver_sha256'],'cases':cases,'foreign_manifest_dependency_consumer':str(foreign_manifest),'cross_package_original_argv_and_custody_passed':True,'nofollow_symlink_dotdot_refused':True,'source_or_artifact_mutation':False,'compiler_Cargo_native_ar_GPU_invoked':False,'new_source_capsule_admission_or_qualification':False,'candidate_against_old_capsule_is_regression_observation_only':True}
 (args.records/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
