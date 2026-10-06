#!/usr/bin/python3 -I -B
"""Root-only guarded standard native driver; no compiler/source modification."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).absolute().parent
PINS=json.loads((R/'INPUT-PINS.json').read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 # These are small independent parent recipe pins. Native/capsule/tool
 # preflight remains in the child and is entirely inside the CPU/wall meter.
 for name,row in PINS['files'].items():assert sha(Path(name))==row['sha256'],name
 approved=json.loads(Path(PINS['Root_guarded_admission']).read_bytes())
 assert set(approved)=={'source','source_seal','native_admission_sha256','guard_sha256','driver_sha256','guard_control_result_sha256','wall_seconds','CPU_seconds','Root_reviewed_current_guard_and_native_recipe'}
 assert approved['source']==PINS['source'] and approved['source_seal']==PINS['source_seal'] and approved['Root_reviewed_current_guard_and_native_recipe'] is True
 assert approved['wall_seconds']==1800 and approved['CPU_seconds']==3600
 native=Path(PINS['preparation'])/'APPROVED-NATIVE.json';assert sha(native)==approved['native_admission_sha256']
 n=json.loads(native.read_bytes());assert n['source']==PINS['source'] and n['admission']=='one1800wall3600CPU-current-build-only'
 assert Path(n['records']).name=='current-native-records-1800' and not Path(n['records']).exists()
 control=Path(PINS['guard_control_result']);assert sha(control)==approved['guard_control_result_sha256']
 report=json.loads(control.read_bytes());assert report['guard_sha256']==approved['guard_sha256']==PINS['files'][PINS['guard']]['sha256']
 assert report['wrong_job_refused_before_spawn_or_signal'] is True and report['samecgroup_sibling_untouched'] is True
 cases=report['actual_owned_CPU_wall_normal_detached_controls'];assert len(cases)==4 and all(row['samecgroup_sibling_untouched'] for row in cases)
 assert {row['case'] for row in cases}=={'normal','cpu','wall','leader-exit-detached'}
 assert sha(PINS['driver'])==approved['driver_sha256']
 guard=Path(PINS['guard']);spec=importlib.util.spec_from_file_location('actual_cpu_guard',guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 # Baseline precedes invocation of the complete standard driver preflight.
 # The guard safely cleans only its subreaper-owned descendants usingpidfds.
 code=g.run(['/usr/bin/python3','-I','-B',PINS['driver']],Path(PINS['guard_records']),CPU_seconds=3600,wall_seconds=1775,teardown_seconds=20)
 result=json.loads((Path(PINS['guard_records'])/'CPU-GUARD-RESULT.json').read_bytes())
 (R/'LAST-GUARDED-DRIVER-OUTCOME.json').write_text(json.dumps({'source':PINS['source'],'source_seal':PINS['source_seal'],'guard_exit':code,'guard_record':str(Path(PINS['guard_records'])/'CPU-GUARD-RESULT.json'),'guard_record_sha256':sha(Path(PINS['guard_records'])/'CPU-GUARD-RESULT.json'),'actual_CPU_including_teardown':result['CPU_from_before_preflight_through_teardown_seconds'],'current_native_success':result['native_success'],'original_failed_traces_preserved':True,'qualified':False},indent=2)+'\n')
 sys.exit(code)
if __name__=='__main__':main()
