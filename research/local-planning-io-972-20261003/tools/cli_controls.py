"""Run real normal/optimized planner CLIs with owned metadata-only fixtures."""
if not __debug__:
    raise RuntimeError('CLI proof assertions must be enabled')
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(root/'tools'))
import test_validation_plan as fixtures
results=[]

f=fixtures.ValidationPlanTests();f.setUp()
try:
 tools=f.repo/'tools';tools.mkdir()
 for name in ('validation_plan.py','support_record_inputs.py','validation_inputs.json','skip-census.py'):
  shutil.copyfile(root/'tools'/name,tools/name)
 f.put_support_data_reader_fixture();base=f.commit()
 oldscript=tools/'before_validation_plan.py';oldscript.write_bytes((Path(__file__).with_name('original_validation_plan.py')).read_bytes())
 # Real commands are planned only. Native execution is never invoked.
 for flags in ([],['-O']):
  script=str(tools/'validation_plan.py')
  for action in ('build','clippy','publish'):
   args=[sys.executable,*flags,script,'cargo',action,'--packages','memra-lanes','--dry-command']
   result=subprocess.run(args,capture_output=True,timeout=20)
   assert result.returncode==0,(action,result.stderr.decode())
   command=json.loads(result.stdout.decode().splitlines()[-1]);assert command[0]=='cargo'
   oldresult=subprocess.run([sys.executable,*flags,str(oldscript),'cargo',action,'--packages','memra-lanes','--dry-command'],capture_output=True,timeout=20)
   assert oldresult.returncode==0 and oldresult.stdout==result.stdout,(action,oldresult.stderr.decode())
   results.append({'baseline_identical':True,'case':'regular-dry-'+action,'flags':flags,'exit':0,'command':command,'native_executed':False})
  unknown=f.repo/'tools/check-support-states.py';known=unknown.read_bytes();unknown.write_text('# unknown reader\n')
  result=subprocess.run([sys.executable,*flags,script,'local','--repo',str(f.repo),'--base',base],capture_output=True,timeout=5)
  plan=json.loads(result.stdout);assert result.returncode==0 and plan['mode']=='full' and all(plan['jobs'].values()) and not plan['native']['qualification']
  oldresult=subprocess.run([sys.executable,*flags,str(oldscript),'local','--repo',str(f.repo),'--base',base],capture_output=True,timeout=5)
  oldplan=json.loads(oldresult.stdout);assert oldresult.returncode==0 and oldplan['mode']=='full' and oldplan['jobs']==plan['jobs']
  results.append({'case':'unknown-reader-full','flags':flags,'exit':0,'all_jobs':True,'baseline_obligations_identical':True,'native_executed':False})
  unknown.write_bytes(known)
  target=f.repo/'Cargo.toml';old=target.read_bytes();target.unlink();os.mkfifo(target)
  result=subprocess.run([sys.executable,*flags,script,'local','--repo',str(f.repo),'--base',base],capture_output=True,timeout=5)
  assert result.returncode==0,result.stderr.decode()
  plan=json.loads(result.stdout);assert plan['mode']=='full' and all(plan['jobs'].values()) and not plan['native']['qualification']
  results.append({'case':'cargo-root-fifo','flags':flags,'exit':result.returncode,'mode':plan['mode'],'all_jobs':all(plan['jobs'].values()),'native_qualification':False})
  target.unlink();target.write_bytes(old)
  target=f.repo/'crates/memra-server/Cargo.toml';old=target.read_bytes();target.unlink();os.mkfifo(target)
  result=subprocess.run([sys.executable,*flags,script,'cargo','build','--packages','memra-lanes','--dry-command'],capture_output=True,timeout=5)
  assert result.returncode!=0 and b'not a regular file' in result.stderr
  assert result.stdout==b''
  results.append({'case':'unrelated-manifest-fifo','flags':flags,'exit':result.returncode,'no_command_emitted':True,'native_executed':False})
  target.unlink();target.write_bytes(old)
finally:f.doCleanups()
(out/'actual-cli-controls.json').write_text(json.dumps({'cases':results,'owned_root_removed':not f.repo.exists(),'native_qualification':False},indent=2)+'\n')
print(json.dumps({'actual_cli_cases':len(results),'owned_root_removed':not f.repo.exists(),'native_executed':False}))
