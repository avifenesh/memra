from pathlib import Path
import subprocess,json,hashlib,re
R=Path(__file__).resolve().parent;O=R/'boundary-controls';original=(O/'after.rs').read_text()
mutations={
 'overflow-wrap':('n.checked_add(1)','Some(n.wrapping_add(1))','site0_overflow_and_contention'),
 'underflow-success':('v.checked_sub(1)','Some(v.saturating_sub(1))','site6_underflow_and_contention'),
 'queue-one-extra':('events < MAX_EVENT_QUEUE_EVENTS','events <= MAX_EVENT_QUEUE_EVENTS','site8_queue_bound_and_contention'),
 'bytes-one-extra':('*next <= MAX_EVENT_QUEUE_BYTES','*next <= MAX_EVENT_QUEUE_BYTES + 1','site9_byte_bound_and_arithmetic_overflow'),
}
rows=[]
for label,(old,new,test) in mutations.items():
 assert old in original;source=original.replace(old,new,1);p=O/(label+'.rs');p.write_text(source);binary=O/(label+'-tests')
 commands=[['rustc','--test','--edition=2024',str(p),'-o',str(binary)],[str(binary),test,'--exact','--test-threads=1']]
 build=subprocess.run(commands[0],capture_output=True,text=True,timeout=60);(O/(label+'-build.log')).write_text(build.stdout+build.stderr);assert build.returncode==0,(label,build.stderr)
 run=subprocess.run(commands[1],capture_output=True,text=True,timeout=30);raw=run.stdout+run.stderr;(O/(label+'-tests.log')).write_text(raw)
 assert run.returncode!=0 and '0 passed; 1 failed; 0 ignored' in raw and 'assertion' in raw,(label,raw)
 (O/(label+'-commands.json')).write_text(json.dumps(commands,indent=2)+'\n');rows.append({'label':label,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'test':test,'compile_exit':build.returncode,'actual_test_exit':run.returncode,'raw_log':label+'-tests.log'})
# Restore witness executes the original saved binary without overwriting any source.
cmd=[str(O/'after-tests'),'--test-threads=1'];run=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(O/'restored-tests.log').write_text(run.stdout+run.stderr);assert run.returncode==0 and '18 passed; 0 failed; 0 ignored' in run.stdout
assert (O/'after.rs').read_text()==original
(O/'FAULT-RESULTS.json').write_text(json.dumps({'rows':rows,'restored_command':cmd,'restored_exit':run.returncode,'restored_source_sha256':hashlib.sha256(original.encode()).hexdigest(),'limits':'Source-copied expression oracle faults, with real rustc compile and failing executable tests. No original worktree mutation and no production execution claim.'},indent=2)+'\n')
print('Four compiling boundary faults rejected; restored18 controls passed.')
