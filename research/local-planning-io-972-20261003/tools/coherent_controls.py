"""Run independently removed local read guards against real owned IO controls."""
if not __debug__:
    raise RuntimeError('local IO proof assertions must be enabled')
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
source = (root / 'tools/validation_plan.py').read_text()
cls = 'test_validation_local_input_io.LocalInputIO.'
cases = [
 ('remove-prestat-type', source.replace('if not stat.S_ISREG(info.st_mode):', 'if False:'), [cls+'test_fifo_and_directory_readers_refuse_before_content']),
 ('remove-leaf-nofollow', source.replace('os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK', 'os.O_RDONLY | os.O_NONBLOCK'), [cls+'test_stat_to_symlink_replacement_refuses_actual_target_content']),
 ('remove-parent-nofollow', source.replace('os.O_DIRECTORY | os.O_NOFOLLOW', 'os.O_DIRECTORY'), [cls+'test_leaf_and_parent_links_are_refused_before_reading_target']),
 ('remove-nonblock', source.replace(' | os.O_NONBLOCK', ''), [cls+'test_stat_to_fifo_replacement_is_nonblocking_and_refused']),
 ('remove-opened-type', source.replace('if not stat.S_ISREG(os.fstat(leaf).st_mode):', 'if False:'), [cls+'test_stat_to_fifo_replacement_is_nonblocking_and_refused']),
 ('normalize-raw-bytes', source.replace('return source.read()', "return source.read().replace(b'\\r\\n', b'\\n') if binary else source.read()"), [cls+'test_regular_text_and_raw_bytes_preserve_original_semantics']),
 ('leak-parent-fd', source.replace('            os.close(parent)\n\n\ndef git', '            pass\n\n\ndef git'), [cls+'test_descriptor_cleanup_on_success_and_failure']),
]
results=[]
with tempfile.TemporaryDirectory(prefix='memra-local-io-controls-') as owned:
 for name, changed, tests in cases:
  assert changed != source, name
  code = Path(owned) / (name+'.py'); code.write_text(changed)
  runner="import importlib.util,sys,unittest;sys.path.insert(0,sys.argv[1]);s=importlib.util.spec_from_file_location('validation_plan',sys.argv[2]);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(sys.argv[3:]));sys.exit(0 if r.wasSuccessful() else 1)"
  result=subprocess.run([sys.executable,'-c',runner,str(root/'tools'),str(code),*tests],capture_output=True,timeout=15)
  raw=result.stdout+result.stderr;(out/(name+'.log')).write_bytes(raw)
  assert result.returncode==1 and b'FAIL:' in raw and b'ERROR:' not in raw,(name,result.returncode,raw.decode())
  results.append({'control':name,'exit':result.returncode,'changed_source_sha256':hashlib.sha256(changed.encode()).hexdigest(),'log_sha256':hashlib.sha256(raw).hexdigest(),'tests':tests})
(out/'coherent-controls.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'coherent_red_groups':len(results),'errors':0,'owned_root_removed':not Path(owned).exists()}))
