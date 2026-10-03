"""Reproduce support-input refusal and coherent guard-removal controls, CPU only."""
import hashlib, json, pathlib, subprocess, sys, tempfile
if not __debug__:
 raise RuntimeError('support execution control assertions must be enabled')
root=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve(); out.mkdir(parents=True,exist_ok=True)
source=(root/'tools/support_record_inputs.py').read_text()
base=pathlib.Path(__file__).with_name('original_support_record_inputs.py').read_text()
if hashlib.sha256(base.encode()).hexdigest() != 'bb1cb80f4e52ea75104bfb9bd70f6047a8de31d77a060ade656362aa0806f473':
 raise RuntimeError('baseline helper changed')
cls='test_validation_support_execution_inputs.SupportExecutionInputs.'
cases=[
 ('baseline-read-before-type',base,[cls+'test_real_fifo_directory_and_socket_refuse_before_any_content_read',cls+'test_leaf_symlink_to_regular_fifo_or_missing_refuses_before_reads',cls+'test_ancestor_symlink_including_broken_parent_refuses_before_reads']),
 ('removed-parent-discovery',source.replace('any(parent.is_symlink() or (parent.exists() and not parent.is_dir())\n                        for parent in path.parents\n                        if parent.is_relative_to(self.root))', 'False'),[cls+'test_ancestor_symlink_including_broken_parent_refuses_before_reads']),
 ('removed-preflight',source.replace('        tree.validate_inputs(expected)','        pass'),[cls+'test_real_fifo_directory_and_socket_refuse_before_any_content_read']),
 ('removed-nonblocking-open',source.replace(' | os.O_NONBLOCK',''),[cls+'test_fifo_replacement_between_stat_and_open_refuses_without_blocking']),
 ('removed-leaf-nofollow',source.replace('os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK','os.O_RDONLY | os.O_NONBLOCK'),[cls+'test_symlink_replacement_between_stat_and_open_refuses']),
 ('removed-opened-type-check',source.replace('if not stat.S_ISREG(os.fstat(leaf).st_mode):','if False:'),[cls+'test_fifo_replacement_between_stat_and_open_refuses_without_blocking']),
 ('removed-parent-nofollow',source.replace('os.O_DIRECTORY | os.O_NOFOLLOW','os.O_DIRECTORY'),[cls+'test_ancestor_symlink_including_broken_parent_refuses_before_reads']),
 ('normalized-reader-bytes',source.replace("with os.fdopen(self._open_regular(name), 'rb') as source:\n            return source.read()","with os.fdopen(self._open_regular(name), 'rb') as source:\n            return source.read().replace(b'\\r\\n', b'\\n')"),[cls+'test_read_bytes_remains_exact_and_regular_metadata_still_resolves']),
 ('typed-type-error-fallback',source.replace("raise InputContractError('support data reader input contains", "raise UnmodelledReader('support data reader input contains"),[cls+'test_unknown_reader_does_not_turn_fifo_into_typed_fallback']),
]
results=[]
with tempfile.TemporaryDirectory(prefix='memra-support-type-controls-') as tmp:
 for name,text,tests in cases:
  assert name.startswith('baseline') or text != source, name
  helper=pathlib.Path(tmp)/(name+'.py');helper.write_text(text)
  runner="import importlib.util, sys, unittest; sys.path.insert(0,sys.argv[1]); s=importlib.util.spec_from_file_location('support_record_inputs',sys.argv[2]); m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(sys.argv[3:]));sys.exit(0 if r.wasSuccessful() else 1)"
  result=subprocess.run([sys.executable,'-c',runner,str(root/'tools'),str(helper),*tests],capture_output=True,timeout=15)
  (out/(name+'.log')).write_bytes(result.stdout+result.stderr)
  assert result.returncode==1, (name,result.returncode,result.stderr.decode())
  assert b'FAIL:' in result.stderr and b'ERROR:' not in result.stderr, (name,result.stderr.decode())
  results.append({'control':name,'exit':result.returncode,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'tests':tests,'log_sha256':hashlib.sha256(result.stdout+result.stderr).hexdigest()})
(out/'mutation-results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'controls_failed_as_expected':len(results),'errors':0,'scratch_removed':not pathlib.Path(tmp).exists()}))
