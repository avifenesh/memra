"""Actual before/after local consumer reads, with owned FIFO observers."""
if not __debug__:
    raise RuntimeError('local consumer proof assertions must be enabled')
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(root/'tools'))
import test_validation_plan as fixtures
import validation_plan as current
oldsource=(Path(__file__).with_name('original_validation_plan.py')).read_bytes()
results=[]
class ReadObserved(Exception):pass
with tempfile.TemporaryDirectory(prefix='memra-local-before-source-') as owned:
 tools=Path(owned);oldpath=tools/'validation_plan.py';oldpath.write_bytes(oldsource)
 for name in ('support_record_inputs.py','validation_inputs.json','skip-census.py'):
  source=Path(__file__).with_name('original_'+name) if name != 'skip-census.py' else root/'tools'/name
  shutil.copyfile(source,tools/name)
 spec=importlib.util.spec_from_file_location('old_local_reader',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 for case in ('cargo-root','unrelated-cargo','tracked-reader','own-registry'):
  f=fixtures.ValidationPlanTests();f.setUp()
  try:
   f.put_support_data_reader_fixture();f.commit()
   if case=='cargo-root':relative='Cargo.toml';target=f.repo/relative
   elif case=='unrelated-cargo':relative='crates/memra-server/Cargo.toml';target=f.repo/relative
   elif case=='tracked-reader':relative='tools/check-support-states.py';target=f.repo/relative
   else:relative='validation_inputs.json';target=tools/relative
   target.unlink();os.mkfifo(target);realtext=Path.read_text;realbytes=Path.read_bytes;reads=[]
   def text(path,*a,**kw):
    if path==target:reads.append('text');raise ReadObserved()
    return realtext(path,*a,**kw)
   def raw(path,*a,**kw):
    if path==target:reads.append('bytes');raise ReadObserved()
    return realbytes(path,*a,**kw)
   def execute(vp):
    if case=='cargo-root':return vp.make_plan(['Cargo.toml'],vp.Tree(f.repo,f.base),vp.LocalTree(f.repo))
    if case=='unrelated-cargo':return vp.cargo_packages('memra-lanes',f.repo)
    if case=='tracked-reader':return vp.support_record_data_inputs(vp.LocalTree(f.repo))
    return vp.included_inputs(vp.LocalTree(f.repo),vp.workspace(vp.LocalTree(f.repo))[1])
   with mock.patch.object(Path,'read_text',text),mock.patch.object(Path,'read_bytes',raw):
    try:execute(old)
    except ReadObserved:pass
    else:raise AssertionError('old reader must engage FIFO observer: '+case)
   assert len(reads)==1,case
   if case=='own-registry':
    newpath=tools/'new_validation_plan.py';newpath.write_bytes((root/'tools/validation_plan.py').read_bytes());s=importlib.util.spec_from_file_location('new_registry_reader',newpath);vp=importlib.util.module_from_spec(s);s.loader.exec_module(vp)
   else:vp=current
   try:
    value=execute(vp)
    assert case=='cargo-root' and value['mode']=='full' and all(value['jobs'].values()) and not value['native']['qualification'],case
    outcome='full planning expansion'
   except vp.Refused as error:
    assert case!='cargo-root' and 'regular' in str(error),case
    outcome='refused before content'
   results.append({'case':case,'old_observed_read':reads[0],'actual_fifo':True,'no_fifo_io':True,'current_outcome':outcome})
   if case=='own-registry':target.unlink();target.write_bytes((root/'tools/validation_inputs.json').read_bytes())
  finally:f.doCleanups()
(out/'executed-consumers.json').write_text(json.dumps({'old_source_sha256':hashlib.sha256(oldsource).hexdigest(),'new_source_sha256':hashlib.sha256((root/'tools/validation_plan.py').read_bytes()).hexdigest(),'controls':results,'owned_roots_removed':not tools.exists(),'native_qualification':False},indent=2)+'\n')
print(json.dumps({'actual_consumers':len(results),'old_reader_reds':len(results),'current_safe_outcomes':len(results),'owned_roots_removed':not tools.exists()}))

# The original consumers follow regular aliases too. Observe the exact read
# without dereferencing special-file content. Trusted roots have no descendant
# ancestor component, so only Cargo and the tracked reader have this control.
alias_results=[]
with tempfile.TemporaryDirectory(prefix='memra-local-alias-source-') as owned:
 tools=Path(owned)
 for name in ('validation_plan.py','support_record_inputs.py','validation_inputs.json'):
  shutil.copyfile(Path(__file__).with_name('original_'+name),tools/name)
 shutil.copyfile(root/'tools/skip-census.py',tools/'skip-census.py')
 s=importlib.util.spec_from_file_location('old_alias_reader',tools/'validation_plan.py')
 old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
 newpath=tools/'new_validation_plan.py';newpath.write_bytes((root/'tools/validation_plan.py').read_bytes())
 s=importlib.util.spec_from_file_location('new_alias_reader',newpath)
 new=importlib.util.module_from_spec(s);s.loader.exec_module(new)
 for case in ('cargo-root','unrelated-cargo','tracked-reader','own-registry'):
  for kind in ('leaf', 'ancestor') if case in ('unrelated-cargo','tracked-reader') else ('leaf',):
   f=fixtures.ValidationPlanTests();f.setUp()
   try:
    f.put_support_data_reader_fixture();f.commit()
    target={'cargo-root':f.repo/'Cargo.toml','unrelated-cargo':f.repo/'crates/memra-server/Cargo.toml',
            'tracked-reader':f.repo/'tools/check-support-states.py','own-registry':tools/'validation_inputs.json'}[case]
    original=target.read_bytes()
    alias=target if kind=='leaf' else target.parent
    saved=alias.with_name(alias.name+'-saved');alias.rename(saved)
    alias.symlink_to(saved,target_is_directory=kind=='ancestor')
    realtext=Path.read_text;realbytes=Path.read_bytes;reads=[]
    def text(path,*a,**kw):
     if path.resolve()==target.resolve():reads.append('text');raise ReadObserved()
     return realtext(path,*a,**kw)
    def raw(path,*a,**kw):
     if path.resolve()==target.resolve():reads.append('bytes');raise ReadObserved()
     return realbytes(path,*a,**kw)
    def execute(vp):
     if case=='cargo-root':return vp.make_plan(['Cargo.toml'],vp.Tree(f.repo,f.base),vp.LocalTree(f.repo))
     if case=='unrelated-cargo':return vp.cargo_packages('memra-lanes',f.repo)
     if case=='tracked-reader':return vp.support_record_data_inputs(vp.LocalTree(f.repo))
     return vp.included_inputs(vp.LocalTree(f.repo),vp.workspace(vp.LocalTree(f.repo))[1])
    with mock.patch.object(Path,'read_text',text),mock.patch.object(Path,'read_bytes',raw):
     try:execute(old)
     except ReadObserved:pass
     else:raise AssertionError('original alias read not observed: '+case)
    assert len(reads)==1,(case,kind)
    vp=new if case=='own-registry' else current
    try:
     value=execute(vp)
     assert case=='cargo-root' and value['mode']=='full' and all(value['jobs'].values())
     outcome='full planning expansion'
    except vp.Refused:outcome='refused before content'
    alias_results.append({'case':case,'alias':kind,'original_observed_read':reads[0],'current_outcome':outcome})
    alias.unlink();saved.rename(alias)
   finally:f.doCleanups()
(out/'executed-alias-consumers.json').write_text(json.dumps({'controls':alias_results,'owned_roots_removed':not tools.exists()},indent=2)+'\n')
assert len(alias_results)==6
print(json.dumps({'actual_original_alias_reads':len(alias_results),'current_safe_outcomes':len(alias_results)}))
