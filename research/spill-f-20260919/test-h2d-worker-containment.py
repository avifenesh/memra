#!/usr/bin/env python3
"""CPU-only timeout containment positive + detached-child red regression."""
from pathlib import Path
import runpy
import sys
import tempfile
import time

worker = Path(__file__).with_name('run-h2d-copies-plumbing.py').resolve()
B = runpy.run_path(str(worker), run_name='test_import')['B']
with tempfile.TemporaryDirectory(prefix='memra-f-containment-') as temp:
    root = Path(temp)
    for detached in [False, True]:
        marker = root/f'{detached}.marker'
        child = ('from pathlib import Path; import time; '
                 f'time.sleep(.7); Path({str(marker)!r}).write_text("survived")')
        script = f'import runpy,sys; from pathlib import Path; m=runpy.run_path({str(worker)!r},run_name="test_import"); '
        if detached:
            script += ('original=m["subprocess"].Popen; '
                       'm["subprocess"].Popen=lambda *a,**kw: original(*a,**dict(kw,start_new_session=True)); ')
        script += f'm["tee_in_collector_group"]([sys.executable,"-c",{child!r}],Path({str(root / (str(detached)+".raw"))!r}))'
        code, expired = B.tee_run([sys.executable, '-c', script], root/f'{detached}.outer',
                                  timeout=.3, echo=False)
        assert expired and code != 0, 'outer timeout did not execute'
        time.sleep(.8)  # Beyond the child's marker deadline; no daemon survives this test.
        assert marker.exists() == detached, 'process-group containment mismatch'
        print(('RED control detected escaped detached child' if detached else
               'PASS collector timeout kills nested probe group'))
print('PASS CPU-only containment regression; no GPU and no rig lock used')
