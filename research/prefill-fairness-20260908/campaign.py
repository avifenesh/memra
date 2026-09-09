"""Restartable serial GPU campaign. Existing boot directories are attached, never relaunched."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/root/prefill-fairness-20260908')
p.add_argument('--root', default='/root/prefill-fairness-gpu-20260909')
p.add_argument('--binary', default='/root/target-prefill/release/memra-server')
p.add_argument('--sha', default='19db5df71812eabfa914ba99f58ff676e5d6a860174c0a75084dfad19a89d8e4')
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--skip-chunk-probes', action='store_true')
a = p.parse_args()
repo, root = pathlib.Path(a.repo), pathlib.Path(a.root)
script = repo / 'research/prefill-fairness-20260908'
base = pathlib.Path('/tmp/qwen-ornith-5090-capacity-20260908')
binary, sha = a.binary, a.sha
profiles = {'ornith': str(base/'ornith-361-composed-private-sampled'), 'qwen': str(base/'qwen-32k-host-cap1')}
longs = {'ornith': str(root/'ornith-r1-off-v2/0-request.json'), 'qwen': str(base/'qwen-128k-chunk1024-cap1/turn1-request.json')}

def alive(run):
    f = run / 'identity.json'
    if not f.exists():
        return False
    ident = json.loads(f.read_text())
    pid = ident.get('server_pid', ident.get('pid'))
    expected = ident.get('server_start_ticks', ident.get('start_ticks'))
    stat = pathlib.Path(f'/proc/{pid}/stat')
    try:
        fields = stat.read_text().split()
        return fields[2] != 'Z' and str(fields[21]) == str(expected)
    except FileNotFoundError:
        return False

def execute(label, command, mixed):
    run = root/label
    print('CELL', label, flush=True)
    if run.exists():
        while alive(run):
            time.sleep(1)
        # The controller writes its cleanup after waiting for its own child.
        for _ in range(20):
            if (run/'cleanup.json').exists():
                break
            time.sleep(.5)
        assert (run/'cleanup.json').exists(), f'incomplete existing boot: {label}'
        print('ATTACHED', label, flush=True)
    else:
        with (root/(label+'.controller.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f'cell failed: {label}; see controller log'
    if mixed:
        summary = json.loads((run/'stage0-summary.json').read_text())
        assert not summary['aborted'] and not summary['faults'], summary
        assert summary['counts'] == {'completed': 21}, summary
        assert summary['server_delta']['step_oom_parks'] == 0, summary
    else:
        summary = json.loads((run/'summary.json').read_text())
    (root/'campaign-progress.json').write_text(json.dumps({'last_completed':label,'summary':summary},indent=2))
    print('PASS', label, flush=True)

for model in ('ornith', 'qwen'):
    for repetition in range(1,a.repetitions+1):
        for arm in ('off','on'):
            label=f'{model}-r{repetition}-{arm}'
            if label=='ornith-r1-off':label+='-v2'
            command=['python3',str(script/'mixed_load.py'),'--model',model,'--label',label,'--arm',arm,
                     '--chunk','1024','--sessions','4','--binary',binary,'--binary-sha256',sha,
                     '--profile-source',profiles[model],'--out-root',str(root)]
            if model=='qwen':command+=['--long-request',longs[model]]
            execute(label,command,True)

for model in ('ornith','qwen'):
    for arm in (() if a.skip_chunk_probes else ('off','on')):
        label=f'{model}-4096-{arm}'
        execute(label,['python3',str(script/'gate_http.py'),'--source',profiles[model],
                       '--binary',binary,'--sha',sha,'--out',str(root/label),
                       '--arm',arm,'--mode','chunk','--chunk','4096','--long-request',longs[model]],False)
    for mode,arm in [('chain','off'),('chain','on'),('solo','off'),('pair','on')]:
        label=f'{model}-{mode}-{arm}'
        execute(label,['python3',str(script/'gate_http.py'),'--source',profiles[model],
                       '--binary',binary,'--sha',sha,'--out',str(root/label),
                       '--arm',arm,'--mode',mode,'--chunk','1024','--long-request',longs[model],'--oracle'],False)
        if mode=='chain' and arm=='on':
            for turn in range(1,5):
                off=json.loads((root/f'{model}-chain-off/turn{turn}-result.json').read_text())
                on=json.loads((root/label/f'turn{turn}-result.json').read_text())
                assert off['output_sha256']==on['output_sha256'], (model,turn,'chain bytes')
        if mode=='pair':
            for request in ('long','small'):
                solo=json.loads((root/f'{model}-solo-off/{request}-result.json').read_text())
                pair=json.loads((root/label/f'{request}-result.json').read_text())
                assert solo['output_sha256']==pair['output_sha256'], (model,request,'pair bytes')
# Isolate the 1024-row wall measurements too. Mixed cells include peer work and
# cannot substitute for a one-request early/middle/late comparison with 4096.
for model in ('ornith', 'qwen'):
    for arm in (() if a.skip_chunk_probes else ('off', 'on')):
        label = f'{model}-1024-{arm}'
        execute(label, ['python3', str(script/'gate_http.py'), '--source', profiles[model],
                        '--binary', binary, '--sha', sha, '--out', str(root/label),
                        '--arm', arm, '--mode', 'chunk', '--chunk', '1024',
                        '--long-request', longs[model]], False)
print('CAMPAIGN_PASS',flush=True)
