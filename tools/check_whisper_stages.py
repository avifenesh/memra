#!/usr/bin/env python3
"""Compare raw native stage tensors against a hash-bound offline Whisper oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--numeric', choices=('f32','f16'), required=True)
    parser.add_argument('--build-receipt', type=Path, help='reuse the source binding for an unchanged previously measured binary')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--fp32-control', type=Path, help='required for F16: matching HF FP32 capture that defines the measured reference floor')
    args = parser.parse_args()
    manifest = json.loads((args.oracle/'manifest.json').read_text())
    threshold = 1e-3
    floor = None
    if args.numeric == 'f16':
        if manifest['dtype'] != 'f16' or args.fp32_control is None:
            raise SystemExit('F16 gate requires HF F16 oracle and matching --fp32-control')
        control = json.loads((args.fp32_control/'manifest.json').read_text())
        if control['dtype'] != 'f32' or any(control[key] != manifest[key] for key in ('input_pcm_sha256','source_repository','source_revision','config_sha256')):
            raise SystemExit('FP16 and FP32 oracle pins differ')
        a_entry = manifest['files']['encoder']
        b_entry = control['files']['encoder']
        a_path = args.oracle/a_entry['path']
        b_path = args.fp32_control/b_entry['path']
        if a_entry['shape'] != b_entry['shape'] or sha(a_path) != a_entry['sha256'] or sha(b_path) != b_entry['sha256']:
            raise SystemExit('reference floor shape/hash mismatch')
        a = np.fromfile(a_path,dtype='<f4')
        b = np.fromfile(b_path,dtype='<f4')
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise SystemExit('invalid reference floor tensors')
        threshold = float(np.abs(a-b).max())
        floor = {'max_abs':threshold,'hf_fp16_manifest_sha256':sha(args.oracle/'manifest.json'),'hf_fp32_manifest_sha256':sha(args.fp32_control/'manifest.json'),
                 'rule':'owner 2026-09-09: native F16 versus HF F16, bounded by same-fixture HF F16 versus HF F32 delta'}
    elif manifest['dtype'] != 'f32':
        raise SystemExit('F32 gate requires HF F32 oracle')
    # The capture must name the mel it consumed. The same binary scores 0.25 against HF F16 on
    # the reference log-mel and 0.6796875 on the native mel; a receipt that omits the input
    # cannot tell an encoder gate from a stacked frontend-and-encoder one.
    mel_entry = manifest['files']['log-mel']
    native_input = args.native/'input-mel.f32'
    if not native_input.exists():
        raise SystemExit('capture predates input pinning: rerun the stage runner to bank input-mel.f32')
    native_input_sha = sha(native_input)
    if native_input_sha != mel_entry['sha256']:
        raise SystemExit('native encoder consumed '+native_input_sha+', not the reference log-mel '+mel_entry['sha256'])
    names = ['conv1' ,'conv2',*[f'layer-{i:02d}' for i in range(32)],'encoder']
    rows = []
    for name in names:
        entry = manifest['files'][name]
        path = args.oracle/entry['path']
        if sha(path) != entry['sha256']:
            raise SystemExit('oracle hash mismatch: '+name)
        native = args.native/(name+'.f32')
        a = np.fromfile(native,dtype='<f4')
        b = np.fromfile(path,dtype='<f4')
        if a.shape != b.shape or a.size != int(np.prod(entry['shape'])) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise SystemExit('invalid shape or non-finite stage: '+name)
        delta = np.abs(a-b)
        i = int(delta.argmax())
        row = {'stage':name,'shape':entry['shape'],'max_abs':float(delta[i]),'mean_abs':float(delta.mean()),
               'worst_flat_index':i,'native_worst_value':float(a[i]),'oracle_worst_value':float(b[i]),
               'native_sha256':sha(native),'oracle_sha256':entry['sha256']}
        rows.append(row)
        print(name, 'max_abs='+str(row['max_abs']), 'mean_abs='+str(row['mean_abs']))
    source_files = ['crates/memra-reference/src/speech/encoder.rs','crates/memra-reference/src/speech/matrix.rs','crates/memra-reference/src/bin/whisper-stage.rs']
    source_hashes = {n:sha(Path(n)) for n in source_files}
    if args.build_receipt:
        build = json.loads(args.build_receipt.read_text())
        if build['binary_sha256'] != sha(args.binary):
            raise SystemExit('binary does not match the build receipt')
        source_hashes = build['native_source_sha256']
    receipt = {'schema':'memra-whisper-stage-gate-v1','status':'passed' if rows[-1]['max_abs'] <= threshold else 'failed',
        'gate':'final encoder output','reference_floor':floor,'threshold_abs':threshold,'native_numeric':args.numeric,
        'oracle_dtype':manifest['dtype'],'oracle_manifest_sha256':sha(args.oracle/'manifest.json'),
        'input_pcm_sha256':manifest['input_pcm_sha256'],'binary_sha256':sha(args.binary),
        'native_input_sha256':native_input_sha,'native_input_is_reference_log_mel':True,
        'gate_scope':'encoder only: native encoder consumed the reference log-mel, so the frontend is not stacked into this number',
        'native_source_sha256':source_hashes,
        'first_intermediate_above_threshold':next((x['stage'] for x in rows if x['max_abs']>threshold),None),
        'stages':rows,'scope':'2-second synthetic CPU fixture, normal 30-second padded window; no WER, GPU or serving qualification'}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print('final gate',receipt['status'])
    raise SystemExit(0 if receipt['status']=='passed' else 1)


if __name__ == '__main__':
    main()
