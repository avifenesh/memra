#!/usr/bin/env python3
"""Check cached native decoder logits against a pinned same-precision HF capture."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('oracle','native','binary','receipt'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--numeric',choices=('f32','f16'),required=True)
    p.add_argument('--fp32-control',type=Path)
    args=p.parse_args();manifest=json.loads((args.oracle/'manifest.json').read_text())
    if manifest['dtype']!=args.numeric:raise SystemExit('oracle numeric class mismatch')
    if (args.native/'numeric.txt').read_text().strip()!=args.numeric.upper():raise SystemExit('native numeric class mismatch')
    tokens=list(map(int,(args.native/'input-tokens.txt').read_text().split()))
    if tokens!=manifest['input_tokens']:raise SystemExit('input-token sequence mismatch')
    entry=manifest['files']['encoder-input']
    if sha(args.oracle/entry['path'])!=entry['sha256'] or sha(args.native/'encoder-input.f32')!=entry['sha256']:
        raise SystemExit('encoder-input identity mismatch')
    control=None
    if args.numeric=='f16':
        if args.fp32_control is None:raise SystemExit('FP16 requires the matching HF FP32 control')
        control=json.loads((args.fp32_control/'manifest.json').read_text())
        if control['dtype']!='f32':raise SystemExit('control is not an FP32 capture')
        # encoder_manifest_sha256 must match: a floor measured across two different encoder
        # oracles carries encoder FP16 error, not the decoder rounding this gate is about.
        for key in ('input_tokens','input_pcm_sha256','source_revision','encoder_manifest_sha256'):
            if control[key]!=manifest[key]:raise SystemExit('reference precision controls differ in '+key)
        if control['raw_argmax']!=manifest['raw_argmax']:raise SystemExit('reference precisions disagree on argmax')
    truth=control or manifest
    native_steps=[line.split('\t') for line in (args.native/'steps.tsv').read_text().splitlines()[1:]]
    if len(native_steps)!=len(tokens):raise SystemExit('native step census mismatch')
    rows=[];argmax_ok=True;worst_native=0.0;worst_floor=0.0;worst_truth=0.0
    for step,token in enumerate(tokens):
        if list(map(int,native_steps[step][:2]))!=[step,token]:raise SystemExit('native step order mismatch')
        stage=f'step-{step:02d}-logits';entry=manifest['files'][stage];reference=args.oracle/entry['path']
        if sha(reference)!=entry['sha256']:raise SystemExit('reference logit hash mismatch')
        native=args.native/(stage+'.f32');a=np.fromfile(native,dtype='<f4');b=np.fromfile(reference,dtype='<f4')
        if a.shape!=b.shape or a.size!=int(np.prod(entry['shape'])) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise SystemExit('invalid logit extent or non-finite values')
        delta=float(np.abs(a-b).max());floor=None;truth_delta=None
        if control is not None:
            c_entry=control['files'][stage];c_path=args.fp32_control/c_entry['path']
            if c_entry['shape']!=entry['shape'] or sha(c_path)!=c_entry['sha256']:raise SystemExit('control shape/hash mismatch')
            c=np.fromfile(c_path,dtype='<f4')
            if not np.isfinite(c).all():raise SystemExit('non-finite reference control')
            floor=float(np.abs(b-c).max());truth_delta=float(np.abs(a-c).max())
            worst_floor=max(worst_floor,floor);worst_truth=max(worst_truth,truth_delta)
        worst_native=max(worst_native,delta)
        argmax=int(a.argmax());same_argmax=argmax==truth['raw_argmax'][step];argmax_ok&=same_argmax
        if argmax!=int(native_steps[step][2]):raise SystemExit('native argmax receipt mismatch')
        row={'step':step,'input_token':token,'max_abs':delta,'mean_abs':float(np.abs(a-b).mean()),
             'reference_precision_floor':floor,'native_versus_fp32_truth':truth_delta,'argmax_match':same_argmax,
             'native_argmax':argmax,'oracle_argmax':truth['raw_argmax'][step],
             'native_sha256':sha(native),'oracle_sha256':entry['sha256']}
        if control is None:row['status']='passed' if delta<=1e-3 and same_argmax else 'failed'
        rows.append(row);print(step,delta,floor,same_argmax)
    if control is None:
        gate={'rule':'native F32 versus HF F32, fixed 1e-3 absolute logit bound','threshold_abs':1e-3,'max_abs':worst_native}
        passed=all(r['status']=='passed' for r in rows)
    else:
        # Two distinct FP16 rounding programs cannot be required to agree more closely than
        # their shared distance from FP32. Both bounds come from the reference captures only;
        # neither is derived from native error. Same shape as the owner's encoder F16 rule.
        gate={'rule':'owner 2026-09-09: native F16 bounded at fixture level by the same-fixture HF F16 versus HF F32 delta, '
                     'and native F16 must be no further from the FP32 truth than HF F16 is',
              'threshold_abs':worst_floor,'max_abs':worst_native,'native_versus_fp32_truth_max_abs':worst_truth,
              'agreement_within_reference_noise':worst_native<=worst_floor,
              'accuracy_at_least_reference':worst_truth<=worst_floor,
              'steps_above_per_step_floor':sum(1 for r in rows if r['max_abs']>r['reference_precision_floor'])}
        passed=gate['agreement_within_reference_noise'] and gate['accuracy_at_least_reference'] and argmax_ok
        for row in rows:row['status']='passed' if passed else 'failed'
    sources=['crates/memra-reference/src/speech/encoder.rs','crates/memra-reference/src/speech/decoder.rs','crates/memra-reference/src/speech/matrix.rs','crates/memra-reference/src/bin/whisper-stage.rs']
    receipt={'schema':'memra-whisper-decoder-gate-v1','status':'passed' if passed else 'failed','numeric':args.numeric,
             'gate':gate,'argmax_match_count':sum(1 for r in rows if r['argmax_match']),'step_count':len(rows),
             'oracle_manifest_sha256':sha(args.oracle/'manifest.json'),'binary_sha256':sha(args.binary),
             'native_source_sha256':{name:sha(Path(name)) for name in sources},'steps':rows,
             'scope':'12-step bounded cached math on a two-second synthetic clip, same encoder input and forced tokens; no transcript quality, throughput or serving qualification'}
    if control is not None:receipt['fp32_control_manifest_sha256']=sha(args.fp32_control/'manifest.json')
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print('decoder gate',receipt['status'])
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
