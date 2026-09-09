#!/usr/bin/env python3
"""Gate native Whisper stages against the pinned real-audio rental oracle.

The oracle stores, per 30-second window, the faster-whisper log-mel, the CT2 encoder
output (FP16 widened to F32) and the beam-1 token sequence. Windows shorter than 3000
frames are zero-filled in feature space by faster-whisper, while the native frontend
writes the analytic silence floor there, so mel is compared over content frames and the
encoder is fed the oracle's own window mel.
"""
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'

SAMPLES_PER_FRAME=160
WINDOW_SAMPLES=480000
WINDOW_FRAMES=3000


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--stage',choices=('mel','encoder'),required=True)
    p.add_argument('--numeric',choices=('f32','f16'),default='f16')
    p.add_argument('--clips',help='comma separated clip ids; default every ct2 clip')
    p.add_argument('--windows',type=int,help='cap windows per clip')
    p.add_argument('--work',type=Path,required=True,help='scratch directory, caller deletes it')
    args=p.parse_args()
    manifest=json.loads((args.oracle/'MANIFEST.json').read_text())
    ct2=next(b for b in manifest['backends'] if b['backend']=='ct2')
    wanted=set(args.clips.split(',')) if args.clips else None
    clips=[c for c in ct2['clips'] if wanted is None or c['clip_id'] in wanted]
    if wanted and len(clips)!=len(wanted):raise SystemExit('unknown clip id requested')
    args.work.mkdir(parents=True,exist_ok=True)
    rows=[];passed=True
    for clip in clips:
        pcm_entry=clip['pcm']
        pcm_path=args.oracle/pcm_entry['path']
        if sha(pcm_path)!=pcm_entry['sha256']:raise SystemExit('oracle pcm hash mismatch: '+clip['clip_id'])
        pcm=np.fromfile(pcm_path,dtype='<i2').astype(np.float32)/32768.0
        if pcm.size!=clip['samples']:raise SystemExit('pcm extent mismatch: '+clip['clip_id'])
        clip_features=None
        if args.stage=='mel':
            source=args.work/'clip.pcm.f32';pcm.tofile(source)
            out=args.work/'clip.mel.f32'
            if out.exists():out.unlink()
            subprocess.run([str(args.binary),'mel-clip',str(args.checkpoint),str(source),str(out)],
                           check=True,stdout=subprocess.DEVNULL)
            clip_features=np.fromfile(out,dtype='<f4')
            frames=pcm.size//SAMPLES_PER_FRAME+1
            if clip_features.size!=128*frames:raise SystemExit('native clip mel extent mismatch: '+clip['clip_id'])
            if clip_features.size!=int(np.prod(clip['log_mel']['shape'])):
                raise SystemExit('native clip frame count disagrees with the oracle: '+clip['clip_id'])
            clip_features=clip_features.reshape(128,frames)
            if not np.isfinite(clip_features).all():raise SystemExit('non-finite native clip mel')
        for window in clip['windows'][:args.windows]:
            entry=window['log_mel']
            reference=args.oracle/entry['path']
            if sha(reference)!=entry['sha256']:raise SystemExit('oracle mel hash mismatch')
            ref=np.fromfile(reference,dtype='<f4').reshape(entry['shape'])
            content=min(window['segment_size_frames'],WINDOW_FRAMES)
            if clip_features is not None:
                seek=window['seek_frame']
                clip_mel=np.zeros((128,WINDOW_FRAMES),dtype=np.float32)
                clip_mel[:,:content]=clip_features[:,seek:seek+content]
            row={'clip':clip['clip_id'],'window':window['index'],'seek_frame':window['seek_frame'],
                 'content_frames':content,'oracle_sha256':entry['sha256']}
            if args.stage=='mel':
                delta=float(np.abs(clip_mel[:,:content]-ref[:,:content]).max())
                tail=float(np.abs(ref[:,content:]).max()) if content<WINDOW_FRAMES else 0.0
                row.update({'max_abs':delta,'threshold_abs':1e-3,
                            'oracle_pad_is_zero':tail==0.0,
                            'status':'passed' if delta<=1e-3 and tail==0.0 else 'failed'})
            else:
                entry=window['encoder']
                reference=args.oracle/entry['path']
                if sha(reference)!=entry['sha256']:raise SystemExit('oracle encoder hash mismatch')
                ref=np.fromfile(reference,dtype='<f4')
                # Feed the oracle's own window mel so this measures the encoder alone.
                mel_source=args.oracle/window['log_mel']['path']
                out=args.work/'encoder'
                if out.exists():
                    for f in out.iterdir():f.unlink()
                    out.rmdir()
                subprocess.run([str(args.binary),'encoder',str(args.checkpoint),str(mel_source),str(out),args.numeric],
                               check=True,stdout=subprocess.DEVNULL)
                native=np.fromfile(out/'encoder.f32',dtype='<f4')
                if native.size!=ref.size or not np.isfinite(native).all():raise SystemExit('invalid native encoder extent')
                delta=float(np.abs(native-ref).max())
                row.update({'max_abs':delta,'mean_abs':float(np.abs(native-ref).mean()),
                            'numeric':args.numeric,'oracle_meaning':entry.get('meaning'),
                            'native_input_sha256':sha(mel_source)})
                row['status']='measured'
            passed&=row.get('status')!='failed'
            rows.append(row);print(row['clip'],row['window'],row['max_abs'],row['status'],flush=True)
    receipt={'schema':'memra-whisper-real-audio-v1','stage':args.stage,'status':'passed' if passed else 'failed',
             'oracle_manifest_sha256':sha(args.oracle/'MANIFEST.json'),
             'cell_lock_sha256':manifest['cell_lock_sha256'],'binary_sha256':sha(args.binary),
             'clip_count':len(clips),'window_count':len(rows),'windows':rows,
             'scope':'real recorded Hebrew audio against the pinned CT2 rental oracle; frontend and encoder '
                     'numerics only, no WER, no transcription policy, no GPU or serving qualification'}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print('real-audio',args.stage,receipt['status'],len(rows),'windows')
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
