cd /home/avifenesh/projects/memra-runs/ink2-121a-223610e293
for cls in shipped nvfp4-rig; do
  /home/avifenesh/.cache/ink2-target-121a/release/ink2-gpu /home/avifenesh/models/inkling2-mini --class $cls --tape /home/avifenesh/projects/ink2-tapes/tapes/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-$cls.json > batch-$cls.log 2>&1; echo "RC batch-$cls=$?"; tail -n 1 batch-$cls.log
  /home/avifenesh/.cache/ink2-target-121a/release/ink2-gpu /home/avifenesh/models/inkling2-mini --class $cls --context 64 --batch-seq-cap 4096 --tape /home/avifenesh/projects/ink2-tapes/tapes/p1_chat.exact_f32 --batch-check 4,12,7 --out past64-$cls.json > past64-$cls.log 2>&1; echo "RC past64-$cls=$?"; tail -n 1 past64-$cls.log
  /home/avifenesh/.cache/ink2-target-121a/release/ink2-gpu /home/avifenesh/models/inkling2-mini --class $cls --tape /home/avifenesh/projects/ink2-tapes/tapes/p1_chat.exact_f32 --out probe-$cls.json > probe-$cls.log 2>&1; echo "RC probe-$cls=$?"; tail -n 2 probe-$cls.log
done
