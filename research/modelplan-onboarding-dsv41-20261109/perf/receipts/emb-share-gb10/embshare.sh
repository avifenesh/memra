#!/usr/bin/env bash
# GB10 (one lease): emb's shared embedding table, byte for byte across ranks on one card.
#  1. build dsv41-run for 121a from src-emb.tar (emb's head, the table one copy a card);
#  2. dsv41-run --cases (capital, code_pal, needle 3,761 tokens; teacher-forced) over the first 4 layers (Engram
#     layer 1 inside), MEMRA_FULL_PREC=1, on one rank (--devices 0) and on two ranks of the card (--devices 0,0,
#     which now share one table);
#  3. compare every value file of the two-rank run with the one-rank run, and the one-rank run with the run of the
#     build before the review fix (a table a rank; ~/.cache/dsv41-speed/out/gates-emb/run-emb-0, read only).
# Done-file out/result.txt, marker EMBSHARE-DONE.
set -u
R=$HOME/.cache/dsv41-embshare
D=$HOME/.cache/dsv41-speed/data
OLD=$HOME/.cache/dsv41-speed/out/gates-emb
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out
mkdir -p $O; : > $O/result.txt
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" RUSTUP_TOOLCHAIN=1.97.1
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
echo "build start $(date +%T)" >> $O/result.txt
rm -rf $R/src && mkdir -p $R/src $R/bins && cd $R/src && tar -xf $R/src-emb.tar || { echo "untar failed; EMBSHARE-DONE" >> $O/result.txt; exit 1; }
nice -n 19 timeout 60m cargo build --release --locked -j 12 -p memra-engine --bin dsv41-run > $O/build.log 2>&1
rc=$?
echo "build rc $rc $(date +%T)" >> $O/result.txt
if [ $rc -ne 0 ]; then grep -E "^error" -A12 $O/build.log | head -40 >> $O/result.txt; echo "EMBSHARE-DONE" >> $O/result.txt; exit 1; fi
cp -a $R/target/release/dsv41-run $R/bins/ && sha256sum $R/bins/dsv41-run >> $O/result.txt
export MEMRA_FULL_PREC=1
I=$D/dec/ids
printf '%s\n' "capital $I/capital.prompt.i32 $I/capital.gen.i32" "code_pal $I/code_pal.prompt.i32 $I/code_pal.gen.i32" \
  "needle $I/needle.prompt.i32 $I/needle.gen.i32" > $O/cases.txt
for dev in 0 0,0; do
  rm -rf $O/run-$dev; mkdir -p $O/run-$dev
  t=$(date +%s)
  timeout 8m $R/bins/dsv41-run $C --token-map $TM --cases $O/cases.txt --layers 4 --devices $dev --out $O/run-$dev > $O/run-$dev.log 2>&1
  echo "run devices $dev rc $? $(( $(date +%s) - t )) s; $(grep -a -m2 'dsv41-resident\|embed' $O/run-$dev.log | tr '\n' ' ' | cut -c1-200)" >> $O/result.txt
done
cmpset() {  # cmpset A B LABEL: the value files of A against B
  local n=0 bad=0 f
  for f in $(cd $1 && find . -type f \( -name '*.f32' -o -name '*.i32' \) | sort); do
    n=$((n+1)); cmp -s $1/$f $2/$f || { bad=$((bad+1)); echo "  differs: $f" >> $O/result.txt; }
  done
  echo "$3: $n value files compared, $bad differ" >> $O/result.txt
}
cmpset $O/run-0 $O/run-0,0 "two ranks on the card (one shared table) against one rank"
[ -d $OLD/run-emb-0 ] && cmpset $O/run-0 $OLD/run-emb-0 "one rank against the build before the review fix (a table a rank)"
[ -d $OLD/run-emb-0,0 ] && cmpset $O/run-0,0 $OLD/run-emb-0,0 "two ranks against the build before the review fix, two ranks"
echo "EMBSHARE-DONE $(date +%T)" >> $O/result.txt
