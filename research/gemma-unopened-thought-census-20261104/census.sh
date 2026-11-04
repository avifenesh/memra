#!/bin/bash
# E4B unopened-thought census at the gemma-thought lane tip (the parser rule), RTX 5090.
cd $HOME/.cache/memra-fix/engine-dsv4-glm || exit 2
git checkout -q --detach bk/w4-gemma-thought || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$HOME/.cache/memra-fix/target-cuda CARGO_BUILD_JOBS=2
EV=$HOME/.cache/memra-fix4/gemma-census-$GTAG; mkdir -p $EV; S=$EV/status
H=$(git rev-parse HEAD); echo "head $H $(date +%H:%M)" > $S
flock $HOME/.cache/memra-rig-build.lock nice -n 15 cargo build --release -p memra-server --bin memra-server > $EV/build.log 2>&1
echo "build rc=$? stamp=$(grep -a -c "${H:0:12}" $CARGO_TARGET_DIR/release/memra-server)" >> $S
cp $CARGO_TARGET_DIR/release/memra-server $EV/memra-server-${H:0:12}
run() {
    E4B=$GMODEL
    env MEMRA_COMPAT=openai "MEMRA_MODELS=e4b=$E4B" MEMRA_ADDR=127.0.0.1:18172 MEMRA_CTX=8192 \
        $EV/memra-server-${H:0:12} > $EV/server.log 2>&1 &
    pid=$!
    waitfor --http http://127.0.0.1:18172/v1/models --timeout 10m > /dev/null 2>&1 || { echo "not ready" >> $S; kill $pid; return 1; }
    python3 $HOME/.cache/memra-fix4/gemma-census-client.py 18172 $EV/out > $EV/census.log 2>&1
    echo "client rc=$?" >> $S
    kill $pid; wait $pid 2>/dev/null
}
export -f run; export EV S H GMODEL
gpulock run --ttl 40m --for "memra gemma4 unopened-thought census ($GTAG)" --wait 2h -- bash -c run
grep SUMMARY $EV/census.log >> $S
echo "panics=$(grep -c panicked $EV/server.log) done $(date +%H:%M)" >> $S
