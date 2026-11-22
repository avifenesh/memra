#!/usr/bin/env bash
# GB10 box: build hy4f-spec, hy4f-gpu, memra-server from ~/hy4flash/batch/src (CPU only, low
# weight), then copy the binaries to ~/hy4flash/batch/bins-TAG with their sha256.
# usage: build.sh TAG
set -o pipefail
tag=$1
. "$HOME/.cargo/env"
cd "$HOME/hy4flash/batch/src" || exit 2
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CARGO_TARGET_DIR=$HOME/hy4flash/batch/target
export CARGO_BUILD_JOBS=${JOBS:-16}
nice -n 10 cargo build --release -p memra-engine --bin hy4f-spec --bin hy4f-gpu -p memra-server --bin memra-server -p memra-cli --bin memra 2>&1
rc=$?
echo "BUILD_EXIT=$rc"
[ $rc -eq 0 ] || exit $rc
d=$HOME/hy4flash/batch/bins-$tag
mkdir -p $d
cp $CARGO_TARGET_DIR/release/{hy4f-spec,hy4f-gpu,memra-server,memra} $d/
cp HEAD.txt $d/
(cd $HOME/hy4flash/batch && sha256sum bins-$tag/hy4f-spec bins-$tag/hy4f-gpu bins-$tag/memra-server bins-$tag/memra > bins-$tag.sha256)
cat $HOME/hy4flash/batch/bins-$tag.sha256
