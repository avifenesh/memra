#!/bin/bash
# One lease for a landing head: the GPU test binaries and the batching gate (land_gpu.sh), then verify
# checkpoint / serve of both classes and the served c=1/2/4 timing (land_verify.sh), on LAND_BIN.
D=$(dirname "$0")
bash $D/land_gpu.sh
bash $D/land_verify.sh
echo LAND_ALL_DONE
