#!/bin/bash
# One lease for a landing head on LAND_BIN: the GPU test binaries and the batched==solo gate
# (land_gpu.sh), verify checkpoint / serve of both classes and the served c=1/2/4 timing
# (land_verify.sh), lane H's batching gate (batch_gate.sh) and lane S's drafter gates (land_spec.sh).
D=$(dirname "$0")
bash $D/land_gpu.sh
bash $D/land_verify.sh
BIN=$LAND_BIN OUT=$LAND_GATE_OUT bash $D/batch_gate.sh
bash $D/land_spec.sh
echo LAND_ALL_DONE
