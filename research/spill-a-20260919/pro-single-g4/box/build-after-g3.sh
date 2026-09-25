#!/usr/bin/env bash
until grep -q driver-done /root/spill-receipts/a-g3/progress.log; do sleep 20; done
exec bash /root/spill-receipts/a-g4/build.sh 13ec5a7bddea54fcd6a77ba867dbce21eef638a7
