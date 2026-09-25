#!/usr/bin/env bash
until grep -q driver-done /root/spill-receipts/a-g4/progress.log; do sleep 20; done
exec bash /root/spill-receipts/a-s/build.sh c8b5d3c54a3eb9add81bd5ddcf363e4f6783eb05
