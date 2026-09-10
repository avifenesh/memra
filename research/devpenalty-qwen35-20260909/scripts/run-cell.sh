set -eu
cd /root/dp
# Interleaved A/B, three boots per arm, counterbalanced so a within-session ordering or
# thermal effect cannot ride the delta: OFF/ON, ON/OFF, OFF/ON.
bash scripts/boot-block.sh off 1 --reps 5
bash scripts/boot-block.sh on  1 --reps 5
bash scripts/boot-block.sh on  2 --reps 5
bash scripts/boot-block.sh off 2 --reps 5
bash scripts/boot-block.sh off 3 --reps 5
bash scripts/boot-block.sh on  3 --reps 5
echo CELL_COMPLETE
