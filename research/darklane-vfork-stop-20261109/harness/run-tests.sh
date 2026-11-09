#!/bin/bash
# Run the darklane vfork tests on <host>. Usage: run-tests.sh PHASE   (PHASE = before | after)
set -u
PHASE=$1
D=$WORK/darklane
B="bash $D/build.sh"
$B "$PHASE-vfork" --lib -- --exact darklane::tests::stop_acknowledges_a_leader_parked_in_vfork_on_a_child_stopped_before_exec --nocapture
$B "$PHASE-stress" --lib -- --exact darklane::tests::stop_resume_cycles_on_a_constantly_spawning_shell_job --ignored --nocapture
$B "$PHASE-repeated" --lib -- --exact darklane::tests::repeated_stop_resume_has_fresh_acknowledgements_and_stopped_shutdown --nocapture
$B "$PHASE-darklane" --lib -- darklane:: --nocapture
echo done > "$D/$PHASE.DONE"
