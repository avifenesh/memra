#!/bin/bash
# Re-base the private build worktree onto the newer lane tip (the prime seq_end fix the
# coordinator named) and rebuild. Kept as a FILE rather than an inline ssh command because the
# previous two attempts used `pkill -f "s37h-..."`, whose pattern matched the launching shell's
# own command line and killed the launcher before it did anything. A script's pkill pattern
# cannot match the caller.
set -u
OUT=/root/s37h-rebase.txt; : > $OUT
exec >>$OUT 2>&1
echo "date=$(date -Is)"

# Stop the queued measurement job first: it must not copy a binary out of a half-written build.
for p in $(pgrep -f "bash /root/s37h-spec.sh"); do echo "stopping queued sweep $p"; kill "$p"; done
sleep 2
for p in $(pgrep -f "flock -w 43200"); do
  own=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  [ -z "$own" ] || [ -d "/proc/$own" ] || { echo "reaping orphan flock $p"; kill "$p"; }
done

cd /root/memra-src
git show-ref > /root/s37h-refs-b2.txt
# EXPLICIT refspec, unique ref name. A wildcard against a bundle plus a global fetch.prune
# deletes every local branch the bundle does not carry, and this repo holds two other lanes.
git fetch /root/s37h-tip2.bundle "refs/heads/lane/step37-mtp-masked-vocab-20260825:refs/heads/s37h-lanetip2" 2>&1 | tail -2
echo "refs lost (must be empty):"
comm -23 <(cut -d' ' -f2 /root/s37h-refs-b2.txt | sort) <(git show-ref | cut -d' ' -f2 | sort)

cd /root/wt-s37h
echo "dirty before: $(git status --short | tr '\n' ' ')"
git checkout --detach s37h-lanetip2 2>&1 | tail -2
echo "tip now: $(git log -1 --format='%h %s')"
echo "dirty after (my lib.rs patch must survive): $(git status --short | tr '\n' ' ')"
echo "w8_view_on=$(grep -c w8_view_on crates/memra-engine/src/lib.rs) ROWS_TAB_RESTAGE=$(grep -c ROWS_TAB_RESTAGE crates/memra-engine/src/tp.rs)"

source $HOME/.cargo/env
export CARGO_TARGET_DIR=/root/wt-s37h/target
nice -n 19 cargo build --release -j 8 -p memra-server --bin memra-server > /root/s37h-rebuild.log 2>&1
RC1=$?
nice -n 19 cargo build --release -j 8 -p memra-engine --bin run-gen >> /root/s37h-rebuild.log 2>&1
RC2=$?
echo "rebuild rc=$RC1/$RC2 compiled=$(grep -c Compiling /root/s37h-rebuild.log)"
if [ $RC1 -ne 0 ]; then tail -30 /root/s37h-rebuild.log; echo "REBUILD FAILED"; exit 1; fi
cp -f target/release/memra-server /root/s37h-spec-server
[ $RC2 -eq 0 ] && cp -f target/release/run-gen /root/s37h-spec-rungen
# Fingerprint from strings, never cargo's Finished line: a 1.7s "Finished" after a checkout is
# the failed-checkout alarm, not a build.
echo "server md5=$(md5sum /root/s37h-spec-server | cut -c1-12) size=$(stat -c %s /root/s37h-spec-server)"
echo "strings ROWS_TAB_RESTAGE=$(strings -a /root/s37h-spec-server | grep -c ROWS_TAB_RESTAGE) w8-view=$(strings -a /root/s37h-spec-server | grep -c w8-view) step37-defaults=$(strings -a /root/s37h-spec-server | grep -c step37-defaults)"
echo "free=$(df --output=avail -BG /root | tail -1 | tr -dc '0-9')G"
echo "S37H-BUILD-DONE"
cd /root && setsid nohup /root/s37h-spec.sh > /root/s37h-spec-driver.log 2>&1 < /dev/null &
sleep 3
echo "sweep requeued: $(pgrep -f 'bash /root/s37h-spec.sh' | tr '\n' ' ')"
echo "S37H-REBASE-DONE"
