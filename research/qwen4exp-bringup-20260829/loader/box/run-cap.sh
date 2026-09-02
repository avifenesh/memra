#!/bin/bash
# q4e loader-stream lane (memra issue #48): load the real artifact under a cgroup cap that
# reproduces the 180 GB-RAM 2-card box class, and record peak RSS.
# args: <label> <binary> <MemoryMax>
#
# LOCK POLICY (lane law, two owner corrections on 2026-09-02). TWO locks, both mandatory:
#
#   1. /tmp/q48fn-measure.lock -s, waited for however long it takes, no fallback and no
#      timeout. Shared is enough for untimed work: it makes this arm wait behind exclusive
#      holders and blocks new exclusive holders while it loads. A 174 GB load hammers memory
#      bandwidth, so running one beside somebody's timed arms corrupts their receipts even
#      when this arm takes no card of theirs.
#   2. ~/realgate/bin/q4e-load-lock.sh, composed INSIDE the systemd-run scope. The measure
#      lock does not stop this: shared holders load concurrently, and two concurrent loads of
#      this artifact exceed the 353 GB host, so the GLOBAL OOM killer takes one of them
#      (receipted 2026-09-02 00:15:09Z: another lane's run, CONSTRAINT_NONE, anon-rss
#      180.8 GB, while this lane's old-cap230 arm sat at ~185 GiB). A MemoryMax cap does NOT
#      make an arm safe for co-tenants: it bounds this process, and the host still hands it
#      every byte up to the cap.
set -uo pipefail
lbl="$1"; bin="$2"; cap="$3"
out=/root/realgate/loaderout
log="$out/$lbl.log"
{
  printf "# lane\tq4e-loader-stream-20260901 (memra issue #48)\n"
  printf "# label\t%s\n" "$lbl"
  printf "# binary\t%s\n" "$bin"
  printf "# binary_sha256\t%s\n" "$(sha256sum "$bin" | cut -d" " -f1)"
  printf "# cap\tMemoryMax=%s MemorySwapMax=0\n" "$cap"
  printf "# card\tCUDA_VISIBLE_DEVICES=1\n"
  printf "# seams\tMEMRA_Q4E_SEAMS=idxsel\n"
  printf "# ckpt\t/root/data/q48fn-yarn1m\n"
  printf "# started\t%s\n" "$(date -u +%FT%TZ)"
  printf "# free_before\t%s\n" "$(free -g | awk 'NR==2{print $2" total "$3" used "$7" avail (GB)"}')"
  printf "# vram_before\t%s\n" "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' '|')"
} > "$log"

gate() {
  env CUDA_VISIBLE_DEVICES=1 MEMRA_Q4E_SEAMS=idxsel \
    systemd-run --scope --unit="q4e-$lbl" -p MemoryMax="$cap" -p MemorySwapMax=0 \
      /root/realgate/bin/q4e-load-lock.sh "$out/$lbl.gate.log" \
        /usr/bin/time -v "$bin" /root/data/q48fn-yarn1m "$out" --label "$lbl" --mtp \
          --goldens /root/realgate/dump --prompts /root/realgate/shapes/thinkon-prompts.tsv
  local rc=$?
  cat "$out/$lbl.gate.log" 2>/dev/null
  return $rc
}

# Hold the shared lock on fd 9 for the whole arm (append-open: never truncate a co-tenant's
# lock file). No timeout, no fallback: -s waits for an exclusive holder for as long as one
# exists.
exec 9>>/tmp/q48fn-measure.lock
flock -s 9
printf "# lock\tacquired -s after %ss of waiting\n" "$SECONDS" >> "$log"
gate >> "$log" 2>&1
rc=$?
exec 9>&-
{
  printf "# rc\t%s\n" "$rc"
  printf "# finished\t%s\n" "$(date -u +%FT%TZ)"
  printf "# free_after\t%s\n" "$(free -g | awk 'NR==2{print $2" total "$3" used "$7" avail (GB)"}')"
} >> "$log"
exit "$rc"
