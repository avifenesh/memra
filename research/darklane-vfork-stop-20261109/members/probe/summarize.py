#!/usr/bin/env python3
"""One line per probe run: who leads, when ba03102882's code acknowledged, and the worker counter."""
import json
import sys

for line in open(sys.argv[1]):
    r = json.loads(line)
    st = lambda key: ",".join(t.split(":")[1] for t in r[key])
    print(
        f"{r['case']:28} rep={r['rep']} leader={r['leader_comm']:8} ack_ms={r['ack_ms']!s:5} "
        f"err={r['err'] or '-'} ctr stop/ack/late/resumed={r['ctr_stop']}/{r['ctr_ack']}/{r['ctr_late']}/{r['ctr_resumed']} "
        f"cprog before={st('cprog_before')} ack={st('cprog_at_ack')} late={st('cprog_late')} resumed={st('cprog_resumed')} "
        f"exit={r['exit']}"
    )
