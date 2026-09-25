#!/usr/bin/env python3
"""memra#536 stall cell (lane A day 16, OWNER-THREAD-CENSUS.md, pre-registered).

A streaming TENANT decodes one token per tick (`MEMRA_SERVE_SPEC=0` boot) while an INTRUDER request
runs one owner-thread class: a cold prime of a fixed length (`prime`), a host demotion (`demote`,
a fresh 64-token prompt whose seed insert evicts the previous entry into the host tier) or a host
promotion (`promote`, the day-15 alternation: a host hit that promotes and whose insert evicts the other
entry). The tenant's client-side inter-token gaps are the observation; the idle control is the tenant
alone. Order 1 = (idle, arm) x N, order 2 = (arm, idle) x N. The rule line is fixed here and replayed
from receipt.json by `--replay`; no threshold, no verdict: the cell measures a stall per class.

    stall_cell.py --port P --mode prime|demote|promote|capture|restore --server-log LOG --out DIR [--n 5]
    (`capture`, WP-A day 20: the prime arm's intruder on a boot with the prefix cache ON, so its
    grid seed captures; after it returns the same prompt is re-posted untimed and its
    `cached_tokens` recorded, the hit clause of memra#536 Move 2 cell (i). The three day-16
    arms are byte-for-byte the day-16 program.)
    (`restore`, WP-A day 21: ONE fixed intruder prompt of the prime arm's length, posted untimed in
    setup so its grid seed captures and publishes; every timed run then re-posts the SAME prompt
    at the tenant's 24th token, a whole-entry HIT whose restore copies the entry and primes the
    suffix; each intruder's `cached_tokens` is recorded, memra#536 Move 2 cell (ii). The four
    earlier arms are byte-for-byte unchanged.)
    (`demote-long` and `promote-long`, WP-A day 43, `DAY43.md` section 1, OWED item 15: the demote and
    promote arms at 4096-token entries. `demote-long`: each timed intruder is a fresh long prompt of the
    prime arm's length whose insert evicts the resident long entry into the host tier (one untimed long
    seed in setup, so the first timed run demotes too). `promote-long`: two fixed long prompts L_A and
    L_B seeded untimed (L_B's insert demotes L_A); each timed intruder is a CHAIN: a hit on L_A (it
    promotes, and its insert demotes L_B), then, the moment its response returns, a hit on L_B, which
    meets L_B Demoting and parks until its publication, then promotes (its insert demotes L_A again);
    the chained request's e2e is `chain_wall_ms`. The five earlier arms are byte-for-byte unchanged.)
    stall_cell.py --replay DIR/receipt.json

Client-side only: stdlib, no engine binary, no GPU access of its own.
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time

TENANT_PROMPT = ("List nine tide gauge stations from north to south, one per line, terse:")  # 20 tokens or fewer
FIRE_AT = 24          # the intruder fires when the tenant's 24th token has arrived
TENANT_MAX_TOKENS = 160
PRIME_TARGET_TOKENS = 4096
WORDS = ("river stone maple copper harbor signal ladder winter garden meadow anchor beacon canvas "
         "delta ember falcon granite hollow island jasper kettle lantern marble nickel orchid pepper "
         "quartz ribbon saddle timber umber velvet walnut yellow zephyr basket candle dagger engine "
         "fabric gutter hammer ingot jacket kernel locket magnet needle oyster pillar quiver rocket "
         "socket tablet uplink vessel window yonder zenith almond bridge cobalt dinghy pewter").split()
assert len(WORDS) == 64

P_A = ("You are indexing the survey logs of a coastal tide-gauge network. For each of the twelve "
       "stations, ordered north to south, report the gauge type, the datum epoch, the sampling "
       "interval in minutes, the last calibration date, the responsible technician role, and the "
       "anomaly that would force an out-of-cycle calibration. Be systematic and terse; do not skip "
       "a station. After the twelve stations, add a short paragraph on network-wide drift checks.")
P_B = ("Draft the commissioning checklist for a small hydroelectric turbine hall. Cover, in "
       "order: penstock inspection, wicket-gate travel, governor response, generator insulation, "
       "thrust-bearing temperature rise, cooling-water flow, overspeed trip, and grid-synchronization "
       "tests. For each item name the instrument used, the acceptance threshold, the sign-off role, "
       "and the failure symptom that would halt commissioning. Be systematic and terse throughout.")


def words(n, salt):
    return " ".join(WORDS[(i * 7 + salt) % 64] for i in range(n))


def fresh_prompt(n_words, run_id):
    return f"run {run_id}: " + words(n_words, run_id)


# WP-A day 43 (`promote-long`): the two fixed long prompts, the prime arm's length each.
L_A = fresh_prompt(PRIME_TARGET_TOKENS - 4, 3901)
L_B = fresh_prompt(PRIME_TARGET_TOKENS - 4, 3902)


def post(port, body, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.monotonic()
    c.request("POST", "/v1/completions", body=json.dumps(body),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    wall = (time.monotonic() - t0) * 1e3
    c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:200]!r}")
    return json.loads(data), wall


def stream_tenant(port, arrivals, fired, error):
    """Streams the tenant; appends the monotonic arrival time of every SSE token event."""
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        body = {"model": "gate", "prompt": TENANT_PROMPT, "max_tokens": TENANT_MAX_TOKENS,
                "temperature": 0, "stream": True}
        c.request("POST", "/v1/completions", body=json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            error.append(f"tenant HTTP {r.status}: {r.read()[:200]!r}")
            return
        text = []
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            now = time.monotonic()
            ev = json.loads(payload)
            ch = ev.get("choices") or []
            if ch and ch[0].get("text"):  # a token event; usage-only or empty events are not arrivals
                arrivals.append(now)
                text.append(ch[0]["text"])
                if len(arrivals) == FIRE_AT:
                    fired.set()
        c.close()
        arrivals.append(("text", "".join(text)))
    except Exception as e:  # recorded, never inferred
        error.append(f"tenant: {e!r}")
    finally:
        fired.set()


def log_tail(path, offset):
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace"), f.tell()


def server_capture_ms(lines):
    """WP-A day 20: `[prefix-cache] capture published off the tick (..): N tokens complete after P
    poll(s), X ms from submission to completion, ..`: the copy's own duration, per publication."""
    out = []
    for ln in lines.splitlines():
        if "[prefix-cache] capture published off the tick" in ln:
            i = ln.find("poll(s), ")
            if i >= 0:
                num = ln[i + 9:].split("ms")[0].strip()
                try:
                    out.append(float(num))
                except ValueError:
                    pass
    return out


def server_restore_ms(lines):
    """WP-A day 21: `[prefix-cache] restore landed off the tick: N tokens complete after P poll(s),
    X ms from submission to completion, ..`: the copy's own duration, per landed restore."""
    out = []
    for ln in lines.splitlines():
        if "[prefix-cache] restore landed off the tick" in ln:
            i = ln.find("poll(s), ")
            if i >= 0:
                num = ln[i + 9:].split("ms")[0].strip()
                try:
                    out.append(float(num))
                except ValueError:
                    pass
    return out


def server_ms(lines, kind):
    out = []
    for ln in lines.splitlines():
        if f"[prefix-host] {kind}: " in ln:
            # `[prefix-host] demote: 64 tokens, 159.9MB in 37.2ms (...` (worker.rs 11208, 11764)
            i = ln.find(" in ")
            if i >= 0:
                num = ln[i + 4:].split("ms")[0].strip()
                try:
                    out.append(float(num))
                except ValueError:
                    pass
    return out


def one_run(port, mode, arm, run_id, log_path, log_off, promote_toggle):
    arrivals, error = [], []
    fired = threading.Event()
    th = threading.Thread(target=stream_tenant, args=(port, arrivals, fired, error), daemon=True)
    t_start = time.monotonic()
    th.start()
    intruder = None
    if arm != "idle":
        fired.wait(timeout=600)
        if mode in ("prime", "capture"):
            prompt = fresh_prompt(PRIME_TARGET_TOKENS - 4, 1000 + run_id)
        elif mode == "restore":
            prompt = fresh_prompt(PRIME_TARGET_TOKENS - 4, 2100)  # the ONE seeded prompt, a hit every run
        elif mode == "demote":
            prompt = fresh_prompt(72, 2000 + run_id)
        elif mode == "demote-long":
            prompt = fresh_prompt(PRIME_TARGET_TOKENS - 4, 3000 + run_id)
        elif mode == "promote-long":
            prompt = L_A
        else:
            prompt = P_A if promote_toggle[0] % 2 == 0 else P_B
            promote_toggle[0] += 1
        t_i = time.monotonic()
        try:
            resp, wall = post(port, {"model": "gate", "prompt": prompt, "max_tokens": 1, "temperature": 0})
            usage = resp.get("usage", {})
            intruder = {"fired_at_tenant_token": FIRE_AT, "fired_at_ms": (t_i - t_start) * 1e3,
                        "wall_ms": wall, "prompt_tokens": usage.get("prompt_tokens"),
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens",
                                                                                         usage.get("cached_tokens"))}
        except Exception as e:
            intruder = {"error": repr(e), "fired_at_ms": (t_i - t_start) * 1e3}
        if mode == "promote-long" and intruder is not None and "error" not in intruder:
            # The chain: the entry the first hit's insert just demoted, at once (it is Demoting now).
            t_c = time.monotonic()
            try:
                resp2, wall2 = post(port, {"model": "gate", "prompt": L_B, "max_tokens": 1, "temperature": 0})
                usage2 = resp2.get("usage", {})
                intruder["chain_fired_at_ms"] = (t_c - t_start) * 1e3
                intruder["chain_wall_ms"] = wall2
                intruder["chain_prompt_tokens"] = usage2.get("prompt_tokens")
                intruder["chain_cached_tokens"] = (usage2.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", usage2.get("cached_tokens"))
            except Exception as e:
                intruder["chain_error"] = repr(e)
        if mode == "capture" and intruder is not None and "error" not in intruder:
            # The hit clause: the same prompt again, untimed, after the tenant's stream ends (so the
            # re-post never sits beside the timed window); its cached_tokens must equal the seed's
            # published length (the grid-aligned prompt end).
            th.join(timeout=900)
            try:
                resp2, wall2 = post(port, {"model": "gate", "prompt": prompt, "max_tokens": 1, "temperature": 0})
                usage2 = resp2.get("usage", {})
                intruder["repost_wall_ms"] = wall2
                intruder["repost_prompt_tokens"] = usage2.get("prompt_tokens")
                intruder["repost_cached_tokens"] = (usage2.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", usage2.get("cached_tokens"))
            except Exception as e:
                intruder["repost_error"] = repr(e)
    th.join(timeout=900)
    text = ""
    times = []
    for a in arrivals:
        if isinstance(a, tuple):
            text = a[1]
        else:
            times.append(a)
    itl = [(times[i + 1] - times[i]) * 1e3 for i in range(len(times) - 1)]
    tail, new_off = log_tail(log_path, log_off)
    run = {
        "arm": arm, "run_id": run_id, "tenant_tokens": len(times), "ttft_ms": (times[0] - t_start) * 1e3 if times else None,
        "tenant_wall_ms": (times[-1] - t_start) * 1e3 if times else None,
        "itl_ms": [round(x, 3) for x in itl], "tenant_text_sha": __import__("hashlib").sha256(text.encode()).hexdigest()[:16],
        "intruder": intruder, "errors": error,
        "server_demote_ms": server_ms(tail, "demote"), "server_promote_ms": server_ms(tail, "promote"),
        "server_capture_ms": server_capture_ms(tail) if mode == "capture" else [],
        "server_restore_ms": server_restore_ms(tail) if mode == "restore" else [],
        "server_log_lines": [ln for ln in tail.splitlines() if "[prefix-host]" in ln or "[abort]" in ln
                             or "[prefix-cache] capture" in ln or "[prefix-cache] restore" in ln
                             or "[prefix-cache] hit" in ln][:40],
    }
    if itl:
        p50 = statistics.median(itl)
        run["p50"] = p50
        run["max"] = max(itl)
        run["stall_ms"] = max(itl) - p50
    return run, new_off


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(runs):
    idle = [r for r in runs if r["arm"] == "idle"]
    arm = [r for r in runs if r["arm"] != "idle"]
    pool_idle = [x for r in idle for x in r["itl_ms"]]
    pool_arm = [x for r in arm for x in r["itl_ms"]]
    stalls = [r["stall_ms"] for r in arm if "stall_ms" in r]
    return {
        "idle": {"runs": len(idle), "p50": pct(pool_idle, .5), "p95": pct(pool_idle, .95), "p99": pct(pool_idle, .99),
                 "max": max(pool_idle) if pool_idle else None, "n": len(pool_idle)},
        "arm": {"runs": len(arm), "p50": pct(pool_arm, .5), "p95": pct(pool_arm, .95), "p99": pct(pool_arm, .99),
                "max": max(pool_arm) if pool_arm else None, "n": len(pool_arm)},
        "stall_ms": {"per_run": stalls, "median": statistics.median(stalls) if stalls else None,
                     "min": min(stalls) if stalls else None, "max": max(stalls) if stalls else None},
        "server_demote_ms": [x for r in arm for x in r["server_demote_ms"]],
        "server_promote_ms": [x for r in arm for x in r["server_promote_ms"]],
        "server_restore_ms": [x for r in arm for x in r.get("server_restore_ms", [])],
        "intruder_prompt_tokens": [r["intruder"].get("prompt_tokens") for r in arm if r.get("intruder")],
        "intruder_cached_tokens": [r["intruder"].get("cached_tokens") for r in arm if r.get("intruder")],
        "intruder_wall_ms": [round(r["intruder"].get("wall_ms", float("nan")), 1) for r in arm if r.get("intruder")],
        "tenant_text_shas": sorted({r["tenant_text_sha"] for r in runs}),
        "errors": [e for r in runs for e in r["errors"]] + [r["intruder"]["error"] for r in arm if r.get("intruder") and "error" in r["intruder"]]
        + [r["intruder"]["chain_error"] for r in arm if r.get("intruder") and "chain_error" in r["intruder"]],
        "chain_wall_ms": [round(r["intruder"]["chain_wall_ms"], 1) for r in arm
                          if r.get("intruder") and "chain_wall_ms" in r["intruder"]],
    }


def rule_line(tag, mode, n, s):
    f = lambda v: "na" if v is None or v != v else f"{v:.1f}"
    return (f"STALL rule cell={tag} arm={mode} n_per_order={n} pooled={n * 2} "
            f"idle_runs={s['idle']['runs']} idle_p50={f(s['idle']['p50'])} idle_p95={f(s['idle']['p95'])} "
            f"idle_p99={f(s['idle']['p99'])} idle_max={f(s['idle']['max'])} "
            f"arm_runs={s['arm']['runs']} arm_p50={f(s['arm']['p50'])} arm_p95={f(s['arm']['p95'])} "
            f"arm_p99={f(s['arm']['p99'])} arm_max={f(s['arm']['max'])} "
            f"stall_median={f(s['stall_ms']['median'])} stall_min={f(s['stall_ms']['min'])} stall_max={f(s['stall_ms']['max'])} "
            f"server_demote_ms={[round(x, 1) for x in s['server_demote_ms']]} "
            f"server_promote_ms={[round(x, 1) for x in s['server_promote_ms']]} "
            + (f"server_restore_ms={[round(x, 1) for x in s['server_restore_ms']]} "
               f"intruder_cached_tokens={s['intruder_cached_tokens']} " if mode == "restore" else "")
            + (f"intruder_wall_ms={s['intruder_wall_ms']} chain_wall_ms={s.get('chain_wall_ms', [])} "
               if mode == "promote-long" else "")
            + f"intruder_prompt_tokens={s['intruder_prompt_tokens']} tenant_text_identical={len(s['tenant_text_shas']) == 1} "
            f"errors={len(s['errors'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--mode", choices=["prime", "demote", "promote", "capture", "restore", "demote-long", "promote-long"])
    ap.add_argument("--server-log")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--replay")
    a = ap.parse_args()
    if a.replay:
        rec = json.load(open(a.replay))
        s = summarize(rec["runs"])
        line = rule_line(rec["tag"], rec["mode"], rec["n_per_order"], s)
        print(line)
        ok = line == rec["rule_line"]
        print("STALL REPLAY:", "PASS (replay agrees with the harness's rule line)" if ok else "FAIL (rule line differs)")
        return 0 if ok else 1
    if not (a.port and a.mode and a.server_log and a.out):
        ap.error("--port --mode --server-log --out are required")
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"stall-{a.mode}"
    log_off = os.path.getsize(a.server_log)
    runs, setup = [], []
    promote_toggle = [0]
    if a.mode == "promote":
        # Seed E_A then E_B (E_B's insert demotes E_A): untimed setup, recorded.
        for i, p in enumerate((P_A, P_B)):
            resp, wall = post(a.port, {"model": "gate", "prompt": p, "max_tokens": 1, "temperature": 0})
            tail, log_off = log_tail(a.server_log, log_off)
            setup.append({"seed": "AB"[i], "wall_ms": wall, "usage": resp.get("usage"),
                          "server_demote_ms": server_ms(tail, "demote")})
    if a.mode == "promote-long":
        # WP-A day 43: seed L_A then L_B (L_B's insert demotes L_A): untimed setup, recorded.
        for i, p in enumerate((L_A, L_B)):
            resp, wall = post(a.port, {"model": "gate", "prompt": p, "max_tokens": 1, "temperature": 0})
            tail, log_off = log_tail(a.server_log, log_off)
            setup.append({"seed": "AB"[i], "wall_ms": wall, "usage": resp.get("usage"),
                          "server_demote_ms": server_ms(tail, "demote")})
    if a.mode == "demote-long":
        # WP-A day 43: one untimed long seed, so the first timed run's insert demotes it.
        resp, wall = post(a.port, {"model": "gate", "prompt": fresh_prompt(PRIME_TARGET_TOKENS - 4, 2999),
                                   "max_tokens": 1, "temperature": 0})
        tail, log_off = log_tail(a.server_log, log_off)
        setup.append({"seed": True, "wall_ms": wall, "usage": resp.get("usage")})
    if a.mode == "restore":
        # Untimed setup: the ONE intruder prompt is posted once so its grid seed captures and
        # publishes; every timed run's re-post is then a whole-entry hit. Recorded.
        resp, wall = post(a.port, {"model": "gate", "prompt": fresh_prompt(PRIME_TARGET_TOKENS - 4, 2100),
                                   "max_tokens": 1, "temperature": 0})
        tail, log_off = log_tail(a.server_log, log_off)
        setup.append({"seed": True, "wall_ms": wall, "usage": resp.get("usage"),
                      "server_capture_lines": [ln for ln in tail.splitlines() if "[prefix-cache]" in ln][:10]})
    if a.mode in ("prime", "capture"):
        # Untimed calibration of the fresh prompt's token count, recorded (the timed runs use
        # PRIME_TARGET_TOKENS - 4 words plus the run header; this reports what the tokenizer made of it).
        resp, wall = post(a.port, {"model": "gate", "prompt": fresh_prompt(PRIME_TARGET_TOKENS - 4, 999),
                                   "max_tokens": 1, "temperature": 0})
        tail, log_off = log_tail(a.server_log, log_off)
        setup.append({"calibration": True, "wall_ms": wall, "usage": resp.get("usage")})
    run_id = 0
    seq = [("idle", a.mode)] * a.n + [(a.mode, "idle")] * a.n
    for order, pair in enumerate(seq):
        for arm in pair:
            run_id += 1
            r, log_off = one_run(a.port, a.mode, arm, run_id, a.server_log, log_off, promote_toggle)
            r["order"] = 1 if order < a.n else 2
            runs.append(r)
            print(f"  run {run_id:2d} order {r['order']} {arm:8s} tokens={r['tenant_tokens']} "
                  f"p50={r.get('p50', float('nan')):.1f} max={r.get('max', float('nan')):.1f} "
                  f"stall={r.get('stall_ms', float('nan')):.1f} demote={r['server_demote_ms']} promote={r['server_promote_ms']} "
                  f"intruder={r['intruder'] and r['intruder'].get('wall_ms')}", flush=True)
    s = summarize(runs)
    line = rule_line(tag, a.mode, a.n, s)
    rec = {"tag": tag, "mode": a.mode, "n_per_order": a.n, "fire_at": FIRE_AT, "tenant_max_tokens": TENANT_MAX_TOKENS,
           "tenant_prompt": TENANT_PROMPT, "setup": setup, "runs": runs, "summary": s, "rule_line": line}
    json.dump(rec, open(os.path.join(a.out, "receipt.json"), "w"), indent=1)
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
