#!/usr/bin/env python3
"""Offline model of the two prefix-cache policies on the day-15 replay shape (prediction only).

Mirrors worker.rs: pin promotes at lookup time (probation -> protected, LRU refresh,
rebalance demotes protected LRU while protected > target); publication at prefill-done while the
hit entry is still pinned: prepare_snapshot (oversize / leased refusals, room_victim_with(slru,
None) until total <= budget - bytes) then insert into probation with room_victim_with(slru, keep);
unpin at retire refreshes recency and re-enters the segment. lru arm: protected target = budget
(no demotion), victim = global oldest across both segments.
"""
import sys

PER_TOKEN = 29750.0
FIXED = 156716666.66666666
MIN_TOKENS = 64


def entry_bytes(tokens):
    return int(FIXED + PER_TOKEN * tokens)


class Cache:
    def __init__(self, budget, slru, protected_pct=80):
        self.budget = budget
        self.slru = slru
        self.target = budget if not slru else budget * protected_pct // 100
        self.entries = []  # dicts: id, tenant, toks, bytes, seg, last_use, pins
        self.clock = 0
        self.next_id = 1
        self.evictions = 0
        self.log = []

    def now(self):
        self.clock += 1
        return self.clock

    def total(self):
        return sum(e["bytes"] for e in self.entries)

    def seg_bytes(self, seg):
        return sum(e["bytes"] for e in self.entries if e["seg"] == seg)

    def lru(self, seg):
        c = [e for e in self.entries if e["pins"] == 0 and e["seg"] == seg]
        return min(c, key=lambda e: (e["last_use"], e["id"])) if c else None

    def oldest_evictable(self):
        c = [e for e in self.entries if e["pins"] == 0]
        return min(c, key=lambda e: (e["last_use"], e["id"])) if c else None

    def evictable(self):
        return sum(e["bytes"] for e in self.entries if e["pins"] == 0)

    def pinned(self):
        return sum(e["bytes"] for e in self.entries if e["pins"] > 0)

    def rebalance(self):
        while self.seg_bytes("protected") > self.target:
            v = self.lru("protected")
            if v is None:
                break
            v["seg"] = "probation"
            self.log.append(f"demote {v['tenant']}:{v['toks']}")

    def capacity_victim(self):
        if not self.slru:
            return self.oldest_evictable()
        return self.lru("probation")

    def room_victim(self, keep):
        v = self.capacity_victim()
        if not self.slru:
            return v
        if v is not None and v["id"] != keep:
            return v
        return self.lru("protected")

    def remove(self, e, why):
        self.entries.remove(e)
        self.evictions += 1
        self.log.append(f"evict ({why}, {e['seg']}) {e['tenant']}:{e['toks']}")

    def lookup(self, tenant, prompt_len):
        c = [e for e in self.entries if e["tenant"] == tenant and MIN_TOKENS <= e["toks"] <= prompt_len]
        return max(c, key=lambda e: e["toks"]) if c else None

    def pin(self, e):
        if e["seg"] == "probation":
            e["seg"] = "protected"
        e["pins"] += 1
        e["last_use"] = self.now()
        self.rebalance()

    def unpin(self, e):
        e["pins"] -= 1
        if e["pins"] == 0:
            e["last_use"] = self.now()
            self.rebalance()

    def has_key(self, tenant, toks):
        return any(e["tenant"] == tenant and e["toks"] == toks for e in self.entries)

    def prepare_snapshot(self, tenant, b):
        if b > self.budget:
            self.log.append(f"refused oversize {tenant} {b}")
            return False
        target = self.budget - b
        needed = max(0, self.total() - target)
        if needed == 0:
            return True
        if needed > self.evictable():
            self.log.append(f"refused leased {tenant} {b} beside {self.pinned()}")
            return False
        while self.total() > target:
            v = self.room_victim(None)
            if v is None:
                return False
            self.remove(v, "preflight")
        return True

    def insert(self, tenant, toks):
        b = entry_bytes(toks)
        if b > self.budget or b > self.budget - self.pinned():
            self.log.append(f"insert refused {tenant}:{toks}")
            return
        self.rebalance()
        e = {"id": self.next_id, "tenant": tenant, "toks": toks, "bytes": b, "seg": "probation", "last_use": self.now(), "pins": 0}
        self.next_id += 1
        self.entries.append(e)
        self.log.append(f"insert {tenant}:{toks} resident {self.total()}")
        while self.total() > self.budget:
            v = self.room_victim(e["id"])
            if v is None:
                break
            self.remove(v, "insert")

    def request(self, tenant, prompt_len):
        hit = self.lookup(tenant, prompt_len)
        cached = 0
        if hit is not None:
            self.pin(hit)
            cached = hit["toks"]
        if not self.has_key(tenant, prompt_len):
            if self.prepare_snapshot(tenant, entry_bytes(prompt_len)):
                self.insert(tenant, prompt_len)
        if hit is not None:
            self.unpin(hit)
        return cached


def replay(slru, budget, cohort, start, grow, turns, return_every):
    c = Cache(budget, slru)
    rows = []
    conv = {f"cohort-{i+1}": n for i, n in enumerate(cohort)}
    for t, n in conv.items():
        for s in ("seed1", "seed2"):
            cached = c.request(t, n)
            rows.append((s, t, n, cached))
    ret_i = 0
    tenants = list(conv)
    for k in range(1, turns + 1):
        p = start + (k - 1) * grow
        cached = c.request("grow", p)
        rows.append((f"loop{k}", "grow", p, cached))
        if return_every and k % return_every == 0:
            t = tenants[ret_i % len(tenants)]
            ret_i += 1
            conv[t] += grow
            cached = c.request(t, conv[t])
            rows.append((f"return@{k}", t, conv[t], cached))
    if return_every:
        for t in tenants:
            conv[t] += grow
            cached = c.request(t, conv[t])
            rows.append(("final", t, conv[t], cached))
    return rows, c


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2048 * (1 << 20)
    cohort = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "7800,8000,8200,8400").split(",")]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 27300
    turns = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    grow = 300
    ret = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    print(f"budget={budget} protected80={budget*80//100} cohort_bytes={sum(entry_bytes(n) for n in cohort)} e1={entry_bytes(start)} e_last={entry_bytes(start+(turns-1)*grow)} two_last={entry_bytes(start+(turns-1)*grow)+entry_bytes(start+(turns-2)*grow)}")
    out = {}
    for arm in ("slru", "lru"):
        rows, c = replay(arm == "slru", budget, cohort, start, grow, turns, ret)
        computed = sum(p - cached for _, _, p, cached in rows)
        loop_cold = sum(1 for s, _, _, cached in rows if s.startswith("loop") and s != "loop1" and cached == 0)
        ret_cached = sum(cached for s, _, _, cached in rows if s.startswith("return") or s == "final")
        out[arm] = (computed, rows, c)
        print(f"\n== {arm}: computed={computed} loop_cold_after_1={loop_cold} return_cached_sum={ret_cached} evictions={c.evictions}")
        for r in rows:
            print(f"  {r[0]:10s} {r[1]:9s} prompt={r[2]:6d} cached={r[3]:6d} computed={r[2]-r[3]:6d}")
        if "-v" in sys.argv:
            print("\n".join("   " + l for l in c.log))
    a, b = out["slru"][0], out["lru"][0]
    print(f"\nprimary: slru={a} lru={b} diff(slru-lru)={a-b}")


if __name__ == "__main__":
    main()
