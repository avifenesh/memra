"""Trace-free policy simulator for the device prefix cache on a hybrid (GDN) model.

Entry bytes = R + n*b (R = recurrent planes, b = KV bytes per token; Qwen3.8-27B measured:
64 tokens = 158.8 MB, 512 tokens = 172.1 MB). Whole-entry restores only; an entry restores
for a prompt when it is a token prefix ending on the 32-token grid (or at the prompt end).
A cold request publishes its seed at the grid boundary below its prompt end; a cold request
whose prompt shares >= 64 tokens with a resident entry also publishes the LCP boundary.

Policies:
  lru   plain LRU (main)
  gds   GreedyDual-Size: H = L + tokens/MB, victim = min H, L <- H(victim); a new entry is
        admitted only by displacing entries of lower H
  door  doorkeeper: a seed is published only when its prompt's first 64 tokens were seen
        before (resident entry or ghost); LCP splits always publish

Prompts are tuples of (segment_id, tokens) pairs; two prompts share a token prefix exactly as
far as their leading segments match.
"""
import itertools

R = 156.9e6
B = 29700.0
GRID = 32
MIN = 64
FLOOR = 16


def nbytes(n):
    return R + n * B


def plen(p):
    return sum(t for _, t in p)


def lcp(a, b):
    n = 0
    for (sa, ta), (sb, tb) in zip(a, b):
        if sa != sb:
            break
        if ta != tb:
            return n + min(ta, tb)
        n += ta
    return n


def seed_boundary(n):
    if n % GRID == 0:
        return n
    b = (n // GRID) * GRID
    if n - b < FLOOR:
        b -= GRID
    return b if b >= MIN else None


def grid_down(n):
    b = (n // GRID) * GRID
    return b if b >= MIN else None


class Cache:
    def __init__(self, budget, policy):
        self.budget, self.policy = budget, policy
        self.entries = {}  # key -> dict(prompt, n, bytes, last, H)
        self.total = 0.0
        self.clock = 0
        self.L = 0.0
        self.ghosts = set()
        self.evictions = 0
        self.refused = 0

    def value(self, n):
        return n / (nbytes(n) / 1e6)

    def lookup(self, prompt):
        best = None
        pl = plen(prompt)
        for k, e in self.entries.items():
            n = e['n']
            if n <= pl and lcp(e['prompt'], prompt) >= n and (n == pl or (n % GRID == 0 and pl - n >= FLOOR)):
                if best is None or n > self.entries[best]['n']:
                    best = k
        return best

    def touch(self, k):
        self.clock += 1
        e = self.entries[k]
        e['last'] = self.clock
        e['H'] = self.L + self.value(e['n'])

    def best_lcp(self, prompt):
        return max((lcp(e['prompt'], prompt) for e in self.entries.values()), default=0)

    def insert(self, prompt_prefix, n):
        key = (tuple(prompt_prefix), n)
        if key in self.entries:
            self.touch(key)
            return
        size = nbytes(n)
        if size > self.budget:
            self.refused += 1
            return
        self.clock += 1
        h = self.L + self.value(n)
        order = sorted(self.entries, key=lambda k: (self.entries[k]['H'], self.entries[k]['last'])
                       if self.policy == 'gds' else self.entries[k]['last'])
        need = self.total + size - self.budget
        victims, freed = [], 0.0
        for k in order:
            if freed >= need:
                break
            if self.policy == 'gds' and self.entries[k]['H'] > h:
                break
            victims.append(k)
            freed += self.entries[k]['bytes']
        if freed < need:
            self.refused += 1
            return
        for k in victims:
            if self.policy == 'gds':
                self.L = max(self.L, self.entries[k]['H'])
            self.total -= self.entries[k]['bytes']
            del self.entries[k]
            self.evictions += 1
        self.entries[key] = {'prompt': prompt_prefix, 'n': n, 'bytes': size, 'last': self.clock,
                             'H': self.L + self.value(n)}
        self.total += size


def cut(prompt, n):
    out, acc = [], 0
    for s, t in prompt:
        if acc + t >= n:
            out.append((s, n - acc))
            return out
        out.append((s, t))
        acc += t
    return out


def serve(c, prompt):
    """One request: returns cached tokens. Publishes seed / LCP entries on a miss path."""
    pl = plen(prompt)
    k = c.lookup(prompt)
    hit = c.entries[k]['n'] if k is not None else 0
    if k is not None:
        c.touch(k)
    fam = tuple(cut(prompt, MIN)) if pl >= MIN else None
    seen = fam is not None and (fam in c.ghosts or c.best_lcp(prompt) >= MIN)
    # LCP split: shares >= MIN with an entry beyond the hit -> publish the grid boundary at the LCP
    l = c.best_lcp(prompt)
    if l >= MIN and l > hit:
        b = grid_down(min(l, pl))
        if b and b > hit and b < pl and pl - b >= FLOOR:
            c.insert(cut(prompt, b), b)
    sb = seed_boundary(pl)
    if sb and sb > hit and (sb == pl or sb - hit >= FLOOR):
        if c.policy != 'door' or seen:
            c.insert(cut(prompt, sb), sb)
    if fam is not None:
        c.ghosts.add(fam)
    return hit


def run(shape, budget, policy):
    c = Cache(budget, policy)
    computed = cached = 0
    hits = []
    for tag, prompt in shape:
        h = serve(c, prompt)
        cached += h
        computed += plen(prompt) - h
        hits.append((tag, h))
    return computed, cached, c.evictions, c.refused, hits


# ---- shapes ---------------------------------------------------------------------------------
def mixed(conc=8, repeats=6, batch_per_repeat=None):
    """Warm 520-token prompt; c8 unique ~90-token prompts; a repeat (warm + 13 tokens) after
    every `batch_per_repeat` batch requests."""
    out = [('warm', [('P', 520)])]
    uid = itertools.count()
    for i in range(repeats):
        for _ in range(batch_per_repeat):
            u = next(uid)
            out.append(('batch', [(f'U{u}', 10), ('Ptxt', 80 + (u % 8) * 3)]))
        out.append((f'repeat-{i}', [('P', 520), (f'E{i}', 13)]))
    return out


def incident(turns=12, start=10912, grow=160, cohort=(1248, 1344, 1440, 1536), every=3):
    """5090 day-20 replay geometry (PREFIX-CACHE-POLICY.md): four cohort tenants seeded twice,
    one loop growing per turn, a cohort tenant continuing after every third loop turn, every
    cohort tenant once more after the loop."""
    out = []
    conv = {i: [(f'C{i}', n)] for i, n in enumerate(cohort)}
    for _ in range(2):
        for i in conv:
            out.append((f'cohort-{i}', list(conv[i])))
    loop = [('L0', start)]
    nxt = 0
    for t in range(turns):
        out.append((f'loop-{t}', list(loop)))
        loop = loop + [(f'L{t + 1}', grow)]
        if (t + 1) % every == 0:
            i = nxt % len(cohort)
            nxt += 1
            conv[i] = conv[i] + [(f'C{i}x{t}', 96)]
            out.append((f'cohort-{i}-ret', list(conv[i])))
    for i in conv:
        conv[i] = conv[i] + [(f'C{i}end', 96)]
        out.append((f'cohort-{i}-end', list(conv[i])))
    return out


def agentic(turns):
    """prime pass then measured pass over (session, prompt-tokens) turns; turns of one session
    share the earlier turn's prompt as a prefix."""
    by = {}
    shape = []
    for s, n in turns:
        prev = by.get(s)
        p = [(f'S{s}', n)] if prev is None else prev + [(f'S{s}+{n}', n - plen(prev))]
        by[s] = p
        shape.append((s, p))
    return [('prime-' + s, p) for s, p in shape] + [('meas-' + s, p) for s, p in shape]


if __name__ == '__main__':
    import json, sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 2259943424
    shapes = {
        'mixed-14/rep': mixed(batch_per_repeat=14),
        'mixed-30/rep': mixed(batch_per_repeat=30),
        'incident-5090@1GiB': incident(),
    }
    if len(sys.argv) > 2:
        shapes['agentic'] = agentic([tuple(x) for x in json.load(open(sys.argv[2]))])
    for name, shape in shapes.items():
        bud = 1024 * 2**20 if name.startswith('incident') else budget
        for pol in ('lru', 'gds', 'door'):
            comp, cach, ev, ref, hits = run(shape, bud, pol)
            rep = [h for t, h in hits if t.startswith('repeat') or t.startswith('meas') or 'ret' in t or 'end' in t]
            print(f'{name:22s} {pol:5s} budget {bud/1e6:8.0f}MB computed {comp:8d} cached {cach:8d} '
                  f'evict {ev:4d} refused {ref:4d} hits {sum(1 for h in rep if h)}/{len(rep)}')
