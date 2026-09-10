#!/usr/bin/env python3
"""MEMRA_SERVE_DEVPENALTY qualification on the qwen35 class, one RTX 5090 (sm_120).

Four sub-cells per boot, one arm per boot (the flag is a process OnceLock):

  vendor  - the motivating production shape: the client sends NO sampling field, so the
            server resolves Qwen's vendor NON-THINKING arm from models.toml
            (temperature .7, top_p .8, top_k 20, presence_penalty 1.5). c in {1,4,8}.
  pp0     - the same page-task with presence_penalty 0.0 sent explicitly. Penalty-free rows
            never enter the device-penalty path in either arm, so this is the ceiling the
            penalized rows are trying to reach, and it is also a null control: pp0 must not
            move between arms.
  pptg    - the standard pp512/tg128 twin on the vendor shape, c=1.
  greedy  - the exactness instrument. temperature 0 with presence_penalty 1.5 and with 0.0.
            Penalized greedy stays host-side by construction in BOTH arms
            (worker.rs devsample_meta), so the door must not move one byte of either.
"""
import argparse, hashlib, json, subprocess, threading, time, urllib.request, uuid
from pathlib import Path

BASE = 'http://127.0.0.1:8080'
HERE = Path(__file__).resolve().parent
FIX = json.loads((HERE / 'page-task.json').read_text())

SUMMARY_TOKENS = 200
EXTRACT_TOKENS = 60
TG_TOKENS = 128


def messages(with_summary=None):
    m = [
        {'role': 'system', 'content': FIX['system']},
        {'role': 'user', 'content': FIX['user_1']},
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_fetch_1', 'type': 'function',
            'function': {'name': FIX['tool_call']['name'],
                         'arguments': json.dumps(FIX['tool_call']['arguments'])}}]},
        {'role': 'tool', 'tool_call_id': 'call_fetch_1', 'content': FIX['tool_result']},
    ]
    if with_summary is not None:
        m.append({'role': 'assistant', 'content': with_summary})
        m.append({'role': 'user', 'content': FIX['user_2']})
    return m


def call(model, msgs, max_tokens, salt, sampling=None, tools=True, timeout=900):
    body = {
        'model': model, 'messages': msgs, 'max_tokens': max_tokens, 'stream': True,
        'stream_options': {'include_usage': True}, 'cache_salt': salt,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    if tools:
        body['tools'] = FIX['tools']
    if sampling:
        body.update(sampling)
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=data,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    text, usage, chunks = [], None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            ev = json.loads(payload)
            if ev.get('usage'):
                usage = ev['usage']
            for ch in ev.get('choices') or []:
                d = ch.get('delta') or {}
                piece = d.get('content') or d.get('reasoning_content') or d.get('reasoning') or ''
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
                    chunks += 1
    t1 = time.perf_counter()
    if first is None:
        first = t1
    u = usage or {}
    prompt, completion = u.get('prompt_tokens', 0), u.get('completion_tokens', 0)
    cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    ttft, decode_s = first - t0, max(t1 - first, 1e-9)
    body_text = ''.join(text)
    return {
        'ttft_s': ttft, 'total_s': t1 - t0, 'decode_s': decode_s,
        'prompt_tokens': prompt, 'cached_tokens': cached, 'completion_tokens': completion,
        'prefill_tok_s': (prompt / ttft) if ttft > 0 else None,
        'decode_tok_s': ((completion - 1) / decode_s) if completion > 1 else None,
        'stream_chunks': chunks, 'spec': u.get('spec'),
        'server_elapsed_s': u.get('elapsed_s'),
        'text_sha256': hashlib.sha256(body_text.encode()).hexdigest(),
        'text': body_text,
    }


def page_task(model, salt, sampling=None):
    t0 = time.perf_counter()
    t1r = call(model, messages(), SUMMARY_TOKENS, salt, sampling)
    t2r = call(model, messages(t1r['text']), EXTRACT_TOKENS, salt, sampling)
    return {'salt': salt,
            'turn1': {k: v for k, v in t1r.items() if k != 'text'},
            'turn2': {k: v for k, v in t2r.items() if k != 'text'},
            'e2e_s': time.perf_counter() - t0,
            'turn1_text': t1r['text'], 'turn2_text': t2r['text']}


def vram_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0]) if out else -1


class VramWatch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag, self.peak = threading.Event(), 0

    def run(self):
        while not self.stop_flag.is_set():
            self.peak = max(self.peak, vram_mib())
            time.sleep(0.25)


def cell(model, name, sampling, conc, reps, tag, arm, outdir):
    rows = []
    for rep in range(1, reps + 1):
        watch = VramWatch(); watch.start()
        t0 = time.perf_counter()
        results = [None] * conc

        def work(i):
            results[i] = page_task(
                model, f'{tag}-{name}-c{conc}-r{rep}-w{i}-{uuid.uuid4().hex[:8]}', sampling)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
        for t in threads: t.start()
        for t in threads: t.join()
        block_s = time.perf_counter() - t0
        watch.stop_flag.set(); watch.join()
        rows.append({'rep': rep, 'block_wall_s': block_s, 'vram_peak_mib': watch.peak,
                     'tasks': results})
        print(f'  [{arm}] {name} c={conc} rep {rep}: block {block_s:.2f}s peak {watch.peak} MiB',
              flush=True)
    p = Path(outdir) / f'{tag}-{name}-c{conc}.json'
    p.write_text(json.dumps({'model': model, 'arm': arm, 'cell': name, 'sampling': sampling,
                             'concurrency': conc, 'reps': reps, 'tag': tag,
                             'summary_max_tokens': SUMMARY_TOKENS,
                             'extract_max_tokens': EXTRACT_TOKENS,
                             'rows': rows}, ensure_ascii=False, indent=1) + '\n')


def pptg(model, reps, tag, arm, outdir):
    """pp512/tg128 twin on the vendor shape: one ~512-token prompt, 128 decoded tokens."""
    prompt = (HERE / 'pptg-prompt.txt').read_text()
    rows = []
    for rep in range(1, reps + 1):
        r = call(model, [{'role': 'user', 'content': prompt}], TG_TOKENS,
                 f'{tag}-pptg-r{rep}-{uuid.uuid4().hex[:8]}', tools=False)
        rows.append({k: v for k, v in r.items() if k != 'text'})
        print(f'  [{arm}] pptg rep {rep}: prompt {r["prompt_tokens"]} tok, '
              f'pp {r["prefill_tok_s"]:.0f} tok/s, tg {r["decode_tok_s"]:.1f} tok/s '
              f'({r["completion_tokens"]} tok)', flush=True)
    (Path(outdir) / f'{tag}-pptg.json').write_text(
        json.dumps({'model': model, 'arm': arm, 'tag': tag, 'max_tokens': TG_TOKENS,
                    'rows': rows}, indent=1) + '\n')


def greedy(model, tag, arm, outdir):
    """The exactness instrument. Byte identity across arms, both penalty shapes.

    Greedy is argmax, so the same prefix must yield the same bytes on every boot of every
    arm. A fixed cache_salt per case makes the request itself identical too.
    """
    cases = {
        'greedy-pp15': {'temperature': 0.0, 'top_k': 0, 'top_p': 1.0,
                        'presence_penalty': 1.5, 'repetition_penalty': 1.0},
        'greedy-pp00': {'temperature': 0.0, 'top_k': 0, 'top_p': 1.0,
                        'presence_penalty': 0.0, 'repetition_penalty': 1.0},
    }
    out = {}
    for name, samp in cases.items():
        r = call(model, messages(), SUMMARY_TOKENS, f'exactness-{name}', samp)
        out[name] = {'sampling': samp, 'text': r['text'], 'text_sha256': r['text_sha256'],
                     'completion_tokens': r['completion_tokens'],
                     'prompt_tokens': r['prompt_tokens'], 'spec': r['spec'],
                     'decode_tok_s': r['decode_tok_s']}
        print(f'  [{arm}] {name}: {r["completion_tokens"]} tok sha {r["text_sha256"][:16]}',
              flush=True)
    (Path(outdir) / f'{tag}-greedy.json').write_text(
        json.dumps({'model': model, 'arm': arm, 'tag': tag, 'cases': out},
                   ensure_ascii=False, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--control-reps', type=int, default=3)
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    rest = vram_mib()
    (Path(a.outdir) / f'{a.tag}-vram-at-rest.json').write_text(
        json.dumps({'vram_at_rest_mib': rest}) + '\n')
    print(f'{a.tag}: VRAM at rest {rest} MiB', flush=True)

    # one unmeasured warm task so the first measured rep is not paying one-time JIT/alloc
    page_task(a.model, f'{a.tag}-warmup-{uuid.uuid4().hex[:8]}')

    greedy(a.model, a.tag, a.arm, a.outdir)
    for c in (1, 4, 8):
        cell(a.model, 'vendor', None, c, a.reps, a.tag, a.arm, a.outdir)
    for c in (1, 4, 8):
        cell(a.model, 'pp0', {'presence_penalty': 0.0}, c, a.control_reps, a.tag, a.arm, a.outdir)
    pptg(a.model, a.reps, a.tag, a.arm, a.outdir)
    print(f'{a.tag} COMPLETE', flush=True)


if __name__ == '__main__':
    main()
