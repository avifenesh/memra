
# ---------------------------------------------------------------------------------------------------------------
# DSpark draft tapes (the drafting lane's copy of the oracle; everything above this line is run_oracle.py unchanged
# but for the --replay / --spec-win arguments).
#
# What feeds `Transformer.forward_spec(input_ids, main_hidden, start_pos)` in the vendor code (inference/model.py):
#   main_hidden  [b, s, 3 * dim] bf16: for each trunk position of the forward, the concatenation over the target layers
#                (config dspark_target_layer_ids = 37, 38, 39) of `h.mean(dim=2)`, the mean over the hc copies of the
#                residual stream at that layer's INPUT (after its Engram, before its attention), `Transformer.forward`.
#   input_ids    [b]: the token the trunk sampled from the forward's last position (position start_pos + s).
#   start_pos    the first trunk position of that forward.
# A forward_spec at start_pos 0 (the prefill) only seeds each draft stage's window KV with the prompt's main_kv rows
# and returns None. Past it the vendor code takes one trunk position per call (`main_kv.squeeze(1)`): stage KV of
# position start_pos goes to ring slot start_pos % window, the block's rows are [input token, noise x (block - 1)] at
# RoPE positions start_pos + 1 .. start_pos + block, every block row attends ring slots 0 .. min(window, start_pos + 1)
# in slot order and then all block rows (no causal mask inside the block). Row i's logits (the trunk's head on the
# last stage's collapsed stream, plus the Markov head's bias of the previous token) give the token at position
# start_pos + 2 + i; output_ids = [input token, draft 1 .. draft block]; the confidence head scores each row.
#
# Mode A (default): run the trunk greedily as run_oracle.py does and call forward_spec after the prefill and after
# every decode forward; the drafts never feed the trunk, so the generation is the plain greedy one.
# Mode B (--replay TAPE_DIR): no trunk; feed forward_spec the main_hidden and tokens of a mode-A tape (the same
# inputs on another backend: the vendor's own floor for the draft layers).
REC = {"on": False, "step": None}


def _cpu(t):
    return t.detach().to("cpu", copy=True)


def _stage_hook(si):
    def hook(m, i, o):
        if REC["on"]:
            REC["step"][f"h{si}"] = _cpu(o[0][0])  # [block, hc, dim] bf16
            REC["step"][f"pre{si}"] = _cpu(o[1][0].float())  # [block, hc]
    return hook


def _attn_hook(si):
    def hook(m, i, o):
        if REC["on"]:
            REC["step"][f"attn_in{si}"] = _cpu(i[0][0])  # [block, dim] after attn_norm
            REC["step"][f"attn_out{si}"] = _cpu(o[0])
    return hook


def _ffn_hook(si):
    def hook(m, i, o):
        if REC["on"]:
            REC["step"][f"ffn_in{si}"] = _cpu(i[0][0])
            REC["step"][f"ffn_out{si}"] = _cpu(o[0])
    return hook


def _gate_hook(si):
    def hook(m, i, o):
        if REC["on"]:
            REC["step"][f"route_w{si}"] = _cpu(o[0].float())  # [block, topk]
            REC["step"][f"route_i{si}"] = _cpu(o[1])
    return hook


def _markov_hook(m, i, o):
    if REC["on"]:
        REC["step"].setdefault("markov_bias", []).append(o[0][0].detach().float())  # [vocab] fp32, kept on device
        REC["step"].setdefault("markov_embed", []).append(_cpu(o[1][0]))  # [rank] bf16


def _conf_hook(m, i, o):
    if REC["on"]:
        REC["step"]["head_x"] = _cpu(i[0][0])  # [block, dim] bf16: the collapsed last-stage stream (before the norm)


def _main_norm_hook(m, i, o):
    if REC["on"]:
        REC["step"]["main_x"] = _cpu(o[0])  # [1, dim] bf16


for si, layer in enumerate(model.mtp):
    layer.register_forward_hook(_stage_hook(si))
    layer.attn.register_forward_hook(_attn_hook(si))
    layer.ffn.register_forward_hook(_ffn_hook(si))
    layer.ffn.gate.register_forward_hook(_gate_hook(si))
model.mtp[-1].markov_head.register_forward_hook(_markov_hook)
model.mtp[-1].confidence_head.register_forward_hook(_conf_hook)
model.mtp[0].main_norm.register_forward_hook(_main_norm_hook)


def spec_step(tok_in, mh, pos, step, n_full):
    """One forward_spec past the prefill, taped. tok_in: int; mh: [1, 1, 3 * dim] bf16 on the device."""
    REC["step"] = {}
    REC["on"] = True
    out_ids, lg, conf = model.forward_spec(torch.tensor([tok_in], device=a.device), mh, pos)
    REC["on"] = False
    r = REC["step"]
    lg = lg[0].float()  # [block, vocab], the Markov bias already added
    v, i = lg.topk(32, dim=-1)
    bias = torch.stack(r.pop("markov_bias"))  # [block, vocab]
    bv, bi = bias.topk(32, dim=-1)
    r.update(pos=pos, tok_in=tok_in, main_hidden=_cpu(mh[0, 0]), out_ids=out_ids[0].cpu().clone(),
             conf=_cpu(conf[0].float()), lse=torch.logsumexp(lg, -1).cpu(), top_vals=v.cpu(), top_ids=i.cpu(),
             bias_top_vals=bv.cpu(), bias_top_ids=bi.cpu(), markov_embed=torch.stack(r["markov_embed"]),
             full_logits=lg.cpu().clone() if step < n_full else None,
             full_bias=bias.cpu().clone() if step < n_full else None)
    return r


def rings():
    return torch.stack([_cpu(l.attn.window_kv_cache[0]) for l in model.mtp])  # [stages, window, head_dim] bf16


cases = json.load(open(a.cases)) if not a.replay else [{"id": x} for x in a.only.split(",")]
for c in cases:
    if a.only and c["id"] not in a.only.split(","):
        continue
    t1 = time.time()
    steps = []
    with torch.inference_mode():
        if a.replay:
            src = torch.load(os.path.join(a.replay, f"{c['id']}.pt"))
            ids, gen = src["prompt_ids"], src["gen_ids"]
            assert model.forward_spec(torch.tensor([gen[0]], device=a.device), src["prompt_main_hidden"].to(a.device)[None], 0) is None
            ring0 = rings()
            for step, s in enumerate(src["steps"]):
                steps.append(spec_step(s["tok_in"], s["main_hidden"].to(a.device)[None, None], s["pos"], step, a.full_logits))
            tops, t_pre, pmh = None, 0.0, src["prompt_main_hidden"]
        else:
            ids = c.get("prompt_ids") or tok.encode(encode_messages(c["messages"], thinking_mode=c.get("thinking_mode", "chat")))
            gen, tops = [], []
            x = torch.tensor([ids], device=a.device)
            nxt, logits, mh = model(x, 0)
            t_pre = time.time() - t1
            pmh = _cpu(mh[0])  # [s, 3 * dim] bf16
            assert model.forward_spec(nxt, mh, 0) is None
            ring0 = rings()
            pos = len(ids)
            for step in range(a.max_new):
                lg = logits[0].float().cpu()
                v, i = lg.topk(32)
                tops.append((v.clone(), i.clone()))
                t_id = int(lg.argmax())
                gen.append(t_id)
                if t_id == tok.eos_token_id or pos + 1 >= a.max_seq:
                    break
                nxt, logits, mh = model(torch.tensor([[t_id]], device=a.device), pos)
                steps.append(spec_step(int(nxt[0]), mh, pos, step, a.full_logits))
                pos += 1
    dt = time.time() - t1
    # the prefix of each draft the plain greedy generation agrees with (step k's input token is gen[k + 1])
    acc = []
    for k, s in enumerate(steps):
        d, n = s["out_ids"].tolist()[1:], 0
        while n < len(d) and k + 2 + n < len(gen) and d[n] == gen[k + 2 + n]:
            n += 1
        acc.append(n if k + 2 + len(d) <= len(gen) else -1)  # -1: the generation ended inside the block
    rec = {"id": c["id"], "prompt_ids": ids, "gen_ids": gen, "text": tok.decode(gen), "prompt_main_hidden": pmh,
           "ring_after_prefill": ring0, "ring_final": rings(), "steps": steps, "greedy_prefix": acc,
           "top_vals": torch.stack([t[0] for t in tops]) if tops else None,
           "top_ids": torch.stack([t[1] for t in tops]) if tops else None, "prefill_s": t_pre, "total_s": dt,
           "config": {k: cfg[k] for k in cfg if k.startswith("dspark") or k in ("n_mtp_layers", "window_size")}}
    torch.save(rec, os.path.join(a.out, f"{c['id']}.pt"))
    full = [x for x in acc if x >= 0]
    print(json.dumps({"id": c["id"], "prompt_len": len(ids), "gen": len(gen), "spec_steps": len(steps),
                      "greedy_prefix_mean": round(sum(full) / max(1, len(full)), 3), "prefill_s": round(t_pre, 1),
                      "total_s": round(dt, 1), "text": rec["text"][:120]}), flush=True)
print("ORACLE-DONE", flush=True)
