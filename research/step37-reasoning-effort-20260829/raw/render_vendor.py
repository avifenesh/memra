import jinja2, json, sys
tpl_src = open("/tmp/model-ct.jinja").read()
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
env.filters["fromjson"] = json.loads
tpl = env.from_string(tpl_src)
msgs = [{"role":"user","content":"What is 17*23? Reply with the number only."}]
for effort in [None, "low", "medium", "high"]:
    kw = dict(messages=msgs, add_generation_prompt=True, bos_token="")
    if effort is not None:
        kw["reasoning_effort"] = effort
    out = tpl.render(**kw)
    name = effort or "default"
    open(f"/home/ubuntu/re-lane/vendor-{name}.txt","w").write(out)
    print(f"=== {name} ({len(out)} bytes) ===")
    print(repr(out))
