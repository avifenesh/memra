# Parameterize the banked instruments for this lane: own port, own directory.
# The originals in ~ are another lane's and are never modified.
import ast, os
OUT = "/home/ubuntu/ppn-cell"
p = open("/home/ubuntu/probe.py").read()
p = p.replace("import json, sys, time, urllib.request",
              "import json, os, sys, time, urllib.request", 1)
p = p.replace('open("/home/ubuntu/prompts.json")',
              'open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"))', 1)
p = p.replace('"http://127.0.0.1:18400/v1/chat/completions"',
              'os.environ.get("MEMRA_URL", "http://127.0.0.1:18401/v1/chat/completions")', 1)
assert "MEMRA_URL" in p and "abspath" in p, "probe.py patch did not apply"
open(os.path.join(OUT, "probe.py"), "w").write(p)

s = open("/home/ubuntu/steady.py").read()
s = s.replace("import json, re, statistics, subprocess, sys",
              "import json, os, re, statistics, subprocess, sys", 1)
s = s.replace('"/home/ubuntu/probe.py"',
              'os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe.py")', 1)
assert "abspath" in s, "steady.py patch did not apply"
open(os.path.join(OUT, "steady.py"), "w").write(s)

for f in ("probe.py", "steady.py"):
    ast.parse(open(os.path.join(OUT, f)).read())
print("instruments patched and parse-clean")
