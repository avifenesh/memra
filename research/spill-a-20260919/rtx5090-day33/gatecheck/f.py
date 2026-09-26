import sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
ref = next((i for i, l in enumerate(lines) if "MEMRA_KV_HOST_FAULT=contract-promote-spans" in l and "promote refused" in l), None)
subs = [i for i, l in enumerate(lines) if "promote submitted off the tick:" in l and "f32 spans filled on the copy stream" in l]
after = [i for i in subs if ref is not None and i > ref]
print(f"refusal at {ref}, filled span submissions {len(subs)}, after the refusal {len(after)}")
sys.exit(0 if ref is not None and after else 1)
