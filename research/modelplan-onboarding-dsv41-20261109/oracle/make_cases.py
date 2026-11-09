"""Oracle cases: short chat prompts across skills, plus a ~3k-token needle prompt that drives the indexer past its
top-512 (encoder ratio-2 layers past 1024 tokens, decoder ratio-1 layers past 512)."""
import json, random, sys

r = random.Random(1017)
words = "amber birch cobalt delta ember fjord granite harbor indigo juniper kestrel lumen meadow nimbus onyx prairie quartz raven sierra tundra umber velvet willow xenon yarrow zephyr".split()
facts = []
for i in range(260):
    facts.append(f"Record {i:03d}: the {r.choice(words)} {r.choice(words)} code is {r.randint(10000, 99999)}.")
needle = 187
q = facts[needle].split(": ", 1)[1].split(" code")[0]
long_text = "\n".join(facts)
copy_src = "The quick brown fox jumps over the lazy dog while seven wizards quietly juggle twelve purple boxes near the old mill by the river at dawn."
cases = [
    # the tokens `memra model verify checkpoint` runs its native side on (oracle_to_checkpoint_tape.py)
    {"id": "tape-1234", "prompt_ids": [1, 2, 3, 4]},
    {"id": "capital", "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}]},
    {"id": "gsm", "messages": [{"role": "user", "content": "A baker makes 24 muffins per tray and bakes 7 trays. She sells 150 muffins and gives away 9. How many muffins are left? Give the final number only."}]},
    {"id": "code", "messages": [{"role": "user", "content": "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Code only."}]},
    {"id": "copy", "messages": [{"role": "user", "content": f"Repeat this sentence exactly, nothing else:\n{copy_src}"}]},
    {"id": "hebrew", "messages": [{"role": "user", "content": "מה בירת ישראל? ענה במילה אחת."}]},
    {"id": "needle", "messages": [{"role": "user", "content": f"{long_text}\n\nWhat is the code in Record {needle:03d}? Answer with the number only."}]},
]
json.dump(cases, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("needle answer:", facts[needle])
