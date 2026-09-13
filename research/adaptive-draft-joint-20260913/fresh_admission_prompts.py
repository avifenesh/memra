"""Deterministic fictional test inputs; no customer data or model generation."""
from pathlib import Path
import hashlib
import json

def write(directory):
    directory.mkdir(exist_ok=False)
    names = ['Aster','Bracken','Cairn','Dapple']
    for i, name in enumerate(names):
        events = [
            'RUN, START, RUN, HOLD, FINISH, RESUME, FINISH, HOLD, RESET, START',
            'START, HOLD, RUN, FINISH, RESUME, RUN, HOLD, RESET, FINISH, START',
            'RESET, START, RUN, RUN, HOLD, RESUME, HOLD, FINISH, RESET, RUN',
            'START, RUN, HOLD, HOLD, RESET, START, FINISH, RUN, FINISH, RESET'][i]
        cases = {
            'inventory': f'A fictional workshop tracks {name} enamel tiles. Opening stock is {40+i*3} boxes. Morning delivery adds 18; twelve go to repair bay; two of those twelve return unopened; afternoon delivery adds seven; five are quarantined and cannot be used. Reconcile physical and available stock. Explain every movement, distinguish returns from new deliveries, and propose three checks for the ledger.',
            'routing': f'Plan a route in the fictional {name} depot network. Undirected travel times: Depot-Orchard {4+i}, Depot-Quarry {7-i}, Orchard-Quarry 2, Orchard-Harbor {8-i}, Quarry-Harbor {3+i}, Harbor-Depot 9. Start at Depot, visit Orchard and Harbor, return to Depot. Quarry is optional and revisits are allowed. Compare at least three routes, show sums, and explain which is shortest. The cargo is ceramic sample jars, with no real-world safety constraints.',
            'schema': f'Normalize these fictional {name} exhibit records into JSON followed by a validation explanation. Fields: label, material, original_year, repaired_year, display_ready. Records: Willow bowl, clay, 1912, repaired 2004, ready; Lark frame, oak, year unknown, repaired 1998, not ready; Pebble lamp, brass, 1931, never repaired, ready; Thistle box, tin, 1906, repaired 2011, ready; Moss plaque, stone, year unknown, never repaired, not ready. Use null for unknown years and never invent dates. Explain the distinction between unknown and not applicable.',
            'terminology': f'Rewrite a fictional {name} instrument manual for new operators. Preserve these defined names exactly: amber dial, silver latch, meadow chamber, ripple indicator. Source: Before the silver latch is released, inspect the ripple indicator. A steady indicator permits access to the meadow chamber. A blinking indicator requires turning the amber dial clockwise once, waiting for a steady indicator, then releasing the silver latch. Closing the chamber requires re-engaging the latch. State the sequence clearly, explain each condition, and add a short glossary without adding new operational rules.',
            'state': f'Execute this fictional {name} state machine and explain the trace. States are DORMANT, READY, BUSY, PAUSED. START maps DORMANT to READY. RUN maps READY to BUSY. HOLD maps BUSY to PAUSED. RESUME maps PAUSED to BUSY. FINISH maps BUSY to READY. RESET maps any state to DORMANT. An event with no defined transition leaves the state unchanged and increments an ignored-event counter. Begin DORMANT with counter zero. Events: {events}. Show every state and counter update, then explain two invalid events.',
            'evidence': f'Summarize conflicting fictional observations from the {name} archive, separating observation from inference. Note A: at 08:10 the east cabinet appeared unlocked; the writer did not test its latch. Note B: at 08:15 a caretaker says the east cabinet was locked after cleaning but records no locking time. Note C: at 08:20 an inventory count finds all twelve copper seals present. Note D: a clock near the cabinet is known to run four minutes fast, but none of the notes names the clock used. Explain what can and cannot be concluded, list unresolved questions and suggest records that could resolve them. Do not accuse anyone or invent missing facts.'
        }
        for family, prompt in cases.items():
            # Long explanations keep the fixed128-token measurement inside the answer.
            (directory/f'{family}-{i}.txt').write_text(prompt+' Write at least 220 words.\n', encoding='utf-8')
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob('*.txt'))}
    assert len(manifest) == 24
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    import sys
    write(Path(sys.argv[1]))
