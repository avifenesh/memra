"""Second untouched six-family synthetic test, generated after dense fitting."""
from pathlib import Path
import hashlib
import json

def write(directory):
    directory.mkdir(exist_ok=False)
    for i,name in enumerate(['Fern','Grove','Hazel','Iris']):
        cases={
            'chronology':f'Reconcile this fictional {name} observatory chronology. Local noon is12:00. The north sensor reports a flash at11:{40+i:02d} on a clock known to run seven minutes fast. The south sensor reports a flash at11:{35+i:02d} on a clock known to run two minutes slow. A courier reports hearing a bell three minutes after the north flash but gives no clock reading. Convert observations to the reference clock, explain which could refer to the same event, and separate certain ordering from unsupported assumptions.',
            'conversion':f'Explain a conversion worksheet for the fictional {name} craft guild. One plume equals four reed-lengths; one reed-length equals three pebble-lengths. A frame needs{5+i} plumes of trim plus{7+i} reed-lengths. Stock is{90+i*10} pebble-lengths. Convert each quantity into pebble-lengths, determine surplus or deficit, and show two independent checks. These are invented units; do not substitute real physical units.',
            'grammar':f'Parse messages in the fictional {name} protocol. Grammar: a message is OPEN, then one or more ITEM tokens, then optionally CHECK, then CLOSE. No other tokens are allowed. Assess these sequences independently: OPEN ITEM CLOSE; OPEN CLOSE; OPEN ITEM CHECK ITEM CLOSE; OPEN ITEM ITEM CHECK CLOSE; ITEM OPEN CLOSE; OPEN ITEM CHECK CLOSE. Identify the first invalid token or missing element for each rejected sequence, show accepted parses, and explain why a checksum marker cannot be followed by another item.',
            'taxonomy':f'Classify fictional objects in the {name} cabinet. Rule priority: an object with both wings and a lantern is a beacon; otherwise one with wheels is a rover; otherwise one with wings is a glider; everything else is a relic. Objects: saffron has wings and wheels; lilac has wheels and a lantern; ochre has wings and a lantern; teal has only a lantern; russet has wings, wheels and a lantern; indigo has no listed feature. Apply rules in order, explain every classification, and show why rule priority matters.',
            'comparison':f'Build a comparison and recommendation from these fictional {name} notebook specifications only. Alder has{64+i*16} pages, sewn binding, water-resistant cover and no index. Birch has96 pages, stapled binding, plain cover and a numbered index. Cedar has80 pages, sewn binding, plain cover and a numbered index. A field recorder values a water-resistant cover first, then sewn binding; an archivist requires an index and prefers sewn binding. Explain each recommendation and any compromises. Do not invent prices, durability measurements or missing features.',
            'reference':f'Resolve references in this fictional {name} workshop account. Mira placed a copper key beside a blue folder. Oren moved the folder to a shelf but left the key. Lina then put a silver key inside the folder. Mira picked up the key that had remained on the table and handed it to Oren. He placed it in a red box. The box was carried to the shelf without being opened. Track each object and person action, distinguish the two keys, and answer where each key is at the end. Explain every pronoun or definite description used in the account.'}
        for family,prompt in cases.items():
            (directory/f'{family}-{i}.txt').write_text(prompt+' Write at least220 words.\n',encoding='utf-8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob('*.txt'))}
    assert len(manifest)==24
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    import sys
    write(Path(sys.argv[1]))
