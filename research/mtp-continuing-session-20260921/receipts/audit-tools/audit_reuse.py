"""Reject cold fallback and verify the exact prefix carried between recorded turns."""
import csv
from pathlib import Path


def audit_reuse(root, turns_per_conversation=8):
    root = Path(root)
    with (root / 'turns.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if not rows or len(rows) % turns_per_conversation:
        raise ValueError('incomplete conversation')
    cached_total = input_total = 0
    previous_prompt = []
    previous_checkpoint = 0
    for index, row in enumerate(rows):
        turn = index + 1
        if int(row['turn']) != turn:
            raise ValueError('turn order changed')
        prompt = [int(x) for x in (root / f'turn-{turn}.prompt.ids').read_text().split()]
        output = [int(x) for x in (root / f'turn-{turn}.output.ids').read_text().split()]
        cached = int(row['cached_tokens'])
        processed = int(row['new_input_tokens'])
        checkpoint = int(row['checkpoint_tokens'])
        first = index % turns_per_conversation == 0
        if len(prompt) != int(row['prompt_tokens']) or len(output) != int(row['output_tokens']):
            raise ValueError('recorded token counts disagree with tapes')
        if cached < 0 or processed <= 0 or cached + processed != len(prompt):
            raise ValueError('cache accounting does not cover the prompt')
        if not cached < checkpoint < len(prompt):
            raise ValueError('checkpoint did not advance within the current prompt')
        if first:
            if cached or row['resumed'] != 'false':
                raise ValueError('independent conversation did not reset its cache')
        elif (cached == 0 or cached != previous_checkpoint or row['resumed'] != 'true'
              or prompt[:cached] != previous_prompt[:cached]):
            raise ValueError('continuing turn did not reuse its exact preceding checkpoint')
        previous_prompt, previous_checkpoint = prompt, checkpoint
        cached_total += cached
        input_total += len(prompt)
    return {'turns': len(rows), 'conversations': len(rows) // turns_per_conversation,
            'cached_tokens': cached_total, 'input_tokens': input_total,
            'new_input_tokens': input_total - cached_total,
            'cached_fraction': cached_total / input_total}
