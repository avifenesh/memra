# mtp10 SXC prompt pool — provenance by reference (raw text NOT banked here)

The 300-prompt owner pool (75 per pool: hermes/claude/codex/eigen) that extends the
mtp10 rank corpus is OWNER SESSION TEXT and stays out of the public repo — the
public-boundary gate caught a live cloud-provider resource identifier inside one
transcript line, which is exactly the class of surface this repo must not carry (mtp9's 48-prompt pool passed the
same gate only because its smaller sample happened to match no pattern).

Where the artifact actually lives:
- Raw pool: box `~/realgate/mtp10/sxc-prompts-300.tsv` (and reproducible on the rig:
  `spec/extract-sxc-prompts.py <sessions_root> out.tsv 300` — deterministic given the
  session store; filters + round-robin interleave documented in the script header).
- Rendered corpus (token ids, chat template on): box
  `~/realgate/mtp10/corpus-prompts-big.tsv`; per-generation resume ledger
  `corpus-ids-big.tsv`; both referenced by the owngen receipt banked here
  (`corpus/owngen-owngen-mtp10.tsv`, binary sha + counts + coverage table inside).
- Ranks sidecar: `ranks-owngen-big.txt.gz` (banked here — id/count pairs only).
