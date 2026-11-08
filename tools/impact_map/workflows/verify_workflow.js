export const meta = {
  name: 'memra-impact-map-verify',
  description: 'Adversarially verify narrowing claims and reported defects of the memra impact map',
  phases: [
    { title: 'Narrowing', detail: 'try to refute every claim that excludes a family, route, device or default' },
    { title: 'Defects', detail: 'try to refute every reported error, dead code and broken gate' },
  ],
}

const H = args.work  // the work dir (stage-0 fact base, slices)
const WT = H + '/wt'
const RULES = `Read-only: ${WT} is a git worktree at origin/main 5ba8a3b11a. Do not edit, build, run cargo or GPU work, or write files. Read with Read; search with rg/grep --threads 4 or fewer, inside ${WT} only. The stage-0 call graph is ${H}/stage0/graph/edges.tsv (src, dst, kind, count, call lines) with nodes in ${H}/stage0/graph/nodes.jsonl; per-arch CUDA line liveness is ${H}/stage0/liveness.json; the family/route/device tables are in ${H}/wave1/scouts.json.`

const NARROW_SCHEMA = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
  claim: { type: 'string' }, refuted: { type: 'boolean' }, reason: { type: 'string' },
  counter_path: { type: 'string' },
  evidence: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, line: { type: 'integer' }, quote: { type: 'string' } }, required: ['file', 'line', 'quote'] } } },
  required: ['claim', 'refuted', 'reason', 'evidence'] } } }, required: ['verdicts'] }

const DEFECT_SCHEMA = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, verdict: { type: 'string', enum: ['confirmed', 'refuted', 'unsure'] }, severity: { type: 'string', enum: ['high', 'medium', 'low'] },
  reason: { type: 'string' }, fix: { type: 'string' },
  evidence: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, line: { type: 'integer' }, quote: { type: 'string' } }, required: ['file', 'line', 'quote'] } } },
  required: ['id', 'verdict', 'severity', 'reason', 'evidence'] } } }, required: ['verdicts'] }

phase('Narrowing')
const narrow = parallel(args.narrow.map((batchFile, i) => () => agent(`${RULES}

You are a skeptic checking an impact map of the memra inference engine. A push gate will SKIP testing a model family, serving route or device whenever a claim below says that code cannot reach it. A wrong exclusion lets a regression ship, so your job is to REFUTE claims.

How the map uses a region: the region's FIRST line (usually the branch condition) is never narrowed, because a changed condition moves every arm; only the lines strictly after it are. So judge the lines after the first line. A claim with "earlier_refutation" was refuted before in a wider form; it now claims less (read its "check").

Each claim names a code region (file, lines), the families/routes/devices that the annotator says can execute it (["*"] = all), optionally a door (env var; default_reached=false means "a naked default launch never executes this region"), the predicate and the basis. For EACH claim:
- Read the region and its predicate, and the callers of the enclosing function (edges.tsv, rg).
- Try to find a concrete way an EXCLUDED family, route or device executes the region: another caller, a shared helper, a predicate that is true for more families than claimed (check the family table), an env var whose default actually takes the arm, a device check that also admits other cards (GB10 builds as 120a today), a trait object or function pointer.
- refuted=true when you find such a path, AND ALSO when you cannot establish the claim from the code within reasonable effort (uncertain = refuted). refuted=false only when the code shows the exclusion holds. Give counter_path when refuted.
- A claim with kind "output-neutral" says changing that item can never change output bytes, timing, memory or serving decisions (logging, metrics, receipts only): refute it if any of its values feed such a decision, or if its side effects (I/O, locks, allocation, panics) can change behavior.
- Every verdict cites evidence {file, line, quote} (quote = exact substring of that line).
Your claims are in ${batchFile} (Read it first; a claim with trace_hint has a concrete counterexample from a real kernel trace: an executed kernel the claim's sets exclude, so check it first). Return one verdict per claim id in that file, none skipped.`, { label: `narrow:${batchFile.split('/').pop()}`, phase: 'Narrowing', schema: NARROW_SCHEMA, effort: 'high' })))

phase('Defects')
const defects = parallel(args.defects.map((batchFile, i) => () => agent(`${RULES}

You are a skeptic checking defect reports about the memra engine. Each item claims a bug, dead code, an unreachable path, a stale doc or flag, or a broken gate. For EACH item, try to REFUTE it: read the cited code and its context; for dead code, search every caller (other crates, bins, tests, macros, string names, pub re-exports); for a bug, check whether the failure scenario can actually happen (guards upstream, invariants, callers); for a stale flag or doc, compare with the current code and docs/FLAGS.md.
verdict: confirmed (the defect is real as described, with evidence), refuted (it is not), unsure (cannot tell). Adjust severity to what the evidence supports (high = wrong output, crash, data loss or a gate that passes broken code; medium = wrong behavior in a narrower case, misleading gate, real maintenance hazard; low = cosmetic, stale text). Give a one-line fix when confirmed. Cite evidence {file, line, quote}.
Your items are in ${batchFile} (Read it first). Return one verdict per item id in that file, none skipped.`, { label: `defects:${batchFile.split('/').pop()}`, phase: 'Defects', schema: DEFECT_SCHEMA, effort: 'high' })))

const [n, d] = await Promise.all([narrow, defects])
return { narrow: n, defects: d }
