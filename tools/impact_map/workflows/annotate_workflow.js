export const meta = {
  name: 'memra-impact-map-annotate',
  description: 'Scout model/route/device tables, then annotate 22 memra slices with predicate regions, entries, effects and findings',
  phases: [
    { title: 'Scout', detail: 'packs truth table, routes and entries, device predicates' },
    { title: 'Annotate', detail: 'one agent per slice, 22 slices' },
  ],
}

const H = args.work  // the work dir (stage-0 fact base, slices)
const WT = H + '/wt'

const COMMON = `You are one agent in a bounded, read-only analysis of the memra inference engine (Rust host + CUDA kernels).
Repository checkout (origin/main 5ba8a3b11a): ${WT}. It is a git worktree; DO NOT edit, build, run cargo, run GPU work, or write files anywhere. Read with the Read tool, search with rg/grep using --threads 4 or fewer (never search outside ${WT} and ${H}).

GOAL. The owner wants qualification to measure only the delta: for any change, which models (families), serving routes, devices and formats can change output or speed. Your labels feed a machine-readable impact map that a push gate consumes. Accuracy rules:
1. Every claim cites evidence: {file, line, quote} where file is repo-relative, line is 1-based, and quote is an EXACT substring of that line (trimmed, max 160 chars). A validator script checks every quote against the file; claims whose quotes fail are dropped.
2. Asymmetry: claiming that something does NOT reach a family/route/device ("narrowing") is the dangerous direction; a wrong exclusion lets a regression ship. Narrow only on evidence you read (a predicate, a cfg, a type that only one loader constructs). When unsure, use ["*"] and set conservative=true. Over-inclusion only costs a test run.
3. Do not restate mechanical facts already in the fact base; add what a script cannot decide: what predicates mean, which families/routes/devices take each arm, effect kinds, defects.
4. Use only the vocabulary ids given (families, routes, devices, formats, effects). ["*"] means all.

STAGE-0 FACT BASE (mechanical, validated):
- ${H}/stage0/graph/nodes.jsonl: one JSON per item (Rust items from a token scanner, cross-checked against a rust-analyzer SCIP index: 16,861/16,861 functions; CUDA functions from a C++ scanner). Fields: id, file, kind, qual, line_start, line_end, cfg, test, hash, idents, strings (Rust), quals/extern_c/launches (CUDA).
- ${H}/stage0/graph/edges.tsv: src, dst, kind (ref = SCIP reference, trait = trait method to every impl, name = name-based fallback for cfg'd-out code, kernel = Rust string literal naming a __global__ kernel, ffi = Rust extern decl to CUDA extern C, launch = CUDA host launch, cucall = CUDA name-based call), count, first line.
- ${H}/stage0/kernels.tsv: every __global__ kernel with its file and its Rust/CUDA launch sites.
- ${H}/stage0/liveness.json: per build variant (120a, 120a-MEMRA_CUTLASS, 120a-MEMRA_DSV4_CUTLASS, 100a, 90a, 89) and per CUDA unit, the exact source line ranges the device and host preprocessor passes keep (from the exact nvcc calls build.rs makes). A CUDA line not live for a variant cannot affect that build.
- ${H}/stage0/vocab.json: vocabulary.
- Per slice: ${H}/slices/<SID>/files.txt (file, first line, last line of your slice), facts.tsv (every non-test item: lines, size, fan-in/out, kernels launched, MEMRA_* env names, top callers and callees outside the slice), hints.tsv (lines matching arch/env/family/route/format/plan patterns; navigation only), candidates.json (mechanical findings candidates in your slice: big_fn, dead_candidate, near_duplicate, exact_duplicate, stale_flag, arch_check_scatter, stale_doc).
Owner rules that matter here are in ${WT}/AGENTS.md (doors, per-device defaults, one numeric program per request, evidence).`

const EVID = { type: 'array', minItems: 1, items: { type: 'object', properties: { file: { type: 'string' }, line: { type: 'integer' }, quote: { type: 'string' } }, required: ['file', 'line', 'quote'] } }
const SLIST = { type: 'array', items: { type: 'string' } }

const PACKS_SCHEMA = { type: 'object', properties: {
  families: { type: 'array', items: { type: 'object', properties: {
    family: { type: 'string' }, aliases: SLIST, engine_types: SLIST,
    load_entry: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, what: { type: 'string' }, evidence: EVID }, required: ['item', 'what', 'evidence'] } },
    traits: SLIST, formats: SLIST, kv_formats: SLIST, routes: SLIST, devices_measured: SLIST, support_state: { type: 'string' },
    models: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, artifact: { type: 'string' }, formats: SLIST, routes: SLIST, devices: SLIST, source: { type: 'string' } }, required: ['id', 'formats', 'routes', 'devices', 'source'] } },
    notes: { type: 'string' }, evidence: EVID }, required: ['family', 'aliases', 'engine_types', 'load_entry', 'traits', 'formats', 'routes', 'support_state', 'models', 'evidence'] } },
  extra_families: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, what: { type: 'string' }, evidence: EVID }, required: ['id', 'what', 'evidence'] } },
  predicates: { type: 'array', items: { type: 'object', properties: { predicate: { type: 'string' }, meaning: { type: 'string' }, true_for: SLIST, false_for: SLIST, evidence: EVID }, required: ['predicate', 'meaning', 'true_for', 'false_for', 'evidence'] } },
  findings: { type: 'array', items: { type: 'object', properties: { category: { type: 'string' }, severity: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' }, quote: { type: 'string' }, title: { type: 'string' }, fix: { type: 'string' } }, required: ['category', 'severity', 'file', 'line', 'quote', 'title', 'fix'] } },
}, required: ['families', 'extra_families', 'predicates', 'findings'] }

const ROUTES_SCHEMA = { type: 'object', properties: {
  routes_vocab: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, meaning: { type: 'string' } }, required: ['id', 'meaning'] } },
  routes: { type: 'array', items: { type: 'object', properties: {
    route: { type: 'string' }, selected_where: EVID, selection_predicate: { type: 'string' }, families: SLIST, devices: SLIST,
    entry_items: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, role: { type: 'string' }, evidence: EVID }, required: ['item', 'role', 'evidence'] } },
    call_chain: { type: 'string' }, default_on: { type: 'boolean' }, door: { type: 'string' }, notes: { type: 'string' } }, required: ['route', 'selected_where', 'selection_predicate', 'families', 'entry_items', 'call_chain', 'default_on'] } },
  loader_dispatch: { type: 'array', items: { type: 'object', properties: { families: SLIST, engine_type: { type: 'string' }, constructor_item: { type: 'string' }, predicate: { type: 'string' }, evidence: EVID }, required: ['families', 'engine_type', 'constructor_item', 'predicate', 'evidence'] } },
  binaries: { type: 'array', items: { type: 'object', properties: { binary: { type: 'string' }, main_item: { type: 'string' }, serves: { type: 'string' }, evidence: EVID }, required: ['binary', 'main_item', 'serves', 'evidence'] } },
  findings: { type: 'array', items: { type: 'object', properties: { category: { type: 'string' }, severity: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' }, quote: { type: 'string' }, title: { type: 'string' }, fix: { type: 'string' } }, required: ['category', 'severity', 'file', 'line', 'quote', 'title', 'fix'] } },
}, required: ['routes_vocab', 'routes', 'loader_dispatch', 'binaries', 'findings'] }

const DEVICE_SCHEMA = { type: 'object', properties: {
  device_predicates: { type: 'array', items: { type: 'object', properties: {
    item: { type: 'string' }, predicate: { type: 'string' },
    truth: { type: 'object', properties: { rtx5090: { type: 'string' }, rtxpro6000: { type: 'string' }, gb10: { type: 'string' }, b200: { type: 'string' }, h100: { type: 'string' }, sm89: { type: 'string' }, cpu: { type: 'string' } }, required: ['rtx5090', 'rtxpro6000', 'gb10', 'b200', 'h100', 'sm89', 'cpu'] },
    used_by_count: { type: 'integer' }, evidence: EVID }, required: ['item', 'predicate', 'truth', 'evidence'] } },
  device_defaults: { type: 'array', items: { type: 'object', properties: {
    file: { type: 'string' }, line: { type: 'integer' }, setting: { type: 'string' }, per_device: { type: 'string' }, env_seam: { type: 'string' }, evidence: EVID }, required: ['file', 'line', 'setting', 'per_device', 'evidence'] } },
  build_cfgs: { type: 'array', items: { type: 'object', properties: { cfg: { type: 'string' }, set_for: SLIST, meaning: { type: 'string' }, evidence: EVID }, required: ['cfg', 'set_for', 'meaning', 'evidence'] } },
  findings: { type: 'array', items: { type: 'object', properties: { category: { type: 'string' }, severity: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' }, quote: { type: 'string' }, title: { type: 'string' }, fix: { type: 'string' } }, required: ['category', 'severity', 'file', 'line', 'quote', 'title', 'fix'] } },
}, required: ['device_predicates', 'device_defaults', 'build_cfgs', 'findings'] }

const REGION = { type: 'object', properties: {
  file: { type: 'string' }, line_start: { type: 'integer' }, line_end: { type: 'integer' }, item: { type: 'string' },
  label: { type: 'string' }, predicate: { type: 'string' },
  predicate_kind: { type: 'string', enum: ['family', 'route', 'device', 'format', 'env_door', 'capability', 'config', 'shape', 'other', 'none'] },
  families: SLIST, routes: SLIST, devices: SLIST, formats: SLIST,
  door: { type: 'object', properties: { env: { type: 'string' }, default_reached: { type: 'boolean' } }, required: ['env', 'default_reached'] },
  effects: { type: 'array', items: { type: 'string', enum: ['numeric', 'speed', 'memory', 'policy', 'observability', 'none'] } },
  narrowing_basis: { type: 'string' }, conservative: { type: 'boolean' }, evidence: EVID,
}, required: ['file', 'line_start', 'line_end', 'label', 'predicate_kind', 'families', 'routes', 'devices', 'formats', 'effects', 'conservative', 'evidence'] }

const ANNOTATE_SCHEMA = { type: 'object', properties: {
  slice: { type: 'string' },
  file_defaults: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, families: SLIST, routes: SLIST, devices: SLIST, formats: SLIST, effects: SLIST, note: { type: 'string' }, narrowing_basis: { type: 'string' }, evidence: EVID }, required: ['file', 'families', 'routes', 'devices', 'formats', 'effects', 'note', 'evidence'] } },
  regions: { type: 'array', items: REGION },
  entries: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, families: SLIST, routes: SLIST, devices: SLIST, role: { type: 'string' }, evidence: EVID }, required: ['item', 'families', 'routes', 'role', 'evidence'] } },
  exact_items: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, why: { type: 'string' }, evidence: EVID }, required: ['item', 'why', 'evidence'] } },
  dynamic_edges: { type: 'array', items: { type: 'object', properties: { from_item: { type: 'string' }, to: { type: 'string' }, mechanism: { type: 'string' }, evidence: EVID }, required: ['from_item', 'to', 'mechanism', 'evidence'] } },
  candidate_verdicts: { type: 'array', items: { type: 'object', properties: { kind: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' }, verdict: { type: 'string', enum: ['confirmed', 'rejected', 'unsure'] }, note: { type: 'string' } }, required: ['kind', 'file', 'line', 'verdict', 'note'] } },
  findings: { type: 'array', items: { type: 'object', properties: {
    category: { type: 'string', enum: ['mergeable', 'reusable', 'decoupling', 'error', 'dead_code', 'unreachable', 'stale_doc', 'stale_flag', 'broken_gate'] },
    severity: { type: 'string', enum: ['high', 'medium', 'low'] }, file: { type: 'string' }, line: { type: 'integer' }, quote: { type: 'string' },
    title: { type: 'string' }, detail: { type: 'string' }, fix: { type: 'string' }, impact_cut: { type: 'string' }, related: SLIST }, required: ['category', 'severity', 'file', 'line', 'quote', 'title', 'detail', 'fix'] } },
  writeup_md: { type: 'string' },
}, required: ['slice', 'file_defaults', 'regions', 'entries', 'exact_items', 'dynamic_edges', 'candidate_verdicts', 'findings', 'writeup_md'] }

const SCOUTS = [
  { key: 'packs', schema: PACKS_SCHEMA, prompt: `TASK: build the model-family truth table.
Read crates/memra-gguf/src/model_packs/*/ (every pack: aliases, config normalization, tensor schema, plan construction, gate manifest, support field), crates/memra-gguf/src/{config.rs,model_plan.rs,execution_manifest.rs,lib.rs} as needed, docs/MODELS.md, docs/models/, docs/support-records.toml, tools/release-roster.tsv, research/tune-data/current-board.json.
For each pack (family id = pack directory name): aliases, the engine model type(s) that load it (find the loader call site: which Rust type/constructor serves this family; cite it), traits (moe, mtp_head, gdn, kda, mla, dsa, swa, gated_attention, shared_expert, vision, audio_asr, audio_tts, encoder_decoder, dense, ...), weight formats of the artifacts the docs and roster actually run, KV formats, routes the family supports (use the route vocab in vocab.json), devices it has been measured on, support_state, and concrete models (id, artifact/quant, formats, routes, devices, source citation).
Also list extra_families: engine code serving a family no pack declares (id prefixed 'x:').
Then predicates: the config fields, plan operations and capability checks that runtime code branches on to tell families apart (for example n_expert > 0, has MTP tensors, attention kind, plan.has_op(...)), each with the families it is true and false for. Engine slices will use this table to turn a predicate into a family set, so be precise and cite where each family's value comes from.
findings: only defects you meet on the way (stale docs, support-state mismatches, packs that claim routes the code lacks).` },
  { key: 'routes', schema: ROUTES_SCHEMA, prompt: `TASK: build the route and entry-point table.
Trace how a request reaches the engine for each route: start at the binaries (crates/memra-server/src/main.rs or bin, memra-cli, engine bins that serve), the HTTP layer (crates/memra-server/src/lib.rs), the worker (crates/memra-server/src/worker.rs: run, admit, step_session and their dispatch), and the engine entry points (crates/memra-engine/src/{model.rs,hybrid.rs,decode.rs,decode_batch.rs,spec.rs,dflash.rs,glm_spec.rs,gemma_spec.rs,tp.rs,pp.rs,vision*.rs,...}). Use facts.tsv of slices S17, S18, S19, S05, S10, S11, S09 to find items.
Output routes_vocab (refine vocab.json routes: keep ids stable, add any route the code has that is missing, drop none silently; say what each means), and per route: where it is selected (cite), the selection predicate, families that can take it, entry items (stage-0 ids from nodes.jsonl), the call chain in one line, default_on (is it reached by a naked default launch), and the env door if any.
loader_dispatch: how a model file becomes an engine type (which predicate picks HybridModel, the DSv4 model, qwen4exp, glm5 TP, mimo, gemma, whisper, tts, ...), with the families each constructor serves.
binaries: each served/gate binary and its main item.
findings: only defects you meet on the way.` },
  { key: 'devices', schema: DEVICE_SCHEMA, prompt: `TASK: build the device-predicate table.
Find every runtime device/arch check and device-keyed default in crates/*/src (not tests): functions and consts that read the built arch (MEMRA_BUILT_CUDA_ARCH, cuda_arch), compute capability, SM count, VRAM, device name or card class, and the cfg flags build.rs sets (memra_portable_cuda, memra_hopper_mma, memra_sm100_tcgen05, memra_cutlass; read crates/memra-engine/build.rs). The hints.tsv files under ${H}/slices/*/ list candidate lines with category 'arch'; ${H}/slices/*/candidates.json has arch_check_scatter counts.
device_predicates: each helper or inline check, with its truth value per device: rtx5090 (sm_120, 82 SMs laptop / 170 SMs desktop, 24-32 GB), rtxpro6000 (sm_120, 188 SMs, 96 GB), gb10 (sm_121, 48 SMs, unified 128 GB; built as 121a only on PR #1025, today it is not an accepted MEMRA_CUDA_ARCH), b200 (sm_100), h100 (sm_90), sm89 (Ada portable build), cpu. Verify SM counts and VRAM thresholds from the code's own constants and comments; do not assume.
device_defaults: every place a default differs by device (setting, value per device, the env seam that overrides it).
build_cfgs: each cfg, which builds set it.
findings: device checks scattered across files that should be one helper (decoupling), checks that disagree with each other, dead device arms, docs that disagree with code.` },
]

const SLICES = args.slices

phase('Scout')
const scouts = await parallel(SCOUTS.map(s => () => agent(`${COMMON}\n\n${s.prompt}`, { label: `scout:${s.key}`, phase: 'Scout', schema: s.schema, effort: 'high' })))
const [packs, routes, devices] = scouts
log(`scouts: packs=${packs ? packs.families.length : 'FAILED'} routes=${routes ? routes.routes.length : 'FAILED'} device_predicates=${devices ? devices.device_predicates.length : 'FAILED'}`)

const compact = (o, n) => { const s = JSON.stringify(o); return s.length > n ? s.slice(0, n) + '...[truncated]' : s }
const packTable = packs ? compact({ families: packs.families.map(f => ({ family: f.family, aliases: f.aliases, engine_types: f.engine_types, traits: f.traits, formats: f.formats, kv_formats: f.kv_formats, routes: f.routes, models: f.models.map(m => ({ id: m.id, formats: m.formats, routes: m.routes, devices: m.devices })) })), extra_families: packs.extra_families.map(x => x.id + ': ' + x.what), predicates: packs.predicates.map(p => ({ predicate: p.predicate, meaning: p.meaning, true_for: p.true_for, false_for: p.false_for })) }, 30000) : 'scout failed: derive family membership yourself from crates/memra-gguf/src/model_packs/'
const routeTable = routes ? compact({ routes_vocab: routes.routes_vocab, routes: routes.routes.map(r => ({ route: r.route, selection_predicate: r.selection_predicate, families: r.families, entry_items: r.entry_items.map(e => e.item + ' (' + e.role + ')'), call_chain: r.call_chain, default_on: r.default_on, door: r.door })), loader_dispatch: routes.loader_dispatch.map(l => ({ families: l.families, engine_type: l.engine_type, constructor_item: l.constructor_item, predicate: l.predicate })) }, 30000) : 'scout failed: derive routes yourself'
const deviceTable = devices ? compact({ device_predicates: devices.device_predicates.map(d => ({ item: d.item, predicate: d.predicate, truth: d.truth })), device_defaults: devices.device_defaults.map(d => ({ at: d.file + ':' + d.line, setting: d.setting, per_device: d.per_device, env_seam: d.env_seam })), build_cfgs: devices.build_cfgs.map(c => ({ cfg: c.cfg, set_for: c.set_for })) }, 20000) : 'scout failed: derive device truth yourself'

phase('Annotate')
const annotated = await parallel(SLICES.map(s => () => agent(`${COMMON}

YOUR SLICE: ${s.id} — ${s.title}. Its exact file ranges are in ${H}/slices/${s.id}/files.txt; read facts.tsv, hints.tsv and candidates.json there first, then read the code (targeted reads by line range; files can be tens of thousands of lines).

SHARED TABLES FROM THE SCOUT PHASE (authoritative vocabulary and family/route/device truth; use them to turn a predicate into sets):
FAMILY TABLE: ${packTable}
ROUTE TABLE: ${routeTable}
DEVICE TABLE: ${deviceTable}

WHAT TO RETURN (schema enforced):
- file_defaults: for each file in your slice, the families/routes/devices/formats/effects any change in it can reach when nothing narrower applies, with the basis. A file named for one family is NOT automatically family-local: check the callers outside the slice (facts.tsv) before narrowing.
- regions: every branch point in your slice that narrows reach by family, route, device, format, env door, capability or config, as a line range (the arm, not just the condition line) with the sets that take that arm and its effect kinds. Include env doors: door.default_reached says whether a naked default launch can take the arm (code reached only behind a default-OFF door affects no default cell). Also label whole items whose reach is narrower than their file default. Large functions (worker::run, admit, generate_spec_inner2, load_from_source_impl, kernel launchers) need their internal arms labeled; that is where most of the precision comes from. Effects: numeric if the region can change output bytes, speed if only time, memory, policy, observability, none.
- entries: items that start a route for a family (forward/decode/verify/prime entry points).
- exact_items: items whose changes are provably output-neutral (pure bookkeeping, logging, metrics), with why.
- dynamic_edges: calls the static graph cannot see (function pointers, trait objects built from config, string-keyed kernel names computed at runtime, env-selected module loads, macro-generated kernels), from which item to what.
- candidate_verdicts: a verdict for EVERY entry of your candidates.json except near_duplicate/dead_candidate lists longer than 60, where you judge the 60 largest and say so in the note of one extra verdict with kind 'coverage'. dead_candidate: confirmed only if nothing calls it (check pub re-exports, macros, string names, other crates, bins, tests are not callers); stale_flag: confirmed if the door still exists past its decide-by date with no receipt.
- findings: the owner asked to mark along the way: mergeable (near-copies that can become one), reusable (pieces other parts reimplement), decoupling (a component that drags unrelated models/routes/devices into its blast radius: say the cut and which cells it removes from the impact set in impact_cut), and errors (real bugs, dead code, unreachable paths, stale docs and flags, broken gates). Each with file:line, quote, severity, a one-line fix. Fix nothing. Real bugs need a concrete failure scenario in detail; do not report style.
- writeup_md: a subsystem write-up for docs/impact-map (400-900 words): what the slice does, its hot paths per family and route, device-specific parts, effect kinds, where the impact set is wide and why, the top findings. Cite file:line.
Be thorough and specific; the slice is large, so prioritize branch points on hot paths and the largest items.`, { label: `annotate:${s.id}`, phase: 'Annotate', schema: ANNOTATE_SCHEMA, effort: 'high' })))

return { scouts: { packs, routes, devices }, annotated: annotated.map((a, i) => a ? a : { slice: SLICES[i].id, failed: true }) }
