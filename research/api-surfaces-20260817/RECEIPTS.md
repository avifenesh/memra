# lane/api-surfaces — real-surface acceptance receipts (2026-08-17)

Two translation surfaces (`/v1/messages` Anthropic Messages, `/v1/responses` OpenAI
Responses) over the chat-completions core. Gates below are the engine law: the actual
client binaries against the actual server, never "loads = works".

Rig: RTX 5090 Laptop (local), serialized behind `flock /tmp/memra-5090.lock`.
Server: `target/release/memra-server` (branch lane/api-surfaces), booted as
`MEMRA_MODELS="m=<model>" MEMRA_ADDR=127.0.0.1:8390 MEMRA_API_KEY=test-key-123
MEMRA_CTX=40960 MEMRA_REQUEST_LEDGER=<jsonl> MEMRA_MODEL_METADATA=<toml>`.
Clients: `claude` CLI 2.1.233 (`ANTHROPIC_BASE_URL=http://127.0.0.1:8390
ANTHROPIC_AUTH_TOKEN=test-key-123 ANTHROPIC_MODEL=m MAX_THINKING_TOKENS=0`),
`codex` CLI 0.147.0 (custom provider, `wire_api="responses"`,
`base_url="http://127.0.0.1:8390/v1"`, `env_key`, `model="m"`,
`model_reasoning_effort="none"`).

## CLI gates (model `qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`)

`claude -p "Respond with exactly this string and nothing else: SURFACE-OK-7391" --model m`

    SURFACE-OK-7391

`claude -p "Call the Bash tool now with command 'echo probe'. ..." --model m --allowedTools "Bash"`

    The Bash tool returned the output: `probe`

(tool_use block emitted by the surface -> Claude Code executed `echo probe` ->
tool_result round-tripped back through translation -> model reported the output.)

`codex exec "reply with exactly: SURFACE-OK-7391"` (run on the earlier qwen35-4b boot,
same branch/server config)

    codex
    SURFACE-OK-7391

`codex exec "Call the exec_command tool now with cmd 'echo probe'. ..."`

     succeeded in 0ms:
    probe

    codex
    The command executed successfully. The output is "probe".

(function_call item -> codex ran the command -> function_call_output round-tripped.)

## Billing parity (ledger diff, same 16-token request on all three routes)

`memra.request-cost.v1` rows for byte-similar prompts (`max_tokens 16`, temp 0), one per
route, identical usage and identical cost:

    route=/v1/chat/completions usage 16/0/16 cost total 0.0000480
    route=/v1/messages         usage 16/0/16 cost total 0.0000480
    route=/v1/responses        usage 16/0/16 cost total 0.0000480

CLI-session rows (9B run): every request the two CLIs fired was metered and priced,
including tool-turn follow-ups and a prefix-cache discount row:

    /v1/messages  completed 200 prompt 21222 cached 0     completion 35 cost 0.0212920
    /v1/messages  completed 200 prompt 21231 cached 0     completion 41 cost 0.0213130
    /v1/messages  completed 200 prompt 21270 cached 18912 completion 34 cost 0.0118820
    /v1/responses completed 200 prompt 7678  cached 0     completion 28 cost 0.0077340
    /v1/responses completed 200 prompt 7760  cached 0     completion 12 cost 0.0077840

A curl-aborted `/v1/messages` stream landed `outcome=abandoned http_status=499` with the
partial prompt+1-token usage — the streaming receipt discipline holds on the new surface.

## Wire-capture findings (mock probe, exact grammars, logging server)

- Claude Code 2.1.233 sends: `thinking:{"type":"adaptive"}` unconditionally,
  `role:"system"` messages inside `messages`, `context_management`, `output_config`,
  `metadata.user_id`, cache_control everywhere, Bearer-only auth
  (ANTHROPIC_AUTH_TOKEN), `anthropic-version: 2023-06-01`, `?beta=true` query — and all
  27 of its tools as PLAIN custom tools (WebSearch included; no server tool types).
- Codex 0.147.0 sends: `store:false`, `stream:true`, `include:["reasoning.encrypted_content"]`,
  `prompt_cache_key=<session uuid>`, `tool_choice:"auto"`, `parallel_tool_calls:false`,
  flattened function tools PLUS `type:"web_search"` and `type:"namespace"` entries
  unconditionally — which is why non-function tools are dropped (logged), not refused.
- Model-compliance note (not a wire finding): qwen35-4b completed every round trip but
  would not follow sentinel/tool instructions inside 21k-token agentic prompts; the 9B
  passed all gates. Surface correctness was already model-independent by then (the mock
  gates + 4B round trips prove the wire; the 9B proves the full loop).
