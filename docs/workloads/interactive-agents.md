# Interactive agents

| | Recommended use |
|---|---|
| **Start with** | Qwen3.8 27B or Ornith 1.5 35B-A3B on a qualified Blackwell configuration |
| **API** | OpenAI Chat Completions, OpenAI Responses, or Anthropic Messages |
| **Prioritize** | Single-stream latency, tools, reasoning controls, and stable multi-turn caching |
| **Read next** | [Serving](../SERVING.md), [API surfaces](../API-SURFACES.md), [Cookbook](../COOKBOOK.md) |

Use the client dialect you already have. Keep model, reasoning, sampling, and cache behavior
explicit, and validate a real tool loop before calling a configuration ready for an agent.
