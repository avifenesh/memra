---
library_name: transformers
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/LICENSE
pipeline_tag: text-generation
---


<img width="600px" src="assets/ornith_logo.png">


# Ornith-1.5-9B


Chirp Chirp! 🐦 We are introducing Ornith-1.5, a major step toward building foundation models through end-to-end self-improvement.

Ornith-1.5 extends Ornith-1.0 (which was developed on top of Qwen3.5 and Gemma4 with additional continued pretraining, mid-training, and post-training) by expanding the self-improvement loop from scaffold and rollout optimization to jointly optimizing task generation, scaffold construction, and solution rollouts. Rather than relying on a fixed set of human-curated tasks and manually designed harnesses, Ornith-1.5 continuously generates new training tasks, discovers effective strategies for solving them, and improves the policy through reinforcement learning. For more details on the task, harness, and rollout reward design, please refer to our [blog](https://ornith.ai/ornith_1_5.html).
<img style="width: 100%; max-width: 900px;" src="assets/ornith_9b_eval.png" alt="Ornith 1.5 9B Benchmark Results" title="Ornith 1.5 9B Benchmark Results">

## Ornith 1.5 9B

This model card documents **Ornith-1.5-9B**, the most lightweight member of the Ornith-1.5 family — a 9B dense model designed for efficient single-GPU deployment, and edge-deployable on mobile devices via its quantized Ornith-1.5-9B-Mobile variant.

### Benchmarks


<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;width:120%;margin:0 auto;padding:16px 0">
<table style="width:120%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr>
<th style="width:28%;padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #FD8E5B;color:#FD8E5B"></th>
<th style="width:14.40%;padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #FD8E5B;color:#FD8E5B;font-size:14px;background:rgba(253, 142, 91, 0.12)">Ornith-1.5-9B</th>
<th style="width:14.40%;padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #FD8E5B;color:#FD8E5B;font-size:14px">Ornith-1.0-9B</th>
<th style="width:14.40%;padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #FD8E5B;color:#FD8E5B;font-size:14px">Qwen3.5-9B</th>
<th style="width:14.40%;padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #FD8E5B;color:#FD8E5B;font-size:14px">Qwen3.6-35B-A3B</th>
<th style="width:14.40%;padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #FD8E5B;color:#FD8E5B;font-size:14px">Gemma-4-31B <sub></sub></th>
</tr></thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#FD8E5B;border-bottom:1px solid rgba(253, 142, 91, 0.2);background:rgba(253, 142, 91, 0.1)">Coding</td></tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">Terminal-Bench 2.1 <sub><small>(Terminus-2)</small></sub></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">46.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">43.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">21.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">52.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">42.1</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">Terminal-Bench 2.1 <sub><small>(Claude Code)</small></sub></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">47</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">40.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">18.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">49.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">SWE-bench Verified</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">70.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">69.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">53.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">73.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">52</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">SWE-bench Pro</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">47.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">42.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">31.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">49.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">35.7</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">SWE-bench Multilingual</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">54.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">52</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">39.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">67.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">51.7</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">NL2Repo</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">32.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">27.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">16.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">29.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">15.5</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">SWE Atlas - QnA</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">20.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">17.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">9.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">15.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#FD8E5B;border-bottom:1px solid rgba(253, 142, 91, 0.2);background:rgba(253, 142, 91, 0.1)">Reasoning</td></tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">HLE <sub><small>(no tools)</small></sub></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">20.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">16.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">14.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">21.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">19.5</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">HLE <sub><small>(with tools)</small></sub></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">30.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">26.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">24.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">28.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">26.5</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">GPQA Diamond</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">86.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">82.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">81.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">86</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">84.3</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#FD8E5B;border-bottom:1px solid rgba(253, 142, 91, 0.2);background:rgba(253, 142, 91, 0.1)">Agentic</td></tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">MCP-Atlas</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">54.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">49.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">46.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">62.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">55</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">Toolathlon-Verified</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">41.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">33.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">29.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">41.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">52.8</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">WideSearch</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">59.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">55.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">53.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">60.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">54.2</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">BrowseComp</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">56.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">44.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">41.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">62</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-</td>
</tr>
<tr>
<td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);">ClawEval</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);font-weight:600;color:#FD8E5B;background:rgba(253, 142, 91, 0.06)">66.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">63.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">53.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">68.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">48.5</td>
</tr>
</tbody>
</table>

<p style="margin-top:12px;font-size:10px;opacity:0.7">
* All results reported for Ornith-1.5 are averaged over five independent runs.<br/>
* Terminal-Bench 2.1 (Terminus-2): We evaluate Terminal-Bench 2.1 using the Harbor/Terminus-2 framework with parser=json, temperature=1.0, top_p=1.0, and a 128K context window. Each run uses a 4-hour timeout with 32 CPU cores and 48GB RAM, and results are averaged over 5 runs. We adjust the Qwen chat template to ensure consistency between training and inference (https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/chat_template.jinja), and modify Harbor to align with vLLM's reasoning_content key.<br/>
* Terminal-Bench 2.1 (Claude Code): We evaluate Terminal-Bench 2.1 using Claude Code 2.1.126 with parser=json, temperature=1.0, top_p=1.0, max_new_tokens=131072. Results are averaged over 5 runs. Again, Qwen chat template needs to be modified.<br/>
* SWE-Bench Verified, Pro and Multilingual: using OpenHands harness with temp=1.0, top_p=0.95, 256k context window. Anti-hacking safeguards are applied throughout evaluation: Git history is removed from the local repository image to prevent access to prior solutions or commits; network access is disabled, preventing the model from retrieving external information or resources.<br/>
* DeepSWE: Evaluated using the Claude Code harness with temperature=1.0, top_p=0.95, and a 256K context window.<br/>
* SWE Atlas QnA: using mini SWE agent harness with temp=1.0, top_p=0.95, 128K context window. Results are averaged over 5 runs.<br/>
* NL2Repo: with temperature=1.0, top_p=1.0, 400K context, 48K output. Access to specified GitHub repositories and pip packages is blocked to prevent reward hacking.<br/>
* HLE: Evaluated using Claude 4.6 Opus as the judge model.<br/>
* MCP-Atlas: All models were evaluated in thinking mode on the 500-task public subset, with a 10-minute timeout per task. We use Claude 4.8 Opus as the judge model.<br/>
* Toolathlon-Verified: We use the official evaluation service with the maximum token limit set to 128K.<br/>
* ClawEval: An agentic code benchmark over real-user task distributions; temp=0.6 and 256K context.<br/>
</p>

</div>


## Quickstart

<div style="border-left:4px solid #FD8E5B;background:rgba(253,142,91,0.1);border-radius:6px;padding:12px 16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.6">
<div style="font-weight:700;color:#FD8E5B;margin-bottom:6px">📝 NOTE</div>
<p style="margin:0 0 10px"><b>Ornith-1.5-9B</b> is a <b>reasoning model</b>: by default the assistant turn opens with a <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">&lt;think&gt; … &lt;/think&gt;</code> block before the final answer. The serving recipes below enable a reasoning parser so the chain-of-thought is returned in a separate <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">reasoning_content</code> field, and a tool-call parser so the model's <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">&lt;tool_call&gt;</code> blocks are surfaced as OpenAI-style <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">tool_calls</code>.</p>
<p style="margin:0 0 6px">Serving Ornith-1.5-9B requires recent runtimes:</p>
<ul style="margin:0 0 10px;padding-left:20px">
<li><b>Transformers</b> ≥ 5.8.1</li>
<li><b>vLLM</b> ≥ 0.19.1</li>
<li><b>SGLang</b> ≥ 0.5.9</li>
</ul>
<p style="margin:0 0 6px">Recommended sampling parameters:</p>
<ul style="margin:0;padding-left:20px">
<li><b>For general tasks:</b> <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">temperature=1.0</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">top_p=0.95</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">top_k=20</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">min_p=0.0</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">presence_penalty=1.5</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">repetition_penalty=1.0</code></li>
<li><b>For precise coding tasks:</b> <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">temperature=0.6</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">top_p=0.95</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">top_k=20</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">min_p=0.0</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">presence_penalty=0.0</code>, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">repetition_penalty=1.0</code></li>
</ul>
</div>

### Serving Ornith-1.5-9B

Ornith-1.5-9B is a dense ~9B model (≈19 GB in bf16), so it serves on a **single 80GB GPU**. The recipes below stand up an OpenAI-compatible server; add `--tensor-parallel-size` / `--tp` if you want to shard across more GPUs.

*  vLLM

```bash
vllm serve ornith-ai/Ornith-1.5-9B --served-model-name Ornith-1.5-9B --host 0.0.0.0 --port 8000 --max-model-len 262144 --gpu-memory-utilization 0.90 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --trust-remote-code
```

*  SGLang


```bash
python -m sglang.launch_server --model-path ornith-ai/Ornith-1.5-9B --served-model-name Ornith-1.5-9B --host 0.0.0.0 --port 8000 --context-length 262144 --mem-fraction-static 0.85 --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

#### For Long-Context

Ornith-1.5-9B handles context windows of up to 262,144 tokens. When a task's combined input and output must go beyond this limit, we suggest extending the effective window with RoPE scaling — YaRN is the technique we validate against, and it is already built into both vLLM and SGLang. With a scaling factor of 4.0, the usable window grows to roughly 1M tokens.

You can turn YaRN on in either of two ways:

- **Edit the checkpoint's `config.json`.** Add a `rope_scaling` block to the model configuration:

  ```json
  {
      "rope_scaling": {
          "rope_type": "yarn",
          "factor": 4.0,
          "original_max_position_embeddings": 262144
      }
  }
  ```

- **Override at launch time.** Leave the checkpoint untouched and extend the serve commands above with the equivalent flags.

  vLLM:

  ```bash
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ornith-ai/Ornith-1.5-9B ... --hf-overrides '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}' --max-model-len 1000000
  ```

  SGLang:

  ```bash
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server ... --json-model-override-args '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}' --context-length 1000000
  ```

<div style="border-left:4px solid #FD8E5B;background:rgba(253,142,91,0.1);border-radius:6px;padding:12px 16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.6">
<div style="font-weight:700;color:#FD8E5B;margin-bottom:6px">📝 NOTE</div>
<p style="margin:0">Open-source runtimes implement YaRN <i>statically</i>: the same scaling factor is applied to every request regardless of its length, which can slightly hurt quality on ordinary-length inputs. Only enable <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">rope_scaling</code> when your workload genuinely needs the longer window, and size <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">factor</code> to match it — the target window is roughly <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">factor</code> × 262,144, so if your requests top out around 524,288 tokens, <code style="background:rgba(253,142,91,0.15);padding:1px 5px;border-radius:4px">factor: 2.0</code> is the better setting.</p>
</div>

### Using Ornith-1.5-9B via the Chat Completions API

Once a vLLM or SGLang server is running, talk to it with any OpenAI-compatible client.

#### Basic Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # any non-empty string works for a local server
)

response = client.chat.completions.create(
    model="Ornith-1.5-9B",
    messages=[
        {"role": "user", "content": "Write a one-line Python lambda that squares a number."}
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
)

message = response.choices[0].message
# reasoning_content holds the <think> trace; content holds the final answer.
print("reasoning:", getattr(message, "reasoning_content", None))
print("answer:", message.content)
```

You can also stream tokens, or hand the model tools — Ornith-1.5-9B emits well-formed function calls that the server parses into the standard `tool_calls` field:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="Ornith-1.5-9B",
    messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
    tools=tools,
    tool_choice="auto",
    temperature=0.6,
    max_tokens=2048,
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
# -> get_weather {"city": "Paris"}
```

You can point any OpenAI-compatible SDK (Python, Node.js, etc.) or `curl` at the same `/v1/chat/completions` endpoint.


## Agentic Usage

Ornith-1.5-9B exposes an OpenAI-compatible endpoint with tool calling, it works out of the box with standard agent frameworks. 

**Examples of using Ornith with agents:**


#### Ollama 
```bash
ollama run ornith-1.5:9b
```


#### Atomic.chat
```bash
# Both runtimes load a GGUF build of Ornith (publish one at ornith-ai/Ornith-1.5-9B-GGUF).

# llama.cpp — serve an OpenAI-compatible API on port 8000.
llama-server -hf hf.co/ornith-ai/Ornith-1.5-9B-GGUF --port 8000 -c 262144
```

#### llama.cpp
```bash
# Both runtimes load a GGUF build of Ornith (publish one at ornith-ai/Ornith-1.5-9B-GGUF).

# llama.cpp — serve an OpenAI-compatible API on port 8000.
llama-server -hf hf.co/ornith-ai/Ornith-1.5-9B-GGUF --port 8000 -c 262144
```

#### Hermes Agent
```bash
# Hermes talks to any OpenAI-compatible endpoint — point it at your Ornith server.
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export MODEL="ornith-ai/Ornith-1.5-9B"
```

#### OpenClaw

```bash
# OpenClaw talks to any OpenAI-compatible endpoint — point it at your Ornith server.
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL="ornith-ai/Ornith-1.5-9B"
```

#### Unsloth Studio

```bash
pip install unsloth

# Load Ornith for fast local inference or fine-tuning (Python):
#   from unsloth import FastLanguageModel
#   model, tokenizer = FastLanguageModel.from_pretrained(
#       "unsloth/Ornith-1.5-9B-GGUF",
#       max_seq_length=262144,
#       load_in_4bit=True,
#   )
```


### Coding CLIs

Ornith-1.5-9B is optimized for terminal-based coding agents. Point any OpenAI-compatible coding CLI at your Ornith-1.5-9B endpoint (set `OPENAI_BASE_URL` and `OPENAI_API_KEY`) to understand large codebases, automate tedious work, and ship faster.

#### OpenCode
```bash
# Register your local Ornith endpoint as a provider in ~/.config/opencode/opencode.json:
#
# {
#   "$schema": "https://opencode.ai/config.json",
#   "provider": {
#     "ornith": {
#       "npm": "@ai-sdk/openai-compatible",
#       "name": "Ornith (local)",
#       "options": { "baseURL": "http://localhost:8000/v1", "apiKey": "EMPTY" },
#       "models": { "ornith-ai/Ornith-1.5-9B": { "name": "Ornith-1.5-9B" } }
#     }
#   }
# }

opencode
```

### Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```