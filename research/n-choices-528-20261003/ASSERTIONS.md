# Independent assertion map

Planning only. Native values must be observed; this map cannot convert a skipped or failing cell into qualification.

| Edge | Raw evidence and actual assertion | Required failing control |
| --- | --- | --- |
| n1 compatibility | exact-base and candidate stock HTTP bodies; canonicalization removes only id/created/fingerprint/time fields; native text token IDs and all content/reasoning/finish/usage fields match | semantic byte/count mutation |
| greedy row identity | every indexed choice matches its n1 greedy control, including terminal reason and producer token hash | changed row text/token hash |
| RNG isolation | actual worker seed witness equals master+i; each sampled row repeats and matches its independent seed+i singleton; every index has independent state | shared seed/RNG witness |
| shared prefill | CHOICE_PRIME reports leader-only rows; one fork record has N choices and N-1 copies; complete producer rows match the shared prompt boundary | missing fork or follower cold-prime record |
| indexed termination | raw JSON and SSE cover exactly0..N-1, one finish per index, exactlyone DONE after all finishes | missing/duplicate row or premature DONE |
| usage/accounting | independent callback token sequence1..M, exact one parent open/terminal/drop; HTTP usage equals callback counts and sum of independent worker row outputs, prompt once and cached count unchanged | coherently changed totals against fixed worker/callback witnesses |
| reservation | native callback sees prompt once and output bound multiplied by N; one fixture policy decline when that sum exceeds the bound | multiplied prompt or unscaled output reserve |
| N slots | real contender offered while four indexed rows are active; no immediate extra admission; group reset then four-choice recovery succeeds | missing offered overlap or successful early contender |
| KV exhaustion | real owned CUDA allocation briefly leaves 1GiB free; idle N8 request yields typed400 context_length_exceeded before prime/copy; freeing only that allocation and its context permits N8 recovery; no CUDA OOM may stand in for admission | a prime/copy record on refused request |
| cancel/recover | raw real TCP reset; one unfinalized parent drop with independently counted partial output below total bound; all child receivers/slots retire and next request succeeds | completed callback, zero/missing drop, leaked slot or failed recovery |
| defaults | bare model/messages request and n-only extension; pinned full vendor profiles loaded, template/default thinking engaged, no decode/mode knobs | explicit caller decode/mode override |

Each cell binds source, ELF, model artifact, compiler/runtime environment, actual hardware, request and helper hashes. Both original failures and positive records remain in the denominator. Full server CPU contracts and strict Clippy are separate from native assertions. This scoped serving proof changes no model support state or release qualification policy.
