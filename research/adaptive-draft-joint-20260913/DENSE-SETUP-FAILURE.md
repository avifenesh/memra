# Dense experiment setup failure, 2026-09-14

No dense-training or second-fresh measurements were collected. The initial
replacement host reached CUDA/bootstrap setup but failed its Hugging Face
connectivity check with:

```
curl: (28) Resolving timed out after 20000 milliseconds
```

Subsequent SSH attempts alternated between connection refusal, a successful
read, and connection timeout. DNS recovery was prepared, but its execution was
not confirmed. The failed rental was destroyed and absence verified. Three
replacement-offer attempts returned `no_such_ask`; none created an instance.
These are infrastructure failures, not model results. Private rental identifiers
and provider logs remain in the operations evidence directory.

The dense protocol, source change, evaluator and second fresh prompt generator
are committed. The cadence change remains uncompiled and GPU-unqualified.
Resume with a fresh dedicated RTX 5090, build/format and bind the source hash,
then run the registered dense -> audit -> fit -> freeze -> fresh2 -> audit chain.
Do not reuse the first fresh test as training, and do not combine confidence or
depth adaptation to bypass a negative admission result.
