# Excluded before model boot

This attempt did not start `memra-server` and performed no GPU model work. The runner appended
optional `env -u` operands after environment assignments, so GNU `env` treated `-u` as the command
and exited with the captured line:

```text
env: '-u': No such file or directory
```

The harness ordering is corrected in the next checkpoint. This directory is retained so the failed
attempt is not silently discarded or mistaken for a model result.
