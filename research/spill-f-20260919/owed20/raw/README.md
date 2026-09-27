# GGUF heads for the OWED 20 census

The first 16 MiB of each pinned file, range-fetched 2026-09-27 at
`stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482`:

    B=https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF/resolve/0b69336d2fd2adfdef9c66e425f7778196c31482
    curl -fL -r 0-16777215 -o Step-3.7-flash-IQ4_XS-00001-of-00003.head.bin "$B/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"
    (the same for shards 2 and 3 and Step3.7-flash-mtp-Q8_0.gguf)

The binaries are not tracked: they carry the tokenizer vocabulary (a 64 MB blob whose strings trip
the public-boundary patterns). Their sha256 are in `HEADS.sha256`; `../census.py` reads them from
`MEMRA_F_HEADS` (default the lane's private store) and refuses a head whose hash differs.
