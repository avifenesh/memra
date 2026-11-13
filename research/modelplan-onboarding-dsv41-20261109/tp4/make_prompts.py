"""The 4-card window's prompt set, from the vendor tapes' real token ids (tapes-v2).

prefill-{4096,32768,131072}.i32   BOS, then the first P - 1 tokens of tf128k_en_moby (Moby-Dick from
                                  chapter 1, 131,072 tokens of raw text)
decode/decode-4096.i32            BOS, then the first 4,095 tokens of tf64k_he_cities (Hebrew
                                  Wikipedia)
conc/conc-NN.i32 (NN = 00..31)    32 prompts of 1,024 tokens: BOS, then 1,023 tokens from distinct
                                  windows, even NN from Moby-Dick, odd NN from the Hebrew text
manifest.json                     every file's length and sha256, and the ids themselves (for the
                                  baselines' prompt_token_ids, the same tokens every engine reads)

usage: python -I make_prompts.py TAPES_DIR CKPT_DIR OUT_DIR
"""

import hashlib
import json
import os
import struct
import sys

import torch


def ids_of(tapes, name):
    t = torch.load(os.path.join(tapes, name + ".pt"), weights_only=False, map_location="cpu")
    return [int(x) for x in t["prompt_ids"]]


def write(path, ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    b = struct.pack("<%di" % len(ids), *ids)
    with open(path, "wb") as f:
        f.write(b)
    return hashlib.sha256(b).hexdigest()


def main():
    tapes, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    moby = ids_of(tapes, "tf128k_en_moby")
    he = ids_of(tapes, "tf64k_he_cities")
    with open(os.path.join(ckpt, "config.json")) as f:
        bos = int(json.load(f)["bos_token_id"])
    man = {"source": tapes, "bos": bos, "moby_len": len(moby), "he_len": len(he), "files": {}}

    def add(rel, ids, what):
        sha = write(os.path.join(out, rel), ids)
        man["files"][rel] = {"len": len(ids), "sha256": sha, "what": what, "ids": ids}

    for p in (4096, 32768, 131072):
        assert len(moby) >= p - 1
        add("prefill-%d.i32" % p, [bos] + moby[:p - 1], "[bos] + tf128k_en_moby[:%d]" % (p - 1))
    add("decode/decode-4096.i32", [bos] + he[:4095], "[bos] + tf64k_he_cities[:4095]")
    L = 1024
    for i in range(32):
        src, nm = (moby, "moby") if i % 2 == 0 else (he, "he")
        off = 4096 + (i // 2) * (L - 1)
        assert off + L - 1 <= len(src)
        add("conc/conc-%02d.i32" % i, [bos] + src[off:off + L - 1],
            "[bos] + %s[%d:%d]" % (nm, off, off + L - 1))
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f)
    for k, v in man["files"].items():
        print(k, v["len"], v["sha256"][:16], v["what"])


if __name__ == "__main__":
    main()
