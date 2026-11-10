#!/usr/bin/env python3
"""red_unwritten.py HEAD_WORKER OUT: the red arm of the failed-copy cell. Writes the head's
crates/memra-server/src/worker.rs with the spec boundary publisher ignoring a failed plane copy
(trunk and draft), so a plane left unwritten is published. The failed-copy cell must fail on it."""
import sys

s = open(sys.argv[1]).read()
trunk = ("                if !spec_plane_copy(engine, &mut k, &l.k, kb)\n"
         "                    || !spec_plane_copy(engine, &mut v, &l.v, vb)\n"
         "                {\n                    return;\n                }\n")
draft = ("        if !spec_plane_copy(engine, &mut k, k_src, kb)\n"
         "            || !spec_plane_copy(engine, &mut v, v_src, vb)\n"
         "        {\n            return None;\n        }\n")
assert s.count(trunk) == 1 and s.count(draft) == 1
s = s.replace(trunk, "                let _ = spec_plane_copy(engine, &mut k, &l.k, kb);\n"
                     "                let _ = spec_plane_copy(engine, &mut v, &l.v, vb);\n")
s = s.replace(draft, "        let _ = spec_plane_copy(engine, &mut k, k_src, kb);\n"
                     "        let _ = spec_plane_copy(engine, &mut v, v_src, vb);\n")
open(sys.argv[2], "w").write(s)
