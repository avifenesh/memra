#!/usr/bin/env python3
"""red_patch.py TREE NAME: memra#1221 part 2's red arms, each the head with one guard disarmed.

  rollback  Cache::rollback's lend check never refuses
  rewind    Cache::rewind_recurrent_from's lend check never refuses
  pooled    KvPlane::into_pooled's lend check never refuses
  release   KvPlane::release_beyond releases lent bytes (no clamp)
  adopt     KvPlane's Drop never hands a lent plane to its loan (it is freed with its owner)
  scratch   the MTP scratch's set_len and set_plane_len lend checks never refuse
review round 1 (CPU arms):
  ledger-double   a charge that moves to another reader is also added to the total (must-fix 1)
  ledger-holder   the holder moves to a remaining reader only with a charge (must-fix 2)
  evictable       a leased co-reader's charge is offered to admission (should-fix 3)
  restore-floor   restore_cache_checkpoint writes without checking the lends (should-fix 4)
  publish-copy    the publication lends without deciding lend or copy first (should-fix 6)
"""
import sys

tree, name = sys.argv[1], sys.argv[2]
KV = f"{tree}/crates/memra-kv/src"
ENG = f"{tree}/crates/memra-engine/src"
SRV = f"{tree}/crates/memra-server/src"


def patch(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, (name, path, old[:70], s.count(old))
    open(path, "w").write(s.replace(old, new))


if name == "rollback":
    patch(f"{KV}/lib.rs", 'kvl.lend_floor_refusal("cache rollback", saved + accept_len)',
          'kvl.lend_floor_refusal("cache rollback", usize::MAX)')
elif name == "rewind":
    patch(f"{KV}/lib.rs", 'kvl.lend_floor_refusal("recurrent rewind", pos)',
          'kvl.lend_floor_refusal("recurrent rewind", usize::MAX)')
elif name == "pooled":
    patch(f"{KV}/plane.rs", 'lend_floor_refusal("into_pooled", 0, self.lent_bytes())',
          'lend_floor_refusal("into_pooled", 0, 0)')
elif name == "release":
    patch(f"{KV}/plane.rs",
          "they stay backed (memra#1221).\n        let keep = release_keep_bytes(keep_bytes, self.lent_bytes());",
          "they stay backed (memra#1221).\n        let keep = release_keep_bytes(keep_bytes, 0);")
elif name == "adopt":
    patch(f"{KV}/plane.rs", "            if Arc::strong_count(&loan) > 1 {\n                loan.adopt(",
          "            if Arc::strong_count(&loan) > usize::MAX - 1 {\n                loan.adopt(")
elif name == "scratch":
    patch(f"{ENG}/spec.rs", '.lend_floor_refusal("MTP scratch length", n)',
          '.lend_floor_refusal("MTP scratch length", usize::MAX)', count=2)
elif name == "ledger-double":
    patch(f"{SRV}/worker.rs", "            dead.bytes = dead.bytes.saturating_sub(charged);\n",
          "            dead.bytes = dead.bytes.saturating_sub(charged);\n            self.total_bytes += charged;\n")
elif name == "ledger-holder":
    patch(f"{SRV}/worker.rs", "                if book.holder != dead.id {\n                    continue;\n                }\n                book.holder = next;",
          "                if book.holder != dead.id || book.charged == 0 {\n                    continue;\n                }\n                book.holder = next;")
elif name == "evictable":
    patch(f"{SRV}/worker.rs", "unleased.saturating_sub(self.charges_kept_by_leases())", "unleased")
elif name == "restore-floor":
    patch(f"{ENG}/pp.rs", "    if let Some(why) = target.checkpoint_lend_refusal(snap, source.is_some()) {\n        return Err(why.into());\n    }\n", "")
elif name == "publish-copy":
    patch(f"{SRV}/worker.rs", "    let planes = spec_publication_planes(engine, cache, draft_plane, pos, planes);\n", "")
else:
    sys.exit(f"unknown red arm {name}")
print(f"red arm {name} applied")
