# glm5 vision on the ppN serving shape — BOX BATTERY GREEN (2026-09-01, slot B)

**All four arms green on the real 3-card PP3 serving shape.** This is the half the rig could not
prove: one card has one serving context, so nothing on the rig could show that rows published
across two CUDA contexts are dereferenceable by the stage that owns them. They are.

## Pins

| pin | value |
|---|---|
| binary | `memra-server-v0.123.0-vppn-gb477de493-dl9e3143f` sha256 `7279cfa724fd7f2b632eb5d0c943da131fb416602b85bac22e081ea44b886793` |
| two-pin | memra `b477de493` (the fix + the peer-review source-side guard) · darklanes `9e3143f` |
| build | off-box via darklanes `serving/build-artifact.sh` (the box has no docker and no cargo), ubuntu-22.04 glibc 2.35 floor, arch 120a; sha verified byte-for-byte on arrival |
| shape | ship recipe: `MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 MEMRA_MOE_RESIDENT_GB=98`, spec ON (dflash2 @ `b33c0347`), `MEMRA_DSA_INDEX_RING` unset |
| fixtures | the card3 requests, sha-pinned in the probe, all 5 matching on-box |
| box | vast 49206484, slot B (18893/8808), handed over by the cache lane at 15:49:10Z |

## The line that makes the window count

```
[glm5-vision] overlay intake: tower dev2 -> intake dev0 (cross_context=true) publish=Auto servable=true glm5_models=1
```

The tower runs on **dev2** (the primary follows the LAST pp stage) while embedding intake owns
**dev0**. That is the configuration that 500'd every image request at launch. `cross_context=false`
here would have made the run a VOID rather than a pass.

## Arms

**A — fix arm: PASS.** `10-cant-hallucinate-greedy` and `11-cant-hallucinate-sampled-vendor-default`
both 200 with **codes 3/3 exact** (`ZK5465`, `QV4655`, `XR0818`). Negatives refused by name: video
400, faked pad 400; the literal `<|image|>` stays plain text. Publication receipt — once per
SESSION, which in this probe is also once per request because each probe request is its own
session (the engine builds the overlay on a session's first prefill tick and reuses it after):
`overlay published to the intake engine: dev2 -> dev0 rows=256 elems=1048576 MiB=4.00 mode=Auto` —
4 MiB, matching the ~5 MiB the design predicted. Spec engaged (33 `[glm5-acc]` bursts).

**B — control: PASS (and it reproduces).** With `MEMRA_VISION_OVERLAY_PUBLISH=0` the boot says
`publish=Never servable=false` plus `IMAGE INPUT DISABLED`, and image requests get a **named 400 at
the HTTP waist** ("image input is not enabled on this deployment") — never the launch-era
mid-prefill 500. A cell whose control stays silent cannot testify; this one speaks.

**C — text-only byte identity with vision armed: PASS.** 1987 bytes, sha `80e1184c61f977be`,
identical across the vision-armed boot and `MEMRA_GLM5_VISION=0`. Non-emptiness asserted **first**:
two empty files compare equal, and this API is key-authed, so a pair of 401s would otherwise have
rendered as "identity".

**D — no decode tax: PASS.** Boot-interleaved V,T,V,T,V,T in one contiguous window, vendor-default
sampled, tok/s from the server's own `usage.elapsed_s`:

| arm | rows | median | spread |
|---|---|---|---|
| vision-armed | 49.60 / 49.09 / 51.16 | **49.60** | 4.17% |
| text-only | 53.25 / 51.49 / 50.05 | **51.49** | 6.21% |

Gap **-3.67%**, inside the arms' own spread. One `system_fingerprint`
(`memra-0.123.0-efb0b421211c`) across all six boots, so the arms differ by env only; spec engaged
in both. **Read as "no measurable tax at this rep count", never as a performance claim** — and note
the absolutes are NOT comparable to the launch's 75-77 rows (different prompt, `reasoning_effort`,
and token cap). There is also no mechanism for a tax: publication is one host round trip per
SESSION in prefill, a text request never builds an overlay, and decode never touches one.

## Two harness bugs this window found in my own code

Both refused loudly rather than banking a false green, which is the only reason they are footnotes:

1. `interleave.sh` called the probe without `--base`, so it inherited the probe's edge default
   (`https://$GLM53_HOST`) and, on a loopback run, booted the whole model and then failed every
   request against `https://127.0.0.1:443`. BASE is now a required input of both drivers.
2. Process selection by `pgrep -f <binary path>` matches the flock wrapper AND the calling shell —
   measured on a rig with zero servers, the cmdline predicate returned **6** matches, the exe
   predicate 0. Both drivers now select by `/proc/<pid>/exe`.

Also: the first arm-D attempt died to SIGHUP when the ssh session detached (cards clean, no
orphans); re-run `setsid nohup`-detached.

## Window close

Slot B stopped by exe, cards back to 1 MiB. Both revive paths restored to their real names, slot A
brought back via `bash onstart-stack.sh` (not `serve-deploy` — darklanes#22), running its original
`v0.123.0-dlbfa6abb-glm53`. `ops/serve-gate.py` **GATE PASS 13/13** including `sampled-spec-engaged`
by LOG receipt and `tools-round-trip tool_calls=1`; independent sampled probe 216 tokens with spec
rounds=105 accepted=111 (0.575). Drain-window END logged in the box operator log.

## What this unblocks, and what it does not

It closes the vision half of the glm5 ship gate on the engine side. It does **not** by itself move
any product surface: the launcher's `MEMRA_GLM5_VISION=0` pin and the text-only modality fact are
darklanes changes that go through facts -> render -> gates -> ripple, and they should cite this file.
