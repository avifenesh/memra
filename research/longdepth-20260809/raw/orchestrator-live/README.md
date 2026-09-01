# Live serving receipts imported 2026-08-09

These are the exact assistant-content files named by the orchestrator's hourly steering message,
plus the three full response JSON files from the same live reproductions. They use the same sampled
HTML task at temperature 0.7.

| response | context cap | completion tokens | first forbidden reasoning character | first forbidden content character | sha256 |
|---|---:|---:|---|---|---|
| `repro-256k.json` | 262144 | 4000 | none | no content emitted | `ecd79bca279b489eb5d745d80fd7c59b5449ba72a48f854a0c505ce0b606462a` |
| `repro2.json` | 262144 | 9632 | char 896: `牌` (`U+724C`) | char 575: `刘` (`U+5218`) | `99eed82a2ddfbb0f675960fd8fc6b012562dd21ec5ec03f43945593155be60da` |
| `repro-131k.json` | 131072 | 8687 | char 261: `ا` (`U+0627`) | char 65: `給` (`U+7D66`) | `8d47c5c789980328cd243b5d57d35a0b14de6593fd9d98678454aa6154d2d4bb` |

The original 8.7K and 9.6K figures are total response lengths, not measured corruption-onset
positions. The full responses show contamination near the beginning of both reasoning and code:
`canvas اجتماع` in the 131072 reasoning, `<meta charset給了UTF-8">` in its content,
`move the mouse,牌照?` in the 262144 reasoning, and `font-size刘备? 14px` in its content. This
corrects the initial depth interpretation: the symptom is context-cap-independent, but the live
receipts do not show a late positional threshold. Longer output merely gives a sampled stream more
opportunities to expose contamination.

The JSON response carries neither the request body nor an exact response-token array. Context cap
and temperature therefore come from the retained steering receipt; character offsets are exact,
while exact completion-token indices come only from the controlled native-completion matrix.

The extracted files remain byte-identical to the steering-named receipts:

- `repro-131k-content.txt`: `173e2f772e901710f9900ca05ed5d3591cd08123cfc25ec9d0451cbc43ed3b02`
- `repro2-content.txt`: `1e5323741b137d99ff656e999eebe953ff3dade59b8621491f32f05c62e873c3`

`inbox-cx-longdepth.md` is the exact steering message that changed the matrix; its SHA-256 is
`7145d13b1cf7a5085f04c2e8eaa3474fbcec99459c6a5aada1c541ad6f38f1bd`.
