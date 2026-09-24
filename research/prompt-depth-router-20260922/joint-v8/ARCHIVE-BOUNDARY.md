# Exact archive boundary for the draft-only C/K/D study

Three compressed research archives are published under
`receipts-v8/`, `parent-v6/` and `diagnostic-v7/`.
Their byte-level SHA-256 digests are respectively
`a54b6e48e1e1067cc27e5716563be446cfda30a5c859e285af0554f321b2e55c`,
`78ba4febde049732027d528a43aa8716dc7ce69658b3b29f8601667da6344d18`
and
`a483cf392fef08606001c956779911d1108ccf3f7229199bf2afc4e4e4988210`.
Their manifests pin 6,666, 13,613 and 164 expanded
members. The model checkpoint and provider/operator
credentials are outside these archives.

The public pre-push scanner matched `provider_name_aws`
in each outer compressed stream and
`runpod_pod_id_bare` in the v7 diagnostic stream.
Exact raw-byte review found short generic provider-name
fragments in gzip data. The v7 fourteen-character decoded
match was assembled across bytes dropped by UTF-8 error
skipping and is absent as a contiguous raw string.

A hosted private expanded-member inspection used Memra
policy/scanner SHA-256
`3b05f725210a8136a64b340f2a425602862a16cbd02ec2578ff4cbc1a50d09c2`
and
`0000058db5d59a35421999679372f43e760bdf7e72fde72b1748e74a396518aa`
at public head
`874a1c2b6ae115e326686980ad65e9d59a24891e`.
Private workflow run `35954215960`,
`joint-v8-inspect` job `107488964296`, reran the
expanded scan with the exact approval file and reported
`status: approved`, **zero unapproved** observations.
It found zero matching rules in expanded native request
data, fitted model files or the four changed Rust source
files. Every source archive has the same 1,321-member
inventory as its reviewed v3 parent and changes only
those four Rust files. Eighteen nested source matches
are unchanged v3 file hashes, including the public
scanner's own policy and allowlist text.

Four archived ELF executables contain generic
provider-name matches in code/constant sections, not
an account, instance, pod or credential. The separate
secret-shaped byte match in their `.strtab` section is
the same previously reviewed
`memra_tokenizer::hf_input::AddedToken` symbol,
matched-byte SHA-256
`b29c57d2934e3ed4a5d580523459a38c436cc997fb7027d400c36c9e1f08190c`.
The exact binary hashes, section offsets, all 29
expanded/compressed rule dispositions and the full
findings file are retained in the private custody lane.
That hosted findings file is SHA-256
`e164cc20db5674a52c548c495c292dcd1a9290e6330fd8f8263cdf77aced4d6e`;
its exact rule-scoped approval file is
`14854cd114e26c60238bd3e73e70ef61562b3f2f9c55d32e16a5a059b3df0fad`.

The public allowlist entries pin only the three exact
outer archive hashes and the rules reported for those
compressed streams. They grant no path-wide or
mutable exception. This archive review verifies the
publication boundary; it does not qualify sampled
distribution parity or a serving setting.
