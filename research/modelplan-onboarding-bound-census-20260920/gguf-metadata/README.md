# GGUF accounting without backing-file access

Residency planning and MTP skip-byte reporting now consume `GgufTensorMetadata`: names and exact
physical byte sizes, with no file, mapping or offsets. The bound adapter emits only selected
physical bindings and deduplicates shared physical names. Ordinary raw GGUF sources retain their
previous rows. Other formats retain their complete census but return no GGUF residency estimate;
this does not mean their tensor inventory is absent.

Residency construction is now fallible, and callers propagate metadata errors. The placement and
byte-count arithmetic is unchanged. The bound GGUF test compares all rows and their total against
the original parsed file, while the scoped Step test confirms that a non-GGUF accounting result
does not discard the full census.

Validation: 47 bound-filter library tests passed; GGUF all-targets Clippy with warnings denied
passed; the actual-source model-memory fixture/arithmetic harness passed 10 tests; Linux-target
engine/server lib/bin/test Clippy passed with warnings denied using `DOCS_RS=1` documentation
stubs. Formatting and whitespace checks passed. Logs and final source hashes are retained here.

The GGUF spill context still needs its own scoped placement interface, and the CPU expert and
remaining source/draft consumers remain pending. Root activation, native tests and performance
qualification are not claimed. The newer upstream transfer/VMM state remains reserved for the
coordinator's next reviewed #542 stack.
