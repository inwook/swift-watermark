# Attacks

One attack is `NAME` or `NAME-OPTION`. Several are comma-separated and run left to right.
For `R`, `L`, `T` and `ST` the option is a percentage, default 100.

| Flag | Attack | Option |
|------|--------|--------|
| `R` | Random renaming (meaningless names) | share of local identifiers |
| `L` | LLM renaming, targets picked at random | share of local identifiers |
| `T` | Targeted renaming: the LLM first picks the identifiers that look watermarked | share of local identifiers |
| `ST` | Syntactic transform with a random subset of the rewriters | share of rewriter rules disclosed |
| `MINI` | The language's minifier: `terser` (js), `python-minifier` (py) | passes, default `MC`: `M` mangle, `C` compress, `F` format only |
| `OBFU` | `javascript-obfuscator` (js only) | none |

Notes:
- `T` spends the same budget as `L` and `R` but costs two LLM calls per sample. If carrier
  identification fails it falls back to random selection, counted in
  `attacks.get_targeted_ident_stats()`.
- `MINI` and `OBFU` shell out to the real tools, and their npm packages install on first use.
  Input a tool refuses comes back unchanged and is counted in
  `attack_tools.get_tool_attack_stats()`. Subtract it before reading a detection rate as robustness.
- `F` is the reprint alone. `M` and `C` include it, so `F` is meaningful only on its own.
