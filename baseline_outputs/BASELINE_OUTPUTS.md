# Baseline outputs

Carrier identification (`eval_carrier_identification()` in `src/eval_stealthiness.py`) reads
SrcMarker and CodeMark outputs from this folder. They are not distributed with the repository.
Produce them by running SrcMarker or CodeMark on each dataset and place the files here.

| lang | dataset | system | file |
|---|---|---|---|
| py | MBXP | CodeMark | `codemark_mbpp.jsonl` |
| py | CSN | CodeMark | `codemark_csn_py.jsonl` |
| js | MBXP | CodeMark | `codemark_mbjsp.jsonl` |
| js | MBXP | SrcMarker | `srcmarker_mbjsp.jsonl` |
| js | CSN | CodeMark | `codemark_csn_js.jsonl` |
| js | CSN | SrcMarker | `srcmarker_csn_js.jsonl` |
| java | MBXP | CodeMark | `codemark_mbjp.jsonl` |
| java | MBXP | SrcMarker | `srcmarker_mbjp.jsonl` |
| java | CSN | CodeMark | `codemark_csn_java.jsonl` |
| java | CSN | SrcMarker | `srcmarker_csn_java.jsonl` |

Each line holds one function before and after watermarking. Only `task_id` (or `id`), `original`
and `watermarked` are read, and a line missing any of them is skipped.

```json
{"task_id": "MBJSP/1",
 "original": "function minCost(cost, m, n) {\n    return cost[0][0] + ...}",
 "watermarked": "function minCost ( Cost , M , OnPermissionLevelChanged ) { return Cost [ 0 ] [ 0 ] + ...}"}
```

A missing file raises `FileNotFoundError` and stops the run. To run only some cells, edit
`lang_datasets` in `eval_carrier_identification()`.
