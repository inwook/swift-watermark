# SWIFT: Robust Code Watermarking via Invariant Canonicalization and LLM-Based Rewriting

A source-code watermarking system for Python, JavaScript and Java. Watermarks are embedded
through semantics-preserving code transformations keyed on an identifier-free canonical form.

## Overview

SWIFT has two layers. Layer 1 (syntactic) applies deterministic rewriters such as `a += b` <-> `a = a + b`,
and Layer 2 (structural) picks among LLM-generated, equivalence-verified variants of a function.
Both are keyed on the identifier-free canonical form, so renaming does not remove the watermark.
Detection combines a binomial (L1) and a negative-binomial (L2) p-value with Fisher's method.

## Installation

Python 3.10 or newer (tested on 3.13).

```bash
pip install -r requirements.txt
```

| Tool | Used by |
|------|---------|
| Node.js 18+ (`node`, `npm`) | JS runner and JS equivalence verifier. `npm install lodash` runs on first use, creating `node_modules/` at the repo root |
| JDK 17+ (`javac`, `java`) | Java runner and Java equivalence verifier |

### LLM API keys

Only Layer 2 embedding and the LLM attacks call an LLM. Layer 1 and all detection run
locally with no key.

```bash
cp .env.example .env    # fill in only the providers you use
```

```
OPENAI_API_KEY=...        # gpt-* models
GEMINI_API_KEY=...        # gemini-* models
ANTHROPIC_API_KEY=...     # claude-* models
HF_API_KEY=...            # HuggingFace-hosted models (e.g. Qwen)
```

`llm_provider.py` loads `.env` at import time. Shell environment variables take precedence,
and a missing key raises only when a model that needs it is requested.

## Quick Start

### CLI

`wmcli.py` watermarks a file or a directory in place.

```bash
python src/wmcli.py embed path/to/source/                 # Layer 1
python src/wmcli.py embed path/to/source/ --layers L1L2   # Layer 1 + Layer 2
python src/wmcli.py detect path/to/source/                # per-file and aggregate p-values
```

The language is detected from the file extension. `--lang` restricts a directory scan to one
language, and `-k` sets the secret key.

### Python API

```python
from watermarker import Watermarker
from stats import format_pvalue

wm = Watermarker(lang="py")                 # Watermarker("py", llm_model="gpt-4o-mini") for L2
watermarked = wm.embed(src_code, do_l1=True, do_l2=False)

det = wm.detect(watermarked, do_l1=True, do_l2=False)
p = Watermarker.calculate_l1_pvalue(det["l1_n"], det["l1_s"])
print(det["l1_n"], det["l1_s"], format_pvalue(p))
```

### Batch evaluation

```bash
cd src
python eval_dataset.py    # configs selected in __main__
python experiment.py      # experiments listed in EXPERIMENTS
```

`eval_dataset.py` builds configs (`eval_fidelity()`, `eval_detection()`, `eval_robustness()`,
`eval_tools()`, `eval_folding()`, `eval_disclosure()`), each a `(title, [setting, ...])` tuple
passed to `evaluation()`:

```python
from eval_dataset import evaluation, eval_detection
evaluation(eval_detection(), n_workers=1)
```

Rows with Layer 2 or an LLM attack need an API key.

`eval_stealthiness.py` runs the A/B test and carrier identification. Carrier identification needs
SrcMarker / CodeMark outputs in `baseline_outputs/`, see
[baseline_outputs/BASELINE_OUTPUTS.md](baseline_outputs/BASELINE_OUTPUTS.md).

## Experiment Settings

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `lang` | `"py"`, `"js"`, `"java"` | required | Target language |
| `dataset` | `"humaneval-x"`, `"MBXP"`, `"mbpp"`, `"mbjp"`, `"mbjsp"`, `"csn"` | required | `"MBXP"` routes to mbpp / mbjsp / mbjp by `lang` |
| `fold` | `1`, `4`, `10`, ... | `1` | Concatenates *k* samples into one task. Java with `pass_check="tests"` needs `fold = 1`, since mid-file imports break `javac` |
| `layers` | `""`, `"L1"`, `"L2"`, `"L1L2"` | `""` | No watermark, syntactic only (SWIFT-Syn), structural only, or both layers (SWIFT) |
| `nsamples` | `0`, `100`, ... | `0` | Samples drawn before folding (`0` = all) |
| `attack` | `""`, `"R-20"`, `"ST-10"`, `"R-40,ST-10"`, ... | `""` | Post-embed attack, see [docs/attacks.md](docs/attacks.md) |
| `pass_check` | `""`, `"syntax"`, `"tests"` | `""` | How `Pass%` is decided. `""` runs no check and prints `-` |

### Attacks

`attack` takes `NAME` or `NAME-OPTION`, comma-separated to chain (`"R-40,ST-10"`).
Flags and options are listed in [docs/attacks.md](docs/attacks.md).

## LLM Prompts

| Prompt | Paper section | Location |
|---|---|---|
| Structural variant generation | §3.3 | `structure_embedder._SYSTEM_PROMPT_TEMPLATE`, `_USER_PROMPT_TEMPLATE` |
| Test-input generation | §3.3 | `generate_test_inputs()` in `equivalence_verifier_{py,js,java}.py` |
| LLM renaming attack | §4.4 | `attacks.SYSTEM_PROMPT_RENAME` |
| Carrier identification judge | §4.5 | `attacks.SYSTEM_PROMPT_IDENT`, `USER_PROMPT_IDENT` |
| Blind A/B preference judge | §4.5 | `eval_stealthiness.SYSTEM_PROMPT_AB`, `USER_PROMPT_AB` |
| Agent restructuring task | §5, App. H.1 | `eval_agent_attack.AGENT_TASKS` |

## Rewrite Strategies

Each strategy has two equivalent forms, `s0` and `s1`, and the watermark picks one per applicable
strategy and function. The full list per language is in [docs/rewriters.md](docs/rewriters.md).

## Datasets

| Dataset | Languages | Size | Hugging Face | GitHub |
|---------|-----------|------|--------------|--------|
| HumanEval-X | py, js, java | 164 per language | [THUDM/humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x) | [THUDM/CodeGeeX](https://github.com/THUDM/CodeGeeX) |
| MBPP | py | 500 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) |
| MBJSP / MBJP | js, java | 938 / 874 | [AmazonScience/mxeval](https://huggingface.co/datasets/AmazonScience/mxeval) | [amazon-science/mxeval](https://github.com/amazon-science/mxeval) |
| CodeSearchNet | py, js, java | 21,948 / 4,441 / 26,909 | [code-search-net/code_search_net](https://huggingface.co/datasets/code-search-net/code_search_net) | [github/CodeSearchNet](https://github.com/github/CodeSearchNet) |

`testset.py` downloads them on first use. HumanEval-X is excluded from the paper due to the space limitation.

## Test Projects

The real-world projects are cloned into `testprojects/` at fixed commits by
`scripts/setup_testprojects.sh`.

| Project | Lang | Repository | Commit |
|---------|------|------------|--------|
| networkx | py | https://github.com/networkx/networkx | `24861327ed3bc9a45c28b38f6e4b2ec6c59e678f` |
| gson | java | https://github.com/google/gson | `8260eddffe413c132b3a38e68446da6bd083220d` |
| axios | js | https://github.com/axios/axios | `ad68e1a484b50086af427f767bbd7d6e3aab7ac3` |
