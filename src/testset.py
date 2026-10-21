import json
import os
import pickle
import re
import textwrap
import ast
import urllib.request
import urllib.error
from datasets import load_dataset as _hf_load_dataset
from typing import List, Dict, Tuple
from paths import OUT_PATH, CACHE_PATH

TESTSET_CACHE_DIR = CACHE_PATH()
TESTSET_CACHE_FILE = CACHE_PATH("testset_cache.pkl")

_testset_cache: Dict[tuple, list] = {}


def _fetch_jsonl(url, fallback_url=None):
    lines = []
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').strip().split('\n')
    except urllib.error.URLError:
        if fallback_url:
            print("[!] Failed to access primary URL. Trying fallback link.")
            req = urllib.request.Request(fallback_url)
            with urllib.request.urlopen(req) as response:
                lines = response.read().decode('utf-8').strip().split('\n')
    return lines


def sanitize_code(code):
    code = code.expandtabs(4)
    code = textwrap.dedent(code)
    return code.strip()


def _validate_testset(testset, lang, mode, nsamples):
    total = len(testset)
    lang = {"python": "py", "javascript": "js"}.get(lang, lang)
    if mode == "exec":
        from runners import get_runner
        runner = get_runner(lang)
        valid = []
        for item in testset:
            test_code = "\n".join(item['test_payload'])
            full = f"{item['code']}\n\n{test_code}"
            if runner.run_code(full)["success"]:
                valid.append(item)
                if nsamples > 0 and len(valid) >= nsamples:
                    break
    elif mode == "syntax":
        from language_parser import get_parser
        parser = get_parser(lang)
        valid = []
        for item in testset:
            ok, _ = parser.check_syntax(item['code'])
            if ok:
                valid.append(item)
                if nsamples > 0 and len(valid) >= nsamples:
                    break
    else:
        return testset

    print(f"[*] Validation: {len(valid)}/{total} passed ({mode})")
    return valid


def _cache_lookup(base_key, validate):
    """Exact cache hit, else the unvalidated base entry. Returns (testset_or_None, needs_validate)."""
    exact_key = base_key + (validate,)
    if exact_key in _testset_cache:
        return list(_testset_cache[exact_key]), False

    if validate:
        base = base_key + (False,)
        if base in _testset_cache:
            return list(_testset_cache[base]), True

    return None, validate


def load_mbpp_testset(nsamples = 0, validate = False):
    base_key = ("mbpp",)
    cache_key = base_key + (validate,)

    testset, needs_v = _cache_lookup(base_key, validate)
    if testset is not None and not needs_v:
        print("[*] Using cached testset (mbpp)")
        return testset[:nsamples] if nsamples > 0 else testset

    if testset is None:
        print("[*] Loading MBPP Dataset...")
        dataset = _hf_load_dataset(
            "parquet",
            data_files={"test": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet"},
            split="test"
        )
        testset = []
        for sample in dataset:
            code = textwrap.dedent(sample['code']).strip()
            testset.append({
                "dataset": "mbpp",
                "task_id": str(sample['task_id']),
                "code": code,
                "test_payload": sample['test_list'],
                "entry_point": None
            })
        if nsamples == 0:
            _testset_cache[base_key + (False,)] = list(testset)
        needs_v = validate
    else:
        print("[*] Using cached testset (mbpp, applying filters)")

    if needs_v:
        testset = _validate_testset(testset, "python", "exec", nsamples)
    if nsamples == 0:
        _testset_cache[cache_key] = testset
    return testset[:nsamples] if nsamples > 0 else testset


def load_mbjsp_testset(nsamples=0, validate=False):
    base_key = ("mbjsp",)
    cache_key = base_key + (validate,)

    testset, needs_v = _cache_lookup(base_key, validate)
    if testset is not None and not needs_v:
        print("[*] Using cached testset (mbjsp)")
        return testset[:nsamples] if nsamples > 0 else testset

    if testset is None:
        print("[*] Loading MBJSP Dataset...")
        url = "https://huggingface.co/datasets/AmazonScience/mxeval/resolve/main/data/mbxp/mbjsp_release_v1.2.jsonl"
        fallback_url = "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbjsp_release_v1.2.jsonl"
        lines = _fetch_jsonl(url, fallback_url)
        testset = []
        for line in lines:
            if not line.strip(): continue
            sample = json.loads(line)
            prompt = sample.get('prompt') or ""
            solution = sample.get('canonical_solution')
            if solution is None: continue
            testset.append({
                "dataset": "mbjsp",
                "task_id": str(sample.get('task_id', '')),
                "code": (prompt + solution).strip(),
                "test_payload": [sample.get('test', '')],
                "entry_point": None
            })
        if nsamples == 0:
            _testset_cache[base_key + (False,)] = list(testset)
        needs_v = validate
    else:
        print("[*] Using cached testset (mbjsp, applying filters)")

    if needs_v:
        testset = _validate_testset(testset, "javascript", "exec", nsamples)
    if nsamples == 0:
        _testset_cache[cache_key] = testset
    return testset[:nsamples] if nsamples > 0 else testset


def load_mbjp_testset(nsamples=0, validate=False):
    base_key = ("mbjp",)
    cache_key = base_key + (validate,)

    testset, needs_v = _cache_lookup(base_key, validate)
    if testset is not None and not needs_v:
        print("[*] Using cached testset (mbjp)")
        return testset[:nsamples] if nsamples > 0 else testset

    if testset is None:
        print("[*] Loading MBJP Dataset...")
        url = "https://huggingface.co/datasets/AmazonScience/mxeval/resolve/main/data/mbxp/mbjp_release_v1.2.jsonl"
        fallback_url = "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbjp_release_v1.2.jsonl"
        lines = _fetch_jsonl(url, fallback_url)
        testset = []
        for line in lines:
            if not line.strip(): continue
            sample = json.loads(line)
            prompt = sample.get('prompt') or ""
            solution = sample.get('canonical_solution')
            if solution is None: continue
            testset.append({
                "dataset": "mbjp",
                "task_id": str(sample.get('task_id', '')),
                "code": (prompt + solution).strip(),
                "test_payload": [sample.get('test', '')],
                "entry_point": None
            })
        if nsamples == 0:
            _testset_cache[base_key + (False,)] = list(testset)
        needs_v = validate
    else:
        print("[*] Using cached testset (mbjp, applying filters)")

    if needs_v:
        testset = _validate_testset(testset, "java", "exec", nsamples)
    if nsamples == 0:
        _testset_cache[cache_key] = testset
    return testset[:nsamples] if nsamples > 0 else testset


# HumanEval-X only: plain HumanEval is Python-only.
def load_humaneval_x_testset(lang="python", nsamples=0, remove_docstring=True, validate=False):
    if lang == "py": lang = "python"
    assert lang in ["python", "java", "js", "go"]

    base_key = ("humaneval-x", lang)
    cache_key = base_key + (validate,)

    testset, needs_v = _cache_lookup(base_key, validate)
    if testset is not None and not needs_v:
        print(f"[*] Using cached testset (humaneval-x-{lang})")
        return testset[:nsamples] if nsamples > 0 else testset

    if testset is None:
        print(f"[*] Loading HumanEval-X ({lang}) Dataset...")
        dataset = _hf_load_dataset("THUDM/humaneval-x", lang, split="test", trust_remote_code=True)
        sorted_data = sorted(dataset, key=lambda x: int(x['task_id'].split('/')[-1]))
        testset = []
        for p in sorted_data:
            if 'entry_point' in p:
                entry_point = p['entry_point']
            else:
                if lang == "python":
                    matches = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", p['prompt'])
                    entry_point = matches[-1] if matches else "unknown"
                else:
                    entry_point = ""
            full_code = p['prompt'] + p['canonical_solution']
            if remove_docstring:
                if lang == "python":
                    pattern_docstring = r"(?m)^.*?(?:\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?''')\n?"
                else:
                    pattern_docstring = r"(?m)^\s*(?:\/\*[\s\S]*?\*\/|\/\/.*?$)\n?"
                full_code = re.sub(pattern_docstring, '', p['prompt'], flags=re.MULTILINE) \
                    + p['canonical_solution']
            if lang == "python":
                test_payload = [p['test'] + f"\ncheck({entry_point})\n"]
            else:
                test_payload = [p['test']]
            testset.append({
                "dataset": f"humaneval-x-{lang}",
                "task_id": p['task_id'],
                "code": full_code,
                "test_payload": test_payload,
                "entry_point": entry_point
            })
        if nsamples == 0:
            _testset_cache[base_key + (False,)] = list(testset)
        needs_v = validate
    else:
        print(f"[*] Using cached testset (humaneval-x-{lang}, applying filters)")

    if needs_v:
        testset = _validate_testset(testset, lang, "exec", nsamples)
    if nsamples == 0:
        _testset_cache[cache_key] = testset
    return testset[:nsamples] if nsamples > 0 else testset


def load_codesearchnet_testset(lang="python", nsamples=0, remove_docstring: bool=True, split="test", do_shuffle: bool=True, validate=False):
    if lang == "py": lang = "python"
    if lang == "js": lang = "javascript"

    base_key = ("csn", lang, split)
    cache_key = base_key + (validate,)

    testset, needs_v = _cache_lookup(base_key, validate)
    if testset is not None and not needs_v:
        print(f"[*] Using cached testset (csn-{lang})")
        return testset[:nsamples] if nsamples > 0 else testset

    if testset is None:
        print(f"[*] Loading CodeSearchNet Dataset ({lang}, {split} split)...")
        assert lang in ["python", "java", "javascript", "go", "ruby", "php"]
        dataset = _hf_load_dataset("code_search_net", lang, split=split, trust_remote_code=True)
        if do_shuffle:
            dataset = dataset.shuffle(seed=0)
        testset = []
        for i, p in enumerate(dataset):
            func_name = p.get('func_name')
            full_code = p.get('whole_func_string', '')
            if not func_name or not full_code:
                continue
            if lang == "python":
                try:
                    ast.parse(full_code)
                except Exception:
                    continue
            if remove_docstring:
                if lang == "python":
                    pattern_docstring = r"(?m)^.*?(?:\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?''')\n?"
                elif lang in ["java", "javascript", "go", "php"]:
                    pattern_docstring = r"(?m)^\s*(?:\/\*[\s\S]*?\*\/|\/\/.*?$)\n?"
                elif lang == "ruby":
                    pattern_docstring = r"(?m)^\s*#.*?$\n?"
                else:
                    pattern_docstring = ""
                if pattern_docstring:
                    full_code = re.sub(pattern_docstring, '', full_code, flags=re.MULTILINE)
            testset.append({
                "dataset": f"codesearchnet-{lang}",
                "task_id": f"CSN/{lang}/{split}/{i}",
                "code": full_code,
                "test_payload": [],
                "entry_point": func_name
            })
        if nsamples == 0:
            _testset_cache[base_key + (False,)] = list(testset)
        needs_v = validate
    else:
        print(f"[*] Using cached testset (csn-{lang}, applying filters)")

    if needs_v:
        testset = _validate_testset(testset, lang, "syntax", nsamples)
    if nsamples == 0:
        _testset_cache[cache_key] = testset
    return testset[:nsamples] if nsamples > 0 else testset


def export_testset_to_text(testset, filename):
    print(f"[*] Total {len(testset)} samples loaded.")
    print(f"[*] Saving to text file: {filename}")

    with open(filename, "w", encoding="utf-8") as f:
        for idx, item in enumerate(testset):
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: {item['dataset']} | Task ID: {item['task_id']}\n")
            f.write(f"Entry Point: {item['entry_point']}\n")

            f.write("-" * 30 + " [ Code ] " + "-" * 30 + "\n")
            f.write(item['code'].strip() + "\n")

            f.write("-" * 30 + " [ Tests ] " + "-" * 30 + "\n")
            for test_line in item['test_payload']:
                f.write(test_line.strip() + "\n")

            f.write("\n\n")

    print("[*] Done.")


def export_testset_to_jsonl(testset, filename):
    """One {"url": task_id, "original_string": code} object per line."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"[*] Total {len(testset)} samples.")
    print(f"[*] Saving to JSONL: {filename}")

    with open(filename, "w", encoding="utf-8") as f:
        for item in testset:
            obj = {
                "url": item['task_id'],
                "original_string": item['code'],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[*] Done. {len(testset)} lines written.")


def aggregate_testset(testset, k=4, lang="py"):
    """Concatenate each group of k consecutive tasks into one task, for more L2 context."""
    print(f"[*] Aggregating testset into clusters of size k={k}...")
    aggregated_set = []

    # all clusters have exactly k items
    for i in range(0, len(testset) // k):
        cluster = testset[i*k:(i+1)*k]

        # Aggregated Task ID
        cluster_id = f"AGG_{cluster[0]['task_id']}_to_{cluster[-1]['task_id']}"

        # Merge source code and test payloads
        merged_code = "\n\n".join([item['code'] for item in cluster])
        merged_tests = []
        if lang == "js":
            # Wrap each test block in an IIFE, since every MBXP/JS block redeclares
            # const _ = require("lodash") and reuses names like arg00.
            for item in cluster:
                block = "\n".join(item['test_payload'])
                merged_tests.append(f";(function(){{\n{block}\n}})();")
        else:
            for item in cluster:
                merged_tests.extend(item['test_payload'])

        aggregated_set.append({
            "dataset": f"{cluster[0]['dataset']}_aggregated",
            "task_id": cluster_id,
            "code": merged_code,
            "test_payload": merged_tests,
            "entry_point": "multi_entry"
        })

    return aggregated_set


def get_testset_stats(testset: List[Dict], lang: str) -> Tuple[float, float]:
    """Return (avg_ncloc, avg_funcs) per test case: non-blank non-comment lines and top-level functions.
    Python uses tokenize rather than ast.parse, which warns on invalid escapes in CSN samples."""
    assert testset
    import tokenize, io
    from language_parser import PythonParser, JSParser, JavaParser

    parser_cls = {
        "py": PythonParser,
        "js": JSParser,
        "java": JavaParser,
    }.get(lang)
    if parser_cls is None:
        raise ValueError(f"get_testset_stats: unsupported lang '{lang}'")
    parser = parser_cls()

    PY_TRIVIAL = (
        tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
        tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
    )

    def strip_comments(code: str) -> str:
        """Blank out comments, preserving newlines, so line counting gives NCLOC."""
        if lang == "py":
            try:
                tokens = list(tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline))
            except (tokenize.TokenError, IndentationError, SyntaxError):
                return code
            lines = code.split('\n')
            for tok in tokens:
                if tok.type != tokenize.COMMENT:
                    continue
                row, col = tok.start
                if 0 <= row - 1 < len(lines):
                    line = lines[row - 1]
                    lines[row - 1] = line[:col] + ' ' * (len(line) - col)
            return '\n'.join(lines)

        try:
            tree = parser.parser.parse(code.encode('utf-8'))
        except Exception:
            return code
        buf = bytearray(code.encode('utf-8'))

        def walk(node):
            if "comment" in node.type:
                for i in range(node.start_byte, node.end_byte):
                    if buf[i] != 0x0A:
                        buf[i] = 0x20
                return
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        try:
            return buf.decode('utf-8')
        except UnicodeDecodeError:
            return code

    def count_py_top_funcs(code: str) -> int:
        """Count top-level def / async def tokens."""
        try:
            tokens = list(tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline))
        except (tokenize.TokenError, IndentationError):
            return 0
        count = 0
        for i, tok in enumerate(tokens):
            if not (tok.type == tokenize.NAME and tok.string == "def"):
                continue
            if tok.start[1] == 0:  # bare def foo():
                count += 1
                continue
            # async def: async must be at col 0
            j = i - 1
            while j >= 0 and tokens[j].type in PY_TRIVIAL:
                j -= 1
            if (j >= 0
                    and tokens[j].type == tokenize.NAME
                    and tokens[j].string == "async"
                    and tokens[j].start[1] == 0):
                count += 1
        return count

    total_ncloc, total_functions, parse_errors = 0, 0, 0
    for item in testset:
        code = item.get("code", "")
        stripped = strip_comments(code)
        total_ncloc += sum(1 for line in stripped.split('\n') if line.strip())
        try:
            if lang == "py":
                total_functions += count_py_top_funcs(code)
            else:
                chunks = parser.get_code_chunks(code)
                total_functions += sum(1 for is_func, _code, _name in chunks if is_func)
        except Exception:
            parse_errors += 1

    if parse_errors:
        print(f"[!] get_testset_stats: {parse_errors}/{len(testset)} samples failed to parse")

    ncloc = total_ncloc / len(testset)
    avg_funcs = total_functions / len(testset)
    return ncloc, avg_funcs


def _format_cache_key(key):
    """Format cache key tuple for display. e.g. ('csn','python','test',True) -> 'csn/python/test validated=True'"""
    *base, validate = key
    base_str = "/".join(str(b) for b in base)
    return f"{base_str} validated={validate}"


def save_testset_cache():
    """Save _testset_cache to disk as a pickle file."""
    os.makedirs(TESTSET_CACHE_DIR, exist_ok=True)
    with open(TESTSET_CACHE_FILE, "wb") as f:
        pickle.dump(_testset_cache, f)
    print(f"[*] Saved testset cache ({len(_testset_cache)} entries)")
    for key, val in _testset_cache.items():
        print(f"    {_format_cache_key(key):40s} ({len(val)} samples)")


def load_testset_cache():
    """Load _testset_cache from disk pickle file."""
    _testset_cache.clear()
    if os.path.exists(TESTSET_CACHE_FILE):
        with open(TESTSET_CACHE_FILE, "rb") as f:
            _testset_cache.update(pickle.load(f))
        print(f"[*] Loaded testset cache ({len(_testset_cache)} entries)")
        for key, val in _testset_cache.items():
            print(f"    {_format_cache_key(key):40s} ({len(val)} samples)")
    else:
        print(f"[*] No testset cache found")


def load_testset(dataset: str, lang: str = "py", nsamples: int = 0, fold: int = 1, validate: bool = False):
    """Load and optionally aggregate a testset.
    validate=True drops samples that fail their tests (benchmarks) or do not parse (CSN)."""
    if dataset == "mbpp":
        testset = load_mbpp_testset(nsamples=nsamples, validate=validate)
    elif dataset == "mbjsp":
        testset = load_mbjsp_testset(nsamples=nsamples, validate=validate)
    elif dataset == "mbjp":
        testset = load_mbjp_testset(nsamples=nsamples, validate=validate)

    elif dataset == "MBXP" and lang == "py":
        testset = load_mbpp_testset(nsamples=nsamples, validate=validate)
    elif dataset == "MBXP" and lang == "js":
        testset = load_mbjsp_testset(nsamples=nsamples, validate=validate)
    elif dataset == "MBXP" and lang == "java":
        testset = load_mbjp_testset(nsamples=nsamples, validate=validate)

    elif dataset == "humaneval-x":
        testset = load_humaneval_x_testset(lang, nsamples=nsamples, validate=validate)
    elif dataset == "csn":
        testset = load_codesearchnet_testset(lang, nsamples=nsamples, validate=validate)
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'")

    if fold > 1:
        testset = aggregate_testset(testset, k=fold, lang=lang)

    return testset


def build_testset_cache():
    ALL_DATASETS = ["MBXP", "csn", "humaneval-x"]
    ALL_LANGS = ["py", "js", "java"]

    for dataset in ALL_DATASETS:
        for lang in ALL_LANGS:
            load_testset(dataset, lang=lang, nsamples=0, validate=True)
    save_testset_cache()


if __name__ == "__main__":
    DATASETS = [
        "MBXP",
        #"csn",
        #"humaneval-x",
    ]
    LANGS = [
        "py",
        #"js",
        #"java",
    ]

    # text: code/tests per sample for manual inspection
    # jsonl: {url, original_string}
    EXPORT_TEXT = True
    EXPORT_JSONL = False

    for dataset in DATASETS:
        for lang in LANGS:
            testset = load_testset(dataset, lang=lang, nsamples=200)
            if EXPORT_TEXT:
                export_testset_to_text(testset, OUT_PATH(f"dump_{dataset}_{lang}.txt"))
            if EXPORT_JSONL:
                export_testset_to_jsonl(testset, OUT_PATH(f"{dataset}_{lang}.jsonl"))
