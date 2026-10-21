from __future__ import annotations

import ast
import builtins as _builtins
import json
import random
import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from attacks import LANG_NAMES, SYSTEM_PROMPT_IDENT, USER_PROMPT_IDENT, parse_watermark_list
from llm_provider import LLMProvider, get_provider, get_usage, model_tag, reset_usage
from paths import BASELINE_PATH, OUT_PATH


SYSTEMS = ["l1", "l1l2"]


def _out_path(*parts) -> Path:
    return Path(OUT_PATH("stealthiness", *parts))


def _samples_path(lang: str, dataset: str, embed_model: str) -> Path:
    return _out_path(f"samples_{lang}_{dataset}_{model_tag(embed_model)}.jsonl")


def build_samples(lang: str, dataset: str, nsample: int, embed_model: str, *, n_workers=1) -> Path:
    """Draw nsample testset functions with a fixed seed, embed l1 and l1l2 into each, and write
    samples_<lang>_<dataset>_<embed tag>.jsonl. Samples whose embedding fails are dropped."""

    from testset import load_testset
    from watermarker import Watermarker

    out_path = _samples_path(lang, dataset, embed_model)
    print(f"[*] build_samples({lang=}, {dataset=}, {nsample=}, {embed_model=})", flush=True)

    testset = load_testset(dataset, lang=lang, nsamples=0, fold=1, validate=False)
    by_id = {item["task_id"]: item for item in testset}
    chosen = random.Random(0).sample(sorted(by_id), min(nsample, len(by_id)))
    records = [{"id": tid, "original": by_id[tid]["code"]} for tid in chosen]
    print(f"[*] sampled {len(records)} from {len(by_id)}", flush=True)

    wms = [Watermarker(lang=lang, llm_model=embed_model) for _ in range(max(1, n_workers))]

    def _embed(idx: int, rec: dict) -> None:
        wm = wms[idx % len(wms)]
        for system, do_l2 in (("l1", False), ("l1l2", True)):
            try:
                rec[system] = wm.embed(rec["original"], do_l1=True, do_l2=do_l2)
            except Exception as e:
                rec[f"{system}_error"] = f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_embed, i, r) for i, r in enumerate(records)]
        for done, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if done % 10 == 0 or done == len(records):
                print(f"  embed: {done}/{len(records)}", flush=True)

    kept = [r for r in records if all(r.get(s) for s in SYSTEMS)]
    print(f"[*] after embed: {len(kept)}/{len(records)} OK", flush=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": True, "lang": lang, "dataset": dataset, "n": len(kept),
                            "nsample_requested": nsample, "embed_model": embed_model}) + "\n")
        for r in kept:
            f.write(json.dumps({k: r[k] for k in ("id", "original", *SYSTEMS)}, ensure_ascii=False) + "\n")
    print(f"[*] wrote {len(kept)} samples -> {out_path}", flush=True)
    return out_path


def _load_samples(path: Path) -> tuple[dict, list[dict]]:
    with path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines or not lines[0].get("_meta"):
        raise ValueError(f"{path}: missing _meta header")
    return lines[0], lines[1:]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: str) -> tuple[str, str]:
    """(verdict, reason) from a {"verdict": ..., "reason": ...} response. The verdict is a single
    uppercase letter, TIE for tie / equal / draw / same, or parse_error."""
    if not raw:
        return "parse_error", "empty response"
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = _JSON_RE.search(text)
    if not m:
        return "parse_error", f"no JSON object in: {raw[:120]!r}"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return "parse_error", f"json decode error: {e}"
    raw_v = str(obj.get("verdict", "")).strip()
    reason = str(obj.get("reason", ""))[:300]
    if raw_v.lower() in ("tie", "equal", "draw", "same"):
        return "TIE", reason
    if len(raw_v) == 1 and raw_v.upper().isalpha():
        return raw_v.upper(), reason
    return "parse_error", f"unrecognized verdict: {raw_v!r}"


@dataclass
class Judgement:
    item_id: str
    system: str
    original_slot: str   # "A" or "B"
    verdict: str         # "A" / "B" / "TIE" / "parse_error"
    reason: str
    winner: str = field(init=False)

    def __post_init__(self) -> None:
        if self.verdict == "TIE":
            self.winner = "tie"
        elif self.verdict == self.original_slot:
            self.winner = "original"
        elif self.verdict in ("A", "B"):
            self.winner = "modified"
        else:
            self.winner = "parse_error"


SYSTEM_PROMPT_AB = textwrap.dedent("""
    You judge whether {lang_name} code looks like genuine human-written source.

    GOAL
    You will be shown two functionally equivalent {lang_name} code blocks,
    labeled A and B. One is original human-written code; the other has been
    mechanically transformed (watermarking, obfuscation, or learned rewriting)
    and may carry unnatural naming, awkward stylistic choices, or non-idiomatic
    constructs. Identify which block is more likely to have been written by a
    human.

    OUTPUT RULE
    - Output ONLY one JSON object using exactly this schema:
      {{"verdict": "A" | "B" | "tie", "reason": "<one short sentence>"}}
    - "A" or "B" indicates the more natural / human-written block.
    - "tie" means the two are genuinely indistinguishable.
    - reason: under ~25 words; cite the specific cue (e.g. naming, comments,
      errors, imports, operator, etc.).
    - No explanations or markdown fences outside the JSON object.
""").strip()

USER_PROMPT_AB = textwrap.dedent("""
    Below are two {lang_name} code blocks. Judge which is more natural.

    <code_a>
    {code_a}
    </code_a>

    <code_b>
    {code_b}
    </code_b>

    Return the JSON verdict now.
""").strip()


def _judge_ab(provider: LLMProvider, rec: dict, system: str, lang: str,
              original_first: bool) -> Judgement:
    original_slot = "A" if original_first else "B"
    try:
        if original_first:
            code_a, code_b = rec["original"], rec[system]
        else:
            code_a, code_b = rec[system], rec["original"]
        lang_name = LANG_NAMES.get(lang, lang)
        raw = provider.generate(SYSTEM_PROMPT_AB.format(lang_name=lang_name),
                                USER_PROMPT_AB.format(lang_name=lang_name, code_a=code_a, code_b=code_b))
        verdict, reason = parse_verdict(raw)
    except Exception as e:
        verdict, reason = "parse_error", f"exception: {type(e).__name__}: {e}"
    return Judgement(item_id=str(rec["id"]), system=system,
                     original_slot=original_slot, verdict=verdict, reason=reason)


def run_ab(lang: str, dataset: str, embed_model: str, provider: LLMProvider, *, n_workers: int = 4) -> list[dict]:
    """Judge every sample of one (lang, dataset) for each system, write
    ab_<lang>_<dataset>_<system>_<embed tag>_<judge tag>.jsonl, and return one summary per system."""

    _, records = _load_samples(_samples_path(lang, dataset, embed_model))
    judge = getattr(provider, "model", None)
    print(f"[*] run_ab({lang}/{dataset}): {len(records)} samples x {len(SYSTEMS)} systems", flush=True)

    summaries: list[dict] = []
    for system in SYSTEMS:
        # Original in slot A for exactly half the records. A string seed is hashed with sha512,
        # so unlike hash() it gives the same order in every process.
        orderings = [i < len(records) // 2 for i in range(len(records))]
        random.Random(system).shuffle(orderings)

        ordered: list[Optional[Judgement]] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_judge_ab, provider, r, system, lang, orderings[i]): i
                    for i, r in enumerate(records)}
            for fut in as_completed(futs):
                ordered[futs[fut]] = fut.result()

        out_path = _out_path(f"ab_{lang}_{dataset}_{system}_{model_tag(embed_model)}_{model_tag(judge)}.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": True, "lang": lang, "dataset": dataset, "system": system,
                                "embed_model": embed_model, "judge": judge, "n": len(ordered)}) + "\n")
            for j in ordered:
                f.write(json.dumps({"id": j.item_id, "system": j.system,
                                    "original_slot": j.original_slot, "verdict": j.verdict,
                                    "winner": j.winner, "reason": j.reason}) + "\n")

        counts = {w: sum(1 for j in ordered if j.winner == w)
                  for w in ("original", "modified", "tie", "parse_error")}
        valid = len(ordered) - counts["parse_error"]
        summaries.append({
            "system": system, "n": len(ordered), "valid": valid,
            "orig_win": counts["original"], "mod_win": counts["modified"],
            "tie": counts["tie"], "err": counts["parse_error"],
            "orig_win_pct": counts["original"] / valid * 100 if valid else None,
        })
    return summaries


def print_summary(results: dict[tuple[str, str], list[dict]], *, nsample: int,
                  embed_model: str, judge: str, start: datetime, end: datetime) -> None:
    """One OrigWin% table: rows are systems, columns datasets. A cell whose valid count is below
    nsample shows that count in parentheses."""
    combos = list(results)
    labels = [f"{lang}/{dataset}" for lang, dataset in combos]
    rows = []
    for system in SYSTEMS:
        cells, vals = [], []
        for combo in combos:
            s = next((x for x in results[combo] if x["system"] == system), None)
            if s is None or s["orig_win_pct"] is None:
                cells.append("-")
                continue
            pct = s["orig_win_pct"]
            vals.append(pct)
            cells.append(f"{pct:.1f}" if s["valid"] == nsample else f"{pct:.1f} (n={s['valid']})")
        rows.append((system, cells, f"{sum(vals) / len(vals):.1f}" if vals else "-"))

    widths = [max(len(label), *(len(r[1][i]) for r in rows)) + 2 for i, label in enumerate(labels)]
    header = f"{'System':<8}" + "".join(f"{label:<{w}}" for label, w in zip(labels, widths)) + "Avg"
    bar = "=" * max(len(header), 60)
    print(f"\n{bar}")
    print(" SWIFT A/B  OrigWin%")
    print(f" embed: {embed_model}   judge: {judge}")
    print(f" start: {start:%Y-%m-%d %H:%M:%S}   end: {end:%Y-%m-%d %H:%M:%S}   "
          f"({str(end - start).split('.')[0]})")
    print(bar)
    print(header)
    print("-" * len(bar))
    for system, cells, avg in rows:
        print(f"{system:<8}" + "".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + avg)
    print(bar)
    usage = get_usage()
    print(f" LLM usage (judge): calls={usage['n_calls']:,}  prompt={usage['prompt_tokens']:,}  "
          f"completion={usage['completion_tokens']:,}\n", flush=True)


def build_samples_all(embed_models: list[str], lang_datasets: list[tuple[str, str]], *, nsample=0, n_workers=1) -> None:
    """Write one samples file per (embed model, lang, dataset)."""
    for embed_model in embed_models:
        for lang, dataset in lang_datasets:
            build_samples(lang, dataset, nsample, embed_model, n_workers=n_workers)


def judge_samples(embed_models: list[str], judges: list[str], lang_datasets: list[tuple[str, str]], *, nsample=0, n_workers=1) -> None:
    """Judge the samples of every embed model with every judge and print one table per pair."""
    for embed_model in embed_models:
        for judge_model in judges:
            provider = get_provider(model=judge_model)
            reset_usage()
            start = datetime.now()
            results = {(lang, dataset): run_ab(lang, dataset, embed_model, provider, n_workers=n_workers)
                       for lang, dataset in lang_datasets}
            print_summary(results, nsample=nsample, embed_model=embed_model, judge=judge_model,
                          start=start, end=datetime.now())


# (lang, dataset, system) -> baseline JSONL under baseline_outputs/, not distributed.
# The line format is in baseline_outputs/BASELINE_OUTPUTS.md.
BASELINE_JSONL = {
    ("py",   "MBXP", "codemark"):  "codemark_mbpp.jsonl",
    ("py",   "csn",  "codemark"):  "codemark_csn_py.jsonl",
    ("js",   "MBXP", "codemark"):  "codemark_mbjsp.jsonl",
    ("js",   "MBXP", "srcmarker"): "srcmarker_mbjsp.jsonl",
    ("js",   "csn",  "codemark"):  "codemark_csn_js.jsonl",
    ("js",   "csn",  "srcmarker"): "srcmarker_csn_js.jsonl",
    ("java", "MBXP", "codemark"):  "codemark_mbjp.jsonl",
    ("java", "MBXP", "srcmarker"): "srcmarker_mbjp.jsonl",
    ("java", "csn",  "codemark"):  "codemark_csn_java.jsonl",
    ("java", "csn",  "srcmarker"): "srcmarker_csn_java.jsonl",
}


# Variable extraction: user-defined variable names only, without builtins, declared function
# and class names, and (JS / Java) package qualifiers, method names and field-access slots.

_PY_BUILTINS = set(dir(_builtins)) | {"True", "False", "None"}
# self / cls are kept: CodeMark renames variables to these names, so they can be carriers.


def _extract_py_vars(code: str) -> set[str]:
    """Variable names in a Python snippet, without builtins, defined function/class names and dunders."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    excluded: set[str] = set(_PY_BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            excluded.add(node.name)
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.arg):
            seen.add(node.arg)
    return {v for v in seen if v not in excluded and not (v.startswith("__") and v.endswith("__"))}


# JS / Java collect every identifier node. property_identifier (JS) and type_identifier (Java)
# are separate node types, so obj.foo contributes only obj.

_JS_BUILTINS = {
    "console", "Math", "Number", "String", "Array", "Object", "Map", "Set",
    "Promise", "JSON", "undefined", "null", "true", "false", "this",
    "arguments", "RegExp", "Date", "parseInt", "parseFloat", "isNaN",
    "isFinite", "Boolean", "Error", "Symbol", "TypeError", "RangeError",
    "globalThis", "NaN", "Infinity", "require", "module", "exports",
}

_JAVA_BUILTINS = {
    "System", "String", "Integer", "Boolean", "Object", "List", "Map", "Set",
    "ArrayList", "HashMap", "HashSet", "LinkedList", "TreeMap", "TreeSet",
    "Math", "Long", "Double", "Float", "Character", "Byte", "Short", "Number",
    "Arrays", "Collections", "Iterator", "Iterable", "Optional", "this",
    "null", "true", "false", "out", "println", "print", "in", "err",
    "Exception", "RuntimeException", "Throwable", "Comparable", "Comparator",
    "StringBuilder", "StringBuffer", "Stream", "Collectors", "Function",
    "Consumer", "Supplier", "Predicate", "Objects", "BigInteger", "BigDecimal",
}

_js_parser_cache = None
_java_parser_cache = None


def _get_js_parser():
    global _js_parser_cache
    if _js_parser_cache is None:
        from language_parser import JSParser
        _js_parser_cache = JSParser().parser
    return _js_parser_cache


def _get_java_parser():
    global _java_parser_cache
    if _java_parser_cache is None:
        from language_parser import JavaParser
        _java_parser_cache = JavaParser().parser
    return _java_parser_cache


def _ts_collect_identifiers(parser, code: str, wrap_on_error: bool = False,
                             skip_subtree_types: Optional[set[str]] = None,
                             skip_field_for_parent: Optional[dict[str, set[str]]] = None,
                             skip_inside_types: Optional[set[str]] = None) -> set[str]:
    """Every identifier token in code, except those under skip_subtree_types, in a field of
    skip_field_for_parent ({parent_type: {field}}), or directly under skip_inside_types.
    wrap_on_error retries inside class __W { ... } for Java method fragments."""
    skip_subtree_types = skip_subtree_types or set()
    skip_field_for_parent = skip_field_for_parent or {}
    skip_inside_types = skip_inside_types or set()

    tree = parser.parse(code.encode("utf-8"))
    if tree.root_node.has_error and wrap_on_error:
        wrapped = "class __W { " + code + " }"
        tree = parser.parse(wrapped.encode("utf-8"))
    names: set[str] = set()

    def walk(node):
        if node.type in skip_subtree_types:
            return
        if node.type == "identifier":
            parent = node.parent
            skip = False
            if parent is not None:
                if parent.type in skip_inside_types:
                    skip = True
                fields = skip_field_for_parent.get(parent.type)
                if fields:
                    for fname in fields:
                        ch = parent.child_by_field_name(fname)
                        if ch is not None and ch.start_byte == node.start_byte and ch.end_byte == node.end_byte:
                            skip = True
                            break
            if not skip:
                names.add(node.text.decode("utf-8", errors="replace"))
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return names


def _collect_declared_names(parser, code: str, wrap_on_error: bool,
                            declarator_types: set[str]) -> set[str]:
    """Names declared by nodes of declarator_types (functions, methods, classes)."""
    tree = parser.parse(code.encode("utf-8"))
    if tree.root_node.has_error and wrap_on_error:
        wrapped = "class __W { " + code + " }"
        tree = parser.parse(wrapped.encode("utf-8"))
    decl: set[str] = set()
    def walk(node):
        if node.type in declarator_types:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                decl.add(name_node.text.decode("utf-8", errors="replace"))
        for c in node.children:
            walk(c)
    walk(tree.root_node)
    return decl


_JS_SKIP_SUBTREE = {"import_statement", "export_statement"}
_JS_SKIP_FIELDS = {
    # member_expression.property is already a property_identifier, kept in case the grammar changes.
    "member_expression": {"property"},
}

_JAVA_SKIP_SUBTREE = {"import_declaration", "package_declaration"}
_JAVA_SKIP_FIELDS = {
    "method_invocation": {"name"},
    "field_access": {"field"},
}
# java.util.Map is one scoped_identifier made of plain identifiers, none of them a variable.
_JAVA_SKIP_INSIDE = {"scoped_identifier"}


def _extract_js_vars(code: str) -> set[str]:
    parser = _get_js_parser()
    raw = _ts_collect_identifiers(
        parser, code,
        skip_subtree_types=_JS_SKIP_SUBTREE,
        skip_field_for_parent=_JS_SKIP_FIELDS,
    )
    decl = _collect_declared_names(parser, code, wrap_on_error=False,
                                    declarator_types={"function_declaration",
                                                      "method_definition",
                                                      "class_declaration"})
    return {n for n in raw if n not in _JS_BUILTINS and n not in decl}


def _extract_java_vars(code: str) -> set[str]:
    parser = _get_java_parser()
    raw = _ts_collect_identifiers(
        parser, code, wrap_on_error=True,
        skip_subtree_types=_JAVA_SKIP_SUBTREE,
        skip_field_for_parent=_JAVA_SKIP_FIELDS,
        skip_inside_types=_JAVA_SKIP_INSIDE,
    )
    decl = _collect_declared_names(parser, code, wrap_on_error=True,
                                    declarator_types={"method_declaration",
                                                      "class_declaration",
                                                      "constructor_declaration"})
    return {n for n in raw if n not in _JAVA_BUILTINS and n not in decl}


_VAR_EXTRACTORS = {
    "py": _extract_py_vars,
    "js": _extract_js_vars,
    "java": _extract_java_vars,
}


@dataclass
class VarIdentItem:
    item_id: str
    system: str
    code: str
    all_vars: list[str]      # shown to the judge
    true_wm: list[str]       # vars(wm) - vars(orig)


@dataclass
class VarIdentJudgement:
    item_id: str
    system: str
    all_vars: list[str]
    true_wm: list[str]
    predicted: Optional[list[str]]    # None on parse_error
    error: str

    @property
    def tp(self) -> int:
        if self.predicted is None: return 0
        return len(set(self.predicted) & set(self.true_wm))
    @property
    def fp(self) -> int:
        if self.predicted is None: return 0
        return len(set(self.predicted) - set(self.true_wm))
    @property
    def fn(self) -> int:
        if self.predicted is None: return 0
        return len(set(self.true_wm) - set(self.predicted))


def _load_baseline_pairs(jsonl_name: str) -> list[dict]:
    """{id, original, watermarked} per line of baseline_outputs/<jsonl_name>.
    Uses the baseline's own original, since index-based pairing with the testset is wrong
    whenever the JSONL is a subset of it."""
    path = Path(BASELINE_PATH(jsonl_name))
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("task_id") or obj.get("id")
            orig = obj.get("original")
            wm = obj.get("watermarked")
            if tid is None or orig is None or wm is None:
                continue
            out.append({"id": str(tid), "original": orig, "watermarked": wm})
    return out


def _build_variable_identification_bucket(lang: str, system: str,
                                      jsonl_name: str,
                                      nsample: int) -> list[VarIdentItem]:
    """nsample rows drawn uniformly from the JSONL rows with a non-empty variable list and ground
    truth, with a fixed seed so every run judges the same rows. Returned in file order."""
    if system == "original":
        raise ValueError("variable-identification has no `original` bucket (empty ground truth)")
    extract = _VAR_EXTRACTORS.get(lang)
    if extract is None:
        raise ValueError(f"variable extraction not implemented for lang={lang!r}")

    pairs = _load_baseline_pairs(jsonl_name)
    order = list(range(len(pairs)))
    random.Random(0).shuffle(order)
    picked: list[tuple[int, VarIdentItem]] = []
    for idx in order:
        rec = pairs[idx]
        ovars = extract(rec["original"])
        wvars = extract(rec["watermarked"])
        true_wm = wvars - ovars
        if not true_wm or not wvars:
            continue
        picked.append((idx, VarIdentItem(
            item_id=rec["id"], system=system, code=rec["watermarked"],
            all_vars=sorted(wvars), true_wm=sorted(true_wm),
        )))
        if len(picked) >= nsample:
            break
    picked.sort(key=lambda x: x[0])
    return [it for _, it in picked]


def _judge_identification(provider: LLMProvider, item: VarIdentItem, lang: str) -> VarIdentJudgement:
    lang_name = LANG_NAMES.get(lang, lang)
    sysp = SYSTEM_PROMPT_IDENT.format(lang_name=lang_name)
    var_list_str = ", ".join(f"`{v}`" for v in item.all_vars)
    usrp = USER_PROMPT_IDENT.format(lang_name=lang_name, code=item.code, var_list=var_list_str)
    raw = provider.generate(sysp, usrp)
    picked, err = parse_watermark_list(raw, set(item.all_vars))
    return VarIdentJudgement(
        item_id=item.item_id, system=item.system,
        all_vars=item.all_vars, true_wm=item.true_wm,
        predicted=picked, error=err,
    )


def run_variable_identification(
    lang: str, dataset: str,
    target_systems: Optional[list[str]] = None,
    nsample: int = 100,
    provider: Optional[LLMProvider] = None,
    *,
    n_workers: int = 4,
) -> list[dict]:
    """Judge each baseline's bucket, write identification_<lang>_<dataset>_<system>_<judge tag>.jsonl
    per system, and print avg picked / correct / true counts with micro precision, recall and F1."""
    if provider is None:
        provider = get_provider(model="gpt-4o-mini")

    if target_systems is None:
        target_systems = [s for s in ("codemark", "srcmarker")
                          if (lang, dataset, s) in BASELINE_JSONL]
    else:
        target_systems = [s for s in target_systems if s != "original"]
    bad = [s for s in target_systems if (lang, dataset, s) not in BASELINE_JSONL]
    if bad:
        raise ValueError(f"no baseline jsonl for ({lang}, {dataset}, {bad})")
    if not target_systems:
        print(f"[!] run_variable_identification: no eligible baselines for ({lang}, {dataset})",
              flush=True)
        return []

    print(f"[*] run_variable_identification ({lang}/{dataset}): systems={target_systems} "
          f"(target n={nsample} each)", flush=True)

    reset_usage()
    summaries: list[dict] = []
    for system in target_systems:
        jsonl_name = BASELINE_JSONL[(lang, dataset, system)]
        items = _build_variable_identification_bucket(lang, system, jsonl_name, nsample)
        print(f"  [{system}] bucket: n={len(items)}/{nsample}", flush=True)
        if not items:
            summaries.append({"system": system, "n": 0})
            continue

        def _task(idx: int, it: VarIdentItem) -> VarIdentJudgement:
            try:
                return _judge_identification(provider, it, lang)
            except Exception as e:
                return VarIdentJudgement(
                    item_id=it.item_id, system=it.system,
                    all_vars=it.all_vars, true_wm=it.true_wm,
                    predicted=None, error=f"exception: {type(e).__name__}: {e}",
                )

        results: list[tuple[int, VarIdentJudgement]] = []
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_task, i, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                j = fut.result()
                with lock:
                    results.append((i, j))
                    done = len(results)
                if done % 20 == 0 or done == len(items):
                    print(f"  [{system}] {done}/{len(items)}", flush=True)

        results.sort(key=lambda x: x[0])
        ordered = [j for _, j in results]

        out_path = _out_path(f"identification_{lang}_{dataset}_{system}_{model_tag(provider.model)}.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": True, "lang": lang, "dataset": dataset,
                                "system": system, "n": len(ordered)}) + "\n")
            for j in ordered:
                f.write(json.dumps({
                    "id": j.item_id, "system": j.system,
                    "all_vars": j.all_vars, "true_wm": j.true_wm,
                    "predicted": j.predicted, "error": j.error,
                    "tp": j.tp, "fp": j.fp, "fn": j.fn,
                }, ensure_ascii=False) + "\n")

        # Parse errors are logged but left out of the averages, so a judge that fails to emit
        # JSON does not pull them toward zero.
        valid = [j for j in ordered if j.predicted is not None]
        err = len(ordered) - len(valid)
        n_valid = len(valid)
        if n_valid == 0:
            summaries.append({
                "system": system, "n": len(ordered), "valid": 0, "err": err,
                "avg_picked": 0.0, "avg_correct": 0.0, "avg_true": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            })
            continue

        sum_picked = sum(len(j.predicted) for j in valid)
        sum_true = sum(len(j.true_wm) for j in valid)
        sum_tp = sum(j.tp for j in valid)
        sum_fp = sum(j.fp for j in valid)
        sum_fn = sum(j.fn for j in valid)
        precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
        recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        summaries.append({
            "system": system, "n": len(ordered),
            "valid": n_valid, "err": err,
            "avg_picked": sum_picked / n_valid,
            "avg_correct": sum_tp / n_valid,
            "avg_true": sum_true / n_valid,
            "precision": precision, "recall": recall, "f1": f1,
        })

    print()
    print(f"=== Variable Identification (lang={lang}, dataset={dataset}) ===", flush=True)
    header = (f"{'System':<12}  {'avgPicked':>9}  {'avgCorrect':>10}  {'avgTrue':>8}  "
              f"{'P':>5}  {'R':>5}  {'F1':>5}  {'N':>4}  {'Err':>4}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        if s.get("valid", 0) == 0:
            print(f"{s['system']:<12}  {'-':>9}  {'-':>10}  {'-':>8}  "
                  f"{'-':>5}  {'-':>5}  {'-':>5}  {s.get('n',0):>4}  {s.get('err',0):>4}",
                  flush=True)
        else:
            print(f"{s['system']:<12}  {s['avg_picked']:>9.2f}  {s['avg_correct']:>10.2f}  "
                  f"{s['avg_true']:>8.2f}  "
                  f"{s['precision']:>5.2f}  {s['recall']:>5.2f}  {s['f1']:>5.2f}  "
                  f"{s['n']:>4}  {s['err']:>4}",
                  flush=True)
    print()

    usage = get_usage()
    print(f"[*] LLM usage: calls={usage['n_calls']} prompt={usage['prompt_tokens']} "
          f"completion={usage['completion_tokens']}", flush=True)
    return summaries


# A/B test: the judge sees a testset function and its SWIFT watermarked version (l1, l1l2) and picks
# the more human-written one. OrigWin% is the share of judgements that pick the original.
def eval_ab_test():
    lang_datasets = [(lang, dataset) for lang in ["py", "java", "js"] for dataset in ["MBXP", "csn"]]

    # Comment out an entry of MODELS_EMBED to reuse its cached samples without re-embedding.
    MODELS_EMBED = [
        "gpt-4o-mini",
        #"gemini-2.5-flash-lite",
        #"Qwen/Qwen2.5-7B-Instruct",
    ]

    build_samples_all(MODELS_EMBED, lang_datasets, nsample=100, n_workers=1)

    # Embed models whose samples files are judged. Each needs its samples built first.
    MODELS_TARGET = [
        "gpt-4o-mini",
        #"gemini-2.5-flash-lite",
        #"Qwen/Qwen2.5-7B-Instruct",
    ]
    # Judge models. Every target is judged by each of them, one summary table per pair.
    MODELS_JUDGE = [
        "gpt-4.1",
        #"claude-sonnet-4-6",
    ]

    judge_samples(MODELS_TARGET, MODELS_JUDGE, lang_datasets, nsample=100, n_workers=1)


# Carrier identification: the judge sees one SrcMarker / CodeMark function and its variable list and
# picks the watermarked variables, scored against vars(watermarked) - vars(original).
def eval_carrier_identification():
    JUDGE_MODEL = "gpt-4.1"
    #JUDGE_MODEL = "claude-sonnet-4-6"

    lang_datasets = [(lang, dataset) for lang in ["py", "java", "js"] for dataset in ["MBXP", "csn"]]
    judge = get_provider(model=JUDGE_MODEL)

    print(f"\n############ run_variable_identification (judge={JUDGE_MODEL}) ############", flush=True)
    for lang, dataset in lang_datasets:
        run_variable_identification(lang=lang, dataset=dataset, nsample=100, n_workers=1, provider=judge)


if __name__ == "__main__":
    eval_ab_test()
    eval_carrier_identification()
