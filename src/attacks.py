import textwrap
import ast
import json
import re
import string
import builtins
import threading

import tree_sitter_javascript as tsjs
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from attack_tools import attack_decompile, attack_minify, attack_obfuscate
from language_parser import get_parser
from llm_provider import LLMProvider, get_provider
from stats import get_rng, stochastic_round

ATTACK_FLAGS = ("R", "L", "T", "ST", "W", "MINI", "OBFU", "DCMP")

# These read their option as a percentage, the rest read it as letters.
_RATIO_FLAGS = ("R", "L", "T", "ST")

_ATTACK_ITEM_RE = re.compile(r"([A-Z]+)(?:-([A-Z0-9]+))?")


def apply_attacks(code: str, attack_flags: str, lang: str, *, llm_model: str | None = None) -> str:
    """
    Apply attack(s) to watermarked code based on flag string.
      "R"    = random renaming      (meaningless names, e.g. v_xkzy_0)
      "L"    = LLM-based renaming   (natural, context-fitting names)
      "T"    = targeted renaming    (LLM locates the exposed carriers first)
      "ST"   = syntactic transform  (attacker knows a portion of the rewriter rules)
      "W"    = LLM structural rewrite   (one call over the whole file)
               "W-S" runs it on W_STRONG_MODEL instead of llm_model
      "MINI" = the language's minifier   (js: terser, py: python-minifier)
               passes: M mangle, C compress, F format-only (default "MC")
      "OBFU" = javascript-obfuscator     (js only)
      "DCMP" = javac + a decompiler      (java only)
               options: V pick Vineflower over CFR, G compile with -g
    MINI, OBFU and DCMP live in attack_tools. They shell out to an external tool
    and raise on a language (or, for DCMP, a dataset) they do not cover.
    llm_model selects the LLM used by L, T and W (None = DEFAULT_MODEL).
    """
    if not attack_flags:
        return code

    for item in attack_flags.replace(" ", "").split(","):
        match = _ATTACK_ITEM_RE.fullmatch(item)
        if not match or match.group(1) not in ATTACK_FLAGS:
            raise ValueError(f"Malformed attack: '{item}' in '{attack_flags}'")
        flag, option = match.group(1), match.group(2)

        if flag in _RATIO_FLAGS:
            if option is None:
                ratio = 1.0
            elif option.isdigit():
                ratio = int(option) / 100.0
            else:
                raise ValueError(f"'{flag}' takes a percentage, got '{option}'")

        if flag == "R":
            code = attack_random_renaming(code, lang=lang, ratio=ratio)
        elif flag == "L":
            provider = get_provider(llm_model)
            code = attack_llm_renaming(code, provider, lang=lang, ratio=ratio)
        elif flag == "T":
            provider = get_provider(llm_model)
            code = attack_targeted_renaming(code, provider, lang=lang, ratio=ratio)
        elif flag == "ST":
            code = attack_transform(code, lang=lang, ratio=ratio)
        elif flag == "W":
            if option not in (None, "S"):
                raise ValueError(f"'W' takes no option or 'S' (strong model), got '{option}'")
            provider = get_provider(W_STRONG_MODEL if option == "S" else llm_model)
            code = attack_llm_rewrite(code, provider, lang=lang)
        elif flag == "MINI":
            passes = option if option is not None else "MC"
            if set(passes) - set("MCF"):
                raise ValueError(f"'MINI' passes are M, C and F, got '{passes}'")
            code = attack_minify(code, lang=lang, mangle="M" in passes,
                                 compress="C" in passes, fmt="F" in passes)
        elif flag == "OBFU":
            if option is not None:
                raise ValueError(f"'OBFU' takes no option, got '{option}'")
            code = attack_obfuscate(code, lang=lang)
        elif flag == "DCMP":
            letters = option or ""
            if set(letters) - set("CVG"):
                raise ValueError(f"'DCMP' options are C, V and G, got '{option}'")
            if {"C", "V"} <= set(letters):
                raise ValueError("'DCMP' takes one decompiler, got both C and V")
            code = attack_decompile(
                code, lang=lang,
                decompiler="vineflower" if "V" in letters else "cfr",
                debug_info="G" in letters)
        else:
            raise ValueError(f"Unknown attack: '{flag}'")

    return code


def attack_random_renaming(original_code: str, lang: str = "py", ratio: float = 1.0, *, verbose: bool = False) -> str:
    """Random variable renaming attack. ratio=0.0~1.0 controls fraction of variables renamed."""
    if lang == "py":
        targets = _find_local_names_py(original_code)
    elif lang in ("js", "java"):
        targets = _find_local_names_ts(original_code, lang)
    else:
        raise ValueError(f"Unsupported language: '{lang}'")

    if not targets:
        if verbose: print("[*] Attack Skipped: No target variables found.")
        return original_code

    sorted_targets = sorted(targets)
    k = stochastic_round(len(sorted_targets) * ratio)
    assert k <= len(sorted_targets)
    selected = get_rng().sample(sorted_targets, k)

    if not selected:
        return original_code

    rename_map = {name: _get_random_varname(i) for i, name in enumerate(selected)}
    if verbose: print(f"[*] Random Rename Mapping ({len(selected)}/{len(sorted_targets)}): {rename_map}")

    if lang == "py":
        return _replace_vars_py(original_code, rename_map, verbose=verbose)
    else:
        return _replace_vars_ts(original_code, lang, rename_map, verbose=verbose)


def _find_local_names_py(original_code: str) -> set:
    try:
        tree = ast.parse(original_code)
    except SyntaxError:
        return set()

    exclude_names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exclude_names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                exclude_names.add(alias.asname if alias.asname else alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exclude_names.add(alias.asname if alias.asname else alias.name)

    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in exclude_names:
            targets.add(node.id)
        elif isinstance(node, ast.arg) and node.arg not in exclude_names:
            targets.add(node.arg)
    return targets


def _find_local_names_ts(original_code: str, lang: str) -> set:
    if lang == "js":
        ts_lang = Language(tsjs.language())
    elif lang == "java":
        ts_lang = Language(tsjava.language())
    else:
        return set()

    parser = Parser(ts_lang)
    src_bytes = bytes(original_code, 'utf8')
    tree = parser.parse(src_bytes)

    declared_names = set()
    exclude_names = set()

    def collect_declarations(node):
        if node.type == 'variable_declarator':
            name_node = node.child_by_field_name('name')
            if name_node and name_node.type == 'identifier':
                name = name_node.text.decode('utf8')
                value_node = node.child_by_field_name('value')
                if value_node and value_node.type in ('arrow_function', 'function_expression', 'function'):
                    exclude_names.add(name)
                else:
                    declared_names.add(name)
        if lang == "js" and node.type == 'formal_parameters':
            for child in node.named_children:
                if child.type == 'identifier':
                    declared_names.add(child.text.decode('utf8'))
        if lang == "java" and node.type == 'formal_parameter':
            name_node = node.child_by_field_name('name')
            if name_node and name_node.type == 'identifier':
                declared_names.add(name_node.text.decode('utf8'))
        if node.type in ('function_declaration', 'method_declaration', 'class_declaration'):
            name_node = node.child_by_field_name('name')
            if name_node:
                exclude_names.add(name_node.text.decode('utf8'))
        if lang == "js":
            # A name used as an object key escapes local scope: renaming it changes the key or forces {x} to expand into {x: y}.
            if node.type == 'shorthand_property_identifier':
                exclude_names.add(node.text.decode('utf8'))
            elif node.type == 'pair':
                key = node.child_by_field_name('key')
                val = node.child_by_field_name('value')
                if key and val and val.type == 'identifier' and key.text == val.text:
                    exclude_names.add(key.text.decode('utf8'))
        for child in node.children:
            collect_declarations(child)

    collect_declarations(tree.root_node)
    return declared_names - exclude_names


def _get_random_varname(idx: int, length: int = 4) -> str:
    return f"v_{''.join(get_rng().choices(string.ascii_lowercase, k=length))}_{idx}"


def _replace_vars_py(original_code: str, rename_map: dict, *, verbose: bool = False) -> str:
    """Apply rename_map to Python code via AST transformation."""
    try:
        tree = ast.parse(original_code)
    except SyntaxError:
        if verbose: print("[!] Replace failed: Syntax error.")
        return original_code

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in rename_map:
                return ast.copy_location(ast.Name(id=rename_map[node.id], ctx=node.ctx), node)
            return node
        def visit_arg(self, node):
            if node.arg in rename_map:
                node.arg = rename_map[node.arg]
            return node

    new_tree = Renamer().visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except Exception as e:
        if verbose: print(f"[!] Replace failed during unparsing: {e}")
        return original_code


def _replace_vars_ts(original_code: str, lang: str, rename_map: dict, *, verbose: bool = False) -> str:
    """Apply rename_map to JS/Java code via tree-sitter text replacement.
    Only replaces identifier nodes in safe positions (not properties, methods, types)."""
    if lang not in ("js", "java"):
        return original_code

    ts_lang = Language(tsjs.language()) if lang == "js" else Language(tsjava.language())
    parser = Parser(ts_lang)
    src_bytes = bytes(original_code, 'utf8')
    tree = parser.parse(src_bytes)

    replacements = []

    def is_safe_replacement(node):
        # Java method names and accessed fields are plain identifiers. JS properties and Java types have their own node types.
        parent = node.parent
        if not parent:
            return True

        if parent.type == 'method_invocation':
            if parent.child_by_field_name('name') == node:
                return False

        if parent.type == 'field_access':
            if parent.child_by_field_name('field') == node:
                return False

        return True

    def collect_replacements(node):
        if node.type == 'identifier':
            name = node.text.decode('utf8')
            if name in rename_map and is_safe_replacement(node):
                replacements.append((node.start_byte, node.end_byte, rename_map[name]))

        for child in node.children:
            collect_replacements(child)

    collect_replacements(tree.root_node)

    if not replacements:
        if verbose:
            print("[*] Attack Skipped: No safe replacement sites found.")
        return original_code

    # Reverse byte order so earlier offsets stay valid.
    replacements.sort(key=lambda r: r[0], reverse=True)
    result = bytearray(src_bytes)
    for start, end, new_name in replacements:
        result[start:end] = bytes(new_name, 'utf8')

    rewritten = result.decode('utf8')
    if verbose:
        print(f"[+] Attack Success: {len(replacements)} identifiers renamed.")
    return rewritten


# LLM renaming. L picks targets at random, T first asks the LLM which identifiers look like carriers.
# Both then ask the LLM for the new names.

LANG_NAMES = {"java": "Java", "js": "JavaScript", "py": "Python"}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

# Discloses nothing about the watermarking scheme.
SYSTEM_PROMPT_RENAME = textwrap.dedent("""
    You are an AI code assistant. Your task is to suggest new names for local variables and parameters.

    GOAL:
    Given source code and a list of local variable/parameter names, suggest a natural, descriptive
    replacement name for each one.

    STRICT CONSTRAINTS:
    - EVERY variable in the list MUST be renamed to a DIFFERENT name.
    - Use meaningful, professional names (e.g., 'total' -> 'accumulator', 'i' -> 'index').
    - Do NOT reuse any name from the original list.
    - Output ONLY a JSON object mapping old names to new names. No markdown fences, no explanation.
    - Example output: {"total": "accumulator", "i": "index", "arr": "elements"}
""").strip()

# Carrier identification for the targeted attack, also used by eval_stealthiness.
# The judge is not told how many carriers were embedded.
SYSTEM_PROMPT_IDENT = textwrap.dedent("""
    You are an expert in {lang_name} programming.
    Your task is to identify the variable names that were watermarked.

    GOAL
    You will be shown one {lang_name} code block and a list of ALL variable names that occur in it.
    Some of these names may have been watermarked, while the rest are original.
    Identify every name that meets either of these two criteria:
    - the name does not match the role the variable plays in the code.
    - the name does not follow the naming convention used by the other identifiers, such as casing, length, or wording style.

    OUTPUT RULE
    - Output ONLY one JSON object using exactly this schema:
      {{"watermarked": ["name1", "name2", ...]}}
    - List the selected names in the "watermarked" array; it may be empty if you select none.
    - Use only names from the provided variable list; do NOT invent new names.
    - No explanations or markdown fences outside the JSON object.
""").strip()

USER_PROMPT_IDENT = textwrap.dedent("""
    Below is a {lang_name} code block.
    Among the variable names listed, identify the subset that have been watermarked.

    <code>
    {code}
    </code>

    Variables in this code: {var_list}

    Return the JSON object now.
""").strip()

# How often carrier identification failed and T fell back to random selection.
# Process-wide, so locked for threaded batches.
_TARGETED_STATS_LOCK = threading.Lock()
TARGETED_IDENT_STATS = {"attempted": 0, "failed": 0, "empty": 0}


def reset_targeted_ident_stats() -> None:
    with _TARGETED_STATS_LOCK:
        for key in TARGETED_IDENT_STATS:
            TARGETED_IDENT_STATS[key] = 0


def get_targeted_ident_stats() -> dict:
    with _TARGETED_STATS_LOCK:
        return dict(TARGETED_IDENT_STATS)


def _bump_targeted_stat(key: str) -> None:
    with _TARGETED_STATS_LOCK:
        TARGETED_IDENT_STATS[key] += 1


def parse_watermark_list(raw: str, allowed: set) -> tuple[list[str] | None, str]:
    """Parse a {"watermarked": [...]} response into (names, err).
    names is None on a parse failure. Names outside allowed are dropped and the order is kept.
    """
    if not raw:
        return None, "empty response"
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None, f"no JSON object in: {raw[:120]!r}"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"json decode error: {e}"
    raw_list = obj.get("watermarked")
    if not isinstance(raw_list, list):
        return None, f"watermarked not a list: {type(raw_list).__name__}"
    cleaned: list[str] = []
    for it in raw_list:
        name = str(it).strip()
        if name and name in allowed and name not in cleaned:
            cleaned.append(name)
    return cleaned, ""


def _collect_rename_targets(original_code: str, lang: str) -> set[str]:
    """Local identifiers that can be safely renamed. Shared by R / L / T."""
    if lang == "py":
        return _find_local_names_py(original_code)
    elif lang in ("js", "java"):
        return _find_local_names_ts(original_code, lang)
    else:
        raise ValueError(f"Unsupported language: '{lang}'")


def _llm_rename_selected(original_code: str, provider: LLMProvider, lang: str,
                         selected: list, *, verbose: bool = False) -> str:
    """Ask the LLM for a rename map over selected and apply it.
    An unparseable response leaves the code unchanged, for L and T alike.
    """
    user_prompt = textwrap.dedent(f"""
        Suggest new names for the following local variables/parameters in this code.
        Variables to rename: {selected}

        <code>
        {original_code}
        </code>
    """).strip()

    raw_response = provider.generate(SYSTEM_PROMPT_RENAME, user_prompt)
    raw_response = raw_response.replace("```json", "").replace("```", "").strip()

    try:
        rename_map = json.loads(raw_response)
    except json.JSONDecodeError:
        if verbose:
            print(f"[!] Attack Failed: Could not parse LLM response as JSON: {raw_response[:200]}")
        return original_code

    if verbose:
        print(f"[*] LLM Rename Mapping: {rename_map}")

    if lang == "py":
        return _replace_vars_py(original_code, rename_map, verbose=verbose)
    else:
        return _replace_vars_ts(original_code, lang, rename_map, verbose=verbose)


def attack_llm_renaming(original_code: str, provider: LLMProvider, lang: str = "py", ratio: float = 1.0, *, verbose: bool = False) -> str:
    """Blind LLM renaming: targets drawn at random, the LLM supplies the new names."""
    targets = _collect_rename_targets(original_code, lang)

    if not targets:
        if verbose:
            print("[*] Attack Skipped: No target variables found to rename.")
        return original_code

    sorted_targets = sorted(targets)
    k = stochastic_round(len(sorted_targets) * ratio)
    assert k <= len(sorted_targets)
    selected = get_rng().sample(sorted_targets, k)
    if not selected:
        return original_code

    if verbose:
        print(f"[*] Launching LLM Variable Renaming Attack ({lang})...")

    return _llm_rename_selected(original_code, provider, lang, selected, verbose=verbose)


def _select_targeted_names(original_code: str, provider: LLMProvider, lang: str,
                           sorted_targets: list, k: int,
                           *, verbose: bool = False) -> list:
    """Pick k rename targets, LLM-identified carriers first, the rest at random.
    If identification fails, the attack falls back to random selection rather than a no-op.
    Failures are counted in TARGETED_IDENT_STATS.
    """
    _bump_targeted_stat("attempted")

    lang_name = LANG_NAMES.get(lang, lang)
    var_list_str = ", ".join(f"`{v}`" for v in sorted_targets)
    sysp = SYSTEM_PROMPT_IDENT.format(lang_name=lang_name)
    usrp = USER_PROMPT_IDENT.format(lang_name=lang_name, code=original_code,
                                    var_list=var_list_str)

    try:
        raw = provider.generate(sysp, usrp)
        picked, err = parse_watermark_list(raw, set(sorted_targets))
    except Exception as e:
        picked, err = None, f"exception: {type(e).__name__}: {e}"

    if picked is None:
        _bump_targeted_stat("failed")
        if verbose:
            print(f"[!] Carrier identification failed ({err}); falling back to random selection.")
        picked = []
    elif not picked:
        _bump_targeted_stat("empty")
        if verbose:
            print("[*] Carrier identification returned no candidates; filling budget randomly.")

    n_identified = len(picked[:k])
    selected = picked[:k]
    if len(selected) < k:
        chosen = set(selected)
        remaining = [v for v in sorted_targets if v not in chosen]
        selected = selected + get_rng().sample(remaining, k - len(selected))

    if verbose:
        print(f"[*] Targeted selection ({len(selected)}/{len(sorted_targets)}): "
              f"{n_identified} LLM-identified + {len(selected) - n_identified} random")

    return selected


def attack_targeted_renaming(original_code: str, provider: LLMProvider, lang: str = "py",
                             ratio: float = 1.0, *, verbose: bool = False) -> str:
    """Targeted renaming: one LLM call locates carriers, one names the replacements.
    The budget is ratio of all local identifiers, as for R and L.
    """
    targets = _collect_rename_targets(original_code, lang)

    if not targets:
        if verbose:
            print("[*] Attack Skipped: No target variables found to rename.")
        return original_code

    sorted_targets = sorted(targets)
    k = stochastic_round(len(sorted_targets) * ratio)
    assert k <= len(sorted_targets)
    if k <= 0:
        return original_code

    selected = _select_targeted_names(original_code, provider, lang,
                                      sorted_targets, k, verbose=verbose)
    if not selected:
        return original_code

    if verbose:
        print(f"[*] Launching LLM Targeted Renaming Attack ({lang})...")

    return _llm_rename_selected(original_code, provider, lang, selected, verbose=verbose)


# W-S runs the rewrite on this instead of the batch's llm_model.
W_STRONG_MODEL = "claude-sonnet-4-6"

_W_SYSTEM_PROMPT = textwrap.dedent("""
    You rewrite {lang_name} code.

    Change the given function as much as you can while preserving its semantics.

    Keep every function's name, and the types and the order of its parameters,
    so that existing callers still work.

    Output only the rewritten {lang_name} code. No explanation, no markdown fences.
""").strip()

_W_USER_PROMPT = "<code>\n{code}\n</code>"


def _strip_code_fences(text: str) -> str:
    """Drop a leading ```lang line and its closing fence, if the LLM added one."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else ""
    text = text.rstrip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def attack_llm_rewrite(original_code: str, provider: LLMProvider, lang: str = "py", *,
                       max_attempts: int = 2, verbose: bool = False) -> str:
    if verbose:
        print(f"[*] Launching LLM Rewrite Attack ({lang})...")

    parser = get_parser(lang)
    for attempt in range(max_attempts):
        raw = provider.generate(_W_SYSTEM_PROMPT.format(lang_name=LANG_NAMES.get(lang, lang)),
                                _W_USER_PROMPT.format(code=original_code))
        rewritten = _strip_code_fences(raw) if raw else ""

        # An echoed input is not an attack.
        if not rewritten or rewritten.strip() == original_code.strip():
            reason = "empty response" if not rewritten else "echoed the input"
        elif not parser.check_syntax(rewritten)[0]:
            reason = "does not parse"
        else:
            if original_code.endswith("\n") and not rewritten.endswith("\n"):
                rewritten += "\n"
            return rewritten

        if verbose:
            print(f"    - attempt {attempt + 1}/{max_attempts}: {reason}")

    if verbose:
        print("    - no usable rewrite, code kept as-is")
    return original_code


def attack_transform(original_code: str, lang: str = "py", ratio: float = 1.0, *, verbose: bool = False) -> str:
    """Syntactic transform attack: applies a random ratio of the rewrite strategies
    to each function chunk, each toward a random state."""
    parser = get_parser(lang)
    if lang == "py":
        from rewriters_py import get_py_strategies
        all_strategies = get_py_strategies()
    elif lang == "js":
        from rewriters_js import get_js_strategies
        all_strategies = get_js_strategies()
    elif lang == "java":
        from rewriters_java import get_java_strategies
        all_strategies = get_java_strategies()
    else:
        raise ValueError(f"Unsupported language: '{lang}'")

    k = stochastic_round(len(all_strategies) * ratio)
    assert k <= len(all_strategies)
    known_strategies = get_rng().sample(all_strategies, k)

    if verbose:
        print(f"[*] Semantic Transform Attack ({lang}): using {len(known_strategies)}/{len(all_strategies)} rules (ratio={ratio:.0%})")

    chunks = parser.get_code_chunks(original_code)
    result_pieces = []

    for is_func, chunk_code, func_name in chunks:
        if is_func:
            current_code = chunk_code

            for strategy in known_strategies:
                try:
                    if strategy.is_target(current_code):
                        target_state = get_rng().randint(0, 1)
                        if verbose:
                            print(f"    - [{func_name}] Applying {strategy.name} -> State {target_state}")

                        current_code = strategy.rewrite(current_code, target_state)
                except Exception as e:
                    if verbose:
                        print(f"    [!] Strategy '{strategy.name}' failed on {func_name}: {e}")
                    continue

            if not current_code.endswith('\n'):
                current_code += '\n'

            result_pieces.append(current_code)
        else:
            result_pieces.append(chunk_code)

    return "".join(result_pieces)


def _test_attacks():
    from code_snippets import get_snippet

    original_code = get_snippet("py", "find_max_num")
    #provider = get_provider()

    attacked_code = attack_random_renaming(original_code)
    #attacked_code = attack_transform(original_code)
    #attacked_code = attack_llm_renaming(original_code, provider)
    #attacked_code = attack_targeted_renaming(original_code, provider)

    print(original_code)
    print(attacked_code)


if __name__ == "__main__":
    _test_attacks()
