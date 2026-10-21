import re
import json
import time
import textwrap

from typing import List, Tuple, Optional
from llm_provider import get_provider
from runners import JSRunner
from language_parser import JSParser
from paths import OUT_PATH


def _extract_func_name(code: str) -> Optional[str]:
    """Extract the first function name from JS code using JSParser."""
    parser = JSParser()
    chunks = parser.get_code_chunks(code)
    for is_func, chunk_code, func_name in chunks:
        if is_func and func_name:
            return func_name
    return None


def _build_test_harness(func_code: str, func_name: str, test_cases_js: str) -> str:
    """Build a JS file that runs all test cases and outputs JSON results to stdout."""
    return textwrap.dedent(f"""\
        {func_code}

        const __tests = {test_cases_js};
        const __results = __tests.map(function(args) {{
            try {{
                const r = {func_name}(...args);
                return {{ v: JSON.stringify(r) === undefined ? "undefined" : JSON.stringify(r), e: null }};
            }} catch(e) {{
                return {{ v: null, e: e.constructor.name }};
            }}
        }});
        process.stdout.write(JSON.stringify(__results));
    """)


def _parse_results(stdout: str) -> Optional[List[dict]]:
    """Parse JSON results from node stdout."""
    try:
        results = json.loads(stdout)
        if isinstance(results, list):
            return results
    except (json.JSONDecodeError, TypeError):
        pass
    return None


class EquivalenceVerifier_JS:
    def __init__(self, provider, timeout: float = 5.0, *, verbose: bool = True):
        self.provider = provider
        self.verbose = verbose
        self.base_timeout = timeout
        self.runner = JSRunner(timeout=timeout)

        self.stats = {
            "total_original": 0, "total_variants": 0,
            "passed": 0,
            "failed_syntax": 0, "failed_exec": 0, "failed_value": 0,
            "failed_exception": 0, "failed_timeout": 0, "failed_unknown": 0,
            "baseline_failed": 0,
            "time_baseline": 0.0, "time_verify": 0.0,
            "total_test_generated": 0,
            "time_test_gen": 0.0,
        }

    def generate_test_inputs(self, code: str, ninput: int = 10) -> Optional[str]:
        """Generate test inputs via LLM. Returns JS array literal string, or None on failure."""

        system_prompt = textwrap.dedent("""
            You are a QA engineer who outputs ONLY valid JavaScript array literals.

            Rules:
            - No comments, no explanations, no markdown fences.
            - Output must be a single JavaScript array.
            - Each element must be a positional-arguments array for one function call.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            Analyze the following JavaScript function and provide {ninput} diverse input examples.

            CRITICAL RULES:

            1. Output ONLY a JavaScript array of argument arrays:
            [ args_array, args_array, ... ]

            2. Each args_array represents positional arguments for one function call.

            - Example (multiple arguments including an array):

            Function:
            function process(x, arr, flag) {{
                if (flag) return x + arr.reduce((a,b) => a+b, 0);
                return x - arr.reduce((a,b) => a+b, 0);
            }}

            Valid output:
            [
                [1, [1, 2, 3], true],
                [5, [], false],
                [0, [10], true],
                [-3, [4, -1], false]
            ]

            3. If the function expects a SINGLE array as an argument, wrap it as the first positional argument.
            - RIGHT:  [[1, 2, 3]]
            - WRONG:  [1, 2, 3]

            Example:

            Function:
            function func(arr) {{ ... }}

            Valid output example:
            [
                [[1, 2, 3]],
                [[0]],
                [[]]
            ]

            4. EXAMPLES BASED ON SIGNATURE:
            - Case A: function firstDigit(num)
              Valid output: [ [0], [5], [10], [12345], [-1] ]

            - Case B: function convert(list, bool)
              Valid output: [ [[1, 2, 3], true], [[], true], [[[1, 2], [3, 4]], false] ]

            - Case C: function sortMatrix(matrix)
              Valid output: [ [[[1, 2], [3, 4]]], [[[]]], [[[10, 20]]] ]

            5. Include edge cases such as empty inputs, single-element inputs, and boundary values when applicable.
            6. No comments, no explanations, no markdown fences.
            7. Output ONLY the JavaScript array.

            <code>
            {code}
            </code>
        """).strip()

        start_t = time.perf_counter()

        raw_text = self.provider.generate(system_prompt, user_prompt)
        test_cases_js, parse_err = self._parse_test_inputs(raw_text)

        # Retry once with error feedback
        if test_cases_js is None and parse_err:
            retry_prompt = (
                f"Your previous output could not be parsed.\n"
                f"Error: {parse_err}\n\n"
                f"Please try again. Output ONLY a valid JavaScript array of argument arrays, "
                f"with no expressions, no variables, no comments — only literals.\n\n"
                f"<code>\n{code}\n</code>"
            )
            raw_text = self.provider.generate(system_prompt, retry_prompt)
            test_cases_js, parse_err = self._parse_test_inputs(raw_text)

        self.stats["time_test_gen"] += time.perf_counter() - start_t

        if test_cases_js is None:
            if self.verbose and parse_err:
                print(f"[!] JS test input parsing error: {parse_err}")
            return None

        return test_cases_js

    def _parse_test_inputs(self, raw_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the LLM response into a JS array literal. Returns (literal, None) or (None, error)."""
        try:
            clean_text = raw_text.replace("```javascript", "").replace("```js", "").replace("```", "").strip()

            match = re.search(r"\[\s*(\[.*\])\s*\]", clean_text, re.DOTALL)
            if not match:
                return None, "No valid array pattern found"

            js_literal = match.group(0).strip()

            # Validate by running through Node
            validation_code = f"process.stdout.write(JSON.stringify({js_literal}))"
            res = self.runner.run_code(validation_code)
            if not res["success"]:
                return None, f"JS validation failed: {res['stderr']}"

            parsed = json.loads(res["stdout"])
            if not isinstance(parsed, list):
                return None, "Result is not an array"
            if not all(isinstance(args, list) for args in parsed):
                return None, "Every test case must be an array of positional arguments"

            self.stats["total_test_generated"] += len(parsed)
            return js_literal, None

        except Exception as e:
            return None, str(e)

    def _run_with_harness(self, func_code: str, func_name: str, test_cases_js: str, timeout: float | None = None) -> dict:
        """Run function with test harness. Returns {success, results, error, duration}."""
        harness = _build_test_harness(func_code, func_name, test_cases_js)

        if timeout is not None:
            saved_timeout = self.runner.timeout
            self.runner.timeout = timeout

        start_t = time.perf_counter()
        res = self.runner.run_code(harness)
        duration = time.perf_counter() - start_t

        if timeout is not None:
            self.runner.timeout = saved_timeout

        if not res["success"]:
            error_type = "Timeout" if res["stderr"] == "Timeout" else f"ExecError: {res['stderr'][:200]}"
            return {"success": False, "results": None, "error": error_type, "duration": duration}

        results = _parse_results(res["stdout"])
        if results is None:
            return {"success": False, "results": None, "error": f"OutputParseError: {res['stdout'][:200]}", "duration": duration}

        return {"success": True, "results": results, "error": None, "duration": duration}

    def check_equivalence(self, original_code: str, variants: List[str], test_cases_js: str) -> List[bool]:
        """Return per-variant bools, True when the variant matches the original on every
        input in test_cases_js (a JS array literal)."""
        VERIFIER_LOGGING = True
        log_path = OUT_PATH("equivalence_test_results_js.log")

        self.stats["total_original"] += 1
        self.stats["total_variants"] += len(variants)

        verify_start_t = time.perf_counter()

        if not variants or not test_cases_js:
            return [False] * len(variants)

        func_name = _extract_func_name(original_code)
        if not func_name:
            if self.verbose:
                print(f"[!] Could not extract function name from original code")
            return [False] * len(variants)

        # Baseline
        baseline = self._run_with_harness(original_code, func_name, test_cases_js, timeout=self.base_timeout)
        self.stats["time_baseline"] += baseline["duration"]

        if not baseline["success"]:
            if self.verbose:
                print(f"[!] Baseline failed: {baseline['error']}")
            self.stats["baseline_failed"] += len(variants)
            return [False] * len(variants)

        expected = baseline["results"]

        all_errors = all(r.get("e") is not None for r in expected)
        if all_errors:
            if self.verbose:
                print(f"[!] All baseline cases threw exceptions. Check test cases.")
            self.stats["baseline_failed"] += len(variants)
            return [False] * len(variants)

        variant_timeout = max(baseline["duration"] * 3, 2.0)

        detailed_results = []
        for idx, v_code in enumerate(variants):
            vr = self._run_with_harness(v_code, func_name, test_cases_js, timeout=variant_timeout)

            if not vr["success"]:
                detailed_results.append((False, vr["error"]))
                continue

            actual = vr["results"]
            if len(actual) != len(expected):
                detailed_results.append((False, f"ResultCountMismatch: expected {len(expected)}, got {len(actual)}"))
                continue

            mismatch = None
            for i, (exp, act) in enumerate(zip(expected, actual)):
                if exp.get("e") != act.get("e"):
                    mismatch = f"ExceptionMismatch at case {i}: expected {exp.get('e')}, got {act.get('e')}"
                    break
                if exp.get("e") is None and exp.get("v") != act.get("v"):
                    mismatch = f"ValueMismatch at case {i}: expected {exp.get('v')}, got {act.get('v')}"
                    break

            if mismatch:
                detailed_results.append((False, mismatch))
            else:
                detailed_results.append((True, None))

        self.stats["time_verify"] += time.perf_counter() - verify_start_t

        for ok, err in detailed_results:
            if ok:
                self.stats["passed"] += 1
            elif err:
                if "Timeout" in err:
                    self.stats["failed_timeout"] += 1
                elif "SyntaxError" in err or "Syntax error" in err:
                    self.stats["failed_syntax"] += 1
                elif "ExecError" in err:
                    self.stats["failed_exec"] += 1
                elif "ExceptionMismatch" in err:
                    self.stats["failed_exception"] += 1
                elif "ValueMismatch" in err:
                    self.stats["failed_value"] += 1
                else:
                    self.stats["failed_unknown"] += 1

        if VERIFIER_LOGGING:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"[JS Verifier log]\n")
                f.write(f"Timestamp     : {time.ctime()}\n")
                f.write(f"Function name : {func_name}\n")
                f.write(f"Test cases    : {test_cases_js[:200]}...\n")
                f.write(f"Baseline time : {baseline['duration']:.4f}s\n")
                f.write(f"Variant timeout: {variant_timeout:.2f}s\n")
                f.write(f"Passed        : {sum(1 for r, e in detailed_results if r)}/{len(variants)}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Original code:\n{original_code.strip()}\n\n")

                for idx, (v_code, (ok, err)) in enumerate(zip(variants, detailed_results)):
                    status = "PASS" if ok else f"FAIL ({err})"
                    f.write(f"VARIANT #{idx:02} {status}\n")
                    f.write(f"{v_code.strip()}\n")
                    f.write("-" * 30 + "\n")
                f.write("=" * 80 + "\n\n")

        final = [r for r, e in detailed_results]
        if self.verbose:
            print(f"[*] JS verification: {sum(final)}/{len(variants)} passed.")
        return final

    @staticmethod
    def check_verifiable(code: str) -> Tuple[bool, List[str]]:
        """See check_verifiable_js()."""
        return check_verifiable_js(code)


# Globals always available in Node.js without require/import.
_JS_BUILTINS = {
    # Primitive values
    'undefined', 'NaN', 'Infinity',

    # Constructors & wrapper types
    'Object', 'Array', 'String', 'Number', 'Boolean', 'Symbol', 'BigInt',
    'Function', 'RegExp', 'Date',

    # Error types
    'Error', 'TypeError', 'RangeError', 'ReferenceError', 'SyntaxError',
    'URIError', 'EvalError', 'AggregateError',

    # Collections
    'Map', 'Set', 'WeakMap', 'WeakSet',

    # Async & concurrency
    'Promise',

    # Proxy & reflection
    'Proxy', 'Reflect',

    # Typed arrays & binary data
    'ArrayBuffer', 'SharedArrayBuffer', 'DataView',
    'Int8Array', 'Uint8Array', 'Uint8ClampedArray',
    'Int16Array', 'Uint16Array', 'Int32Array', 'Uint32Array',
    'Float32Array', 'Float64Array', 'BigInt64Array', 'BigUint64Array',

    # Utility objects
    'JSON', 'Math', 'Intl',

    # Global functions
    'parseInt', 'parseFloat', 'isNaN', 'isFinite',
    'encodeURI', 'decodeURI', 'encodeURIComponent', 'decodeURIComponent',
    'eval', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    'setImmediate', 'clearImmediate', 'queueMicrotask',

    # Console (output only)
    'console',

    # Special identifiers (always in scope)
    'arguments', 'this',
    'true', 'false', 'null',
}

# Array / String / Number prototype members. A parameter that only touches these is
# a plain JSON value, so ParamAttr accepts it. Mutating members (push, sort, splice,
# ...) are excluded: the harness does not compare post-call argument state, so a
# variant that copies before mutating would pass while behaving differently.
_SAFE_PARAM_MEMBERS = {
    # Array, non-mutating
    'length', 'slice', 'concat', 'join', 'indexOf', 'lastIndexOf', 'includes',
    'find', 'findIndex', 'filter', 'map', 'reduce', 'reduceRight', 'forEach',
    'some', 'every', 'flat', 'flatMap', 'entries', 'keys', 'values', 'at',
    # String
    'charAt', 'charCodeAt', 'codePointAt', 'split', 'substring', 'substr',
    'toUpperCase', 'toLowerCase', 'trim', 'trimStart', 'trimEnd',
    'replace', 'replaceAll', 'match', 'matchAll', 'search',
    'startsWith', 'endsWith', 'padStart', 'padEnd', 'repeat',
    # Number
    'toFixed', 'toPrecision', 'toExponential',
    # Universal
    'toString', 'valueOf',
}

# Non-deterministic calls.
_NONDETERMINISTIC = {
    ('Math', 'random'),
    ('Date', 'now'),
    ('performance', 'now'),
    ('crypto', 'randomBytes'),
    ('crypto', 'getRandomValues'),
}


def _ts_parse(code: str):
    """Parse JS code with tree-sitter, returning (tree, src_bytes)."""
    import tree_sitter_javascript as tsjs
    from tree_sitter import Language, Parser
    lang = Language(tsjs.language())
    parser = Parser(lang)
    src_bytes = code.encode('utf-8')
    tree = parser.parse(src_bytes)
    return tree, src_bytes


def _get_node_text(node, src_bytes: bytes) -> str:
    return src_bytes[node.start_byte:node.end_byte].decode('utf-8')


def _collect_defined_names(root_node, src_bytes: bytes) -> set:
    """Collect names bound in the code: parameters, var/let/const, nested function and
    class declarations, for-in/for-of variables, and catch parameters."""
    names = set()

    def walk(node):
        if node.type == 'variable_declarator':
            name_node = node.child_by_field_name('name')
            if name_node:
                if name_node.type == 'identifier':
                    names.add(_get_node_text(name_node, src_bytes))
                elif name_node.type in ('array_pattern', 'object_pattern'):
                    _collect_pattern_names(name_node, src_bytes, names)

        if node.type == 'formal_parameters':
            for child in node.named_children:
                if child.type == 'identifier':
                    names.add(_get_node_text(child, src_bytes))
                elif child.type == 'assignment_pattern':
                    left = child.child_by_field_name('left')
                    if left and left.type == 'identifier':
                        names.add(_get_node_text(left, src_bytes))
                    elif left and left.type in ('array_pattern', 'object_pattern'):
                        _collect_pattern_names(left, src_bytes, names)
                elif child.type == 'rest_pattern':
                    for c in child.named_children:
                        if c.type == 'identifier':
                            names.add(_get_node_text(c, src_bytes))

        if node.type in ('function_declaration', 'class_declaration'):
            name_node = node.child_by_field_name('name')
            if name_node:
                names.add(_get_node_text(name_node, src_bytes))

        if node.type == 'catch_clause':
            param = node.child_by_field_name('parameter')
            if param and param.type == 'identifier':
                names.add(_get_node_text(param, src_bytes))

        if node.type in ('for_in_statement',):
            left = node.child_by_field_name('left')
            if left:
                if left.type == 'identifier':
                    names.add(_get_node_text(left, src_bytes))
                # for (const x of ...): left is a lexical_declaration
                elif left.type in ('lexical_declaration', 'variable_declaration'):
                    for child in left.named_children:
                        if child.type == 'variable_declarator':
                            vn = child.child_by_field_name('name')
                            if vn and vn.type == 'identifier':
                                names.add(_get_node_text(vn, src_bytes))

        for child in node.children:
            walk(child)

    walk(root_node)
    return names


def _collect_pattern_names(pattern_node, src_bytes: bytes, names: set):
    """Recursively collect identifier names from destructuring patterns."""
    for child in pattern_node.named_children:
        if child.type == 'identifier':
            names.add(_get_node_text(child, src_bytes))
        elif child.type == 'shorthand_property_identifier_pattern':
            names.add(_get_node_text(child, src_bytes))
        elif child.type == 'pair_pattern':
            # { key: value }: value is the binding
            value = child.child_by_field_name('value')
            if value and value.type == 'identifier':
                names.add(_get_node_text(value, src_bytes))
            elif value and value.type in ('array_pattern', 'object_pattern'):
                _collect_pattern_names(value, src_bytes, names)
        elif child.type == 'rest_pattern':
            for c in child.named_children:
                if c.type == 'identifier':
                    names.add(_get_node_text(c, src_bytes))
        elif child.type in ('array_pattern', 'object_pattern'):
            _collect_pattern_names(child, src_bytes, names)
        elif child.type == 'assignment_pattern':
            left = child.child_by_field_name('left')
            if left and left.type == 'identifier':
                names.add(_get_node_text(left, src_bytes))


def _collect_param_names(root_node, src_bytes: bytes) -> set:
    """Collect parameter names of the top-level function."""
    params = set()
    # Find the first formal_parameters node (top-level function)
    def find_params(node, depth=0):
        if node.type == 'formal_parameters' and depth <= 2:
            for child in node.named_children:
                if child.type == 'identifier':
                    params.add(_get_node_text(child, src_bytes))
                elif child.type == 'assignment_pattern':
                    left = child.child_by_field_name('left')
                    if left and left.type == 'identifier':
                        params.add(_get_node_text(left, src_bytes))
                elif child.type == 'rest_pattern':
                    for c in child.named_children:
                        if c.type == 'identifier':
                            params.add(_get_node_text(c, src_bytes))
            return
        for child in node.children:
            find_params(child, depth + 1)
    find_params(root_node)
    return params


def check_verifiable_js(code: str) -> Tuple[bool, List[str]]:
    """Check whether a JavaScript function can run standalone on JSON inputs with a
    deterministic result. Returns (ok, reasons), listing every rejection reason.
    """
    reasons = []
    seen = set()

    tree, src_bytes = _ts_parse(code)
    if tree.root_node.has_error:
        return False, ["SyntaxError: code has parse errors"]

    js_parser = JSParser()
    chunks = js_parser.get_code_chunks(code)
    func_name = None
    for is_func, chunk_code, fn in chunks:
        if is_func and fn:
            func_name = fn
            break
    if not func_name:
        return False, ["NoFunction: no function definition found"]

    defined_names = _collect_defined_names(tree.root_node, src_bytes)
    defined_names.add(func_name)  # function's own name (for recursion)
    param_names = _collect_param_names(tree.root_node, src_bytes)

    known_names = defined_names | _JS_BUILTINS

    # External references: identifiers in read position that are neither local nor builtin.
    def _check_identifiers(node):
        if node.type == 'identifier':
            name = _get_node_text(node, src_bytes)
            parent = node.parent

            # obj.PROP: PROP is not a free reference
            if parent and parent.type == 'member_expression':
                prop_node = parent.child_by_field_name('property')
                if prop_node and prop_node.id == node.id:
                    return

            # Skip declaration sites (variable_declarator name, function name, etc.)
            if parent and parent.type == 'variable_declarator':
                name_node = parent.child_by_field_name('name')
                if name_node and name_node.id == node.id:
                    return
            if parent and parent.type in ('function_declaration', 'function', 'class_declaration'):
                name_node = parent.child_by_field_name('name')
                if name_node and name_node.id == node.id:
                    return

            # Skip label identifiers (e.g. break LABEL)
            if parent and parent.type in ('labeled_statement', 'break_statement', 'continue_statement'):
                return

            # Skip property keys in object literals: { key: value }
            if parent and parent.type == 'pair':
                key_node = parent.child_by_field_name('key')
                if key_node and key_node.id == node.id:
                    return

            # Skip shorthand property identifiers in patterns
            if parent and parent.type == 'shorthand_property_identifier_pattern':
                return

            if name not in known_names:
                r = f"ExternalRef: '{name}'"
                if r not in seen:
                    reasons.append(r); seen.add(r)

        for child in node.children:
            _check_identifiers(child)

    _check_identifiers(tree.root_node)

    # A non-builtin member on a parameter implies an object argument that cannot be generated.
    def _check_param_attr(node):
        if node.type == 'member_expression':
            obj_node = node.child_by_field_name('object')
            # Walk through chained access: a.b.c -> root is 'a'
            root = obj_node
            while root and root.type == 'member_expression':
                root = root.child_by_field_name('object')
            if root and root.type == 'identifier':
                root_name = _get_node_text(root, src_bytes)
                if root_name in param_names:
                    prop_node = node.child_by_field_name('property')
                    prop_name = _get_node_text(prop_node, src_bytes) if prop_node else '?'
                    if prop_name not in _SAFE_PARAM_MEMBERS:
                        r = f"ParamAttr: parameter '{root_name}' has attribute access '.{prop_name}'"
                        if r not in seen:
                            reasons.append(r); seen.add(r)

        for child in node.children:
            _check_param_attr(child)

    _check_param_attr(tree.root_node)

    # Use of this requires an object context.
    def _check_this(node):
        if node.type == 'this':
            r = "ThisUsage: function uses 'this'"
            if r not in seen:
                reasons.append(r); seen.add(r)
        for child in node.children:
            _check_this(child)

    _check_this(tree.root_node)

    # Parameters called as functions.
    def _check_callable_param(node):
        if node.type == 'call_expression':
            callee = node.child_by_field_name('function')
            if callee and callee.type == 'identifier':
                name = _get_node_text(callee, src_bytes)
                if name in param_names:
                    r = f"CallableParam: parameter '{name}' used as callable"
                    if r not in seen:
                        reasons.append(r); seen.add(r)

        for child in node.children:
            _check_callable_param(child)

    _check_callable_param(tree.root_node)

    # require() / import: module dependency.
    def _check_require_import(node):
        if node.type == 'call_expression':
            callee = node.child_by_field_name('function')
            if callee and callee.type == 'identifier' and _get_node_text(callee, src_bytes) == 'require':
                args = node.child_by_field_name('arguments')
                arg_text = _get_node_text(args, src_bytes) if args else ''
                r = f"Require: require{arg_text}"
                if r not in seen:
                    reasons.append(r); seen.add(r)

        if node.type == 'import_statement':
            r = f"Import: {_get_node_text(node, src_bytes)[:60]}"
            if r not in seen:
                reasons.append(r); seen.add(r)

        for child in node.children:
            _check_require_import(child)

    _check_require_import(tree.root_node)

    # Non-deterministic calls.
    def _check_nondeterministic(node):
        if node.type == 'call_expression':
            callee = node.child_by_field_name('function')
            if callee and callee.type == 'member_expression':
                obj = callee.child_by_field_name('object')
                prop = callee.child_by_field_name('property')
                if obj and prop and obj.type == 'identifier':
                    pair = (_get_node_text(obj, src_bytes), _get_node_text(prop, src_bytes))
                    if pair in _NONDETERMINISTIC:
                        r = f"NonDeterministic: '{pair[0]}.{pair[1]}()'"
                        if r not in seen:
                            reasons.append(r); seen.add(r)

        for child in node.children:
            _check_nondeterministic(child)

    _check_nondeterministic(tree.root_node)

    return (len(reasons) == 0), reasons


def _test_verifier():
    provider = get_provider()
    verifier = EquivalenceVerifier_JS(provider, verbose=True)

    # Test case 1: simple function
    code1 = textwrap.dedent("""\
        function add(a, b) {
            return a + b;
        }
    """).strip()

    # Correct variant
    variant1_ok = textwrap.dedent("""\
        function add(a, b) {
            const result = a + b;
            return result;
        }
    """).strip()

    # Incorrect variant
    variant1_bad = textwrap.dedent("""\
        function add(a, b) {
            return a - b;
        }
    """).strip()

    # Syntax error variant
    variant1_syntax = "function add(a, b { return a + b; }"

    # Infinite loop variant
    variant1_loop = textwrap.dedent("""\
        function add(a, b) {
            while(true) {}
            return a + b;
        }
    """).strip()

    print("\n" + "=" * 60)
    print(" Test 1: Simple function (add)")
    print("=" * 60)

    test_inputs = verifier.generate_test_inputs(code1)
    print(f"Generated test inputs: {test_inputs}")

    if test_inputs:
        results = verifier.check_equivalence(
            code1,
            [variant1_ok, variant1_bad, variant1_syntax, variant1_loop],
            test_inputs
        )
        labels = ["correct", "wrong value", "syntax error", "infinite loop"]
        for label, ok in zip(labels, results):
            status = "PASS" if ok else "FAIL"
            print(f"  {label:20s}: {status}")

    # Test case 2: array function
    code2 = textwrap.dedent("""\
        function reverseArray(arr) {
            const result = [];
            for (let i = arr.length - 1; i >= 0; i--) {
                result.push(arr[i]);
            }
            return result;
        }
    """).strip()

    variant2_ok = textwrap.dedent("""\
        function reverseArray(arr) {
            return arr.slice().reverse();
        }
    """).strip()

    variant2_bad = textwrap.dedent("""\
        function reverseArray(arr) {
            return arr;
        }
    """).strip()

    print("\n" + "=" * 60)
    print(" Test 2: Array function (reverseArray)")
    print("=" * 60)

    test_inputs2 = verifier.generate_test_inputs(code2)
    print(f"Generated test inputs: {test_inputs2}")

    if test_inputs2:
        results2 = verifier.check_equivalence(
            code2,
            [variant2_ok, variant2_bad],
            test_inputs2
        )
        labels2 = ["correct (slice+reverse)", "wrong (identity)"]
        for label, ok in zip(labels2, results2):
            status = "PASS" if ok else "FAIL"
            print(f"  {label:30s}: {status}")

    # Test case 3: string function
    code3 = textwrap.dedent("""\
        const isPalindrome = (s) => {
            const reversed = s.split('').reverse().join('');
            return s === reversed;
        }
    """).strip()

    variant3_ok = textwrap.dedent("""\
        const isPalindrome = (s) => {
            for (let i = 0; i < Math.floor(s.length / 2); i++) {
                if (s[i] !== s[s.length - 1 - i]) return false;
            }
            return true;
        }
    """).strip()

    variant3_bad = textwrap.dedent("""\
        const isPalindrome = (s) => {
            return true;
        }
    """).strip()

    print("\n" + "=" * 60)
    print(" Test 3: Arrow function (isPalindrome)")
    print("=" * 60)

    test_inputs3 = verifier.generate_test_inputs(code3)
    print(f"Generated test inputs: {test_inputs3}")

    if test_inputs3:
        results3 = verifier.check_equivalence(
            code3,
            [variant3_ok, variant3_bad],
            test_inputs3
        )
        labels3 = ["correct (loop)", "wrong (always true)"]
        for label, ok in zip(labels3, results3):
            status = "PASS" if ok else "FAIL"
            print(f"  {label:30s}: {status}")

    print("\n" + "=" * 60)
    print(" Verifier Stats")
    print("=" * 60)
    for key, value in verifier.stats.items():
        if isinstance(value, float):
            print(f"  {key:<24}: {value:>8.2f}")
        else:
            print(f"  {key:<24}: {value:>8}")


if __name__ == "__main__":
    _test_verifier()
