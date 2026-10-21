import math
import re
import ast
import textwrap
import os
import multiprocessing
import copy
import time

from typing import List, Tuple, Dict, Any, Optional
from llm_provider import get_provider
from paths import OUT_PATH

BASE_NAMESPACE: Dict[str, Any] = {
    "math": math,
    "re": re,
    "List": List,
    "Tuple": Tuple,
    "Dict": Dict,
    "Any": Any,
    "Optional": Optional,
    "__builtins__": __builtins__,
}

def _verifier_process_worker(variant_code: str, func_name: str, test_cases: List, expected_data: List) -> Tuple[bool, Optional[str]]:
    try:
        # AST parse / syntax check.
        try:
            ast.parse(variant_code)
        except SyntaxError as se:
            return False, f"SyntaxError: {se.msg} (at line {se.lineno})"
        except Exception as e:
            return False, f"ParsingError: {str(e)}"

        # Define the function in a fresh namespace.
        ns = BASE_NAMESPACE.copy()
        try:
            exec(variant_code, ns)
        except Exception as e:
            return False, f"ExecutionError: Def Phase {str(e)}"

        target_func = ns.get(func_name)
        if not target_func or not callable(target_func):
            return False, f"ExecutionError: Function '{func_name}' not found in generated code."

        if expected_data is None:
            # baseline case
            baseline_results = []

            for i, args in enumerate(test_cases):
                if args is None or not isinstance(args, (list, tuple)):
                    baseline_results.append((None, True, "InvalidInputError"))
                    continue

                try:
                    res = target_func(*copy.deepcopy(args))
                    baseline_results.append((res, False, None))
                except Exception as e:
                    baseline_results.append((None, True, type(e).__name__))
            return True, baseline_results

        for i, args in enumerate(test_cases):
            # expected_data tuple: (result, is_exception, exception_name).
            exp_res, exp_is_excep, exp_excep_name = expected_data[i]

            # Deep-copy inputs to guard against in-place mutation.
            safe_args = copy.deepcopy(args)

            curr_res, curr_is_excep, curr_excep_name = None, False, None

            try:
                curr_res = target_func(*safe_args)
            except Exception as e:
                curr_is_excep = True
                curr_excep_name = type(e).__name__

            if curr_is_excep != exp_is_excep:
                expected_desc = f"exception {exp_excep_name}" if exp_is_excep else f"value {exp_res}"
                got_desc = f"exception {curr_excep_name}" if curr_is_excep else f"value {curr_res}"
                return False, f"ExceptionMismatch at Case {i}, arg: {args}. Expected {expected_desc}, got {got_desc}"

            if curr_is_excep:
                if curr_excep_name != exp_excep_name:
                    return False, f"ExceptionMismatch at Case {i}, arg: {args}. Expected exception {exp_excep_name}, got {curr_excep_name}"
            elif curr_res != exp_res:
                    return False, f"ValueMismatch at Case {i}, arg: {args}. Expected {exp_res}, got {curr_res}"

        return True, None
    except Exception as e:
        return False, f"WorkerError: {str(e)}"


class EquivalenceVerifier_Py:
    def __init__(self, provider, *, verbose: bool = True, n_workers: int | None = None):
        self.provider = provider
        self.verbose = verbose

        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) // 2)
        self.pool = None
        self._init_pool()

        self.stats = {
            "total_original": 0, "total_variants": 0,
            "passed": 0,
            "failed_syntax": 0, "failed_exec": 0, "failed_value": 0,
            "failed_exception": 0, "failed_timeout": 0, "failed_worker": 0, "failed_unknown": 0,
            "baseline_failed": 0,
            "time_baseline": 0.0, "time_verify": 0.0, "time_timeout_wait": 0.0,

            "total_test_generated" : 0,
            "time_test_gen": 0.0,
        }

    def _init_pool(self):
        if self.verbose:
            print(f"[*] Initializing verifier process pool with {self.n_workers} workers.")
        self.pool = multiprocessing.Pool(processes=self.n_workers)

    def _refresh_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
        self._init_pool()

    def __del__(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

    def generate_test_inputs(self, code: str, ninput: int = 10) -> list:
        system_prompt = textwrap.dedent("""
            You are a QA engineer who outputs ONLY pure Python data structures.

            Rules:
            - No comments, no explanations, no markdown fences.
            - Output must be a single Python list.
            - Each element must be a positional-arguments list for one function call.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            Analyze the following Python function and provide {ninput} diverse input examples.

            CRITICAL RULES:

            1. Output ONLY a Python list of argument lists:
            [ args_list, args_list, ... ]

            2. Each args_list represents positional arguments for one function call.

            - Example (multiple arguments including a list):

            Function:
            def process(x, arr, flag):
                if flag:
                    return x + sum(arr)
                return x - sum(arr)

            Valid output:
            [
                [1, [1, 2, 3], True],
                [5, [], False],
                [0, [10], True],
                [-3, [4, -1], False]
            ]

            3. If the function expects a SINGLE list as an argument, wrap it as the first positional argument.
            - RIGHT:  [[1, 2, 3]]
            - WRONG:  [1, 2, 3]

            Example:

            Function:
            def func(arr):
                ...

            Valid output example:
            [
                [[1, 2, 3]],
                [[0]],
                [[]]
            ]

            4. EXAMPLES BASED ON SIGNATURE:
            - Case A: def first_Digit(Num):
              Valid output: [ [0], [5], [10], [12345], [-1] ]

            - Case B: def convert(List, Bool):
              Valid output: [ [[1, 2, 3], True], [[], True], [[(1, 2), (3, 4)], False] ]

            - Case C: def sort_matrix(Matrix):
              Valid output: [ [[[1, 2], [3, 4]]], [[[]]], [[[10, 20]]] ]

            5. Include edge cases such as empty inputs, single-element inputs, and boundary values when applicable.
            6. No comments, no explanations, no markdown fences.
            7. Output ONLY the Python list.

            <code>
            {code}
            </code>
        """).strip()

        start_t = time.perf_counter()

        raw_text = self.provider.generate(system_prompt, user_prompt)
        test_cases, parse_err = self._parse_test_inputs(raw_text)

        # Retry once with error feedback
        if test_cases is None and parse_err:
            retry_prompt = (
                f"Your previous output could not be parsed.\n"
                f"Error: {parse_err}\n\n"
                f"Please try again. Output ONLY a valid Python list of argument lists, "
                f"with no expressions, no variables, no comments — only literals.\n\n"
                f"<code>\n{code}\n</code>"
            )
            raw_text = self.provider.generate(system_prompt, retry_prompt)
            test_cases, parse_err = self._parse_test_inputs(raw_text)

        self.stats["time_test_gen"] += time.perf_counter() - start_t

        if test_cases is None:
            if self.verbose and parse_err:
                print(f"[!] Parsing Error: {parse_err}")
            return []

        self.stats["total_test_generated"] += len(test_cases)
        return test_cases

    def _parse_test_inputs(self, raw_text: str):
        """Parse LLM response into test cases. Returns (list, None) on success, (None, error_msg) on failure."""
        try:
            clean_text = raw_text.replace("```python", "").replace("```", "").strip()

            match = re.search(r"\[\s*(\[.*\])\s*\]", clean_text, re.DOTALL)
            if not match:
                return None, "No valid list pattern found"

            list_str = match.group(0).strip()

            # Fix invalid escape sequences (e.g. \d, \w, \s from regex patterns)
            list_str = re.sub(r'\\(?![\\\'\"abfnrtvxuUN0-7])', r'\\\\', list_str)

            test_cases = ast.literal_eval(list_str)

            if not isinstance(test_cases, list):
                return None, "Result is not a list"

            if not all(isinstance(args, list) for args in test_cases):
                return None, "Every test case must be a list of positional arguments"

            return test_cases, None

        except Exception as e:
            return None, str(e)

    def check_equivalence(self, original_code: str, variants: List[str], test_cases: List) -> List[bool]:
        VERIFIER_LOGGING = True
        log_path = OUT_PATH("equivalence_test_results.log")
        original_timeout_limit = 5.0

        self.stats["total_original"] += 1
        self.stats["total_variants"] += len(variants)

        verify_start_t = time.perf_counter()

        if not variants or not test_cases:
            return [False] * len(variants)

        try:
            orig_tree = ast.parse(original_code)
            orig_func_name = next(node.name for node in ast.walk(orig_tree) if isinstance(node, ast.FunctionDef))
        except Exception as e:
            if self.verbose:
                print(f"[!] Original code parsing failed: {str(e)}")
            return [False] * len(variants)

        if not self.pool:
            self._init_pool()

        expected_data = [] 
        total_start_t = time.perf_counter()

        try:
            baseline_task = self.pool.apply_async(_verifier_process_worker, (original_code, orig_func_name, test_cases, None))
            success, result = baseline_task.get(timeout = original_timeout_limit)

            self.stats["time_baseline"] += time.perf_counter() - total_start_t

            if not success:
                raise Exception(f"Baseline generation logic failed: {result}")

            expected_data = result

            baseline_errors = sum(1 for r, is_ex, name in expected_data if is_ex)
            if baseline_errors == len(test_cases):
                if self.verbose:
                    print(f"[!] Critical: All baseline cases failed ({baseline_errors}/{len(test_cases)}). Check test cases.")
                self.stats["baseline_failed"] += len(variants)
                return [False] * len(variants)
        except multiprocessing.TimeoutError:
            if self.verbose:
                print(f"[!] Baseline Timed Out! Original code is too slow or has an infinite loop.")
            self.stats["time_timeout_wait"] += original_timeout_limit
            self.stats["baseline_failed"] += len(variants)
            self._refresh_pool()
            return [False] * len(variants)
        except Exception as e:
            if self.verbose:
                print(f"[!] Original code execution failed: {e}")
            self.stats["baseline_failed"] += len(variants)
            return [False] * len(variants)

        total_orig_duration = time.perf_counter() - total_start_t
        timeout_limit = max(total_orig_duration * 1.5, 2.0)

        detailed_results = [(False, "Pending")] * len(variants)
        pending_indices = list(range(len(variants)))

        testing_start_t = time.perf_counter()

        while pending_indices:
            if not self.pool:
                self._init_pool()

            async_tasks = []
            current_batch_indices = []
            for idx in pending_indices:
                v_code = variants[idx]
                assert len(test_cases) == len(expected_data)
                task = self.pool.apply_async(_verifier_process_worker, (v_code, orig_func_name, test_cases, expected_data))
                async_tasks.append(task)
                current_batch_indices.append(idx)

            timed_out = False
            for i, task in enumerate(async_tasks):
                target_idx = current_batch_indices[i]
                try:
                    if timed_out: continue
                    res, err = task.get(timeout=timeout_limit)
                    detailed_results[target_idx] = (res, err)
                except multiprocessing.TimeoutError:
                    if self.verbose:
                        print(f"\n[!] CRITICAL: Variant {target_idx} Timed Out ({timeout_limit:.2f}s). Killing pool.")
                    detailed_results[target_idx] = (False, "Timeout")
                    self.stats["time_timeout_wait"] += timeout_limit
                    self._refresh_pool()
                    timed_out = True
                    pending_indices = current_batch_indices[i + 1:]
                    break
                except Exception as e:
                    detailed_results[target_idx] = (False, f"Worker Fatal Error: {str(e)}")

            if not timed_out:
                pending_indices = []

        testing_duration = time.perf_counter() - testing_start_t
        self.stats["time_verify"] += time.perf_counter() - verify_start_t

        for res, err in detailed_results:
            if res:
                self.stats["passed"] += 1
            elif err:
                if "SyntaxError" in err or "ParsingError" in err:
                    self.stats["failed_syntax"] += 1
                elif "ExecutionError" in err:
                    self.stats["failed_exec"] += 1
                elif "Timeout" in err:
                    self.stats["failed_timeout"] += 1
                elif "ExceptionMismatch" in err:
                    self.stats["failed_exception"] += 1
                elif "ValueMismatch" in err:
                    self.stats["failed_value"] += 1
                elif "WorkerError" in err:
                    self.stats["failed_worker"] += 1
                else:
                    self.stats["failed_unknown"] += 1

        if VERIFIER_LOGGING:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write(f"[Verifier log]\n")
                f.write(f"Timestamp     : {time.ctime()}\n")
                f.write(f"Function name : {orig_func_name}\n")
                f.write(f"Testcases     : {test_cases}\n")
                f.write(f"Baseline time : {total_orig_duration:.4f}s\n")
                f.write(f"Testing time  : {testing_duration:.4f}s\n")
                f.write(f"Timeout limit : {timeout_limit:.2f}s\n")
                f.write(f"Passed        : {sum(1 for r, e in detailed_results if r)}/{len(variants)}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Original code:\n{original_code.strip()}\n\n")

                for idx, (v_code, (res, err)) in enumerate(zip(variants, detailed_results)):
                    status = "[PASS]" if res else f"[FAIL] (Reason: {err})"
                    f.write(f"VARIANT #{idx:02} {status}\n")
                    f.write(f"{v_code.strip()}\n")
                    f.write("-" * 30 + "\n")
                f.write("="*80 + "\n\n")

        final_bools = [r for r, e in detailed_results]
        if self.verbose:
            print(f"[*] Verification complete: {sum(final_bools)}/{len(variants)} passed.")
        return final_bools

    @staticmethod
    def check_verifiable(code: str) -> Tuple[bool, List[str]]:
        """Check whether a Python function can run standalone on literal inputs with a
        deterministic result. Returns (ok, reasons), listing every rejection reason.
        """
        return _check_verifiable_py(code)

def _get_defined_names(func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> set:
    """Collect all names defined within a function (params, locals, nested defs, imports)."""
    names = set()
    for arg in func_def.args.args + func_def.args.posonlyargs + func_def.args.kwonlyargs:
        names.add(arg.arg)
    if func_def.args.vararg:
        names.add(func_def.args.vararg.arg)
    if func_def.args.kwarg:
        names.add(func_def.args.kwarg.arg)

    for node in ast.walk(func_def):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        names.add(elt.id)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    names.add(item.optional_vars.id)
        elif isinstance(node, ast.comprehension):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        names.add(elt.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                names.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)

    return names


def _check_verifiable_py(code: str) -> Tuple[bool, List[str]]:
    # Modules that are safe for pure computation
    SAFE_MODULES = {
        'math', 'cmath', 're', 'string', 'collections', 'datetime', 'time',
        'itertools', 'functools', 'operator', 'copy', 'decimal', 'fractions',
        'statistics', 'heapq', 'bisect', 'array', 'enum', 'dataclasses',
        'typing', 'types', 'abc', 'numbers', 'hashlib', 'hmac',
        'json', 'csv', 'textwrap', 'unicodedata', 'difflib',
        'random', 'secrets',
    }

    # Builtin calls that are unsafe (I/O, process control, dynamic execution)
    UNSAFE_BUILTINS = {
        'open', 'input', 'print',
        'exit', 'quit',
        'eval', 'exec', 'compile', '__import__',
        'globals', 'locals', 'vars',
        'breakpoint',
    }

    # Module prefixes that indicate I/O, network, DB, or OS access
    UNSAFE_MODULE_PREFIXES = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'tempfile', 'glob',
        'socket', 'http', 'urllib', 'requests', 'aiohttp',
        'sqlite3', 'pymysql', 'psycopg2', 'sqlalchemy',
        'smtplib', 'ftplib', 'telnetlib',
        'multiprocessing', 'threading', 'concurrent',
        'signal', 'ctypes',
        'tkinter', 'pygame',
    }

    reasons = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e.msg} (line {e.lineno})"]

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break

    if func_def is None:
        return False, ["NoFunction: no function definition found"]

    # Instance/class methods
    if func_def.args.args:
        first_arg = func_def.args.args[0].arg
        if first_arg in ('self', 'cls'):
            reasons.append(f"InstanceMethod: first parameter is '{first_arg}'")

    # Dotted default arguments (np.sum, tf.float32)
    all_defaults = func_def.args.defaults + func_def.args.kw_defaults
    for d in all_defaults:
        if d is None:
            continue
        if isinstance(d, ast.Attribute):
            reasons.append(f"UnsafeDefault: dotted default argument '{ast.unparse(d)}'")
        elif isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
            reasons.append(f"UnsafeDefault: call in default argument '{ast.unparse(d)}'")

    seen = set()  # deduplicate identical reasons within the same function
    for node in ast.walk(func_def):
        # Unsafe imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split('.')[0]
                if mod_root not in SAFE_MODULES:
                    r = f"UnsafeImport: '{alias.name}'"
                    if r not in seen:
                        reasons.append(r); seen.add(r)

        if isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split('.')[0]
                if mod_root not in SAFE_MODULES:
                    r = f"UnsafeImport: 'from {node.module}'"
                    if r not in seen:
                        reasons.append(r); seen.add(r)

        # Unsafe builtins and calls into I/O / OS modules
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in UNSAFE_BUILTINS:
                r = f"UnsafeCall: '{func.id}()'"
                if r not in seen:
                    reasons.append(r); seen.add(r)

            if isinstance(func, ast.Attribute):
                parts = []
                current = func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    root = parts[-1]
                    if root in UNSAFE_MODULE_PREFIXES:
                        full_name = '.'.join(reversed(parts))
                        r = f"UnsafeCall: '{full_name}()'"
                        if r not in seen:
                            reasons.append(r); seen.add(r)

        # Attribute access without a call (os.environ, sys.argv)
        if isinstance(node, ast.Attribute):
            current = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name) and current.id in UNSAFE_MODULE_PREFIXES:
                parts = []
                c = node
                while isinstance(c, ast.Attribute):
                    parts.append(c.attr)
                    c = c.value
                if isinstance(c, ast.Name):
                    parts.append(c.id)
                full_name = '.'.join(reversed(parts))
                r = f"UnsafeAccess: '{full_name}'"
                if r not in seen:
                    reasons.append(r); seen.add(r)

    # External references: names neither defined locally, builtin nor safe stdlib
    import builtins as _builtins
    builtin_names = set(dir(_builtins))
    safe_top_level = SAFE_MODULES | {
        'np', 'numpy',
        # typing names commonly used in annotations
        'List', 'Dict', 'Tuple', 'Set', 'Optional', 'Union', 'Any',
        'Callable', 'Iterable', 'Iterator', 'Generator', 'Sequence',
        'Mapping', 'MutableMapping', 'Type', 'ClassVar',
    }
    known_names = _get_defined_names(func_def) | builtin_names | safe_top_level
    known_names.add(func_def.name)  # function's own name (for recursion)

    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known_names:
                r = f"ExternalRef: '{node.id}'"
                if r not in seen:
                    reasons.append(r); seen.add(r)

    # Generators cannot be tested by a plain call
    for node in ast.walk(func_def):
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            r = "Generator: function uses yield"
            if r not in seen:
                reasons.append(r); seen.add(r)
            break

    # Attribute access on a parameter implies a non-literal object argument
    param_names = set()
    for arg in func_def.args.args + func_def.args.posonlyargs + func_def.args.kwonlyargs:
        param_names.add(arg.arg)
    if func_def.args.vararg:
        param_names.add(func_def.args.vararg.arg)
    if func_def.args.kwarg:
        param_names.add(func_def.args.kwarg.arg)
    param_names.discard('self')
    param_names.discard('cls')

    for node in ast.walk(func_def):
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in param_names:
                r = f"ParamAttr: parameter '{root.id}' has attribute access '.{node.attr}'"
                if r not in seen:
                    reasons.append(r); seen.add(r)

    # Parameters called as functions
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in param_names:
                r = f"CallableParam: parameter '{node.func.id}' used as callable"
                if r not in seen:
                    reasons.append(r); seen.add(r)

    # Non-deterministic calls
    NONDETERMINISTIC_CALLS = {
        ('time', 'time'), ('time', 'monotonic'), ('time', 'perf_counter'),
        ('time', 'process_time'), ('time', 'time_ns'),
        ('random', 'random'), ('random', 'randint'), ('random', 'randrange'),
        ('random', 'choice'), ('random', 'choices'), ('random', 'sample'),
        ('random', 'shuffle'), ('random', 'uniform'), ('random', 'gauss'),
        ('random', 'getrandbits'), ('random', 'seed'),
        ('datetime', 'now'), ('datetime', 'utcnow'), ('datetime', 'today'),
    }
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            val = node.func.value
            if isinstance(val, ast.Name):
                if (val.id, attr) in NONDETERMINISTIC_CALLS:
                    r = f"NonDeterministic: '{val.id}.{attr}()'"
                    if r not in seen:
                        reasons.append(r); seen.add(r)
            elif isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
                # e.g. datetime.datetime.now()
                if (val.attr, attr) in NONDETERMINISTIC_CALLS:
                    r = f"NonDeterministic: '{val.value.id}.{val.attr}.{attr}()'"
                    if r not in seen:
                        reasons.append(r); seen.add(r)

    return (len(reasons) == 0), reasons


def _test_verifier():
    from code_snippets import get_snippets

    infinite_loop_variant = textwrap.dedent("""\
        def find_Max_Num(arr): 
            a = 0
            while True:
                a += 1
            return a
    """).strip()

    provider = get_provider()
    verifier = EquivalenceVerifier_Py(provider, verbose=True)

    snippets = get_snippets("py")
    variants = list(snippets.values()) + [infinite_loop_variant]

    print(f"\n{'Name':<25} | {'Status'} | {'Cases'}")
    print("-" * 80)

    for name, code in snippets.items():
        tests = verifier.generate_test_inputs(code)
        print(f"{tests}")

        if not tests:
            print(f"{name:<25} | [!] Test Gen Failed")
            continue

        result = verifier.check_equivalence(code, variants, tests)
        status_list = list(map(lambda a : "PASS" if a else "FAIL", result))
        print(f"{name:<25} | {str(status_list):<40} | {len(tests)} cases")

    print("\n[Verifier Stat]")
    print("="*40)
    for key, value in verifier.stats.items():
        if isinstance(value, float):
            print(f" {key:<24} : {value:>8.1f} ")
        else:
            print(f" {key:<24} : {value:>8} ")
    print("="*40 + "\n")

if __name__ == "__main__":
    _test_verifier()