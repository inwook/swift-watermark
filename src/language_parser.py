import textwrap
import ast
from identifier_abstraction import abstract_for_ast_dump, abstract_for_tree_sitter
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_javascript as tsjavascript
import tree_sitter_java as tsjava

class LanguageParser(ABC):
    """Interface for multi-language code syntax analysis."""
    
    def __init__(self, abstractor: Callable[[str], str]):
        self._abstractor = abstractor

    def abstract_code(self, src: str) -> str:
        return self._abstractor(src)

    @abstractmethod
    def get_code_chunks(self, src: str) -> List[Tuple[bool, str, Optional[str]]]:
        """List of (is_function, chunk_code, function_name or None)."""
        pass

    @abstractmethod
    def dump_structure(self, src: str) -> str:
        """Structural skeleton of the code, used as the hash fingerprint."""
        pass

    @abstractmethod
    def check_syntax(self, src: str) -> Tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        pass


# Wrapper nodes that stand for their only child: (x || y) dumps the same as
# x || y. tree-sitter keeps them, Python's ast does not.
_TRANSPARENT_NODE_TYPES = frozenset({"parenthesized_expression"})

# Comment node types across the tree-sitter grammars used here. A name absent
# from a grammar simply never matches, so one shared set is enough.
_IGNORED_NODE_TYPES = frozenset({
    "comment",          # js
    "line_comment",     # java
    "block_comment",    # java
})


def _dump_ts_structure(node, transparent=_TRANSPARENT_NODE_TYPES, ignored=_IGNORED_NODE_TYPES) -> str:
    """Structure dump of a tree-sitter node, skipping parentheses and comments.
    Same shape as tree-sitter's own str(node), named children only, each prefixed by its field name.
    """
    if node.type in transparent and node.named_child_count == 1:
        return _dump_ts_structure(node.named_child(0), transparent, ignored)

    parts = []
    for i, child in enumerate(node.children):
        if not child.is_named or child.type in ignored:
            continue
        text = _dump_ts_structure(child, transparent, ignored)
        field = node.field_name_for_child(i)
        parts.append(f"{field}: {text}" if field else text)

    # JS new Foo has no arguments child, so add one to dump like new Foo().
    if node.type == "new_expression" and node.child_by_field_name("arguments") is None:
        parts.append("arguments: (arguments)")
    if not parts:
        return f"({node.type})"
    return "(" + node.type + " " + " ".join(parts) + ")"


def _ts_compute_structure_metrics(root_node, branch_types, nest_types,
                             stmt_suffixes, stmt_extra_types, shortcircuit_ops):
    """Walk a tree-sitter subtree and compute (cc, stmt, nest).
    Shared between JSParser and JavaParser.
    """
    cc = 1
    stmt = 0
    max_depth = 0

    def walk(node, depth):
        nonlocal cc, stmt, max_depth
        ntype = node.type
        if ntype in branch_types:
            cc += 1
        elif ntype == "binary_expression":
            for child in node.children:
                if child.type in shortcircuit_ops:
                    cc += 1
                    break
        if ntype in stmt_extra_types or any(ntype.endswith(s) for s in stmt_suffixes):
            stmt += 1
        new_depth = depth + (1 if ntype in nest_types else 0)
        if new_depth > max_depth:
            max_depth = new_depth
        for child in node.children:
            walk(child, new_depth)

    walk(root_node, 0)
    return cc, stmt, max_depth


class PythonParser(LanguageParser):
    def __init__(self):
        super().__init__(abstract_for_ast_dump)

    def get_code_chunks(self, src: str) -> List[Tuple[bool, str, Optional[str]]]:
        try:
            tree = ast.parse(src)
        except Exception as e:
            print(f"[!] AST Parsing failed: {str(e)}. Treating as Non-Function chunk.")
            return [(False, src, None)]

        lines = src.splitlines(keepends=True)

        func_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        func_nodes.sort(key=lambda x: x.lineno)

        def get_node_range(node) -> Tuple[int, int]:
            # Start line: the decorator if present, else the def line.
            start = node.lineno - 1
            if hasattr(node, 'decorator_list') and node.decorator_list:
                start = min(start, node.decorator_list[0].lineno - 1)

            end = getattr(node, 'end_lineno', None)
            if end is None:
                end = node.lineno
            return start, end

        chunks = []
        last_idx = 0

        for node in func_nodes:
            start_line, end_line = get_node_range(node)

            # Non-function gap.
            if start_line > last_idx:
                gap_code = "".join(lines[last_idx:start_line])
                chunks.append((False, gap_code, None))

            func_code = "".join(lines[start_line:end_line])
            chunks.append((True, func_code, node.name))
            
            last_idx = end_line

        # Non-function tail.
        if last_idx < len(lines):
            tail_code = "".join(lines[last_idx:])
            chunks.append((False, tail_code, None))
            
        return chunks

    def dump_structure(self, src: str) -> str:
        clean_src = textwrap.dedent(src)
        tree = ast.parse(clean_src)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)

    def check_syntax(self, src: str) -> Tuple[bool, str]:
        try:
            ast.parse(src)
            return True, ""
        except Exception as e:
            return False, str(e)

    def compute_structure_metrics(self, chunk_code: str) -> Tuple[int, int, int]:
        """(cc, stmt, nest) for a function chunk."""
        try:
            tree = ast.parse(textwrap.dedent(chunk_code))
        except SyntaxError:
            return (0, 0, 0)

        func_node = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_node = node
                break
        if func_node is None:
            return (0, 0, 0)

        BRANCH_TYPES = (ast.If, ast.For, ast.AsyncFor, ast.While,
                        ast.IfExp, ast.ExceptHandler)
        NEST_TYPES = (ast.If, ast.For, ast.AsyncFor, ast.While,
                      ast.Try, ast.With, ast.AsyncWith)

        cc = 1
        stmt = 0
        for n in ast.walk(func_node):
            if n is func_node:
                continue
            if isinstance(n, ast.stmt):
                stmt += 1
            if isinstance(n, BRANCH_TYPES):
                cc += 1
            elif isinstance(n, ast.BoolOp):
                # Each and/or connection adds one independent path.
                cc += max(0, len(n.values) - 1)

        def depth(node, current: int = 0) -> int:
            d = current
            for child in ast.iter_child_nodes(node):
                sub = current + (1 if isinstance(child, NEST_TYPES) else 0)
                d = max(d, depth(child, sub))
            return d

        nest = depth(func_node)
        return cc, stmt, nest


class JSParser(LanguageParser):
    def __init__(self):
        super().__init__(abstract_for_tree_sitter)
        self.JS_LANGUAGE = Language(tsjavascript.language())
        self.parser = Parser(self.JS_LANGUAGE)

    def get_code_chunks(self, src: str) -> List[Tuple[bool, str, Optional[str]]]:
        src_bytes = src.encode("utf-8")
        tree = self.parser.parse(src_bytes)
        root_node = tree.root_node

        # Identify top-level function/method declarations and arrow / function
        # expressions assigned to a const/let/var declarator.
        func_types = {"function_declaration", "method_definition"}
        func_assign_types = {"arrow_function", "function_expression", "function"}

        chunks = []
        last_idx = 0

        for node in root_node.children:
            is_func_node = node.type in func_types

            # const foo = (...) => {...} / var foo = function(...) {...}
            if not is_func_node and node.type in ("lexical_declaration", "variable_declaration"):
                for declarator in node.named_children:
                    if declarator.type == "variable_declarator":
                        value = declarator.child_by_field_name("value")
                        if value and value.type in func_assign_types:
                            is_func_node = True
                            break

            if is_func_node:
                start_byte = node.start_byte
                end_byte = node.end_byte

                # Non-function gap.
                if start_byte > last_idx:
                    gap_code = src_bytes[last_idx:start_byte].decode("utf-8")
                    if gap_code:
                        chunks.append((False, gap_code, None))

                func_code = src_bytes[start_byte:end_byte].decode("utf-8")

                func_name = None
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = src_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                elif node.type in ("lexical_declaration", "variable_declaration"):
                    for declarator in node.named_children:
                        if declarator.type == "variable_declarator":
                            dn = declarator.child_by_field_name("name")
                            if dn:
                                func_name = src_bytes[dn.start_byte:dn.end_byte].decode("utf-8")
                            break

                chunks.append((True, func_code, func_name))
                last_idx = end_byte

        # Non-function tail.
        if last_idx < len(src_bytes):
            tail_code = src_bytes[last_idx:].decode("utf-8")
            if tail_code:
                chunks.append((False, tail_code, None))

        return chunks if chunks else [(False, src, None)]

    def dump_structure(self, src: str) -> str:
        clean_src = textwrap.dedent(src)
        tree = self.parser.parse(bytes(clean_src, "utf8"))
        return _dump_ts_structure(tree.root_node)

    def check_syntax(self, src: str) -> Tuple[bool, str]:
        tree = self.parser.parse(src.encode("utf-8"))
        # tree-sitter does not raise on errors, it marks them in has_error.
        if tree.root_node.has_error:
            return False, "Syntax error detected by tree-sitter parser"
        return True, ""

    def compute_structure_metrics(self, chunk_code: str) -> Tuple[int, int, int]:
        """(cc, stmt, nest) for a function chunk."""
        tree = self.parser.parse(chunk_code.encode("utf-8"))
        if tree.root_node.has_error:
            return (0, 0, 0)
        return _ts_compute_structure_metrics(
            tree.root_node,
            branch_types={"if_statement", "for_statement", "for_in_statement",
                          "while_statement", "do_statement", "switch_case",
                          "catch_clause", "ternary_expression"},
            nest_types={"if_statement", "for_statement", "for_in_statement",
                        "while_statement", "do_statement", "switch_statement",
                        "try_statement", "catch_clause"},
            stmt_suffixes=("_statement",),
            stmt_extra_types={"lexical_declaration", "variable_declaration"},
            shortcircuit_ops={"&&", "||"},
        )


class JavaParser(LanguageParser):
    def __init__(self):
        super().__init__(abstract_for_tree_sitter)
        self.JAVA_LANGUAGE = Language(tsjava.language())
        self.parser = Parser(self.JAVA_LANGUAGE)

    def get_code_chunks(self, src: str) -> List[Tuple[bool, str, Optional[str]]]:
        src_bytes = src.encode("utf-8")
        tree = self.parser.parse(src_bytes)
        
        query_str = """
        [
            (method_declaration) @func
            (constructor_declaration) @func
        ]
        """
        q = Query(self.JAVA_LANGUAGE, query_str)
        cursor = QueryCursor(q)
        func_nodes = []
        if hasattr(cursor, 'captures'):
            caps = cursor.captures(tree.root_node)
            if isinstance(caps, list):
                func_nodes = [node for node, name in caps if name == "func"]
            elif isinstance(caps, dict) and "func" in caps:
                nodes = caps["func"]
                func_nodes = nodes if isinstance(nodes, list) else [nodes]
        elif hasattr(cursor, 'matches'):
            for match in cursor.matches(tree.root_node):
                caps = match[1] if isinstance(match, tuple) else match
                if isinstance(caps, dict) and "func" in caps:
                    nodes = caps["func"]
                    if isinstance(nodes, list):
                        func_nodes.extend(nodes)
                    else:
                        func_nodes.append(nodes)
        func_nodes.sort(key=lambda x: x.start_byte)

        # Drop methods nested in a previous one (e.g. inside an anonymous class).
        top_level_nodes = []
        last_end = -1
        for node in func_nodes:
            if node.start_byte < last_end:
                continue
            top_level_nodes.append(node)
            last_end = node.end_byte
        func_nodes = top_level_nodes

        chunks = []
        last_idx = 0

        for node in func_nodes:
            start_byte = node.start_byte
            end_byte = node.end_byte

            # Non-function gap.
            if start_byte > last_idx:
                gap_code = src_bytes[last_idx:start_byte].decode("utf-8")

                # Keep whitespace-only gaps: they can include the next method's indentation.
                if gap_code:
                    chunks.append((False, gap_code, None))

            func_code = src_bytes[start_byte:end_byte].decode("utf-8")
            
            name_node = node.child_by_field_name("name")
            func_name = name_node.text.decode("utf-8") if name_node else None

            chunks.append((True, func_code, func_name))
            last_idx = end_byte

        # Non-function tail.
        if last_idx < len(src_bytes):
            tail_code = src_bytes[last_idx:].decode("utf-8")
            if tail_code:
                chunks.append((False, tail_code, None))

        return chunks if chunks else [(False, src, None)]

    def dump_structure(self, src: str) -> str:
        clean_src = textwrap.dedent(src)
        tree = self.parser.parse(bytes(clean_src, "utf8"))
        return _dump_ts_structure(tree.root_node)

    def check_syntax(self, src: str) -> Tuple[bool, str]:
        tree = self.parser.parse(src.encode("utf-8"))
        if tree.root_node.has_error:
            return False, "Syntax error detected by tree-sitter parser"
        return True, ""

    def compute_structure_metrics(self, chunk_code: str) -> Tuple[int, int, int]:
        """(cc, stmt, nest) for a function chunk."""
        # tree-sitter-java rejects a bare method body, so wrap it in a class.
        wrapped = "class __W { " + chunk_code + " }"
        tree = self.parser.parse(wrapped.encode("utf-8"))
        if tree.root_node.has_error:
            # Fall back to a direct parse, partial trees are still walkable.
            tree = self.parser.parse(chunk_code.encode("utf-8"))
        return _ts_compute_structure_metrics(
            tree.root_node,
            branch_types={"if_statement", "for_statement", "enhanced_for_statement",
                          "while_statement", "do_statement",
                          "switch_block_statement_group", "catch_clause",
                          "ternary_expression"},
            nest_types={"if_statement", "for_statement", "enhanced_for_statement",
                        "while_statement", "do_statement", "switch_expression",
                        "switch_statement", "try_statement", "catch_clause"},
            stmt_suffixes=("_statement",),
            stmt_extra_types={"local_variable_declaration"},
            shortcircuit_ops={"&&", "||"},
        )


def get_parser(lang: str) -> LanguageParser:
    if lang == "py":
        return PythonParser()
    elif lang == "js":
        return JSParser()
    elif lang == "java":
        return JavaParser()
    else:
        raise ValueError(f"Unsupported language: '{lang}'")


def _test_python_parser():
    sample_code_py = textwrap.dedent('''\
        import math
        import sys

        # Set global variables
        GLOBAL_VAR = 100

        @staticmethod
        def calculate_area(radius):
            """Calculates the area of a circle."""
            area = math.pi * (radius ** 2)
            return area

        def greet(name, n):
            """
            This is docstring.
            """
            for i in range(n):
                print("Hello, " + name)
            return True

        # End of code
        print("All functions defined.")
    ''')

    parser = PythonParser()
    chunks = parser.get_code_chunks(sample_code_py)

    print("="*40)
    print("[1] Testing: get_code_chunks()")
    print("="*40)
    
    for idx, (is_func, code, name) in enumerate(chunks):
        chunk_type = f"Function ('{name}')" if is_func else "Non-Function"
        print(f"\n--- Chunk {idx + 1} : {chunk_type} ---")
        print(code.strip())

    print("\n" + "="*40)
    print("[2] Testing: dump_structure()")
    print("="*40)

    for is_func, code, name in chunks:
        print("\n[Target Code]")
        print(code.strip())
        print("\n[AST Dump String]")
        print(parser.dump_structure(code))

    print("\n" + "="*40)
    print("[3] Testing: abstract_code()")
    print("="*40)
    
    for is_func, code, name in chunks:
        abstracted = parser.abstract_code(code)
        print("\n[Abstracted Code]")
        print(abstracted.strip())


def _test_js_parser():
    sample_code_js = textwrap.dedent('''\
        import { axios } from 'axios';

        const GLOBAL_VAL = 100;

        const add = (a, b) => {
            return a + b;
        };
                                     
        /**
         * Calculates the average of positive values within an array of objects.
         * @param {Array} dataArray - The array of data objects to process.
         * @returns {number} The calculated average (returns 0 if the array is empty).
         */
        function calculateStats(dataArray) {
            let sum = 0;
            for (const item of dataArray) {
                if (item && item.value > 0) {
                    sum += item.value;
                }
            }
            return sum / dataArray.length || 0;
        }

        class Calculator {
            multiply(a, b) {
                return a * b;
            }
        }

        console.log("Done");
    ''')

    parser = JSParser()
    chunks = parser.get_code_chunks(sample_code_js)

    print("="*40)
    print("[1] Testing: get_code_chunks()")
    print("="*40)
    
    for idx, (is_func, code, name) in enumerate(chunks):
        chunk_type = f"Function ('{name}')" if is_func else "Non-Function"
        print(f"\n--- Chunk {idx + 1} : {chunk_type} ---")
        print(code.strip())

    print("\n" + "="*40)
    print("[2] Testing: dump_structure()")
    print("="*40)

    for is_func, code, name in chunks:
        print("\n[Target Code]")
        print(code.strip())
        print("\n[Dump String]")
        print(parser.dump_structure(code))

    print("\n" + "="*40)
    print("[3] Testing: check_syntax()")
    print("="*40)
    is_valid, msg = parser.check_syntax(sample_code_js)
    print(f"Syntax Valid: {is_valid}")
    if not is_valid:
        print(f"Error Message: {msg}")


def _test_java_parser():
    sample_code_java = textwrap.dedent('''\
        import java.util.List;

        public class MyClass {
            private int counter = 0;

            public MyClass() {
                this.counter = 1;
            }

            /**
             * Adds two integers.
             */
            public int add(int a, int b) {
                return a + b;
            }

            public void run() {
                System.out.println("Running...");
            }
        }
    ''')

    parser = JavaParser()
    chunks = parser.get_code_chunks(sample_code_java)
    
    print("\n" + "="*40)
    print("[1] Testing JavaParser: get_code_chunks()")
    print("="*40)
    
    for idx, (is_func, code, name) in enumerate(chunks):
        chunk_type = f"Function ('{name}')" if is_func else "Non-Function"
        print(f"\n--- Chunk {idx + 1} : {chunk_type} ---")
        print(code.strip())

    print("\n" + "="*40)
    print("[2] Testing JavaParser: dump_structure()")
    print("="*40)
    dump_str = parser.dump_structure(sample_code_java)
    print(dump_str)

    print("\n" + "="*40)
    print("[3] Testing JavaParser: check_syntax()")
    print("="*40)
    is_valid, msg = parser.check_syntax(sample_code_java)
    print(f"Syntax Valid: {is_valid}")
    is_valid, msg = parser.check_syntax("public class {")
    print(f"Invalid Syntax Valid: {is_valid}")
    if not is_valid:
        print(f"Error Message: {msg}")


def _test_structure_metrics():
    """Print (cc, stmt, nest) for hand-picked chunks in each language.

    Sanity check after touching the branch / nest / statement type sets:
    cc should rise with each branch, nest only with real nesting depth.
    """
    py_samples = [
        ("trivial", "def f(x):\n    return x + 1"),
        ("loop_only",
         textwrap.dedent("""\
             def total(xs):
                 s = 0
                 for x in xs:
                     s += x
                 return s
         """)),
        ("nested_branch",
         textwrap.dedent("""\
             def fizz_buzz(n):
                 ns = []
                 for i in range(n):
                     if i % 11 == 0 or i % 13 == 0:
                         ns.append(i)
                 s = ''.join(map(str, ns))
                 ans = 0
                 for c in s:
                     ans += (c == '7')
                 return ans
         """)),
        ("deeply_nested",
         textwrap.dedent("""\
             def grid_search(matrix):
                 best = None
                 for row in matrix:
                     for v in row:
                         if v is not None:
                             if best is None or v > best:
                                 best = v
                 return best
         """)),
    ]
    js_samples = [
        ("trivial", "function f(x) { return x + 1; }"),
        ("nested_branch",
         "function fizzBuzz(n) {\n  let ans = 0;\n  for (let i = 0; i < n; i++) {\n"
         "    if (i % 11 === 0 || i % 13 === 0) {\n      const s = String(i);\n"
         "      for (const c of s) ans += (c === '7') ? 1 : 0;\n    }\n  }\n  return ans;\n}"),
    ]
    java_samples = [
        ("trivial", "public int f(int x) { return x + 1; }"),
        ("nested_branch",
         "public int fizzBuzz(int n) {\n  int ans = 0;\n  for (int i = 0; i < n; i++) {\n"
         "    if (i % 11 == 0 || i % 13 == 0) {\n      String s = String.valueOf(i);\n"
         "      for (char c : s.toCharArray()) if (c == '7') ans++;\n    }\n  }\n  return ans;\n}"),
    ]

    print("\n" + "=" * 40)
    print("[4] Testing: compute_structure_metrics()")
    print("=" * 40)
    for lang_name, parser, samples in [("PYTHON", PythonParser(), py_samples),
                                       ("JAVASCRIPT", JSParser(), js_samples),
                                       ("JAVA", JavaParser(), java_samples)]:
        print(f"--- {lang_name} ---")
        for label, code in samples:
            cc, stmt, nest = parser.compute_structure_metrics(code)
            print(f"  {label:<18} cc={cc:>2}  stmt={stmt:>2}  nest={nest:>2}")
        print()


def _test_ts_structure_dump():
    JS, JAVA = JSParser(), JavaParser()
    J = "class A { int f(int a, int b) { %s } }"

    # Parentheses and comments must not reach the dump.
    for parser, a, b in [
        (JS,   "function f(a,b){ return (a || b); }", "function f(a,b){ return a || b; }"),
        (JS,   "function f(a){ /* x */ return a; }",  "function f(a){ // hi\n return a; }"),
        (JAVA, J % "return ((a + b));",               J % "/* x */ return a + b;"),
        (JAVA, J % "// hi\n return a;",              J % "return a;"),
    ]:
        assert parser.dump_structure(a) == parser.dump_structure(b), a
    assert JS.dump_structure("function f(a){return a;}") != \
           JS.dump_structure("function f(a){return -a;}")

    # Code with neither still dumps exactly as tree-sitter prints it.
    src = "function f(a,b){ return a+b; }"
    assert JS.dump_structure(src) == str(JS.parser.parse(bytes(src, "utf8")).root_node)
    print("\n[5] dump_structure() invariance: ok")


if __name__ == "__main__":
    _test_python_parser()
    _test_js_parser()
    _test_java_parser()
    _test_structure_metrics()
    _test_ts_structure_dump()
