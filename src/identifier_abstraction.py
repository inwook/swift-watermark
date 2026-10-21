import ast
import builtins

def abstract_for_ast_dump(src: str, *, verbose: bool = False) -> str:
    """Identifier abstraction for PythonParser."""
    abstractions = [
        anonymize_variable,
        #sanitize_literal,
    ]

    current_code = src
    for f in abstractions:
        try:
            current_code = f(current_code)
            if verbose:
                print(f"[Abstraction] {f.__name__} applied.")
        except Exception as e:
            if verbose:
                print(f"[Warning] {f.__name__} failed: {e}")
            continue

    return current_code


def abstract_for_tree_sitter(src: str, *, verbose: bool = False) -> str:
    """Identifier abstraction for JavaParser/JSParser. A no-op, since their dump_structure() prints node types only."""
    return src

def sanitize_literal(src: str) -> str:
    try:
        tree = ast.parse(src)
        class LiteralSanitizer(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    node.value = 0
                elif isinstance(node.value, str):
                    node.value = ""
                return node
        return ast.unparse(LiteralSanitizer().visit(tree))
    except:
        return src

def remove_docstring(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    class DocstringRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            # First statement is a docstring iff it is an Expr whose value is a string constant.
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
            return node

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
            return node

        def visit_Module(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
            return node

    return ast.unparse(DocstringRemover().visit(tree))

def rename_variable(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    class InvariantTransformer(ast.NodeTransformer):
        def __init__(self):
            self.var_map = {}   # old name -> normalized name
            self.counter = 0

        def _get_new_name(self, old_name):
            if old_name not in self.var_map:
                self.var_map[old_name] = f"var_{self.counter}"
                self.counter += 1
            return self.var_map[old_name]

        def visit_arg(self, node):
            node.arg = self._get_new_name(node.arg)
            return node

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                node.id = self._get_new_name(node.id)
            elif isinstance(node.ctx, ast.Load):
                if node.id in self.var_map:
                    node.id = self.var_map[node.id]
            elif isinstance(node.ctx, ast.Del):
                if node.id in self.var_map:
                    node.id = self.var_map[node.id]
            return node

        def visit_FunctionDef(self, node):
            # Preserve the function name.
            self.generic_visit(node)
            return node

    transformer = InvariantTransformer()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


_BUILTIN_NAMES = frozenset(dir(builtins))


def _identifier_slots(node) -> list:
    """Identifiers ast.dump() prints as raw strings.
    Attribute.attr and keyword.arg are excluded, since they name the callee's interface.
    """
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return [node.name] if node.name else []
    if isinstance(node, ast.MatchMapping):
        return [node.rest] if node.rest else []
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return list(node.names)
    return []


def anonymize_variable(original_code: str, *, verbose: bool = False) -> str:
    # Collapse every identifier the chunk owns to one token, so the fingerprint depends on structure alone.
    try:
        tree = ast.parse(original_code)
    except SyntaxError:
        return original_code

    names, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.name)
                if alias.asname:
                    imported.add(alias.asname)
        names.update(_identifier_slots(node))

    # Builtins and imports come from outside the chunk, so renaming cannot change them.
    targets = names - _BUILTIN_NAMES - imported

    VAR_STRING = "_var_"

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in targets:
                return ast.copy_location(ast.Name(id=VAR_STRING, ctx=node.ctx), node)
            return node

        def visit_arg(self, node):
            if node.arg in targets:
                node.arg = VAR_STRING
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name in targets:
                node.name = VAR_STRING
            return node

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_ExceptHandler(self, node):
            self.generic_visit(node)
            if node.name in targets:
                node.name = VAR_STRING
            return node

        visit_MatchAs = visit_ExceptHandler
        visit_MatchStar = visit_ExceptHandler

        def visit_MatchMapping(self, node):
            self.generic_visit(node)
            if node.rest in targets:
                node.rest = VAR_STRING
            return node

        def _visit_scope_decl(self, node):
            # Keep the count: global a, b and global a differ structurally.
            node.names = [VAR_STRING if n in targets else n for n in node.names]
            return node

        visit_Global = _visit_scope_decl
        visit_Nonlocal = _visit_scope_decl

        def visit_Attribute(self, node):
            node.value = self.visit(node.value)
            return node

    new_tree = Renamer().visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except Exception:
        return original_code

def _test_abstraction():
    from code_snippets import get_snippet

    test_code = get_snippet("py", "find_max_num")
    print("[Abstracted Code]")
    print(abstract_for_ast_dump(test_code, verbose=True))


def _test_identifier_slots():
    # Renaming anything the chunk owns must not change the abstraction.
    for a, b in [
        ("def outer(n):\n def helper(v):\n  return v*2\n return helper(n)\n",
         "def A(B):\n def C(D):\n  return D*2\n return C(B)\n"),
        ("def g(p):\n try:\n  return int(p)\n except ValueError as e:\n  return str(e)\n",
         "def A(B):\n try:\n  return int(B)\n except ValueError as C:\n  return str(C)\n"),
        ("def bump(step):\n global counter\n counter = counter + step\n return counter\n",
         "def A(B):\n global C\n C = C + B\n return C\n"),
        ("def m(v):\n match v:\n  case [x, *rest]:\n   return x, rest\n  case _:\n   return 0\n",
         "def A(B):\n match B:\n  case [C, *D]:\n   return C, D\n  case _:\n   return 0\n"),
    ]:
        assert abstract_for_ast_dump(a) == abstract_for_ast_dump(b), a
    # Builtins are kept.
    assert abstract_for_ast_dump("def f(x): return len(x)\n") != \
           abstract_for_ast_dump("def f(x): return sum(x)\n")
    print("[Identifier slots] ok")


if __name__ == "__main__":
    _test_abstraction()
    _test_identifier_slots()
