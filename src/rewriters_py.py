import ast
import copy

from rewriters_base import RewriteStrategy

def get_py_strategies():
    return [
        PyAugAssignStrategy(),
        PyCompareSwapStrategy(),
        PyTernaryStrategy(),
        PyRangeStyleStrategy(),
        PyListInitStrategy(),
        PyDictInitStrategy(),
        PyListCompStyleStrategy(),
        PyReturnVarStrategy(),
        PyPairAssignStrategy(),
        PyNegateIfElseStrategy(),
        PyChainedCompareStrategy(),
        PyNotInStrategy(),
        PyIsCompareStrategy(),
        PyYodaEqualsStrategy(),
        PyDeMorganStrategy(),
        PyNegateIfCallStrategy(),
    ]

class ASTRewriteStrategy(RewriteStrategy):
    def is_target(self, src: str) -> bool:
        tree = ast.parse(src)
        return any(self._is_s0(node) or self._is_s1(node) for node in ast.walk(tree))

    def get_current_state(self, src: str) -> int:
        """Return 0 or 1 for the state found, -1 if neither or both are present.
        Raises NotImplementedError if _is_s0/_is_s1 are not implemented."""

        if not (hasattr(self, '_is_s0') and hasattr(self, '_is_s1')):
            raise NotImplementedError(
                f"Strategy '{self.name}' must implement '_is_s0' and '_is_s1' "
                f"to use the default get_current_state.\n"
                f"Alternatively, override get_current_state() manually."
            )

        try:
            tree = ast.parse(src)
        except SyntaxError:
            return -1

        found_s0, found_s1 = False, False

        for node in ast.walk(tree):
            if self._is_s0(node): 
                found_s0 = True
            elif self._is_s1(node): 
                found_s1 = True

        if found_s0 and found_s1: return -1
        if found_s0: return 0
        elif found_s1: return 1
        return -1

class PyAugAssignStrategy(ASTRewriteStrategy):
    """
    s0: a += 1
    s1: a = a + 1
    On set/list/dict, a op= b mutates in place while a = a op b builds a new object,
    which differs when the target is aliased. Sites whose RHS is a container literal are skipped.
    A container held in a variable (a = a - other_set) cannot be detected.
    """
    @property
    def name(self): return "AugAssign"

    SAFE_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)

    _MUTABLE_CONTAINER_TYPES = (ast.Set, ast.List, ast.ListComp, ast.SetComp, ast.Dict, ast.DictComp)

    def _is_s0(self, node: ast.AST) -> bool:
        if not (isinstance(node, ast.AugAssign) and isinstance(node.op, self.SAFE_OPS)):
            return False
        # skip: a -= {v} where RHS is a set/list literal (in-place vs new object)
        if isinstance(node.value, self._MUTABLE_CONTAINER_TYPES):
            return False
        return True

    def _is_s1(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, self.SAFE_OPS):
                # skip: a = a - {v} where RHS operand is a set/list literal
                if isinstance(node.value.right, self._MUTABLE_CONTAINER_TYPES):
                    return False
                try:
                    lhs_code = ast.unparse(target).strip()
                    rhs_left_code = ast.unparse(node.value.left).strip()
                    return lhs_code == rhs_left_code
                except:
                    return False
        return False


    def _to_s0(self, node: ast.AST) -> ast.AST:
        assert self._is_s1(node)
        # a = a + b -> a += b

        return ast.AugAssign(
            target=node.targets[0],
            op=node.value.op,
            value=node.value.right
        )

    def _to_s1(self, node: ast.AST) -> ast.AST:
        assert self._is_s0(node)
        # a += b -> a = a + b

        return ast.Assign(
            targets=[node.target],
            value=ast.BinOp(left=node.target, op=node.op, right=node.value)
        )

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self

        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)  # children first

                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                elif i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)

                return node

        transformer = _Transformer()
        new_tree = transformer.visit(tree)

        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)

class PyCompareSwapStrategy(ASTRewriteStrategy):
    """
    s0: a < b     a <= b 
    s1: b > a     b >= a 
    Swapping operands reverses evaluation order, which matters if both have side effects (A() < B()).
    """

    @property
    def name(self): return "CompareSwap"

    # Operator-flip map (Lt <-> Gt, LtE <-> GtE).
    OP_SWAP_MAP = {
        ast.Lt: ast.Gt,
        ast.LtE: ast.GtE,
        ast.Gt: ast.Lt,
        ast.GtE: ast.LtE
    }

    def _is_s0(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Lt, ast.LtE))
        )

    def _is_s1(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Gt, ast.GtE))
        )

    def _swap_compare(self, node: ast.Compare) -> ast.Compare:
        current_op_type = type(node.ops[0])
        new_op_class = self.OP_SWAP_MAP.get(current_op_type)

        assert new_op_class

        return ast.Compare(
            left=node.comparators[0],     # original RHS becomes LHS
            ops=[new_op_class()],         # operator flipped
            comparators=[node.left]       # original LHS becomes RHS
        )

    def _to_s0(self, node: ast.AST) -> ast.AST:
        assert self._is_s1(node)
        return self._swap_compare(node)

    def _to_s1(self, node: ast.AST) -> ast.AST:
        assert self._is_s0(node)
        return self._swap_compare(node)

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self
        
        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                
                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                elif i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)
                
                return node

        transformer = _Transformer()
        new_tree = transformer.visit(tree)
        
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)


class PyTernaryStrategy(ASTRewriteStrategy):
    """
    s0: x = A if C else B
    s1: if C: x=A else: x=B
    """
    @property
    def name(self): return "Ternary"

    def _is_s0(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Assign) and 
                len(node.targets) == 1 and 
                isinstance(node.value, ast.IfExp))

    def _is_s1(self, node: ast.AST) -> bool:
        if not (isinstance(node, ast.If) and len(node.body) == 1 and len(node.orelse) == 1):
            return False
        
        stmt_if = node.body[0]
        stmt_else = node.orelse[0]

        # Both branches must be plain Assign nodes.
        if not (isinstance(stmt_if, ast.Assign) and isinstance(stmt_else, ast.Assign)):
            return False

        # Same target name on both sides, compared by id rather than unparse.
        try:
            target_if = stmt_if.targets[0]
            target_else = stmt_else.targets[0]
            if isinstance(target_if, ast.Name) and isinstance(target_else, ast.Name):
                return target_if.id == target_else.id
        except (AttributeError, IndexError):
            return False
            
        return False

    def _to_s0(self, node: ast.If) -> ast.Assign:
        assert self._is_s1(node)

        # Structure is guaranteed by _is_s1, so index directly.
        target = node.body[0].targets[0]
        
        return ast.Assign(
            targets=[target],
            value=ast.IfExp(
                test=node.test,
                body=node.body[0].value,
                orelse=node.orelse[0].value
            )
        )

    def _to_s1(self, node: ast.Assign) -> ast.If:
        assert self._is_s0(node)

        target = node.targets[0]
        if_exp = node.value
        
        return ast.If(
            test=if_exp.test,
            body=[ast.Assign(targets=[target], value=if_exp.body)],
            orelse=[ast.Assign(targets=[target], value=if_exp.orelse)]
        )

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self

        class _Transformer(ast.NodeTransformer):
            def visit_Assign(self, node):
                self.generic_visit(node)
                
                if i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)
                return node

            def visit_If(self, node):
                self.generic_visit(node)
                
                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                return node

        transformer = _Transformer()
        new_tree = transformer.visit(tree)
        
        ast.fix_missing_locations(new_tree)
        
        return ast.unparse(new_tree)


class PyRangeStyleStrategy(ASTRewriteStrategy):
    """
    s0: range(n)
    s1: range(0, n)
    """
 
    @property
    def name(self): return "RangeStyle"

    def _is_s0(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call) and 
            isinstance(node.func, ast.Name) and node.func.id == 'range' and
            len(node.args) == 1
        )

    def _is_s1(self, node: ast.AST) -> bool:
        if (isinstance(node, ast.Call) and 
            isinstance(node.func, ast.Name) and node.func.id == 'range' and
            len(node.args) == 2):
            
            # First argument must be the literal 0.
            arg0 = node.args[0]
            return isinstance(arg0, ast.Constant) and arg0.value == 0
            
        return False

    def _to_s0(self, node: ast.Call) -> ast.Call:
        """range(0, n) -> range(n)"""
        return ast.Call(
            func=node.func,
            args=[node.args[1]], 
            keywords=node.keywords
        )

    def _to_s1(self, node: ast.Call) -> ast.Call:
        """range(n) -> range(0, n)"""
        return ast.Call(
            func=node.func,
            args=[ast.Constant(value=0), node.args[0]], 
            keywords=node.keywords
        )

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self
        
        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                
                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                elif i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)
                
                return node

        transformer = _Transformer()
        new_tree = transformer.visit(tree)
        
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)


class PyListInitStrategy(ASTRewriteStrategy):
    """
    s0: []
    s1: list()
    """
    @property
    def name(self): return "ListInit"

    def _is_s0(self, node: ast.AST) -> bool:
        return isinstance(node, ast.List) and len(node.elts) == 0  # []

    def _is_s1(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id == 'list' and len(node.args) == 0 and len(node.keywords) == 0
        return False

    def _to_s0(self, node: ast.Call) -> ast.AST:
        return ast.List(elts=[], ctx=ast.Load())

    def _to_s1(self, node: ast.AST) -> ast.Call:
        return ast.Call(func=ast.Name(id='list', ctx=ast.Load()), args=[], keywords=[])

    def rewrite(self, src: str, i: int) -> str:
        tree = ast.parse(src)
        strategy = self
        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and strategy._is_s1(node): return strategy._to_s0(node)
                elif i == 1 and strategy._is_s0(node): return strategy._to_s1(node)
                return node
        new_tree = _Transformer().visit(tree)
        return ast.unparse(ast.fix_missing_locations(new_tree))

class PyDictInitStrategy(ASTRewriteStrategy):
    """
    s0: {}
    s1: dict()
    """
    @property
    def name(self): return "DictInit"

    def _is_s0(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Dict) and len(node.keys) == 0  # {}

    def _is_s1(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id == 'dict' and len(node.args) == 0 and len(node.keywords) == 0
        return False

    def _to_s0(self, node: ast.Call) -> ast.AST:
        return ast.Dict(keys=[], values=[])

    def _to_s1(self, node: ast.AST) -> ast.Call:
        return ast.Call(func=ast.Name(id='dict', ctx=ast.Load()), args=[], keywords=[])

    def rewrite(self, src: str, i: int) -> str:
        tree = ast.parse(src)
        strategy = self
        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and strategy._is_s1(node): return strategy._to_s0(node)
                elif i == 1 and strategy._is_s0(node): return strategy._to_s1(node)
                return node
        new_tree = _Transformer().visit(tree)
        return ast.unparse(ast.fix_missing_locations(new_tree))



class PyListCompStyleStrategy(ASTRewriteStrategy):
    """
    s0: [x for x in s]
    s1: list(x for x in s)
    """
    @property
    def name(self): return "ListCompStyle"

    def _is_s0(self, node: ast.AST) -> bool:
        """Check: [...]"""
        return isinstance(node, ast.ListComp)

    def _is_s1(self, node: ast.AST) -> bool:
        """Check: list(...) with 1 GeneratorExp arg"""
        return (
            isinstance(node, ast.Call) and 
            isinstance(node.func, ast.Name) and node.func.id == 'list' and
            len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp)
        )

    def _to_s0(self, node: ast.Call) -> ast.ListComp:
        """list(gen) -> [comp]"""
        gen_exp = node.args[0]
        return ast.ListComp(
            elt=gen_exp.elt, 
            generators=gen_exp.generators
        )

    def _to_s1(self, node: ast.ListComp) -> ast.Call:
        """[comp] -> list(gen)"""
        gen_exp = ast.GeneratorExp(
            elt=node.elt, 
            generators=node.generators
        )
        return ast.Call(
            func=ast.Name(id='list', ctx=ast.Load()),
            args=[gen_exp],
            keywords=[]
        )

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self
        
        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                elif i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)
                return node

        transformer = _Transformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        return ast.unparse(new_tree)


class PyReturnVarStrategy(ASTRewriteStrategy):
    """
    s0: return <expr>
    s1: result = <expr>; return result
    """

    CANDIDATE_NAMES = ['result', 'ret', 'res', '_result', '_ret', '_res']

    @property
    def name(self): return "ReturnVar"

    def _pick_var_name(self, func_node) -> str:
        """Pick a natural variable name not already used in the function body."""
        used = {n.id for n in ast.walk(func_node) if isinstance(n, ast.Name)}
        for name in self.CANDIDATE_NAMES:
            if name not in used:
                return name
        return '_result_var'

    def _is_s0_body(self, body: list) -> bool:
        # <expr> must not be a simple variable or constant.

        if not body: return False
        last_stmt = body[-1]
        
        if not isinstance(last_stmt, ast.Return) or last_stmt.value is None:
            return False
            
        if isinstance(last_stmt.value, (ast.Name, ast.Constant)):
            return False

        return True

    def _is_s1_body(self, body: list) -> bool:
        if len(body) < 2: return False
        stmt_assign, stmt_return = body[-2], body[-1]

        if not (isinstance(stmt_assign, ast.Assign) and isinstance(stmt_return, ast.Return)):
            return False
        
        if (len(stmt_assign.targets) != 1 or 
            not isinstance(stmt_assign.targets[0], ast.Name) or
            stmt_return.value is None or
            not isinstance(stmt_return.value, ast.Name)):
            return False

        return stmt_assign.targets[0].id == stmt_return.value.id

    def is_target(self, src: str) -> bool:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_s0_body(node.body) or self._is_s1_body(node.body):
                    return True
        return False

    def _to_s0_body(self, body: list) -> list:
        """var = expr; return var  ->  return expr (drops the temp regardless of name)."""
        stmt_assign = body[-2]
        new_return = ast.Return(value=stmt_assign.value)
        return body[:-2] + [new_return]

    def _to_s1_body(self, body: list, var_name: str = "result") -> list:
        """
        return expr  ->  result = expr; return result
        """
        last_stmt = body[-1]
        
        assign_node = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=last_stmt.value
        )
        return_node = ast.Return(
            value=ast.Name(id=var_name, ctx=ast.Load())
        )

        return body[:-1] + [assign_node, return_node]

    def get_current_state(self, src: str) -> int:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return -1

        if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return -1

        node = tree.body[0]

        if self._is_s0_body(node.body):
            return 0 
        elif self._is_s1_body(node.body):
            return 1
        else:
            return -1

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return src

        node = tree.body[0]

        if i == 0 and self._is_s1_body(node.body):
            node.body = self._to_s0_body(node.body)
        elif i == 1 and self._is_s0_body(node.body):
            var_name = self._pick_var_name(node)
            node.body = self._to_s1_body(node.body, var_name)

        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


class PyPairAssignStrategy(ASTRewriteStrategy):
    """
    s0: a = 1; b = 2  (Expanded / Split)
    s1: a, b = 1, 2   (Compressed / Pair)
    """
    @property
    def name(self): return "PairAssign"

    def _is_simple_assign(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Assign): return False
        if len(node.targets) != 1: return False
        if not isinstance(node.targets[0], ast.Name): return False
        return True

    def _refs_name(self, expr: ast.AST, name: str) -> bool:
        for n in ast.walk(expr):
            if isinstance(n, ast.Name) and n.id == name:
                return True
        return False

    def _is_safe_pair(self, a: ast.AST, b: ast.AST) -> bool:
        """The split form equals the pair form iff the names differ and the second RHS does not read the first name,
        since tuple assignment evaluates every RHS before binding."""
        if not (self._is_simple_assign(a) and self._is_simple_assign(b)):
            return False
        first_name = a.targets[0].id
        second_name = b.targets[0].id
        if first_name == second_name:
            return False
        if self._refs_name(b.value, first_name):
            return False
        return True

    def _is_pair_assign(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Assign): return False
        if len(node.targets) != 1: return False

        target = node.targets[0]
        value = node.value

        if not (isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)):
            return False

        if len(target.elts) != 2 or len(value.elts) != 2:
            return False

        if not all(isinstance(t, ast.Name) for t in target.elts): return False

        # Splitting changes semantics if the second RHS reads the first name, so reject it.
        first_name = target.elts[0].id
        second_name = target.elts[1].id
        if first_name == second_name:
            return False
        if self._refs_name(value.elts[1], first_name):
            return False

        return True

    def get_current_state(self, src: str) -> int:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return -1

        for node in ast.walk(tree):
            if hasattr(node, 'body') and isinstance(node.body, list):
                body = node.body
                i = 0
                while i < len(body):
                    stmt = body[i]

                    # Check s1: Pair Assign (a, b = X, Y)
                    if self._is_pair_assign(stmt):
                        return 1

                    # Check s0: Consecutive Simple Assigns (a=X; b=Y), safe pair only
                    if i + 1 < len(body):
                        next_stmt = body[i+1]
                        if self._is_safe_pair(stmt, next_stmt):
                            return 0

                    i += 1
        return -1

    def is_target(self, src: str) -> bool:
        return self.get_current_state(src) != -1

    def _to_s0_body(self, body: list) -> list:
        """s1 (Pair) -> s0 (Split): a, b = 1, 2  ->  a = 1; b = 2"""
        new_body = []
        for stmt in body:
            if self._is_pair_assign(stmt):
                target_tuple = stmt.targets[0]
                value_tuple = stmt.value
                for t, v in zip(target_tuple.elts, value_tuple.elts):
                    new_body.append(ast.Assign(targets=[t], value=v))
            else:
                new_body.append(stmt)
        return new_body

    def _to_s1_body(self, body: list) -> list:
        """s0 (Split) -> s1 (Pair): a = X; b = Y  ->  a, b = X, Y"""
        new_body = []
        i = 0
        while i < len(body):
            stmt = body[i]

            if i + 1 < len(body):
                next_stmt = body[i+1]
                if self._is_safe_pair(stmt, next_stmt):
                    new_target = ast.Tuple(
                        elts=[stmt.targets[0], next_stmt.targets[0]],
                        ctx=ast.Store()
                    )
                    new_value = ast.Tuple(
                        elts=[stmt.value, next_stmt.value],
                        ctx=ast.Load()
                    )

                    new_assign = ast.Assign(
                        targets=[new_target],
                        value=new_value
                    )
                    new_body.append(new_assign)
                    i += 2
                    continue

            new_body.append(stmt)
            i += 1

        return new_body

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        for node in ast.walk(tree):
            if hasattr(node, 'body') and isinstance(node.body, list):
                if i == 0:
                    node.body = self._to_s0_body(node.body)
                elif i == 1:
                    node.body = self._to_s1_body(node.body)

        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


class PyNegateIfElseStrategy(ASTRewriteStrategy):
    """
    s0: if a == b: A; else: B
    s1: if a != b: B; else: A
    Uses the ==/!= operator flip to avoid the parenthesis accumulation that
    not()-wrapping causes across round-trips. Operator-disjoint with
    CompareSwap (<, <=, >, >=). Excludes patterns owned by Ternary
    (both branches assign the same variable).
    """

    OP_FLIP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}

    @property
    def name(self): return "NegateIfElse"

    def _is_basic_eq_if(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        if not node.orelse:
            return False
        # Exclude elif chains (orelse is a single If).
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            return False
        # Condition must be a single ==/!= comparison.
        test = node.test
        if not isinstance(test, ast.Compare):
            return False
        if len(test.ops) != 1 or len(test.comparators) != 1:
            return False
        if not isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
            return False
        return True

    def _is_ternary_pattern(self, node: ast.If) -> bool:
        """Detect patterns owned by Ternary (both branches assign the same name)."""
        if len(node.body) != 1 or len(node.orelse) != 1:
            return False
        s_if, s_else = node.body[0], node.orelse[0]
        if not (isinstance(s_if, ast.Assign) and isinstance(s_else, ast.Assign)):
            return False
        try:
            t_if = s_if.targets[0]
            t_else = s_else.targets[0]
            if isinstance(t_if, ast.Name) and isinstance(t_else, ast.Name):
                return t_if.id == t_else.id
        except (AttributeError, IndexError):
            pass
        return False

    def _is_valid_target(self, node: ast.AST) -> bool:
        if not self._is_basic_eq_if(node):
            return False
        if self._is_ternary_pattern(node):
            return False
        return True

    def _is_s0(self, node: ast.AST) -> bool:
        return self._is_valid_target(node) and isinstance(node.test.ops[0], ast.Eq)

    def _is_s1(self, node: ast.AST) -> bool:
        return self._is_valid_target(node) and isinstance(node.test.ops[0], ast.NotEq)

    def _flip(self, node: ast.If) -> ast.If:
        old_op_type = type(node.test.ops[0])
        new_test = ast.Compare(
            left=node.test.left,
            ops=[self.OP_FLIP[old_op_type]()],
            comparators=node.test.comparators
        )
        return ast.If(test=new_test, body=node.orelse, orelse=node.body)

    def _to_s0(self, node: ast.AST) -> ast.AST:
        return self._flip(node)

    def _to_s1(self, node: ast.AST) -> ast.AST:
        return self._flip(node)

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self

        class _Transformer(ast.NodeTransformer):
            def visit_If(self, node):
                self.generic_visit(node)
                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                elif i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)
                return node

        new_tree = _Transformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)


class PyChainedCompareStrategy(ASTRewriteStrategy):
    """
    s0: a < x < b       (chained comparison)
    s1: a < x and x < b (split with and)
    """

    @property
    def name(self): return "ChainedCompare"

    def _is_s0(self, node: ast.AST) -> bool:
        """Chained: Compare with exactly 2 ops (e.g. a < x < b)."""
        return (isinstance(node, ast.Compare) and
                len(node.ops) == 2 and
                len(node.comparators) == 2)

    def _is_s1(self, node: ast.AST) -> bool:
        """Split: BoolOp(And) with two single Compares sharing the middle operand."""
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)):
            return False
        if len(node.values) != 2:
            return False
        left, right = node.values
        if not (isinstance(left, ast.Compare) and isinstance(right, ast.Compare)):
            return False
        if len(left.ops) != 1 or len(right.ops) != 1:
            return False
        if len(left.comparators) != 1 or len(right.comparators) != 1:
            return False
        try:
            return ast.unparse(left.comparators[0]) == ast.unparse(right.left)
        except:
            return False

    def _to_s0(self, node: ast.AST) -> ast.AST:
        """a < x and x < b -> a < x < b"""
        left_cmp, right_cmp = node.values
        return ast.Compare(
            left=left_cmp.left,
            ops=[left_cmp.ops[0], right_cmp.ops[0]],
            comparators=[left_cmp.comparators[0], right_cmp.comparators[0]]
        )

    def _to_s1(self, node: ast.AST) -> ast.AST:
        """a < x < b -> a < x and x < b"""
        mid = node.comparators[0]
        left_cmp = ast.Compare(
            left=node.left,
            ops=[node.ops[0]],
            comparators=[mid]
        )
        # deep copy middle operand to avoid AST node sharing
        right_cmp = ast.Compare(
            left=copy.deepcopy(mid),
            ops=[node.ops[1]],
            comparators=[node.comparators[1]]
        )
        return ast.BoolOp(op=ast.And(), values=[left_cmp, right_cmp])

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self

        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                elif i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)
                return node

        new_tree = _Transformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)


class PyNotInStrategy(ASTRewriteStrategy):
    """
    s0: x not in y
    s1: not x in y
    """

    @property
    def name(self): return "NotIn"

    def _is_s0(self, node: ast.AST) -> bool:
        """x not in y: Compare with NotIn operator"""
        return (isinstance(node, ast.Compare) and
                len(node.ops) == 1 and
                isinstance(node.ops[0], ast.NotIn))

    def _is_s1(self, node: ast.AST) -> bool:
        """not x in y: UnaryOp(Not) wrapping Compare with In operator"""
        if not (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
            return False
        inner = node.operand
        return (isinstance(inner, ast.Compare) and
                len(inner.ops) == 1 and
                isinstance(inner.ops[0], ast.In))

    def _to_s0(self, node: ast.AST) -> ast.AST:
        """not x in y -> x not in y"""
        inner = node.operand
        return ast.Compare(
            left=inner.left,
            ops=[ast.NotIn()],
            comparators=inner.comparators
        )

    def _to_s1(self, node: ast.AST) -> ast.AST:
        """x not in y -> not x in y"""
        return ast.UnaryOp(
            op=ast.Not(),
            operand=ast.Compare(
                left=node.left,
                ops=[ast.In()],
                comparators=node.comparators
            )
        )

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy_instance = self

        class _Transformer(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and strategy_instance._is_s1(node):
                    return strategy_instance._to_s0(node)
                elif i == 1 and strategy_instance._is_s0(node):
                    return strategy_instance._to_s1(node)
                return node

        new_tree = _Transformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)


class PyIsCompareStrategy(ASTRewriteStrategy):
    """
    s0: x is None / x is not None  (identity check)
    s1: x == None / x != None      (equality check)
    Only applies when comparator is None.
    """
    @property
    def name(self): return "IsCompare"

    def _is_s0(self, node):
        if not isinstance(node, ast.Compare): return False
        if len(node.ops) != 1 or len(node.comparators) != 1: return False
        if not isinstance(node.ops[0], (ast.Is, ast.IsNot)): return False
        return isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None

    def _is_s1(self, node):
        if not isinstance(node, ast.Compare): return False
        if len(node.ops) != 1 or len(node.comparators) != 1: return False
        if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)): return False
        return isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None

    def _to_s0(self, node):
        new_op = ast.Is() if isinstance(node.ops[0], ast.Eq) else ast.IsNot()
        return ast.Compare(left=node.left, ops=[new_op], comparators=node.comparators)

    def _to_s1(self, node):
        new_op = ast.Eq() if isinstance(node.ops[0], ast.Is) else ast.NotEq()
        return ast.Compare(left=node.left, ops=[new_op], comparators=node.comparators)

    def rewrite(self, src, i):
        tree = ast.parse(src)
        s = self
        class _T(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and s._is_s1(node): return s._to_s0(node)
                elif i == 1 and s._is_s0(node): return s._to_s1(node)
                return node
        return ast.unparse(ast.fix_missing_locations(_T().visit(tree)))


class PyYodaEqualsStrategy(ASTRewriteStrategy):
    """
    s0: x == 5    (variable first)
    s1: 5 == x    (literal first, Yoda style)
    Only for == and != with one literal operand.
    Excludes None (handled by IsCompare).
    """
    @property
    def name(self): return "YodaEquals"

    def _is_literal(self, node):
        return isinstance(node, ast.Constant) and node.value is not None

    def _is_s0(self, node):
        if not isinstance(node, ast.Compare): return False
        if len(node.ops) != 1 or len(node.comparators) != 1: return False
        if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)): return False
        return not self._is_literal(node.left) and self._is_literal(node.comparators[0])

    def _is_s1(self, node):
        if not isinstance(node, ast.Compare): return False
        if len(node.ops) != 1 or len(node.comparators) != 1: return False
        if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)): return False
        return self._is_literal(node.left) and not self._is_literal(node.comparators[0])

    def _swap(self, node):
        return ast.Compare(left=node.comparators[0], ops=node.ops, comparators=[node.left])

    def _to_s0(self, node): return self._swap(node)
    def _to_s1(self, node): return self._swap(node)

    def rewrite(self, src, i):
        tree = ast.parse(src)
        s = self
        class _T(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and s._is_s1(node): return s._to_s0(node)
                elif i == 1 and s._is_s0(node): return s._to_s1(node)
                return node
        return ast.unparse(ast.fix_missing_locations(_T().visit(tree)))


class PyDeMorganStrategy(ASTRewriteStrategy):
    """
    s0: (not a) or (not b)   (expanded)
    s1: not (a and b)        (compact, De Morgan)
    Applies De Morgan's law. Only when all operands are negated.
    """
    @property
    def name(self): return "DeMorgan"

    def _is_s0(self, node):
        if not isinstance(node, ast.BoolOp): return False
        if len(node.values) < 2: return False
        return all(isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not) for v in node.values)

    def _is_s1(self, node):
        if not (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)): return False
        return isinstance(node.operand, ast.BoolOp) and len(node.operand.values) >= 2

    def _to_s0(self, node):
        """not (a and b) -> not a or not b"""
        inner = node.operand
        negated = [ast.UnaryOp(op=ast.Not(), operand=v) for v in inner.values]
        new_op = ast.Or() if isinstance(inner.op, ast.And) else ast.And()
        return ast.BoolOp(op=new_op, values=negated)

    def _to_s1(self, node):
        """not a or not b -> not (a and b)"""
        inner_vals = [v.operand for v in node.values]
        new_op = ast.And() if isinstance(node.op, ast.Or) else ast.Or()
        return ast.UnaryOp(op=ast.Not(), operand=ast.BoolOp(op=new_op, values=inner_vals))

    def rewrite(self, src, i):
        tree = ast.parse(src)
        s = self
        class _T(ast.NodeTransformer):
            def visit(self, node):
                self.generic_visit(node)
                if i == 0 and s._is_s1(node): return s._to_s0(node)
                elif i == 1 and s._is_s0(node): return s._to_s1(node)
                return node
        return ast.unparse(ast.fix_missing_locations(_T().visit(tree)))


class PyNegateIfCallStrategy(ASTRewriteStrategy):
    """
    s0: if call(...): A else: B
    s1: if not call(...): B else: A
    The test must be a bare call or its negation, so at most one negation is ever added.
    Excludes Ternary-convertible patterns and elif chains.
    """

    @property
    def name(self): return "NegateIfCall"

    def _is_ternary_pattern(self, node: ast.If) -> bool:
        if len(node.body) != 1 or len(node.orelse) != 1:
            return False
        a, b = node.body[0], node.orelse[0]
        if not (isinstance(a, ast.Assign) and isinstance(b, ast.Assign)):
            return False
        try:
            t1, t2 = a.targets[0], b.targets[0]
            if isinstance(t1, ast.Name) and isinstance(t2, ast.Name):
                return t1.id == t2.id
        except (AttributeError, IndexError):
            pass
        return False

    def _is_valid_shape(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        if not node.orelse:
            return False
        # elif chain: after the swap, ast.unparse would still emit it as elif, breaking round-trip.
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            return False
        # Symmetric: a body that is a single nested If would become an elif after the swap.
        if len(node.body) == 1 and isinstance(node.body[0], ast.If):
            return False
        if self._is_ternary_pattern(node):
            return False
        return True

    def _is_s0(self, node: ast.AST) -> bool:
        return self._is_valid_shape(node) and isinstance(node.test, ast.Call)

    def _is_s1(self, node: ast.AST) -> bool:
        if not self._is_valid_shape(node):
            return False
        t = node.test
        return (isinstance(t, ast.UnaryOp)
                and isinstance(t.op, ast.Not)
                and isinstance(t.operand, ast.Call))

    def _to_s1(self, node: ast.If) -> ast.If:
        return ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=node.test),
            body=node.orelse,
            orelse=node.body,
        )

    def _to_s0(self, node: ast.If) -> ast.If:
        return ast.If(
            test=node.test.operand,
            body=node.orelse,
            orelse=node.body,
        )

    def rewrite(self, src: str, i: int) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise

        strategy = self

        class _Transformer(ast.NodeTransformer):
            def visit_If(self, node):
                self.generic_visit(node)
                if i == 0 and strategy._is_s1(node):
                    return strategy._to_s0(node)
                elif i == 1 and strategy._is_s0(node):
                    return strategy._to_s1(node)
                return node

        new_tree = _Transformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)


def _test_py_strategies():
    import textwrap
    sample_py_code = textwrap.dedent("""\
        def process_data(items, max_limit):
            # [1. ListInit / DictInit] Mixed initialization styles
            results = list()
            metadata = {}

            # [2. AugAssign] Mixed assignment styles
            total_score = 0
            total_score += 100
            error_count = 0
            error_count = error_count + 1

            # [3. CompareSwap] Mixed comparison operators
            if total_score <= max_limit:
                metadata['status'] = 'pass'
            if error_count > 10:
                metadata['status'] = 'fail'

            # [4. RangeStyle] Mixed range styles
            for i in range(len(items)):
                pass
            for j in range(0, 5):
                pass

            # [5. ListCompStyle] Mixed list comprehension styles
            doubled = [x * 2 for x in items]
            squared = list(x * x for x in items)

            # [6. PairAssign] Mixed variable assignment styles
            a = 1
            b = 2
            c, d = 3, 4

            # [7. Ternary] Mixed ternary operator styles
            status = "high" if total_score > 100 else "low"
            if error_count == 0:
                msg = "perfect"
            else:
                msg = "needs work"

            # [8 & 9. NegateIfElse / ReturnVar] Mixed return styles
            def is_valid(val):
                # ReturnVar (s0)
                return val > 0

            def is_finished(val):
                # NegateIfElse (s0)
                if val == 100:
                    return True
                else:
                    return False

            def get_total():
                # ReturnVar (s0)
                return total_score + error_count

            def get_meta():
                # ReturnVar (s1)
                _res = metadata
                return _res

            return results
    """)

    strategies = get_py_strategies()

    print("=" * 50)
    print("[ Original Code ]")
    print("=" * 50)
    print(sample_py_code.strip())

    # --------------------------------------------
    # Test 1: Convert all to s0
    # --------------------------------------------
    code_s0 = sample_py_code
    for strategy in strategies:
        try:
            code_s0 = strategy.rewrite(code_s0, i=0)
        except Exception as e:
            print(f"Error in {strategy.name} (i=0): {e}")

    print("\n" + "=" * 50)
    print(code_s0.strip())

    # --------------------------------------------
    # Test 2: Convert all to s1 
    # --------------------------------------------
    code_s1 = sample_py_code
    for strategy in strategies:
        try:
            code_s1 = strategy.rewrite(code_s1, i=1)
        except Exception as e:
            print(f"Error in {strategy.name} (i=1): {e}")

    print("\n" + "=" * 50)
    print(code_s1.strip())

if __name__ == "__main__":
    _test_py_strategies()