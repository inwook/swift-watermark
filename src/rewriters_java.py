import tree_sitter_java as java_lang
from tree_sitter import Language, Parser, Query, QueryCursor

from rewriters_base import RewriteStrategy

def get_java_strategies():
    return [
        JavaAugAssignStrategy(),
        JavaCompareSwapStrategy(),
        JavaBooleanReturnStrategy(),
        JavaTernaryStrategy(),
        JavaNegateIfElseStrategy(),
        JavaPrefixPostfixStrategy(),
        JavaReturnVarStrategy(),
        JavaYodaEqualsStrategy(),
        JavaWhileForSwapStrategy(),
        JavaOptionalBracesStrategy(),
        JavaDiamondInferenceStrategy(),
        JavaArrayTypeStyleStrategy(),
    ]

class TreeSitterJavaStrategy(RewriteStrategy):
    # Query() compilation dominates parse time, so compiled queries are cached per class.
    _query_cache = {}  # { query_str: Query }

    def __init__(self, language_capsule):
        self.LANG = Language(language_capsule)
        self.parser = Parser(self.LANG)

    def _get_query(self, query_str: str) -> Query:
        """Return a cached compiled Query object, creating one if not yet cached."""
        if query_str not in TreeSitterJavaStrategy._query_cache:
            TreeSitterJavaStrategy._query_cache[query_str] = Query(self.LANG, query_str)
        return TreeSitterJavaStrategy._query_cache[query_str]

    def _get_target_nodes(self, src: str, query_str: str) -> list:
        """Generic helper: collect all nodes captured as @target by query_str."""
        try:
            tree = self.parser.parse(bytes(src, "utf8"))
        except SyntaxError:
            return []

        q = self._get_query(query_str)
        target_nodes = []
        cursor = QueryCursor(q)

        if hasattr(cursor, 'captures'):
            caps = cursor.captures(tree.root_node)
            if isinstance(caps, list):
                for node, name in caps:
                    if name == "target":
                        target_nodes.append(node)
            elif isinstance(caps, dict) and "target" in caps:
                nodes = caps["target"]
                if isinstance(nodes, list):
                    target_nodes.extend(nodes)
                else:
                    target_nodes.append(nodes)

        elif hasattr(cursor, 'matches'):
            for match in cursor.matches(tree.root_node):
                caps = match[1] if isinstance(match, tuple) else match
                if isinstance(caps, dict) and "target" in caps:
                    nodes = caps["target"]
                    if isinstance(nodes, list):
                        target_nodes.extend(nodes)
                    else:
                        target_nodes.append(nodes)

        return target_nodes

    def get_current_state(self, src: str) -> int:
        if not (hasattr(self, '_query_s0') and hasattr(self, '_query_s1')):
            raise NotImplementedError(
                f"Strategy '{self.name}' must implement '_query_s0' and '_query_s1'."
            )

        found_s0 = len(self._get_target_nodes(src, self._query_s0)) > 0
        found_s1 = len(self._get_target_nodes(src, self._query_s1)) > 0

        if found_s0 and found_s1:
            return -1
        if found_s0:
            return 0
        elif found_s1:
            return 1
        else:
            return -1

    def is_target(self, src: str) -> bool:
        # @InlineMe promises a one-statement body matching its replacement,
        # so ReturnVar, BooleanReturn and Ternary can break the build.
        if "@InlineMe" in src:
            return False
        return self.get_current_state(src) != -1

    def rewrite(self, src: str, i: int) -> str:
        query_str = self._query_s1 if i == 0 else self._query_s0
        target_nodes = self._get_target_nodes(src, query_str)

        target_nodes.sort(key=lambda n: n.start_byte)

        # Drop nodes inside an already selected range, so nested matches are not rewritten twice.
        unique_nodes = []
        last_end = -1
        for node in target_nodes:
            if node.start_byte < last_end:
                continue
            unique_nodes.append(node)
            last_end = node.end_byte

        new_segments = []
        last_idx = 0
        src_bytes = bytes(src, "utf8")

        for node in unique_nodes:
            replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
            if replacement is not None:
                new_segments.append(src_bytes[last_idx:node.start_byte])
                new_segments.append(bytes(replacement, "utf8"))
                last_idx = node.end_byte

        new_segments.append(src_bytes[last_idx:])
        return b"".join(new_segments).decode("utf8")


class JavaAugAssignStrategy(TreeSitterJavaStrategy):
    """
    s0: a += 1 / a -= 1 / a *= 2 / a /= 2 / a %= 3
    s1: a = a + 1 / a = a - 1 / ...
    += is safe for any RHS. -=, *=, /=, %= need a simple RHS (identifier, literal, ...),
    or expansion changes precedence.
    """
    COMPOUND_TO_BINARY = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%"}
    BINARY_TO_COMPOUND = {v: k for k, v in COMPOUND_TO_BINARY.items()}

    # Simple RHS forms whose expansion preserves precedence under -=/*=//=/%=.
    SAFE_RHS_TYPES = {
        'identifier', 'decimal_integer_literal', 'hex_integer_literal',
        'octal_integer_literal', 'binary_integer_literal',
        'parenthesized_expression', 'method_invocation',
        'field_access', 'array_access',
    }

    def __init__(self):
        super().__init__(java_lang.language())

        self._query_s0 = """
        [
            (assignment_expression left: (identifier) @lhs operator: "+=" right: (_) @rhs) @target
            (assignment_expression left: (identifier) @lhs operator: "-=" right: (_) @rhs) @target
            (assignment_expression left: (identifier) @lhs operator: "*=" right: (_) @rhs) @target
            (assignment_expression left: (identifier) @lhs operator: "/=" right: (_) @rhs) @target
            (assignment_expression left: (identifier) @lhs operator: "%=" right: (_) @rhs) @target
        ]
        """

        self._query_s1 = """
        (assignment_expression
            left: (identifier) @lhs
            operator: "="
            right: (binary_expression
                left: (identifier) @b_lhs
                operator: ["+" "-" "*" "/" "%"]
                right: (_) @rhs)
        ) @target
        """

    @property
    def name(self) -> str:
        return "AugAssign"

    def _has_unsafe_expr(self, node) -> bool:
        """Whether the RHS contains a method call, cast, float literal or ternary.
        Compound assignment casts implicitly (sum += Math.ceil(1.5) compiles for an int sum,
        sum = sum + Math.ceil(1.5) does not), so expansion is unsafe whenever the RHS may change type.
        """
        if node.type in ('method_invocation', 'cast_expression',
                         'decimal_floating_point_literal', 'hex_floating_point_literal',
                         'ternary_expression'):
            return True
        for child in node.children:
            if self._has_unsafe_expr(child):
                return True
        return False

    def _get_compound_op(self, node):
        """Extract the compound-assignment operator from an assignment_expression."""
        for child in node.children:
            if child.type in self.COMPOUND_TO_BINARY:
                return child.type
        return None

    def _get_binary_op(self, node):
        """Extract the binary operator from the RHS binary_expression."""
        right = node.child_by_field_name('right')
        if not right or right.type != 'binary_expression':
            return None
        for child in right.children:
            if child.type in self.BINARY_TO_COMPOUND:
                return child.type
        return None

    def _is_valid_s0(self, node) -> bool:
        """Check whether an s0 node can be safely expanded to s1."""
        op = self._get_compound_op(node)
        if not op:
            return False
        rhs = node.child_by_field_name('right')
        if not rhs or self._has_unsafe_expr(rhs):
            return False
        if op == '+=':
            return True
        return rhs.type in self.SAFE_RHS_TYPES

    def _is_valid_s1(self, node) -> bool:
        """Check whether an s1 node can be safely compressed to s0."""
        lhs = node.child_by_field_name('left')
        right = node.child_by_field_name('right')
        if not (lhs and right and right.type == 'binary_expression'):
            return False
        b_lhs = right.child_by_field_name('left')
        if not b_lhs or lhs.text != b_lhs.text:
            return False
        op = self._get_binary_op(node)
        if not op:
            return False
        rhs = right.child_by_field_name('right')
        if not rhs or self._has_unsafe_expr(rhs):
            return False
        if op == '+':
            return True
        return rhs.type in self.SAFE_RHS_TYPES

    def get_current_state(self, src: str) -> int:
        s0_nodes = self._get_target_nodes(src, self._query_s0)
        found_s0 = any(self._is_valid_s0(n) for n in s0_nodes)

        s1_nodes = self._get_target_nodes(src, self._query_s1)
        found_s1 = any(self._is_valid_s1(n) for n in s1_nodes)

        if found_s0 and found_s1:
            return -1
        if found_s0:
            return 0
        elif found_s1:
            return 1
        else:
            return -1

    def _to_s0(self, node) -> str:
        """s1 (a = a + b) -> s0 (a += b)"""
        if not self._is_valid_s1(node):
            return None
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        lhs = get_text(node.child_by_field_name('left'))
        binary_exp = node.child_by_field_name('right')
        rhs = get_text(binary_exp.child_by_field_name('right'))
        op = self._get_binary_op(node)
        compound_op = self.BINARY_TO_COMPOUND.get(op)
        if not compound_op:
            return None
        return f"{lhs} {compound_op} {rhs}"

    def _to_s1(self, node) -> str:
        """s0 (a += b) -> s1 (a = a + b)"""
        if not self._is_valid_s0(node):
            return None
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        lhs = get_text(node.child_by_field_name('left'))
        rhs = get_text(node.child_by_field_name('right'))
        op = self._get_compound_op(node)
        binary_op = self.COMPOUND_TO_BINARY.get(op)
        if not binary_op:
            return None
        return f"{lhs} = {lhs} {binary_op} {rhs}"

    def _canonical_s0(self, node) -> str:
        """Normalize whitespace in an existing s0 (e.g., a-=b -> a -= b)."""
        if not self._is_valid_s0(node):
            return None
        def get_text(n):
            return n.text.decode('utf8') if n else ""
        lhs = get_text(node.child_by_field_name('left'))
        rhs = get_text(node.child_by_field_name('right'))
        op = self._get_compound_op(node)
        if not op:
            return None
        return f"{lhs} {op} {rhs}"

    def rewrite(self, src: str, i: int) -> str:
        """Override: also canonicalize whitespace of existing s0 nodes during
        normalization to guarantee round-trip stability."""
        if i == 0:
            s1_nodes = [(n, 's1') for n in self._get_target_nodes(src, self._query_s1) if self._is_valid_s1(n)]
            s0_nodes = [(n, 's0') for n in self._get_target_nodes(src, self._query_s0) if self._is_valid_s0(n)]
            all_nodes = s1_nodes + s0_nodes
        else:
            all_nodes = [(n, 's0') for n in self._get_target_nodes(src, self._query_s0) if self._is_valid_s0(n)]

        all_nodes.sort(key=lambda x: x[0].start_byte)
        unique = []
        last_end = -1
        for node, state in all_nodes:
            if node.start_byte < last_end:
                continue
            unique.append((node, state))
            last_end = node.end_byte

        new_segments = []
        last_idx = 0
        src_bytes = bytes(src, 'utf8')
        for node, state in unique:
            if i == 0:
                replacement = self._to_s0(node) if state == 's1' else self._canonical_s0(node)
            else:
                replacement = self._to_s1(node)
            if replacement is not None:
                new_segments.append(src_bytes[last_idx:node.start_byte])
                new_segments.append(bytes(replacement, 'utf8'))
                last_idx = node.end_byte
        new_segments.append(src_bytes[last_idx:])
        return b"".join(new_segments).decode('utf8')


class JavaCompareSwapStrategy(TreeSitterJavaStrategy):
    """
    s0: a < b, a <= b
    s1: b > a, b >= a
    """
    def __init__(self):
        super().__init__(java_lang.language())
        
        self._query_s0 = """
        (binary_expression
            left: (_) @lhs
            operator: ["<" "<="]
            right: (_) @rhs
        ) @target
        """
        
        self._query_s1 = """
        (binary_expression
            left: (_) @lhs
            operator: [">" ">="]
            right: (_) @rhs
        ) @target
        """
        
        self.OP_SWAP_MAP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}

    @property
    def name(self) -> str:
        return "CompareSwap"

    def _swap_compare(self, node) -> str:
        def get_text(n):
            return n.text.decode('utf8') if n else ""
        
        lhs = get_text(node.child_by_field_name('left'))
        rhs = get_text(node.child_by_field_name('right'))

        op = None
        for child in node.children:
            if child.type in self.OP_SWAP_MAP:
                op = child.type
                break

        if not op:
            return None
        
        new_op = self.OP_SWAP_MAP[op]
        return f"{rhs} {new_op} {lhs}"

    def _to_s0(self, node) -> str:
        return self._swap_compare(node)
    
    def _to_s1(self, node) -> str:
        return self._swap_compare(node)


class JavaBooleanReturnStrategy(TreeSitterJavaStrategy):
    """
    s0: return <bool_expr>;
    s1: if (<bool_expr>) { return true; } else { return false; }
    """
    def __init__(self):
        super().__init__(java_lang.language())
        
        # s0: only match return statements whose expression is a comparison.
        self._query_s0 = """
        (return_statement
            (binary_expression
                operator: ["==" "!=" "<" ">" "<=" ">="]
            ) @expr
        ) @target
        """

        # s1: if-else returning true/false.
        self._query_s1 = """
        (if_statement
            condition: (parenthesized_expression) @cond
            consequence: (block (return_statement (true)))
            alternative: (block (return_statement (false)))
        ) @target
        """

    @property
    def name(self) -> str:
        return "BooleanReturn"

    def _to_s0(self, node) -> str:
        """s1 (if-else) -> s0 (return bool). Node is an if_statement."""
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        cond_node = node.child_by_field_name('condition')
        if not cond_node:
            return None

        cond_text = get_text(cond_node).strip()
        # Strip the condition's outer parens so the return reads naturally.
        if cond_text.startswith('(') and cond_text.endswith(')'):
            cond_text = cond_text[1:-1].strip()

        return f"return {cond_text};"

    def _to_s1(self, node) -> str:
        """s0 (return bool) -> s1 (if-else). The return_statement's first named child is the expression."""
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        if len(node.named_children) == 0:
            return None

        expr_text = get_text(node.named_children[0])
        return f"if ({expr_text}) {{\n    return true;\n}} else {{\n    return false;\n}}"


class JavaTernaryStrategy(TreeSitterJavaStrategy):
    """
    s0: x = cond ? a : b;
    s1: if (cond) { x = a; } else { x = b; }
    """
    def __init__(self):
        super().__init__(java_lang.language())

        self._query_s0 = """
        (expression_statement
            (assignment_expression
                operator: "="
                right: (ternary_expression)
            )
        ) @target
        """

        self._query_s1 = """
        (if_statement
            condition: (parenthesized_expression) @cond
            consequence: (block) @then_block
            alternative: (block) @else_block
        ) @target
        """

    @property
    def name(self) -> str:
        return "Ternary"

    def _is_ternary_expandable(self, node) -> bool:
        """Can A = X ? B : C become an if-else? Not when A is textually B or C, which expands to A = A."""
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        assign = None
        for child in node.named_children:
            if child.type == 'assignment_expression':
                assign = child
                break
        if not assign:
            return False

        ternary = assign.child_by_field_name('right')
        if not ternary or ternary.type != 'ternary_expression':
            return False

        return get_text(assign.child_by_field_name('left')) not in (
            get_text(ternary.child_by_field_name('consequence')),
            get_text(ternary.child_by_field_name('alternative')))

    def _is_ternary_convertible(self, node) -> bool:
        """Check that both branches of the if-else are a single assignment to
        the same variable."""
        cons = node.child_by_field_name('consequence')
        alt = node.child_by_field_name('alternative')
        if not (cons and alt):
            return False

        cons_stmts = [c for c in cons.named_children]
        alt_stmts = [c for c in alt.named_children]

        if len(cons_stmts) != 1 or len(alt_stmts) != 1:
            return False
        if cons_stmts[0].type != 'expression_statement' or alt_stmts[0].type != 'expression_statement':
            return False

        cons_expr = cons_stmts[0].named_children[0] if cons_stmts[0].named_children else None
        alt_expr = alt_stmts[0].named_children[0] if alt_stmts[0].named_children else None

        if not (cons_expr and alt_expr):
            return False
        if cons_expr.type != 'assignment_expression' or alt_expr.type != 'assignment_expression':
            return False

        # Must be plain assignment (=). Compound assignments (+=, -=, …) cannot be ternary-converted.
        def _has_simple_assign_op(node):
            for child in node.children:
                if child.type == '=':
                    return True
            return False

        if not _has_simple_assign_op(cons_expr) or not _has_simple_assign_op(alt_expr):
            return False

        lhs1 = cons_expr.child_by_field_name('left')
        lhs2 = alt_expr.child_by_field_name('left')
        if not (lhs1 and lhs2):
            return False

        if lhs1.text != lhs2.text:
            return False

        # A self-assigning branch (A = A) collapses to A = X ? A : C.
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        return get_text(lhs1) not in (
            get_text(cons_expr.child_by_field_name('right')),
            get_text(alt_expr.child_by_field_name('right')))

    def get_current_state(self, src: str) -> int:
        found_s0 = any(self._is_ternary_expandable(n)
                       for n in self._get_target_nodes(src, self._query_s0))

        found_s1 = False
        for node in self._get_target_nodes(src, self._query_s1):
            if self._is_ternary_convertible(node):
                found_s1 = True
                break

        if found_s0 and found_s1:
            return -1
        if found_s0:
            return 0
        elif found_s1:
            return 1
        else:
            return -1

    def _to_s0(self, node) -> str:
        """s1 (if-else assignment) -> s0 (ternary assignment)."""
        if not self._is_ternary_convertible(node):
            return None

        def get_text(n):
            return n.text.decode('utf8') if n else ""

        cond_node = node.child_by_field_name('condition')
        cons = node.child_by_field_name('consequence')
        alt = node.child_by_field_name('alternative')

        cond_text = get_text(cond_node).strip()
        if cond_text.startswith('(') and cond_text.endswith(')'):
            cond_text = cond_text[1:-1].strip()

        cons_assign = cons.named_children[0].named_children[0]
        alt_assign = alt.named_children[0].named_children[0]

        lhs = get_text(cons_assign.child_by_field_name('left'))
        rhs1 = get_text(cons_assign.child_by_field_name('right'))
        rhs2 = get_text(alt_assign.child_by_field_name('right'))

        return f"{lhs} = {cond_text} ? {rhs1} : {rhs2};"

    def _to_s1(self, node) -> str:
        """s0 (ternary assignment) -> s1 (if-else assignment). Node is an
        expression_statement."""
        if not self._is_ternary_expandable(node):
            return None

        def get_text(n):
            return n.text.decode('utf8') if n else ""

        assign = None
        for child in node.named_children:
            if child.type == 'assignment_expression':
                assign = child
                break
        if not assign:
            return None

        lhs = get_text(assign.child_by_field_name('left'))
        ternary = assign.child_by_field_name('right')

        if not ternary or ternary.type != 'ternary_expression':
            return None

        cond = get_text(ternary.child_by_field_name('condition'))
        then_val = get_text(ternary.child_by_field_name('consequence'))
        else_val = get_text(ternary.child_by_field_name('alternative'))

        return f"if ({cond}) {{\n    {lhs} = {then_val};\n}} else {{\n    {lhs} = {else_val};\n}}"


class JavaNegateIfElseStrategy(TreeSitterJavaStrategy):
    """
    Flip the ==/!= operator in the condition and swap the if-else blocks.
    s0: if (a == b) { A } else { B }
    s1: if (a != b) { B } else { A }
    Flipping the operator instead of wrapping in !(...) keeps the round trip textually stable.
    Operator domain is disjoint from CompareSwap (<, <=, >, >=).
    """
    OP_FLIP = {"==": "!=", "!=": "=="}

    def __init__(self):
        super().__init__(java_lang.language())

        # Match plain if-else only (alternative: block), not else-if chains.
        self._query_s0 = """
        (if_statement
            condition: (parenthesized_expression) @cond
            consequence: (block) @then_block
            alternative: (block) @else_block
        ) @target
        """
        self._query_s1 = self._query_s0

    @property
    def name(self) -> str:
        return "NegateIfElse"

    def _get_eq_operator(self, node) -> str:
        """Return "==" or "!=" if the if_statement's condition is a direct equality comparison, else None."""
        cond = node.child_by_field_name('condition')
        if not cond or not cond.named_children:
            return None
        inner = cond.named_children[0]
        if inner.type != 'binary_expression':
            return None
        for child in inner.children:
            if child.type in self.OP_FLIP:
                return child.type
        return None

    def get_current_state(self, src: str) -> int:
        nodes = self._get_target_nodes(src, self._query_s0)
        found_s0, found_s1 = False, False
        for node in nodes:
            op = self._get_eq_operator(node)
            if op == '==':
                found_s0 = True
            elif op == '!=':
                found_s1 = True
        if found_s0 and found_s1:
            return -1
        if found_s0:
            return 0
        elif found_s1:
            return 1
        else:
            return -1

    def _flip(self, node, from_op) -> str:
        """Flip == <-> != and swap the if-else blocks."""
        op = self._get_eq_operator(node)
        if op != from_op:
            return None

        def get_text(n):
            return n.text.decode('utf8') if n else ""

        cond = node.child_by_field_name('condition')
        inner = cond.named_children[0]  # binary_expression

        lhs = get_text(inner.child_by_field_name('left'))
        rhs = get_text(inner.child_by_field_name('right'))
        new_op = self.OP_FLIP[from_op]

        cons_text = get_text(node.child_by_field_name('consequence'))
        alt_text = get_text(node.child_by_field_name('alternative'))

        return f"if ({lhs} {new_op} {rhs}) {alt_text} else {cons_text}"

    def _to_s0(self, node) -> str:
        """s1 -> s0: if (a != b) {B} else {A} -> if (a == b) {A} else {B}"""
        return self._flip(node, '!=')

    def _to_s1(self, node) -> str:
        """s0 -> s1: if (a == b) {A} else {B} -> if (a != b) {B} else {A}"""
        return self._flip(node, '==')


class JavaPrefixPostfixStrategy(TreeSitterJavaStrategy):
    """
    s0: i++;  / i--;   (postfix, standalone statement)
    s1: ++i;  / --i;   (prefix, standalone statement)
    Matches only a standalone expression_statement, so for-update positions
    and embedded expressions like a = i++ are excluded.
    """
    def __init__(self):
        super().__init__(java_lang.language())
        self._query = """
        (expression_statement
            (update_expression) @target
        )
        """

    @property
    def name(self): return "PrefixPostfix"

    def _is_postfix(self, node):
        text = node.text.decode('utf8').strip()
        return text.endswith('++') or text.endswith('--')

    def _is_prefix(self, node):
        text = node.text.decode('utf8').strip()
        return text.startswith('++') or text.startswith('--')

    def get_current_state(self, src):
        nodes = self._get_target_nodes(src, self._query)
        found_s0, found_s1 = False, False
        for node in nodes:
            if self._is_postfix(node): found_s0 = True
            elif self._is_prefix(node): found_s1 = True
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node):
        """prefix -> postfix: ++i -> i++"""
        text = node.text.decode('utf8').strip()
        if text.startswith('++'):
            return text[2:] + '++'
        elif text.startswith('--'):
            return text[2:] + '--'
        return None

    def _to_s1(self, node):
        """postfix -> prefix: i++ -> ++i"""
        text = node.text.decode('utf8').strip()
        if text.endswith('++'):
            return '++' + text[:-2]
        elif text.endswith('--'):
            return '--' + text[:-2]
        return None

    def rewrite(self, src, i):
        nodes = self._get_target_nodes(src, self._query)
        nodes.sort(key=lambda n: n.start_byte)

        new_segments = []
        last_idx = 0
        src_bytes = bytes(src, 'utf8')

        for node in nodes:
            if i == 0 and self._is_prefix(node):
                replacement = self._to_s0(node)
            elif i == 1 and self._is_postfix(node):
                replacement = self._to_s1(node)
            else:
                continue
            if replacement:
                new_segments.append(src_bytes[last_idx:node.start_byte])
                new_segments.append(bytes(replacement, 'utf8'))
                last_idx = node.end_byte

        new_segments.append(src_bytes[last_idx:])
        return b"".join(new_segments).decode('utf8')


class JavaReturnVarStrategy(TreeSitterJavaStrategy):
    """
    s0: return expr;
    s1: T result = expr; return result;   (T = the method's declared return type)
    Targets only the last statement of a method body. Skips a bare identifier or literal,
    and an expr containing an anonymous or local class (see _has_nested_method).
    """
    CANDIDATE_NAMES = ['result', 'ret', 'res', '_result', '_ret', '_res']

    TRIVIAL_TYPES = {
        'identifier', 'decimal_integer_literal', 'hex_integer_literal',
        'octal_integer_literal', 'binary_integer_literal',
        'decimal_floating_point_literal', 'hex_floating_point_literal',
        'string_literal', 'character_literal', 'null_literal', 'true', 'false',
    }

    def __init__(self):
        super().__init__(java_lang.language())

    @property
    def name(self): return "ReturnVar"

    def _pick_var_name(self, body_node):
        """Pick a natural variable name not already used in the body."""
        used = set()
        def _collect(node):
            if node.type == 'identifier':
                used.add(node.text.decode('utf8'))
            for child in node.children:
                _collect(child)
        _collect(body_node)
        for name in self.CANDIDATE_NAMES:
            if name not in used:
                return name
        return '_result_var'

    @staticmethod
    def _has_nested_method(node) -> bool:
        """True when the subtree declares a method, i.e. an anonymous or local class.
        get_code_chunks() makes such a method a chunk of its own, overlapping this one, so rewriting
        around it redraws that chunk's canonical form. Lambdas declare no method.
        """
        if node is None:
            return False
        if node.type == 'method_declaration':
            return True
        return any(JavaReturnVarStrategy._has_nested_method(c) for c in node.children)

    def _find_method_bodies(self, src):
        """Returns (body_node, return_type_text) tuples, excluding constructors (no return type)."""
        tree = self.parser.parse(bytes(src, 'utf8'))
        bodies = []

        def visit(node):
            if node.type == 'method_declaration':
                body = node.child_by_field_name('body')
                if body and body.type == 'block':
                    ret_type = node.child_by_field_name('type')
                    if ret_type:
                        bodies.append((body, ret_type.text.decode('utf8')))
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return bodies

    def _check_s0(self, body_node):
        """Last statement must be return <non-trivial expr>."""
        stmts = [c for c in body_node.named_children]
        if not stmts:
            return None
        last = stmts[-1]
        if last.type != 'return_statement':
            return None
        if not last.named_children:
            return None
        expr = last.named_children[0]
        if expr.type in self.TRIVIAL_TYPES:
            return None
        if self._has_nested_method(expr):
            return None
        return last

    def _check_s1(self, body_node):
        """Last two statements must match T x = expr; return x;"""
        stmts = [c for c in body_node.named_children]
        if len(stmts) < 2:
            return None
        decl, ret = stmts[-2], stmts[-1]

        if decl.type != 'local_variable_declaration' or ret.type != 'return_statement':
            return None

        # An annotation has nowhere to go on a return, so collapsing the declaration
        # would drop a @SuppressWarnings and its effect.
        if any(k.type in ('annotation', 'marker_annotation')
               for c in decl.children if c.type == 'modifiers'
               for k in c.children):
            return None

        declarator = None
        for child in decl.named_children:
            if child.type == 'variable_declarator':
                declarator = child
                break
        if not declarator:
            return None

        var_name_node = declarator.child_by_field_name('name')
        var_value_node = declarator.child_by_field_name('value')
        if not var_name_node or not var_value_node:
            return None

        var_name = var_name_node.text.decode('utf8')

        # Return statement must reference the same variable.
        if not ret.named_children:
            return None
        ret_expr = ret.named_children[0]
        if ret_expr.type != 'identifier' or ret_expr.text.decode('utf8') != var_name:
            return None

        # Skip when the assigned expression itself is trivial.
        if var_value_node.type in self.TRIVIAL_TYPES:
            return None
        if self._has_nested_method(var_value_node):
            return None

        return (decl, ret, var_value_node)

    def get_current_state(self, src):
        bodies = self._find_method_bodies(src)
        found_s0, found_s1 = False, False
        for body, _ in bodies:
            if self._check_s0(body): found_s0 = True
            if self._check_s1(body): found_s1 = True
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def rewrite(self, src, i):
        bodies = self._find_method_bodies(src)
        src_bytes = bytes(src, 'utf8')
        replacements = []

        for body, ret_type_text in bodies:
            if i == 0:
                result = self._check_s1(body)
                if result:
                    decl, ret, value_node = result
                    expr_text = value_node.text.decode('utf8')
                    replacements.append((decl.start_byte, ret.end_byte, f"return {expr_text};"))
            elif i == 1:
                ret_node = self._check_s0(body)
                if ret_node:
                    expr = ret_node.named_children[0]
                    expr_text = expr.text.decode('utf8')
                    # Detect leading indent of the return line.
                    line_start = src_bytes.rfind(b'\n', 0, ret_node.start_byte)
                    indent = ''
                    if line_start != -1:
                        for ch in src_bytes[line_start + 1:ret_node.start_byte]:
                            if ch in (32, 9):
                                indent += chr(ch)
                            else:
                                break
                    var_name = self._pick_var_name(body)
                    replacement = f"{ret_type_text} {var_name} = {expr_text};\n{indent}return {var_name};"
                    replacements.append((ret_node.start_byte, ret_node.end_byte, replacement))

        # Drop nested replacements (inner contained in outer), e.g. when a
        # method inside an anonymous class has a return inside an outer return.
        replacements.sort(key=lambda r: r[0])
        filtered = []
        for start, end, repl in replacements:
            if any(fs <= start and end <= fe for fs, fe, _ in filtered):
                continue
            filtered.append((start, end, repl))

        filtered.sort(key=lambda r: r[0], reverse=True)
        for start, end, replacement in filtered:
            src_bytes = src_bytes[:start] + bytes(replacement, 'utf8') + src_bytes[end:]

        return src_bytes.decode('utf8')


class JavaYodaEqualsStrategy(TreeSitterJavaStrategy):
    """
    s0: x == literal  /  x != literal  (variable first)
    s1: literal == x  /  literal != x  (literal first, Yoda)
    Orthogonal to NegateIfElse, which flips the operator but never reorders operands,
    and operator-disjoint from CompareSwap (<, <=, >, >=).
    """
    LITERAL_TYPES = {
        'decimal_integer_literal', 'hex_integer_literal', 'octal_integer_literal',
        'binary_integer_literal', 'decimal_floating_point_literal', 'hex_floating_point_literal',
        'string_literal', 'character_literal', 'null_literal', 'true', 'false',
    }

    def __init__(self):
        super().__init__(java_lang.language())
        self._query = """
        (binary_expression
            left: (_) @lhs
            operator: ["==" "!="]
            right: (_) @rhs
        ) @target
        """

    @property
    def name(self): return "YodaEquals"

    def _is_literal(self, node):
        return node.type in self.LITERAL_TYPES

    def _is_s0_node(self, node):
        lhs = node.child_by_field_name('left')
        rhs = node.child_by_field_name('right')
        if not (lhs and rhs): return False
        return not self._is_literal(lhs) and self._is_literal(rhs)

    def _is_s1_node(self, node):
        lhs = node.child_by_field_name('left')
        rhs = node.child_by_field_name('right')
        if not (lhs and rhs): return False
        return self._is_literal(lhs) and not self._is_literal(rhs)

    def get_current_state(self, src):
        nodes = self._get_target_nodes(src, self._query)
        found_s0, found_s1 = False, False
        for node in nodes:
            if self._is_s0_node(node): found_s0 = True
            elif self._is_s1_node(node): found_s1 = True
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _swap(self, node):
        lhs = node.child_by_field_name('left')
        rhs = node.child_by_field_name('right')
        if not (lhs and rhs): return None
        op = None
        for child in node.children:
            if child.type in ('==', '!='):
                op = child.type
                break
        if not op: return None
        return f"{rhs.text.decode('utf8')} {op} {lhs.text.decode('utf8')}"

    def _to_s0(self, node):
        return self._swap(node) if self._is_s1_node(node) else None

    def _to_s1(self, node):
        return self._swap(node) if self._is_s0_node(node) else None

    def rewrite(self, src, i):
        nodes = self._get_target_nodes(src, self._query)
        nodes.sort(key=lambda n: n.start_byte)

        unique_nodes = []
        last_end = -1
        for node in nodes:
            if node.start_byte < last_end: continue
            unique_nodes.append(node)
            last_end = node.end_byte

        new_segments = []
        last_idx = 0
        src_bytes = bytes(src, 'utf8')

        for node in unique_nodes:
            replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
            if replacement:
                new_segments.append(src_bytes[last_idx:node.start_byte])
                new_segments.append(bytes(replacement, 'utf8'))
                last_idx = node.end_byte

        new_segments.append(src_bytes[last_idx:])
        return b"".join(new_segments).decode('utf8')


class JavaWhileForSwapStrategy(TreeSitterJavaStrategy):
    """
    s0: while (cond) { body }
    s1: for (; cond;) { body }
    Only a for with neither init nor update is handled. An update would require rewriting
    every continue in the body, and an init would have to be hoisted out of the loop's scope.
    Only the first convertible loop in source order is toggled, which avoids nested-match
    dedup problems and canonical drift.
    """
    def __init__(self):
        super().__init__(java_lang.language())

    @property
    def name(self): return "WhileForSwap"

    def _is_simple_for(self, node):
        """A for_statement with neither init nor update, equivalent to while."""
        return (node.type == 'for_statement'
                and node.child_by_field_name('init') is None
                and node.child_by_field_name('update') is None)

    def _is_matchable_while(self, node):
        """A while_statement with a block body."""
        if node.type != 'while_statement':
            return False
        cond = node.child_by_field_name('condition')
        body = node.child_by_field_name('body')
        return (cond is not None and cond.type == 'parenthesized_expression'
                and body is not None and body.type == 'block')

    def _is_matchable_for(self, node):
        """A simple for with a block body."""
        if not self._is_simple_for(node):
            return False
        cond = node.child_by_field_name('condition')
        body = node.child_by_field_name('body')
        return cond is not None and body is not None and body.type == 'block'

    def _find_first(self, src):
        """Return the first convertible loop in source order as (node, state),
        state 0 for while and 1 for simple for, or (None, -1) if there is none.
        """
        try:
            tree = self.parser.parse(bytes(src, "utf8"))
        except SyntaxError:
            return None, -1

        best = None  # (start_byte, node, state)

        def walk(node):
            nonlocal best
            if node.type == 'while_statement' and self._is_matchable_while(node):
                if best is None or node.start_byte < best[0]:
                    best = (node.start_byte, node, 0)
            elif node.type == 'for_statement' and self._is_matchable_for(node):
                if best is None or node.start_byte < best[0]:
                    best = (node.start_byte, node, 1)
            for c in node.children:
                walk(c)

        walk(tree.root_node)
        if best is None:
            return None, -1
        return best[1], best[2]

    def get_current_state(self, src):
        _, state = self._find_first(src)
        return state

    def _extract_while_cond_text(self, cond_node):
        """Inner text of a while condition, outer parens and whitespace stripped
        (( keys.hasNext() ) -> keys.hasNext()).
        """
        text = cond_node.text.decode('utf8').strip()
        if text.startswith('(') and text.endswith(')'):
            text = text[1:-1].strip()
        return text

    def _canonical_while(self, node):
        """Re-emit a while_statement with canonical spacing: while (<cond>) <body>."""
        cond = node.child_by_field_name('condition')  # parenthesized_expression
        body = node.child_by_field_name('body')       # block
        if cond is None or body is None:
            return None
        return f"while ({self._extract_while_cond_text(cond)}) {body.text.decode('utf8')}"

    def _canonical_for(self, node):
        """Re-emit a simple for_statement with canonical spacing: for (; <cond>;) <body>.
        The for's condition field has no outer parens.
        """
        cond = node.child_by_field_name('condition')
        body = node.child_by_field_name('body')
        if cond is None or body is None:
            return None
        return f"for (; {cond.text.decode('utf8').strip()};) {body.text.decode('utf8')}"

    def _while_to_for(self, node):
        """while (cond) { body } -> for (; cond;) { body }."""
        cond = node.child_by_field_name('condition')
        body = node.child_by_field_name('body')
        if cond is None or body is None:
            return None
        return f"for (; {self._extract_while_cond_text(cond)};) {body.text.decode('utf8')}"

    def _for_to_while(self, node):
        """for (; cond;) { body } -> while (cond) { body }."""
        cond = node.child_by_field_name('condition')
        body = node.child_by_field_name('body')
        if cond is None or body is None:
            return None
        return f"while ({cond.text.decode('utf8').strip()}) {body.text.decode('utf8')}"

    def rewrite(self, src, i):
        """Toggle / canonicalize the first convertible loop in source order.
        All other loops in the chunk are left untouched.
        """
        node, state = self._find_first(src)
        if node is None:
            return src

        if i == 0:
            # Normalize / to s0
            if state == 1:        # existing simple for -> while
                replacement = self._for_to_while(node)
            else:                 # state == 0: canonicalize while spacing
                replacement = self._canonical_while(node)
        else:
            # To s1
            if state == 0:        # existing while -> simple for
                replacement = self._while_to_for(node)
            else:                 # state == 1: canonicalize for spacing
                replacement = self._canonical_for(node)

        if replacement is None:
            return src

        src_b = bytes(src, 'utf8')
        new = src_b[:node.start_byte] + bytes(replacement, 'utf8') + src_b[node.end_byte:]
        return new.decode('utf8')


_OB_SINGLE_STMT_TYPES = {
    'expression_statement', 'return_statement', 'throw_statement',
    'break_statement', 'continue_statement',
}

def _ob_get_indent(src_bytes, node):
    line_start = src_bytes.rfind(b'\n', 0, node.start_byte)
    indent = ''
    start = line_start + 1 if line_start != -1 else 0
    for ch in src_bytes[start:node.start_byte]:
        if ch in (32, 9):
            indent += chr(ch)
        else:
            break
    return indent

def _ob_detect_indent_unit(src_bytes, if_node, inner_node):
    if_indent = _ob_get_indent(src_bytes, if_node)
    between = src_bytes[if_node.child_by_field_name('condition').end_byte:inner_node.start_byte]
    if b'\n' in between:
        inner_indent = _ob_get_indent(src_bytes, inner_node)
        unit = inner_indent[len(if_indent):]
        if unit:
            return unit
    if '\t' in if_indent:
        return '\t'
    return '    '

def _ob_get_prefix(src_bytes, node, cons):
    return src_bytes[node.start_byte:cons.start_byte].decode('utf8').rstrip()

def _ob_has_line_comment(src_bytes, node):
    cons = node.child_by_field_name('consequence')
    cond = node.child_by_field_name('condition')
    if not (cons and cond): return False
    if b'//' in src_bytes[cond.end_byte:cons.start_byte]: return True
    if b'//' in cons.text: return True
    return False

def _ob_is_multiline(node):
    return node.start_point[0] != node.end_point[0]


class JavaOptionalBracesStrategy(TreeSitterJavaStrategy):
    """
    s0: if (cond) stmt;        (no braces)
    s1: if (cond) { stmt; }    (with braces, single statement)
    Only if-without-else. Excludes multiline statements and line comments.
    Preserves original if(...) spacing and indentation style for round-trip correctness.
    """
    def __init__(self):
        super().__init__(java_lang.language())
        self._query = """
        (if_statement
            condition: (parenthesized_expression) @cond
            consequence: (_) @body
        ) @target
        """

    @property
    def name(self): return "OptionalBraces"

    def _is_s0_node(self, node, src_bytes=None):
        if node.child_by_field_name('alternative'): return False
        cons = node.child_by_field_name('consequence')
        if not cons or cons.type not in _OB_SINGLE_STMT_TYPES: return False
        if _ob_is_multiline(cons): return False
        if src_bytes and _ob_has_line_comment(src_bytes, node): return False
        return True

    def _is_s1_node(self, node, src_bytes=None):
        if node.child_by_field_name('alternative'): return False
        cons = node.child_by_field_name('consequence')
        if not cons or cons.type != 'block': return False
        stmts = [c for c in cons.named_children]
        if len(stmts) != 1 or stmts[0].type not in _OB_SINGLE_STMT_TYPES: return False
        if _ob_is_multiline(stmts[0]): return False
        if src_bytes and _ob_has_line_comment(src_bytes, node): return False
        return True

    def get_current_state(self, src):
        nodes = self._get_target_nodes(src, self._query)
        sb = bytes(src, 'utf8')
        found_s0 = any(self._is_s0_node(n, sb) for n in nodes)
        found_s1 = any(self._is_s1_node(n, sb) for n in nodes)
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node, src_bytes):
        if not self._is_s1_node(node, src_bytes): return None
        cons = node.child_by_field_name('consequence')
        stmt = cons.named_children[0]
        indent = _ob_get_indent(src_bytes, node)
        indent_unit = _ob_detect_indent_unit(src_bytes, node, stmt)
        prefix = _ob_get_prefix(src_bytes, node, cons)
        return f"{prefix}\n{indent}{indent_unit}{stmt.text.decode('utf8').strip()}"

    def _to_s1(self, node, src_bytes):
        if not self._is_s0_node(node, src_bytes): return None
        cons = node.child_by_field_name('consequence')
        indent = _ob_get_indent(src_bytes, node)
        indent_unit = _ob_detect_indent_unit(src_bytes, node, cons)
        prefix = _ob_get_prefix(src_bytes, node, cons)
        return f"{prefix} {{\n{indent}{indent_unit}{cons.text.decode('utf8').strip()}\n{indent}}}"

    def _normalize_s0(self, node, src_bytes):
        if not self._is_s0_node(node, src_bytes): return None
        cons = node.child_by_field_name('consequence')
        indent = _ob_get_indent(src_bytes, node)
        indent_unit = _ob_detect_indent_unit(src_bytes, node, cons)
        prefix = _ob_get_prefix(src_bytes, node, cons)
        return f"{prefix}\n{indent}{indent_unit}{cons.text.decode('utf8').strip()}"

    def rewrite(self, src, i):
        nodes = self._get_target_nodes(src, self._query)
        if i == 0:
            targets = [(n, 's1') for n in nodes if self._is_s1_node(n)] + \
                      [(n, 's0') for n in nodes if self._is_s0_node(n)]
        else:
            targets = [(n, 's0') for n in nodes if self._is_s0_node(n)]

        targets.sort(key=lambda x: x[0].start_byte, reverse=True)
        src_bytes = bytearray(src, 'utf8')
        seen = set()
        for node, state in targets:
            rng = (node.start_byte, node.end_byte)
            if rng in seen: continue
            seen.add(rng)
            if i == 0:
                r = self._to_s0(node, bytes(src_bytes)) if state == 's1' else self._normalize_s0(node, bytes(src_bytes))
            else:
                r = self._to_s1(node, bytes(src_bytes))
            if r:
                src_bytes[node.start_byte:node.end_byte] = bytes(r, 'utf8')
        return src_bytes.decode('utf8')


class JavaDiamondInferenceStrategy(TreeSitterJavaStrategy):
    """
    s0: List<String> x = new ArrayList<>();           (diamond)
    s1: List<String> x = new ArrayList<String>();      (explicit)
    Only local_variable_declaration where:
    - LHS has generic type with type_arguments
    - RHS is object_creation_expression with diamond or explicit type_arguments
    - No wildcard types (? extends/super)
    """
    def __init__(self):
        super().__init__(java_lang.language())
        self._query = """
        (local_variable_declaration
            type: (_) @type
            declarator: (variable_declarator
                name: (identifier) @name
                value: (object_creation_expression
                    type: (_) @constructor_type
                    arguments: (argument_list) @args
                ) @rhs
            )
        ) @target
        """

    @property
    def name(self): return "DiamondInference"

    def _get_type_args_node(self, type_node):
        for child in type_node.children:
            if child.type == 'type_arguments':
                return child
        return None

    def _has_wildcard(self, type_args_node):
        if not type_args_node: return False
        return b'?' in type_args_node.text

    def _get_lhs_type_args_text(self, node):
        lhs_type = node.child_by_field_name('type')
        if not lhs_type: return None
        ta = self._get_type_args_node(lhs_type)
        if not ta: return None
        if self._has_wildcard(ta): return None
        return ta.text.decode('utf8')

    def _get_rhs_creation(self, node):
        for child in node.children:
            if child.type == 'variable_declarator':
                value = child.child_by_field_name('value')
                if value and value.type == 'object_creation_expression':
                    return value
        return None

    def _get_constructor_type_node(self, creation_node):
        return creation_node.child_by_field_name('type')

    def _is_diamond(self, creation_node):
        ct = self._get_constructor_type_node(creation_node)
        if not ct: return False
        ta = self._get_type_args_node(ct)
        if not ta: return False
        return len(ta.named_children) == 0

    def _is_explicit(self, creation_node):
        ct = self._get_constructor_type_node(creation_node)
        if not ct: return False
        ta = self._get_type_args_node(ct)
        if not ta: return False
        if len(ta.named_children) == 0: return False
        if self._has_wildcard(ta): return False
        return True

    def _is_s0_node(self, node):
        lhs_ta = self._get_lhs_type_args_text(node)
        if not lhs_ta: return False
        creation = self._get_rhs_creation(node)
        return creation and self._is_diamond(creation)

    def _is_s1_node(self, node):
        lhs_ta = self._get_lhs_type_args_text(node)
        if not lhs_ta: return False
        creation = self._get_rhs_creation(node)
        if not creation or not self._is_explicit(creation): return False
        # LHS and RHS type arguments must match (otherwise diamond is unsafe).
        ct = self._get_constructor_type_node(creation)
        rhs_ta = self._get_type_args_node(ct)
        if not rhs_ta: return False
        if rhs_ta.text.decode('utf8') != lhs_ta: return False
        return True

    def get_current_state(self, src):
        nodes = self._get_target_nodes(src, self._query)
        found_s0 = any(self._is_s0_node(n) for n in nodes)
        found_s1 = any(self._is_s1_node(n) for n in nodes)
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node):
        """s1 -> s0: new ArrayList<String>() -> new ArrayList<>()"""
        if not self._is_s1_node(node): return None
        creation = self._get_rhs_creation(node)
        ct = self._get_constructor_type_node(creation)
        ta = self._get_type_args_node(ct)
        src_bytes = node.text
        ta_start = ta.start_byte - node.start_byte
        ta_end = ta.end_byte - node.start_byte
        return (src_bytes[:ta_start] + b'<>' + src_bytes[ta_end:]).decode('utf8')

    def _to_s1(self, node):
        """s0 -> s1: new ArrayList<>() -> new ArrayList<String>()"""
        if not self._is_s0_node(node): return None
        lhs_ta = self._get_lhs_type_args_text(node)
        creation = self._get_rhs_creation(node)
        ct = self._get_constructor_type_node(creation)
        ta = self._get_type_args_node(ct)
        src_bytes = node.text
        ta_start = ta.start_byte - node.start_byte
        ta_end = ta.end_byte - node.start_byte
        return (src_bytes[:ta_start] + bytes(lhs_ta, 'utf8') + src_bytes[ta_end:]).decode('utf8')

    def rewrite(self, src, i):
        nodes = self._get_target_nodes(src, self._query)
        if i == 0:
            targets = [n for n in nodes if self._is_s1_node(n)]
        else:
            targets = [n for n in nodes if self._is_s0_node(n)]
        targets.sort(key=lambda n: n.start_byte, reverse=True)
        src_bytes = bytearray(src, 'utf8')
        seen = set()
        for node in targets:
            rng = (node.start_byte, node.end_byte)
            if rng in seen: continue
            seen.add(rng)
            r = self._to_s0(node) if i == 0 else self._to_s1(node)
            if r:
                src_bytes[node.start_byte:node.end_byte] = bytes(r, 'utf8')
        return src_bytes.decode('utf8')


class JavaArrayTypeStyleStrategy(TreeSitterJavaStrategy):
    """
    s0: int[] arr = ...;   (Java style, brackets on the type)
    s1: int arr[] = ...;   (C style, brackets on the declarator)
    Guards: local_variable_declaration only, 1D only, all declarators on the same side
    (int a[], b; declares different types), and var skipped (var a[] does not compile).
    Checkstyle ArrayTypeStyle discourages the s1 form, which fails builds that enforce it.
    """
    def __init__(self):
        super().__init__(java_lang.language())
        self._query = "(local_variable_declaration) @target"

    @property
    def name(self): return "ArrayTypeStyle"

    def _declarators(self, lvd_node):
        return [c for c in lvd_node.children if c.type == 'variable_declarator']

    def _decl_dims_count(self, decl_node):
        """Number of [ in the declarator's own dimensions field (0 if none)."""
        dims = decl_node.child_by_field_name('dimensions')
        if dims is None:
            return 0
        return dims.text.decode('utf8').count('[')

    def _type_dims_count(self, type_node):
        """Number of [ in an array_type's dimensions field (0 if not array)."""
        if type_node is None or type_node.type != 'array_type':
            return 0
        dims = type_node.child_by_field_name('dimensions')
        if dims is None:
            return 0
        return dims.text.decode('utf8').count('[')

    def _is_var_keyword(self, type_node):
        return (type_node is not None
                and type_node.type == 'type_identifier'
                and type_node.text == b'var')

    def _is_valid_s0(self, lvd_node):
        """Type has 1D [] and no declarator has its own dimensions."""
        type_node = lvd_node.child_by_field_name('type')
        if type_node is None:
            return False
        if self._type_dims_count(type_node) != 1:
            return False
        if self._is_var_keyword(type_node):
            return False
        decls = self._declarators(lvd_node)
        if not decls:
            return False
        for d in decls:
            if self._decl_dims_count(d) != 0:
                return False
        return True

    def _is_valid_s1(self, lvd_node):
        """Type is not array_type and every declarator has 1D own dimensions."""
        type_node = lvd_node.child_by_field_name('type')
        if type_node is None:
            return False
        if type_node.type == 'array_type':
            return False  # mixed, some brackets on the type already
        if self._is_var_keyword(type_node):
            return False
        decls = self._declarators(lvd_node)
        if not decls:
            return False
        for d in decls:
            if self._decl_dims_count(d) != 1:
                return False
        return True

    def get_current_state(self, src):
        nodes = self._get_target_nodes(src, self._query)
        has_s0 = any(self._is_valid_s0(n) for n in nodes)
        has_s1 = any(self._is_valid_s1(n) for n in nodes)
        if has_s0 and has_s1:
            return -1
        if has_s0:
            return 0
        if has_s1:
            return 1
        return -1

    def _element_text_from_array_type(self, type_node):
        """Element-type text for an array_type (int[] -> int, List<String>[] -> List<String>)."""
        dims_node = type_node.child_by_field_name('dimensions')
        type_text = type_node.text.decode('utf8')
        offset = dims_node.start_byte - type_node.start_byte
        return type_text[:offset].rstrip()

    def _first_name_start(self, lvd_node):
        decls = self._declarators(lvd_node)
        if not decls:
            return None
        name = decls[0].child_by_field_name('name')
        return name.start_byte if name is not None else None

    def _edits_s0_to_s1(self, lvd_node):
        """Move [] from the type to every declarator name, normalizing spacing.
        The replacement spans type.start to first_name.start, so non-canonical whitespace (int [] arr) is absorbed.
        """
        type_node = lvd_node.child_by_field_name('type')
        element_text = self._element_text_from_array_type(type_node)
        first_name_start = self._first_name_start(lvd_node)
        if first_name_start is None:
            return []
        edits = [(type_node.start_byte, first_name_start, f'{element_text} ')]
        for d in self._declarators(lvd_node):
            name = d.child_by_field_name('name')
            if name is None:
                continue
            edits.append((name.end_byte, name.end_byte, '[]'))
        return edits

    def _edits_s1_to_s0(self, lvd_node):
        """Move [] from every declarator to the type, normalizing spacing.
        Each declarator's dimensions go with any preceding whitespace, so arr [] = x leaves no stray space.
        """
        type_node = lvd_node.child_by_field_name('type')
        type_text = type_node.text.decode('utf8')
        first_name_start = self._first_name_start(lvd_node)
        if first_name_start is None:
            return []
        edits = [(type_node.start_byte, first_name_start, f'{type_text}[] ')]
        for d in self._declarators(lvd_node):
            dims = d.child_by_field_name('dimensions')
            name = d.child_by_field_name('name')
            if dims is None or name is None:
                continue
            edits.append((name.end_byte, dims.end_byte, ''))
        return edits

    def _edits_canonicalize_s0(self, lvd_node):
        """Canonicalize an existing s0 form to one space between type and name,
        so String [] arr and String[] arr do not differ after a round trip.
        """
        type_node = lvd_node.child_by_field_name('type')
        element_text = self._element_text_from_array_type(type_node)
        first_name_start = self._first_name_start(lvd_node)
        if first_name_start is None:
            return []
        return [(type_node.start_byte, first_name_start, f'{element_text}[] ')]

    def rewrite(self, src, i):
        nodes = self._get_target_nodes(src, self._query)
        all_edits = []
        if i == 0:
            for n in nodes:
                if self._is_valid_s1(n):
                    all_edits.extend(self._edits_s1_to_s0(n))
                elif self._is_valid_s0(n):
                    all_edits.extend(self._edits_canonicalize_s0(n))
        else:
            for n in nodes:
                if self._is_valid_s0(n):
                    all_edits.extend(self._edits_s0_to_s1(n))

        all_edits.sort(key=lambda e: (e[0], e[1]))
        non_overlapping = []
        last_end = -1
        for s, e, t in all_edits:
            if s < last_end:
                continue
            non_overlapping.append((s, e, t))
            last_end = max(last_end, e)

        out = []
        last = 0
        src_b = bytes(src, 'utf8')
        for s, e, t in non_overlapping:
            out.append(src_b[last:s])
            out.append(bytes(t, 'utf8'))
            last = e
        out.append(src_b[last:])
        return b"".join(out).decode('utf8')


def _test_cornercase():
    """Tests a strategy must refuse. Every row is a defect we actually hit."""
    S = {st.name: st for st in get_java_strategies()}

    def f(body):
        return "class A {\n  void f() {\n" + body + "\n  }\n}\n"

    INLINE = ('class A {\n  @InlineMe(replacement = "g(x)")\n'
              '  int f(int x) {\n    return g(x);\n  }\n}\n')
    ANNOTATED = ('class A {\n  List f() {\n    @SuppressWarnings("unchecked")\n'
                 '    List r = (List) g();\n    return r;\n  }\n}\n')

    CASES = [
        # `A = X ? A : C` expands to `A = A;` -- SelfAssignment (gson).
        ("Ternary", False, f('    tz = tz.length() >= 5 ? tz : tz + "00";')),
        # The other branch, and a LHS that is not a bare identifier.
        ("Ternary", False, f("    this.x = c ? y : this.x;")),
        # Same shape from the s1 side, which normalize() would otherwise collapse.
        ("Ternary", False, f("    if (c) {\n      x = x;\n    } else {\n      x = y;\n    }")),
        ("Ternary", True, f("    bonus = total > 100 ? 50 : 0;")),

        # @InlineMe promises a one-statement body, which ReturnVar, BooleanReturn
        # and Ternary each reshape. Gated once for all strategies.
        ("ReturnVar", False, INLINE),
        ("Ternary", False, INLINE),
        ("BooleanReturn", False, INLINE),

        # A return cannot carry an annotation (JLS 9.7.4), so collapsing the
        # declaration would delete the @SuppressWarnings and surface 14 warnings.
        ("ReturnVar", False, ANNOTATED),
        ("ReturnVar", True, "class A {\n  int f() {\n    int r = g();\n    return r;\n  }\n}\n"),

        # -= expands only over a simple RHS, or precedence changes.
        ("AugAssign", False, f("    x -= a + b;")),
        ("AugAssign", True, f("    x -= n;")),
        # Mixed bracket sides cannot be re-emitted on one side.
        ("ArrayTypeStyle", False, f("    int a[], b;")),
        ("ArrayTypeStyle", True, f("    int[] a;")),
    ]

    for name, is_carrier, code in CASES:
        st = S[name]
        head = " ".join(x.strip() for x in code.splitlines()[1:3])
        assert st.is_target(code) == is_carrier, f"{name}: {head}"
        if is_carrier:
            for v in (0, 1):
                assert st.get_current_state(st.rewrite(code, v)) == v, f"{name} v={v}: {head}"
        else:
            assert st.normalize(code) == code, f"{name} normalize touched it: {head}"

    print("[OK] _test_cornercase")


def _test_java_strategies():
    strategies = get_java_strategies()

    sample_java_code = """
public class TestCode {
    public static int processData(int[] items, int maxLimit) {
        // [1. AugAssign] mixed assignment forms
        int totalScore = 0;
        totalScore += 50;

        int errorCount = 0;
        errorCount = errorCount + 1;

        // [2. CompareSwap] mixed comparison directions
        if (totalScore <= maxLimit) {
            System.out.println("pass");
        }
        if (errorCount > 10) {
            System.out.println("fail");
        }

        // [3. BooleanReturn] mixed boolean return forms
        boolean isValid = checkValue(totalScore);

        // [4. Ternary] mixed ternary / if-else assignment
        int bonus;
        bonus = totalScore > 100 ? 50 : 0;
        int penalty;
        if (errorCount == 0) {
            penalty = 100;
        } else {
            penalty = 0;
        }

        // [5. NegateIfElse] flip ==/!=
        if (totalScore == maxLimit) {
            System.out.println("exact");
        } else {
            System.out.println("not exact");
        }

        return totalScore;
    }

    public static boolean checkValue(int val) {
        if (val > 100) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean isPositive(int val) {
        return val > 0;
    }
}
"""

    print("="*60)
    print("[ Original Java Code ]")
    print("="*60)
    print(sample_java_code.strip())

    # Normalize/compress to s0.
    code_s0 = sample_java_code
    for strategy in strategies:
        code_s0 = strategy.rewrite(code_s0, i=0)

    print("\n" + "="*60)
    print("[ Pipeline Result: i=0 (Target: s0 - Compress) ]")
    print("="*60)
    print(code_s0.strip())

    # Expand to s1.
    code_s1 = sample_java_code
    for strategy in strategies:
        code_s1 = strategy.rewrite(code_s1, i=1)

    print("\n" + "="*60)
    print("[ Pipeline Result: i=1 (Target: s1 - Expand) ]")
    print("="*60)
    print(code_s1.strip())


if __name__ == "__main__":
    _test_cornercase()
    _test_java_strategies()
