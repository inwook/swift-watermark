import tree_sitter_javascript as js_lang
from tree_sitter import Language, Parser, Query, QueryCursor

from rewriters_base import RewriteStrategy

def get_js_strategies():
    return [
        JSAugAssignStrategy(),
        JSCompareSwapStrategy(),
        JSArrayInitStrategy(),
        JSObjectInitStrategy(),
        JSBooleanReturnStrategy(),
        JSArrowFunctionReturnStrategy(),
        JSPrefixPostfixStrategy(),
        JSNegateIfElseStrategy(),
        JSTernaryStrategy(),
        JSYodaEqualsStrategy(),
        JSReturnVarStrategy(),
        JSWhileForSwapStrategy(),
        JSBooleanCastStrategy(),
        JSPropertyShorthandStrategy(),
        JSOptionalBracesStrategy(),
    ]

# Tree-sitter base for the JS strategies (s0/s1 stay internal to each strategy).
class TreeSitterJSStrategy(RewriteStrategy):
    # Query() compilation dominates parse time, so compiled queries are cached per class.
    _query_cache = {}  # { query_str: Query }

    def __init__(self, language_capsule):
        self.LANG = Language(language_capsule)  # type: ignore
        self.parser = Parser(self.LANG)

    def _get_query(self, query_str: str) -> Query:
        """Return a cached compiled Query object, creating one if not yet cached."""
        if query_str not in TreeSitterJSStrategy._query_cache:
            TreeSitterJSStrategy._query_cache[query_str] = Query(self.LANG, query_str)
        return TreeSitterJSStrategy._query_cache[query_str]

    def _get_target_nodes(self, src: str, query_str: str) -> list:
        """Collect all @target nodes regardless of which tree-sitter API version
        (captures or matches, list or dict shape) is in effect."""
        try:
            tree = self.parser.parse(bytes(src, "utf8"))
        except SyntaxError:
            return []

        q = self._get_query(query_str)
        target_nodes = []
        cursor = QueryCursor(q)

        # API form A: captures()
        if hasattr(cursor, 'captures'):
            caps = cursor.captures(tree.root_node)
            if isinstance(caps, list):  # [(node, name), ...]
                for node, name in caps:
                    if name == "target":
                        target_nodes.append(node)
            elif isinstance(caps, dict) and "target" in caps:  # {"target": [nodes]}
                nodes = caps["target"]
                if isinstance(nodes, list): target_nodes.extend(nodes)
                else: target_nodes.append(nodes)

        # API form B: matches()
        elif hasattr(cursor, 'matches'):
            for match in cursor.matches(tree.root_node):
                # match may be (pattern_index, captures_dict) or just captures.
                caps = match[1] if isinstance(match, tuple) else match
                if isinstance(caps, dict) and "target" in caps:
                    nodes = caps["target"]
                    if isinstance(nodes, list): target_nodes.extend(nodes)
                    else: target_nodes.append(nodes)

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
        return self.get_current_state(src) != -1

    def rewrite(self, src: str, i: int) -> str:
        if not (hasattr(self, '_to_s0') and hasattr(self, '_to_s1')):
            raise NotImplementedError("Must implement internal _to_s0 and _to_s1 methods")

        query_str = self._query_s1 if i == 0 else self._query_s0
        target_nodes = self._get_target_nodes(src, query_str)

        # Keep only outermost matches: an outer replacement embeds the inner text verbatim,
        # so rewriting both would corrupt byte offsets.
        target_nodes.sort(key=lambda n: n.start_byte)
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


class JSAugAssignStrategy(TreeSitterJSStrategy):
    def __init__(self):
        super().__init__(js_lang.language())

        self._query_s0 = """
        (augmented_assignment_expression
            left: (identifier) @lhs 
            "+=" 
            right: (_) @rhs
        ) @target
        """
        
        self._query_s1 = """
        (assignment_expression 
            left: (identifier) @lhs 
            "=" 
            right: (binary_expression 
                left: (identifier) @b_lhs 
                "+" 
                right: (_) @rhs)
        ) @target
        """

    @property
    def name(self) -> str:
        return "AugAssign"

    def _to_s0(self, node) -> str:
        """s1 ( a = a + b ) -> s0 ( a += b )"""
        def get_text(n): return n.text.decode('utf8') if n else ""

        lhs = get_text(node.child_by_field_name('left'))
        binary_exp = node.child_by_field_name('right')

        b_lhs = get_text(binary_exp.child_by_field_name('left'))
        rhs = get_text(binary_exp.child_by_field_name('right'))

        if lhs == b_lhs:
            return f"{lhs} += {rhs}"
        return None

    def _has_unsafe_rhs(self, node) -> bool:
        """RHS containing a ternary (or other expression of lower precedence
        than +) would change semantics on expansion:
        e.g., sum += a ? 1 : 0  ->  sum = sum + a ? 1 : 0   (wrong precedence)."""
        rhs = node.child_by_field_name('right')
        if not rhs: return True
        return rhs.type == 'ternary_expression'

    def _to_s1(self, node) -> str:
        """s0 ( a += b ) -> s1 ( a = a + b )"""
        if self._has_unsafe_rhs(node):
            return None
        def get_text(n): return n.text.decode('utf8') if n else ""

        lhs = get_text(node.child_by_field_name('left'))
        rhs = get_text(node.child_by_field_name('right'))
        return f"{lhs} = {lhs} + {rhs}"

    def get_current_state(self, src: str) -> int:
        """Skip += patterns whose RHS is a ternary (handled by _has_unsafe_rhs)."""
        s0_nodes = self._get_target_nodes(src, self._query_s0)
        found_s0 = any(not self._has_unsafe_rhs(n) for n in s0_nodes)

        s1_nodes = self._get_target_nodes(src, self._query_s1)
        found_s1 = len(s1_nodes) > 0

        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1


class JSArrayInitStrategy(TreeSitterJSStrategy):
    """
    s0: []           (literal)
    s1: new Array()  (constructor)
    new Array without parentheses is the same expression and reads as s1.
    """
    def __init__(self):
        super().__init__(js_lang.language())
        self._query_s0 = "(array) @target"
        self._query_s1 = """
        (new_expression
            constructor: (identifier) @name
        ) @target
        """

    @property
    def name(self): return "ArrayInit"

    def get_current_state(self, src):
        found_s0 = any(len(n.named_children) == 0
                       for n in self._get_target_nodes(src, self._query_s0))
        found_s1 = any(self._is_array_new(n)
                       for n in self._get_target_nodes(src, self._query_s1))
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _is_array_new(self, node):
        cn = node.child_by_field_name('constructor')
        args = node.child_by_field_name('arguments')
        if not cn or cn.text.decode('utf8') != 'Array': return False
        return args is None or len(args.named_children) == 0

    def _to_s0(self, node):
        if self._is_array_new(node): return "[]"
        return None

    def _to_s1(self, node):
        if node.type == 'array' and len(node.named_children) == 0:
            return "new Array()"
        return None

    def rewrite(self, src, i):
        if i == 0:
            nodes = [n for n in self._get_target_nodes(src, self._query_s1) if self._is_array_new(n)]
        else:
            nodes = [n for n in self._get_target_nodes(src, self._query_s0)
                     if n.type == 'array' and len(n.named_children) == 0]
        nodes.sort(key=lambda n: n.start_byte, reverse=True)
        new_src = bytearray(src, 'utf8')
        for node in nodes:
            replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
            if replacement:
                new_src[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
        return new_src.decode('utf8')


class JSObjectInitStrategy(TreeSitterJSStrategy):
    """
    s0: {}            (literal)
    s1: new Object()  (constructor)
    new Object without parentheses is the same expression and reads as s1.
    """
    def __init__(self):
        super().__init__(js_lang.language())
        self._query_s0 = "(object) @target"
        self._query_s1 = """
        (new_expression
            constructor: (identifier) @name
        ) @target
        """

    @property
    def name(self): return "ObjectInit"

    def get_current_state(self, src):
        found_s0 = any(len(n.named_children) == 0
                       for n in self._get_target_nodes(src, self._query_s0))
        found_s1 = any(self._is_object_new(n)
                       for n in self._get_target_nodes(src, self._query_s1))
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _is_object_new(self, node):
        cn = node.child_by_field_name('constructor')
        args = node.child_by_field_name('arguments')
        if not cn or cn.text.decode('utf8') != 'Object': return False
        return args is None or len(args.named_children) == 0

    def _to_s0(self, node):
        if self._is_object_new(node): return "{}"
        return None

    def _to_s1(self, node):
        if node.type == 'object' and len(node.named_children) == 0:
            return "new Object()"
        return None

    def rewrite(self, src, i):
        if i == 0:
            nodes = [n for n in self._get_target_nodes(src, self._query_s1) if self._is_object_new(n)]
        else:
            nodes = [n for n in self._get_target_nodes(src, self._query_s0)
                     if n.type == 'object' and len(n.named_children) == 0]
        nodes.sort(key=lambda n: n.start_byte, reverse=True)
        new_src = bytearray(src, 'utf8')
        for node in nodes:
            replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
            if replacement:
                new_src[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
        return new_src.decode('utf8')


class JSCompareSwapStrategy(TreeSitterJSStrategy):
    def __init__(self):
        super().__init__(js_lang.language())
        # No field names (left:, operator:), so match by structural order.
        self._query_s0 = """
        [
            (binary_expression (_) "<" (_))
            (binary_expression (_) "<=" (_))
        ] @target
        """
        self._query_s1 = """
        [
            (binary_expression (_) ">" (_))
            (binary_expression (_) ">=" (_))
        ] @target
        """
        self.OP_SWAP_MAP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}

    @property
    def name(self) -> str: return "CompareSwap"

    def _swap_compare(self, node) -> str:
        if len(node.named_children) < 2: return None
        lhs = node.named_children[0].text.decode('utf8')
        rhs = node.named_children[1].text.decode('utf8')
        
        op = None
        for child in node.children:
            if child.type in self.OP_SWAP_MAP:
                op = child.type
                break
        if not op: return None
        
        new_op = self.OP_SWAP_MAP[op]
        return f"{rhs} {new_op} {lhs}"

    def _to_s0(self, node) -> str: return self._swap_compare(node)
    def _to_s1(self, node) -> str: return self._swap_compare(node)



class JSBooleanReturnStrategy(TreeSitterJSStrategy):
    def __init__(self):
        super().__init__(js_lang.language())
        self._query_s0 = """
        (return_statement
          [
            (binary_expression (_) "<" (_))
            (binary_expression (_) "<=" (_))
            (binary_expression (_) ">" (_))
            (binary_expression (_) ">=" (_))
            (binary_expression (_) "==" (_))
            (binary_expression (_) "===" (_))
            (binary_expression (_) "!=" (_))
            (binary_expression (_) "!==" (_))
            (unary_expression "!" (_))
          ]
        ) @target
        """
        self._query_s1 = """
        (if_statement
          (parenthesized_expression)
          (_)
          (else_clause)
        ) @target
        """

    @property
    def name(self) -> str: return "BooleanReturn"

    def _to_s0(self, node) -> str:
        if len(node.named_children) < 3: return None
        cond_node = node.named_children[0]
        cons_node = node.named_children[1]
        alt_node = node.named_children[2] # else_clause

        def clean_text(text):
            return text.replace(" ", "").replace("\n", "").replace("{", "").replace("}", "").replace(";", "").replace("else", "")

        cons_clean = clean_text(cons_node.text.decode('utf8'))
        alt_clean = clean_text(alt_node.text.decode('utf8'))

        if cons_clean == "returntrue" and alt_clean == "returnfalse":
            return f"return {cond_node.text.decode('utf8')};"
        return None

    def _to_s1(self, node) -> str:
        if len(node.named_children) == 0: return None
        cond_text = node.named_children[0].text.decode('utf8')
        return f"if ({cond_text}) {{\n    return true;\n}} else {{\n    return false;\n}}"


class JSArrowFunctionReturnStrategy(TreeSitterJSStrategy):
    def __init__(self):
        super().__init__(js_lang.language())
        self._query = "(arrow_function) @target"

    @property
    def name(self) -> str: return "ArrowFunctionReturn"

    def _is_s0(self, node):
        """Expression body: => expr"""
        if len(node.named_children) < 2: return False
        return node.named_children[-1].type != "statement_block"

    def _is_s1(self, node):
        """Block body with single return: => { return expr; }"""
        if len(node.named_children) < 2: return False
        body = node.named_children[-1]
        if body.type != "statement_block": return False
        return len(body.named_children) == 1 and body.named_children[0].type == "return_statement"

    def get_current_state(self, src: str) -> int:
        target_nodes = self._get_target_nodes(src, self._query)
        found_s0, found_s1 = False, False
        for node in target_nodes:
            if self._is_s0(node): found_s0 = True
            elif self._is_s1(node): found_s1 = True
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node) -> str:
        if not self._is_s1(node): return None
        body = node.named_children[-1]
        return_stmt = body.named_children[0]
        if not return_stmt.named_children:
            return None
        ret_val = return_stmt.named_children[0]
        ret_val_text = ret_val.text.decode('utf8')

        params_text = ""
        for child in node.children:
            if child.type == "=>": break
            params_text += child.text.decode('utf8') + " "
        params_text = params_text.strip()

        if ret_val.type == "object":
            return f"{params_text} => ({ret_val_text})"
        return f"{params_text} => {ret_val_text}"

    def _to_s1(self, node) -> str:
        if not self._is_s0(node): return None
        body = node.named_children[-1]

        params_text = ""
        for child in node.children:
            if child.type == "=>": break
            params_text += child.text.decode('utf8') + " "
        params_text = params_text.strip()

        body_text = body.text.decode('utf8')
        if body.type == "parenthesized_expression" and body.named_children and body.named_children[0].type == "object":
            body_text = body.named_children[0].text.decode('utf8')

        return f"{params_text} => {{\n    return {body_text};\n}}"

    def rewrite(self, src: str, i: int) -> str:
        # Re-parse after each replacement to handle nested arrow functions correctly
        while True:
            nodes = self._get_target_nodes(src, self._query)

            # Find innermost (highest start_byte) applicable node
            target = None
            for node in sorted(nodes, key=lambda n: n.start_byte, reverse=True):
                if i == 0 and self._is_s1(node):
                    replacement = self._to_s0(node)
                elif i == 1 and self._is_s0(node):
                    replacement = self._to_s1(node)
                else:
                    continue
                if replacement:
                    target = (node, replacement)
                    break

            if not target:
                break

            node, replacement = target
            src_bytes = bytearray(src, 'utf8')
            src_bytes[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
            src = src_bytes.decode('utf8')

        return src


class JSPrefixPostfixStrategy(TreeSitterJSStrategy):
    """
    s0: i++;  / i--;   (postfix, standalone statement)
    s1: ++i;  / --i;   (prefix, standalone statement)
    Matches only inside an expression_statement, so for-update positions and
    embedded sub-expressions are excluded.
    """
    def __init__(self):
        super().__init__(js_lang.language())
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
        text = node.text.decode('utf8').strip()
        if text.startswith('++'): return text[2:] + '++'
        elif text.startswith('--'): return text[2:] + '--'
        return None

    def _to_s1(self, node):
        text = node.text.decode('utf8').strip()
        if text.endswith('++'): return '++' + text[:-2]
        elif text.endswith('--'): return '--' + text[:-2]
        return None

    def rewrite(self, src, i):
        nodes = self._get_target_nodes(src, self._query)
        nodes.sort(key=lambda n: n.start_byte, reverse=True)
        new_src = bytearray(src, 'utf8')
        processed = set()
        for node in nodes:
            rng = (node.start_byte, node.end_byte)
            if rng in processed: continue
            if i == 0 and self._is_prefix(node):
                replacement = self._to_s0(node)
            elif i == 1 and self._is_postfix(node):
                replacement = self._to_s1(node)
            else:
                continue
            if replacement:
                new_src[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
                processed.add(rng)
        return new_src.decode('utf8')


class JSNegateIfElseStrategy(TreeSitterJSStrategy):
    """
    s0: if (a === b) { A } else { B }
    s1: if (a !== b) { B } else { A }
    Flip ===/!== or ==/!= and swap the if-else blocks. Operator-disjoint with CompareSwap (<, <=, >, >=).
    Excludes ternary-convertible patterns (both branches assign the same variable),
    BooleanReturn patterns (return true/false) and else-if chains.
    """
    OP_FLIP = {"===": "!==", "!==": "===", "==": "!=", "!=": "=="}

    def __init__(self):
        super().__init__(js_lang.language())
        self._query = """
        (if_statement
            condition: (parenthesized_expression) @cond
            consequence: (statement_block) @then_block
            alternative: (else_clause (statement_block) @else_block)
        ) @target
        """

    @property
    def name(self): return "NegateIfElse"

    def _get_eq_operator(self, node):
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

    def _get_else_block(self, node):
        alt = node.child_by_field_name('alternative')
        if not alt: return None
        for child in alt.named_children:
            if child.type == 'statement_block':
                return child
        return None

    def _is_ternary_pattern(self, node):
        cons = node.child_by_field_name('consequence')
        alt_block = self._get_else_block(node)
        if not cons or not alt_block: return False

        cons_stmts = [c for c in cons.named_children]
        alt_stmts = [c for c in alt_block.named_children]
        if len(cons_stmts) != 1 or len(alt_stmts) != 1: return False
        if cons_stmts[0].type != 'expression_statement' or alt_stmts[0].type != 'expression_statement':
            return False

        c_expr = cons_stmts[0].named_children[0] if cons_stmts[0].named_children else None
        a_expr = alt_stmts[0].named_children[0] if alt_stmts[0].named_children else None
        if not c_expr or not a_expr: return False
        if c_expr.type != 'assignment_expression' or a_expr.type != 'assignment_expression':
            return False

        lhs1 = c_expr.child_by_field_name('left')
        lhs2 = a_expr.child_by_field_name('left')
        if not lhs1 or not lhs2: return False
        return lhs1.text == lhs2.text

    def _is_boolean_return_pattern(self, node):
        cons = node.child_by_field_name('consequence')
        alt_block = self._get_else_block(node)
        if not cons or not alt_block: return False

        def clean(text):
            return text.replace(" ", "").replace("\n", "").replace("{", "").replace("}", "").replace(";", "")

        c = clean(cons.text.decode('utf8'))
        a = clean(alt_block.text.decode('utf8'))
        return (c == "returntrue" and a == "returnfalse") or \
               (c == "returnfalse" and a == "returntrue")

    def _is_valid_target(self, node):
        op = self._get_eq_operator(node)
        if not op: return False
        if self._is_ternary_pattern(node): return False
        if self._is_boolean_return_pattern(node): return False
        return True

    def get_current_state(self, src):
        nodes = self._get_target_nodes(src, self._query)
        found_s0, found_s1 = False, False
        for node in nodes:
            if not self._is_valid_target(node): continue
            op = self._get_eq_operator(node)
            if op in ('===', '=='): found_s0 = True
            elif op in ('!==', '!='): found_s1 = True
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _flip(self, node, from_ops):
        op = self._get_eq_operator(node)
        if op not in from_ops: return None
        if not self._is_valid_target(node): return None

        def get_text(n):
            return n.text.decode('utf8') if n else ""

        cond = node.child_by_field_name('condition')
        inner = cond.named_children[0]
        lhs = get_text(inner.child_by_field_name('left'))
        rhs = get_text(inner.child_by_field_name('right'))
        new_op = self.OP_FLIP[op]

        cons_text = get_text(node.child_by_field_name('consequence'))
        alt_text = get_text(self._get_else_block(node))

        return f"if ({lhs} {new_op} {rhs}) {alt_text} else {cons_text}"

    def _to_s0(self, node):
        return self._flip(node, ('!==', '!='))

    def _to_s1(self, node):
        return self._flip(node, ('===', '=='))

    def rewrite(self, src, i):
        nodes = self._get_target_nodes(src, self._query)
        # Keep only outermost matches, as in the base class rewrite().
        nodes.sort(key=lambda n: n.start_byte)
        unique_nodes = []
        last_end = -1
        for node in nodes:
            if node.start_byte < last_end:
                continue
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


class JSTernaryStrategy(TreeSitterJSStrategy):
    """
    s0: x = cond ? a : b;
    s1: if (cond) { x = a; } else { x = b; }
    Targets only assignment + ternary inside an expression_statement.
    """
    def __init__(self):
        super().__init__(js_lang.language())
        self._query_s0 = """
        (expression_statement
            (assignment_expression
                "="
                right: (ternary_expression)
            )
        ) @target
        """
        self._query_s1 = """
        (if_statement
            condition: (parenthesized_expression) @cond
            consequence: (statement_block) @then_block
            alternative: (else_clause (statement_block) @else_block)
        ) @target
        """

    @property
    def name(self): return "Ternary"

    def _is_ternary_convertible(self, node):
        cons = node.child_by_field_name('consequence')
        alt = node.child_by_field_name('alternative')
        if not cons or not alt: return False

        alt_block = None
        for child in alt.named_children:
            if child.type == 'statement_block':
                alt_block = child
                break
        if not alt_block: return False

        cons_stmts = [c for c in cons.named_children]
        alt_stmts = [c for c in alt_block.named_children]
        if len(cons_stmts) != 1 or len(alt_stmts) != 1: return False
        if cons_stmts[0].type != 'expression_statement' or alt_stmts[0].type != 'expression_statement':
            return False

        c_expr = cons_stmts[0].named_children[0] if cons_stmts[0].named_children else None
        a_expr = alt_stmts[0].named_children[0] if alt_stmts[0].named_children else None
        if not c_expr or not a_expr: return False
        if c_expr.type != 'assignment_expression' or a_expr.type != 'assignment_expression':
            return False

        def has_simple_assign(expr):
            for child in expr.children:
                if child.type == '=': return True
            return False
        if not has_simple_assign(c_expr) or not has_simple_assign(a_expr):
            return False

        lhs1 = c_expr.child_by_field_name('left')
        lhs2 = a_expr.child_by_field_name('left')
        if not lhs1 or not lhs2: return False
        return lhs1.text == lhs2.text

    def get_current_state(self, src):
        found_s0 = len(self._get_target_nodes(src, self._query_s0)) > 0
        found_s1 = any(self._is_ternary_convertible(n)
                       for n in self._get_target_nodes(src, self._query_s1))
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node):
        if not self._is_ternary_convertible(node): return None
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        cond = node.child_by_field_name('condition')
        cond_text = get_text(cond).strip()
        if cond_text.startswith('(') and cond_text.endswith(')'):
            cond_text = cond_text[1:-1].strip()

        cons = node.child_by_field_name('consequence')
        alt = node.child_by_field_name('alternative')
        alt_block = None
        for child in alt.named_children:
            if child.type == 'statement_block':
                alt_block = child
                break

        c_assign = cons.named_children[0].named_children[0]
        a_assign = alt_block.named_children[0].named_children[0]

        lhs = get_text(c_assign.child_by_field_name('left'))
        rhs1 = get_text(c_assign.child_by_field_name('right'))
        rhs2 = get_text(a_assign.child_by_field_name('right'))

        return f"{lhs} = {cond_text} ? {rhs1} : {rhs2};"

    def _to_s1(self, node):
        def get_text(n):
            return n.text.decode('utf8') if n else ""

        assign = None
        for child in node.named_children:
            if child.type == 'assignment_expression':
                assign = child
                break
        if not assign: return None

        lhs = get_text(assign.child_by_field_name('left'))
        ternary = assign.child_by_field_name('right')
        if not ternary or ternary.type != 'ternary_expression': return None

        cond = get_text(ternary.child_by_field_name('condition'))
        then_val = get_text(ternary.child_by_field_name('consequence'))
        else_val = get_text(ternary.child_by_field_name('alternative'))

        return f"if ({cond}) {{\n    {lhs} = {then_val};\n}} else {{\n    {lhs} = {else_val};\n}}"

    def rewrite(self, src, i):
        query = self._query_s1 if i == 0 else self._query_s0
        nodes = self._get_target_nodes(src, query)
        nodes.sort(key=lambda n: n.start_byte, reverse=True)
        new_src = bytearray(src, 'utf8')
        processed = set()
        for node in nodes:
            rng = (node.start_byte, node.end_byte)
            if rng in processed: continue
            replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
            if replacement:
                new_src[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
                processed.add(rng)
        return new_src.decode('utf8')


class JSYodaEqualsStrategy(TreeSitterJSStrategy):
    """
    s0: x === literal  /  x !== literal  (variable first)
    s1: literal === x  /  literal !== x  (literal first, Yoda)
    Targets ===, !==, ==, !=. Swaps operands only, the operator is preserved.
    Operator-disjoint with CompareSwap and orthogonal to NegateIfElse.
    """
    LITERAL_TYPES = {'number', 'string', 'null', 'undefined', 'true', 'false'}
    TARGET_OPS = {'===', '!==', '==', '!='}

    def __init__(self):
        super().__init__(js_lang.language())
        self._query = """
        [
            (binary_expression (_) "===" (_))
            (binary_expression (_) "!==" (_))
            (binary_expression (_) "==" (_))
            (binary_expression (_) "!=" (_))
        ] @target
        """

    @property
    def name(self): return "YodaEquals"

    def _is_literal(self, node):
        return node.type in self.LITERAL_TYPES

    def _is_s0_node(self, node):
        lhs = node.child_by_field_name('left')
        rhs = node.child_by_field_name('right')
        if not lhs or not rhs: return False
        return not self._is_literal(lhs) and self._is_literal(rhs)

    def _is_s1_node(self, node):
        lhs = node.child_by_field_name('left')
        rhs = node.child_by_field_name('right')
        if not lhs or not rhs: return False
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
        if not lhs or not rhs: return None
        op = None
        for child in node.children:
            if child.type in self.TARGET_OPS:
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
        nodes.sort(key=lambda n: n.start_byte, reverse=True)
        new_src = bytearray(src, 'utf8')
        processed = set()
        for node in nodes:
            rng = (node.start_byte, node.end_byte)
            if rng in processed: continue
            replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
            if replacement:
                new_src[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
                processed.add(rng)
        return new_src.decode('utf8')


class JSReturnVarStrategy(TreeSitterJSStrategy):
    """
    s0: return expr;
    s1: const result = expr; return result;
    Targets only the last statement of a top-level function body. Skips a trivial expr (identifier/literal).
    May compete with BooleanReturn on return <compare>.
    """
    CANDIDATE_NAMES = ['result', 'ret', 'res', '_result', '_ret', '_res']
    TRIVIAL_TYPES = {'identifier', 'number', 'string', 'null', 'undefined', 'true', 'false'}

    def __init__(self):
        super().__init__(js_lang.language())

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

    def _find_function_bodies(self, src):
        """Return bodies of top-level functions only. Nested functions are excluded
        to avoid interfering with ArrowFunctionReturn."""
        tree = self.parser.parse(bytes(src, 'utf8'))
        bodies = []
        for node in tree.root_node.children:
            if node.type in ('function_declaration', 'function_expression'):
                body = node.child_by_field_name('body')
                if body and body.type == 'statement_block':
                    bodies.append(body)
            elif node.type in ('lexical_declaration', 'variable_declaration'):
                for declarator in node.named_children:
                    if declarator.type == 'variable_declarator':
                        value = declarator.child_by_field_name('value')
                        if value and value.type in ('arrow_function', 'function_expression'):
                            body = value.child_by_field_name('body')
                            if body and body.type == 'statement_block':
                                bodies.append(body)
        return bodies

    def _check_s0(self, body_node):
        stmts = [c for c in body_node.named_children]
        if not stmts: return None
        last = stmts[-1]
        if last.type != 'return_statement': return None
        if not last.named_children: return None
        expr = last.named_children[0]
        if expr.type in self.TRIVIAL_TYPES: return None
        # Exclude the comma operator (sequence_expression), whose comma would act as a separator after the rewrite.
        if expr.type == 'sequence_expression': return None
        return last

    def _check_s1(self, body_node):
        stmts = [c for c in body_node.named_children]
        if len(stmts) < 2: return None
        decl, ret = stmts[-2], stmts[-1]

        if decl.type not in ('lexical_declaration', 'variable_declaration'):
            return None
        if ret.type != 'return_statement':
            return None

        declarator = None
        for child in decl.named_children:
            if child.type == 'variable_declarator':
                declarator = child
                break
        if not declarator: return None

        var_name_node = declarator.child_by_field_name('name')
        var_value_node = declarator.child_by_field_name('value')
        if not var_name_node or not var_value_node: return None

        var_name = var_name_node.text.decode('utf8')

        if not ret.named_children: return None
        ret_expr = ret.named_children[0]
        if ret_expr.type != 'identifier' or ret_expr.text.decode('utf8') != var_name:
            return None

        if var_value_node.type in self.TRIVIAL_TYPES: return None
        return (decl, ret, var_value_node)

    def get_current_state(self, src):
        bodies = self._find_function_bodies(src)
        found_s0, found_s1 = False, False
        for body in bodies:
            if self._check_s0(body): found_s0 = True
            if self._check_s1(body): found_s1 = True
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def rewrite(self, src, i):
        bodies = self._find_function_bodies(src)
        src_bytes = bytearray(src, 'utf8')
        replacements = []

        for body in bodies:
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
                    line_start = bytes(src_bytes).rfind(b'\n', 0, ret_node.start_byte)
                    indent = ''
                    if line_start != -1:
                        for ch in src_bytes[line_start + 1:ret_node.start_byte]:
                            if ch in (32, 9):
                                indent += chr(ch)
                            else:
                                break
                    var_name = self._pick_var_name(body)
                    replacement = f"const {var_name} = {expr_text};\n{indent}return {var_name};"
                    replacements.append((ret_node.start_byte, ret_node.end_byte, replacement))

        replacements.sort(key=lambda r: r[0], reverse=True)
        for start, end, replacement in replacements:
            src_bytes[start:end] = bytes(replacement, 'utf8')
        return src_bytes.decode('utf8')


class JSWhileForSwapStrategy(TreeSitterJSStrategy):
    """
    s0: while (cond) { body }
    s1: for (; cond;) { body }
    Targets only simple for-loops with neither init nor update.
    """
    def __init__(self):
        super().__init__(js_lang.language())
        self._query_s0 = """
        (while_statement
            condition: (parenthesized_expression) @cond
            body: (statement_block) @body
        ) @target
        """
        self._query_s1 = """
        (for_statement
            condition: (_) @cond
            body: (statement_block) @body
        ) @target
        """

    @property
    def name(self): return "WhileForSwap"

    def _is_simple_for(self, node):
        """Check the for-loop has no init, no update and a real condition
        (rejects the infinite for(;;))."""
        initializer = node.child_by_field_name('initializer')
        increment = node.child_by_field_name('increment')
        condition = node.child_by_field_name('condition')
        # tree-sitter parses empty for-fields as empty_statement (";").
        init_empty = initializer is None or initializer.type == 'empty_statement'
        has_condition = condition is not None and condition.type != 'empty_statement'
        return init_empty and increment is None and has_condition

    def get_current_state(self, src):
        found_s0 = len(self._get_target_nodes(src, self._query_s0)) > 0
        found_s1 = any(self._is_simple_for(n)
                       for n in self._get_target_nodes(src, self._query_s1))
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node):
        if not self._is_simple_for(node): return None
        def get_text(n):
            return n.text.decode('utf8') if n else ""
        cond = node.child_by_field_name('condition')
        body = node.child_by_field_name('body')
        if not cond or not body: return None
        return f"while ({get_text(cond)}) {get_text(body)}"

    def _to_s1(self, node):
        def get_text(n):
            return n.text.decode('utf8') if n else ""
        cond = node.child_by_field_name('condition')
        body = node.child_by_field_name('body')
        if not cond or not body: return None
        cond_text = get_text(cond).strip()
        if cond_text.startswith('(') and cond_text.endswith(')'):
            cond_text = cond_text[1:-1].strip()
        return f"for (; {cond_text};) {get_text(body)}"

    def rewrite(self, src, i):
        # Re-parse after each replacement to keep byte offsets consistent
        # when nested while/for loops are present.
        while True:
            query = self._query_s1 if i == 0 else self._query_s0
            nodes = self._get_target_nodes(src, query)
            if i == 0:
                nodes = [n for n in nodes if self._is_simple_for(n)]

            # Process the deepest (last-starting) node first.
            target = None
            for node in sorted(nodes, key=lambda n: n.start_byte, reverse=True):
                replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
                if replacement:
                    target = (node, replacement)
                    break

            if not target:
                break

            node, replacement = target
            src_bytes = bytearray(src, 'utf8')
            src_bytes[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
            src = src_bytes.decode('utf8')

        return src


class JSBooleanCastStrategy(TreeSitterJSStrategy):
    """
    s0: !!x        (double negation, idiomatic JS boolean coercion)
    s1: Boolean(x) (explicit conversion)
    """
    # Primary expressions bind tighter than unary !, so !! needs no parens.
    _PRIMARY_TYPES = {
        'identifier', 'number', 'string', 'template_string',
        'true', 'false', 'null', 'undefined', 'regex',
        'member_expression', 'subscript_expression', 'call_expression',
        'this', 'super', 'parenthesized_expression',
        'object', 'array', 'new_expression',
    }

    def __init__(self):
        super().__init__(js_lang.language())
        # !!expr: unary ! wrapping another unary !
        self._query_s0 = """
        (unary_expression
            operator: "!"
            argument: (unary_expression
                operator: "!"
                argument: (_) @inner
            )
        ) @target
        """
        # Boolean(expr): call to Boolean with single argument
        self._query_s1 = """
        (call_expression
            function: (identifier) @fn
            arguments: (arguments (_) @arg)
        ) @target
        """

    @property
    def name(self): return "BooleanCast"

    def _is_boolean_call(self, node):
        fn = node.child_by_field_name('function')
        args = node.child_by_field_name('arguments')
        if not fn or fn.type != 'identifier': return False
        if fn.text.decode('utf8') != 'Boolean': return False
        if not args: return False
        # Allow exactly one argument.
        named = [c for c in args.named_children]
        return len(named) == 1

    def get_current_state(self, src):
        found_s0 = len(self._get_target_nodes(src, self._query_s0)) > 0
        s1_nodes = self._get_target_nodes(src, self._query_s1)
        found_s1 = any(self._is_boolean_call(n) for n in s1_nodes)
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node):
        """Boolean(x) -> !!x (parens added only for non-primary expressions)"""
        if not self._is_boolean_call(node): return None
        args = node.child_by_field_name('arguments')
        arg = args.named_children[0]
        arg_text = arg.text.decode('utf8')
        if arg.type in self._PRIMARY_TYPES:
            return f"!!{arg_text}"
        return f"!!({arg_text})"

    def _to_s1(self, node):
        """!!x -> Boolean(x) (unwrap outer parens on x since Boolean() already provides them)"""
        # node is the outer unary_expression (!!x)
        # inner is the inner unary_expression (!x), inner's argument is x
        inner = node.child_by_field_name('argument')
        if not inner: return None
        x = inner.child_by_field_name('argument')
        if not x: return None
        # Unwrap a redundant outer parenthesized_expression so !!(a||b) -> Boolean(a||b)
        if x.type == 'parenthesized_expression' and x.named_children:
            inner_expr = x.named_children[0]
            return f"Boolean({inner_expr.text.decode('utf8')})"
        return f"Boolean({x.text.decode('utf8')})"

    def rewrite(self, src, i):
        if i == 0:
            query = self._query_s1
            nodes = self._get_target_nodes(src, query)
            nodes = [n for n in nodes if self._is_boolean_call(n)]
        else:
            query = self._query_s0
            nodes = self._get_target_nodes(src, query)

        nodes.sort(key=lambda n: n.start_byte, reverse=True)
        new_src = bytearray(src, 'utf8')
        processed = set()
        for node in nodes:
            rng = (node.start_byte, node.end_byte)
            if rng in processed: continue
            replacement = self._to_s0(node) if i == 0 else self._to_s1(node)
            if replacement:
                new_src[node.start_byte:node.end_byte] = bytes(replacement, 'utf8')
                processed.add(rng)
        return new_src.decode('utf8')


class JSPropertyShorthandStrategy(TreeSitterJSStrategy):
    """
    s0: { x }       (shorthand property)
    s1: { x: x }    (explicit property, key == value)
    """
    def __init__(self):
        super().__init__(js_lang.language())
        self._query_pair = """
        (pair
            key: (property_identifier) @key
            value: (identifier) @val
        ) @target
        """
        self._query_shorthand = "(shorthand_property_identifier) @target"

    @property
    def name(self): return "PropertyShorthand"

    def _is_same_key_val(self, node):
        key = node.child_by_field_name('key')
        val = node.child_by_field_name('value')
        return key and val and key.text == val.text

    def get_current_state(self, src):
        found_s0 = len(self._get_target_nodes(src, self._query_shorthand)) > 0
        found_s1 = any(self._is_same_key_val(n) for n in self._get_target_nodes(src, self._query_pair))
        if found_s0 and found_s1: return -1
        if found_s0: return 0
        if found_s1: return 1
        return -1

    def _to_s0(self, node):
        """{ x: x } -> { x }"""
        if not self._is_same_key_val(node): return None
        return node.child_by_field_name('key').text.decode('utf8')

    def _to_s1(self, node):
        """{ x } -> { x: x }"""
        name = node.text.decode('utf8')
        return f"{name}: {name}"

    def rewrite(self, src, i):
        if i == 0:
            nodes = [n for n in self._get_target_nodes(src, self._query_pair) if self._is_same_key_val(n)]
        else:
            nodes = list(self._get_target_nodes(src, self._query_shorthand))
        nodes.sort(key=lambda n: n.start_byte, reverse=True)
        new_src = bytearray(src, 'utf8')
        for node in nodes:
            r = self._to_s0(node) if i == 0 else self._to_s1(node)
            if r:
                new_src[node.start_byte:node.end_byte] = bytes(r, 'utf8')
        return new_src.decode('utf8')


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


def _ob_terminate(stmt_text):
    # Only a semicolon counts as terminated, not a trailing }. Every type in _OB_SINGLE_STMT_TYPES
    # can end in an object or function literal (x = {}), and all of them accept a semicolon.

    stmt_text = stmt_text.strip()
    return stmt_text if stmt_text.endswith(';') else stmt_text + ';'


class JSOptionalBracesStrategy(TreeSitterJSStrategy):
    """
    s0: if (cond) stmt;            (no braces)
    s1: if (cond) { stmt; }        (with braces, single statement)
    Only if-without-else. Excludes multiline statements and line comments.
    s0 always ends in a semicolon, since dropping the braces also drops the } that ASI
    (Automatic Semicolon Insertion) was using to terminate the statement.
    """
    def __init__(self):
        super().__init__(js_lang.language())
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
        if not cons or cons.type != 'statement_block': return False
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
        body = _ob_terminate(stmt.text.decode('utf8'))
        return f"{prefix}\n{indent}{indent_unit}{body}"

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
        body = _ob_terminate(cons.text.decode('utf8'))
        return f"{prefix}\n{indent}{indent_unit}{body}"

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


def _test_js_strategies():
    strategies = get_js_strategies()
    
    sample_js_code = """
function processData(items, max_limit) {
    // [1. ArrayInit / ObjectInit] mixed initialization forms
    let results = new Array();
    let metadata = {};

    // [2. AugAssign] mixed assignment forms
    let totalScore = 0;
    totalScore += 50;

    let errorCount = 0;
    errorCount = errorCount + 1;

    // [3. CompareSwap] mixed comparison directions
    if (totalScore <= max_limit) {
        metadata.status = "pass";
    }
    if (errorCount > 10) {
        metadata.status = "fail";
    }

    // [4. ArrowFunctionReturn] mixed arrow-function return forms
    let doubled = items.map(x => {
        return x * 2;
    });
    let squared = items.map(x => x * x);

    // [5. BooleanReturn] mixed boolean return forms
    let isFinished = function(val) {
        if (val === 100) {
            return true;
        } else {
            return false;
        }
    };
    
    let isInvalid = function(val) {
        return val < 0;
    };

    return results;
}
"""

    print("="*60)
    print("[ Original Code ]")
    print("="*60)
    print(sample_js_code.strip())

    # Test 1: apply all strategies with i=0 (Normalize / Compress to s0).
    code_s0 = sample_js_code
    for strategy in strategies:
        code_s0 = strategy.rewrite(code_s0, i=0)

    print("\n" + "="*60)
    print("[ Pipeline Result: i=0 (Target: s0 - Compress) ]")
    print("- new Array() -> []")
    print("- a = a + b   -> a += b")
    print("- a > b       -> b < a")
    print("- => { return x; } -> => x")
    print("- if(c) return true else return false -> return c;")
    print("="*60)
    print(code_s0.strip())

    # Test 2: apply all strategies with i=1 (Expand to s1).
    code_s1 = sample_js_code
    for strategy in strategies:
        code_s1 = strategy.rewrite(code_s1, i=1)

    print("\n" + "="*60)
    print("[ Pipeline Result: i=1 (Target: s1 - Expand) ]")
    print("- []     -> new Array(), {} -> new Object()")
    print("- a += b -> a = a + b")
    print("- a <= b -> b >= a")
    print("- => x   -> => { return x; }")
    print("- return c; -> if(c) { return true; } else { return false; }")
    print("="*60)
    print(code_s1.strip())

if __name__ == "__main__":
    _test_js_strategies()
