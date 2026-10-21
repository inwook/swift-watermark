import os
import re
import json
import time
import textwrap
import subprocess
import tempfile

from typing import List, Tuple, Optional, Dict, Any

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from llm_provider import get_provider
from language_parser import JavaParser
from paths import TMP_PATH, OUT_PATH

_JAVA_LANGUAGE = Language(tsjava.language())

_PRIMITIVES = {'int', 'long', 'short', 'byte', 'double', 'float', 'boolean', 'char'}
_PRIM_TO_BOX = {
    'int': 'Integer', 'long': 'Long', 'short': 'Short', 'byte': 'Byte',
    'double': 'Double', 'float': 'Float', 'boolean': 'Boolean', 'char': 'Character',
}
_BOX_TO_PRIM = {v: k for k, v in _PRIM_TO_BOX.items()}


def _normalize_type(s: str) -> str:
    """Collapse whitespace inside a type expression."""
    return re.sub(r'\s+', '', s)


def _strip_array_suffix(t: str) -> Tuple[str, int]:
    """Return (base_type, ndim) for an array type, or (t, 0) for non-arrays."""
    ndim = 0
    while t.endswith('[]'):
        t = t[:-2]
        ndim += 1
    return t, ndim


def _is_supported_type(type_str: str, allow_void: bool = False) -> bool:
    """Whitelist: primitives, String, arrays of those, and List<T>/ArrayList<T> of wrappers or String."""
    t = _normalize_type(type_str)
    if not t:
        return False
    if allow_void and t == 'void':
        return True

    base, _ndim = _strip_array_suffix(t)
    if base in _PRIMITIVES:
        return True
    if base in _BOX_TO_PRIM:
        return True
    if base == 'String':
        return True

    m = re.match(r'^(List|ArrayList)<([^<>]+)>$', base)
    if m:
        inner = m.group(2)
        if inner in _BOX_TO_PRIM or inner == 'String':
            return True
        return False

    return False


def _ts_parse(code: str):
    parser = Parser(_JAVA_LANGUAGE)
    b = code.encode('utf-8')
    return parser.parse(b), b


def _node_text(node, src_bytes: bytes) -> str:
    return src_bytes[node.start_byte:node.end_byte].decode('utf-8')


def _walk_find_first(node, node_type: str):
    if node.type == node_type:
        return node
    for c in node.children:
        r = _walk_find_first(c, node_type)
        if r is not None:
            return r
    return None


def _parse_method_signature(method_code: str) -> Optional[Dict[str, Any]]:
    """Extract name, return_type, params, is_static and has_type_params, or None if it does not parse."""
    # A bare top-level method_declaration sometimes parses with errors.
    wrapped = f"class __Tmp {{\n{method_code}\n}}"
    tree, src = _ts_parse(wrapped)
    method = _walk_find_first(tree.root_node, 'method_declaration')
    if method is None:
        return None

    name_node = method.child_by_field_name('name')
    type_node = method.child_by_field_name('type')
    params_node = method.child_by_field_name('parameters')
    tparams_node = method.child_by_field_name('type_parameters')

    if name_node is None or type_node is None or params_node is None:
        return None

    name = _node_text(name_node, src)
    return_type = _node_text(type_node, src)

    is_static = False
    for c in method.children:
        if c.type == 'modifiers':
            mtext = _node_text(c, src)
            if re.search(r'\bstatic\b', mtext):
                is_static = True
            break

    params: List[Tuple[str, str]] = []
    for c in params_node.named_children:
        if c.type == 'formal_parameter':
            ptype_node = c.child_by_field_name('type')
            pname_node = c.child_by_field_name('name')
            if ptype_node is None or pname_node is None:
                return None
            ptype = _node_text(ptype_node, src)
            pname = _node_text(pname_node, src)
            # int x[]: dimensions attached to the name
            dims_node = c.child_by_field_name('dimensions')
            if dims_node is not None:
                ptype = ptype + _node_text(dims_node, src)
            params.append((ptype, pname))
        elif c.type == 'spread_parameter':
            # varargs as an array of its type
            t = None
            nm = None
            for sc in c.children:
                if sc.type not in ('modifiers', ',', '...'):
                    if t is None:
                        t = _node_text(sc, src)
                    elif sc.type == 'variable_declarator':
                        nn = sc.child_by_field_name('name')
                        if nn is not None:
                            nm = _node_text(nn, src)
            if t is None or nm is None:
                return None
            params.append((t + '[]', nm))

    return {
        'name': name,
        'return_type': return_type,
        'params': params,
        'is_static': is_static,
        'has_type_params': tparams_node is not None,
    }


def _force_static(method_code: str) -> str:
    """Ensure the method has the static modifier. Prepends if missing."""
    # Only check the signature portion (before the opening brace of the body)
    brace = method_code.find('{')
    head = method_code[:brace] if brace >= 0 else method_code
    if re.search(r'\bstatic\b', head):
        return method_code
    m = re.match(r'^(\s*)', method_code)
    indent = m.group(1)
    return indent + 'static ' + method_code[len(indent):]


def _signature_matches(variant_code: str, ref_sig: Dict[str, Any]) -> bool:
    """Check that a variant has the same (name, return_type, param types) as ref."""
    v = _parse_method_signature(variant_code)
    if v is None:
        return False
    if v['name'] != ref_sig['name']:
        return False
    if _normalize_type(v['return_type']) != _normalize_type(ref_sig['return_type']):
        return False
    if len(v['params']) != len(ref_sig['params']):
        return False
    for (vt, _vn), (rt, _rn) in zip(v['params'], ref_sig['params']):
        if _normalize_type(vt) != _normalize_type(rt):
            return False
    return True


# _HARNESS_IMPORTS and _JAVA_KNOWN_NAMES must stay in sync: a name is known only if
# the harness imports its package, otherwise javac fails with "cannot find symbol".
_HARNESS_IMPORTS = (
    "java.util.*", "java.util.stream.*", "java.util.function.*",
    "java.util.regex.*", "java.io.*", "java.math.*", "java.net.*",
    "java.nio.charset.*", "java.time.*", "java.time.format.*",
    "java.time.temporal.*",
)

_JAVA_KNOWN_NAMES = {
    # java.lang (implicitly imported)
    'String', 'Object', 'Integer', 'Long', 'Short', 'Byte', 'Double', 'Float',
    'Boolean', 'Character', 'Number', 'Math', 'System', 'StringBuilder',
    'StringBuffer', 'CharSequence', 'Comparable', 'Iterable', 'Runnable',
    'Thread', 'Class', 'Enum', 'Void', 'Record', 'Cloneable', 'AutoCloseable',
    'Throwable', 'Exception', 'RuntimeException', 'Error',
    'IllegalArgumentException', 'IllegalStateException', 'NullPointerException',
    'IndexOutOfBoundsException', 'ArrayIndexOutOfBoundsException',
    'StringIndexOutOfBoundsException', 'NumberFormatException',
    'ArithmeticException', 'ClassCastException', 'UnsupportedOperationException',
    'InterruptedException', 'CloneNotSupportedException', 'AssertionError',
    'NegativeArraySizeException', 'StackOverflowError', 'OutOfMemoryError',
    'SecurityException', 'Deprecated', 'Override', 'SuppressWarnings',
    'FunctionalInterface', 'SafeVarargs',
    # java.util
    'List', 'ArrayList', 'LinkedList', 'Map', 'HashMap', 'TreeMap',
    'LinkedHashMap', 'Set', 'HashSet', 'TreeSet', 'LinkedHashSet',
    'Collection', 'Collections', 'Arrays', 'Iterator', 'ListIterator',
    'Optional', 'OptionalInt', 'OptionalLong', 'OptionalDouble', 'Objects',
    'Comparator', 'Queue', 'Deque', 'ArrayDeque', 'PriorityQueue', 'Stack',
    'Vector', 'Random', 'Scanner', 'StringJoiner', 'StringTokenizer', 'UUID',
    'BitSet', 'EnumMap', 'EnumSet', 'AbstractMap', 'SortedMap', 'SortedSet',
    'NavigableMap', 'NavigableSet', 'Spliterator', 'Spliterators', 'Base64',
    'NoSuchElementException', 'ConcurrentModificationException',
    'EmptyStackException', 'InputMismatchException', 'Date', 'Calendar',
    'Locale', 'Formatter', 'IntSummaryStatistics', 'DoubleSummaryStatistics',
    'LongSummaryStatistics',
    # java.util.stream / function
    'Stream', 'IntStream', 'LongStream', 'DoubleStream', 'Collectors',
    'Collector', 'StreamSupport', 'Function', 'BiFunction', 'Supplier',
    'Consumer', 'BiConsumer', 'Predicate', 'BiPredicate', 'UnaryOperator',
    'BinaryOperator', 'IntFunction', 'ToIntFunction', 'ToLongFunction',
    'ToDoubleFunction', 'IntPredicate', 'IntUnaryOperator', 'IntBinaryOperator',
    # java.util.regex
    'Pattern', 'Matcher', 'MatchResult', 'PatternSyntaxException',
    # java.io
    'IOException', 'UncheckedIOException', 'FileNotFoundException', 'File',
    'InputStream', 'OutputStream', 'Reader', 'Writer', 'BufferedReader',
    'BufferedWriter', 'InputStreamReader', 'OutputStreamWriter',
    'StringReader', 'StringWriter', 'PrintWriter', 'PrintStream',
    'ByteArrayInputStream', 'ByteArrayOutputStream', 'Serializable',
    # java.math
    'BigDecimal', 'BigInteger', 'MathContext', 'RoundingMode',
    # java.net
    'URL', 'URI', 'MalformedURLException', 'URISyntaxException',
    'URLEncoder', 'URLDecoder',
    # java.nio.charset
    'Charset', 'StandardCharsets',
    # java.time
    'LocalDate', 'LocalDateTime', 'LocalTime', 'Instant', 'Duration', 'Period',
    'ZonedDateTime', 'ZoneOffset', 'ZoneId', 'DayOfWeek', 'Month', 'Year',
    'ChronoUnit', 'DateTimeFormatter', 'DateTimeParseException',
}

# Identifier positions that are declaration sites or member names, not free
# references. method_invocation.name is handled separately: it is a free
# reference only when the invocation has no receiver object.
_DECL_NAME_PARENTS = {
    'formal_parameter', 'spread_parameter', 'variable_declarator',
    'catch_formal_parameter', 'enhanced_for_statement', 'resource',
    'method_declaration', 'class_declaration', 'interface_declaration',
    'enum_declaration', 'record_declaration', 'type_parameter',
    'labeled_statement', 'annotation_type_declaration',
}


def _field_name_of(node) -> Optional[str]:
    """Return the field name this node occupies in its parent, or None."""
    parent = node.parent
    if parent is None:
        return None
    for i in range(parent.child_count):
        if parent.child(i) == node:
            return parent.field_name_for_child(i)
    return None


def _java_collect_defined_names(root, src_bytes: bytes) -> set:
    """Names bound inside the method: params, locals, loop/catch vars, lambdas."""
    names = set()

    def visit(node):
        t = node.type
        if t in ('formal_parameter', 'spread_parameter', 'catch_formal_parameter',
                 'variable_declarator', 'enhanced_for_statement', 'resource',
                 'type_parameter'):
            nm = node.child_by_field_name('name')
            if nm is not None:
                names.add(_node_text(nm, src_bytes))
            elif t == 'type_parameter' and node.child_count:
                names.add(_node_text(node.child(0), src_bytes))
        elif t == 'lambda_expression':
            params = node.child_by_field_name('parameters')
            if params is not None:
                if params.type == 'identifier':
                    names.add(_node_text(params, src_bytes))
                else:
                    for c in params.children:
                        if c.type == 'identifier':
                            names.add(_node_text(c, src_bytes))
        for c in node.children:
            visit(c)

    visit(root)
    return names


def _java_free_references(code: str, extra_known: Optional[set] = None) -> List[str]:
    """Identifiers/types the standalone harness cannot resolve, so javac would fail."""
    tree, src_bytes = _ts_parse("class __W {\n" + code + "\n}")
    root = tree.root_node
    if root.has_error:
        return []

    method = _walk_find_first(root, 'method_declaration')
    if method is None:
        method = _walk_find_first(root, 'constructor_declaration')
    if method is None:
        return []

    defined = _java_collect_defined_names(method, src_bytes)
    own = method.child_by_field_name('name')
    if own is not None:
        defined.add(_node_text(own, src_bytes))
    known = defined | _JAVA_KNOWN_NAMES | (extra_known or set())

    unresolved, seen = [], set()

    def report(name):
        if name and name not in known and name not in seen:
            seen.add(name)
            unresolved.append(name)

    def visit(node):
        t = node.type

        # Qualified names: only the leftmost component must resolve.
        if t in ('scoped_type_identifier', 'scoped_identifier'):
            leftmost = node
            while leftmost.type in ('scoped_type_identifier', 'scoped_identifier'):
                if leftmost.child_count == 0:
                    break
                leftmost = leftmost.child(0)
            report(_node_text(leftmost, src_bytes))
            return  # do not descend: the trailing parts are members

        # Integer::intValue: the part after :: is a member of the qualifier.
        if t == 'method_reference':
            if node.child_count:
                visit(node.child(0))
            return

        if t == 'type_identifier':
            report(_node_text(node, src_bytes))

        elif t == 'identifier':
            field = _field_name_of(node)
            parent_type = node.parent.type if node.parent else ''
            if parent_type == 'method_invocation':
                if field == 'name':
                    # An unqualified call targets the enclosing class, which the harness lacks.
                    if node.parent.child_by_field_name('object') is None:
                        report(_node_text(node, src_bytes))
                elif field == 'object':
                    report(_node_text(node, src_bytes))
            elif parent_type == 'field_access':
                if field == 'object':
                    report(_node_text(node, src_bytes))
                # field == 'field' is a member name, not a free reference
            elif parent_type in _DECL_NAME_PARENTS and field == 'name':
                pass  # declaration site
            elif parent_type in ('break_statement', 'continue_statement'):
                pass  # loop label
            elif parent_type in ('marker_annotation', 'annotation') and field == 'name':
                report(_node_text(node, src_bytes))
            else:
                report(_node_text(node, src_bytes))

        for c in node.children:
            visit(c)

    # throws types must resolve too, since a custom checked exception breaks the harness signature.
    for c in method.children:
        if c.type == 'throws':
            visit(c)
    body = method.child_by_field_name('body')
    if body is not None:
        visit(body)
    # Parameter and return types are already covered by the whitelist check,
    # but generic/annotation parts of the header can still reference unknowns.
    params = method.child_by_field_name('parameters')
    if params is not None:
        visit(params)

    return unresolved


def check_verifiable_java(code: str, enclosing_class: Optional[str] = None
                         ) -> Tuple[bool, List[str]]:
    """Check whether a Java method can be verified standalone. Returns (ok, reasons).
    enclosing_class lets a static self-call like Zigzag.zigzag(...) resolve.
    """
    reasons: List[str] = []

    sig = _parse_method_signature(code)
    if sig is None:
        return False, ["ParseError: could not parse method signature"]

    if sig['has_type_params']:
        reasons.append("GenericMethod: has type parameters")

    if not _is_supported_type(sig['return_type'], allow_void=True):
        reasons.append(f"UnsupportedReturn: {sig['return_type']}")

    for ptype, pname in sig['params']:
        if not _is_supported_type(ptype, allow_void=False):
            reasons.append(f"UnsupportedParam: {pname} ({ptype})")

    # @Override methods cannot be made static
    head = code[:code.find('{')] if '{' in code else code
    if '@Override' in head:
        reasons.append("HasOverride: @Override cannot be force-static")

    # this./super. reference instance state unavailable in static harness
    if re.search(r'\bthis\.', code):
        reasons.append("UsesThis: this. reference")
    if re.search(r'\bsuper\.', code):
        reasons.append("UsesSuper: super. reference")

    # Free identifiers / types the standalone harness cannot resolve.
    extra = {enclosing_class} if enclosing_class else None
    for name in _java_free_references(code, extra)[:5]:
        reasons.append(f"ExternalRef: '{name}'")

    return (len(reasons) == 0), reasons


def _java_str_escape(s: str) -> str:
    out = []
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == '\\':
            out.append('\\\\')
        elif c == '\n':
            out.append('\\n')
        elif c == '\r':
            out.append('\\r')
        elif c == '\t':
            out.append('\\t')
        elif c == '\b':
            out.append('\\b')
        elif c == '\f':
            out.append('\\f')
        elif ord(c) < 0x20:
            out.append(f'\\u{ord(c):04x}')
        else:
            out.append(c)
    return ''.join(out)


def _java_char_literal(s: str) -> str:
    if len(s) != 1:
        raise ValueError(f"char requires length-1 string, got {s!r}")
    c = s
    if c == "'":
        return "'\\''"
    if c == '\\':
        return "'\\\\'"
    if c == '\n':
        return "'\\n'"
    if c == '\r':
        return "'\\r'"
    if c == '\t':
        return "'\\t'"
    if ord(c) < 0x20 or ord(c) > 0x7e:
        return f"'\\u{ord(c):04x}'"
    return f"'{c}'"


def _to_java_literal(value, java_type: str) -> str:
    """Convert a json.loads value to a Java literal of the given type. Raises ValueError on mismatch."""
    t = _normalize_type(java_type)

    if value is None:
        # Allow null only for object-ish types
        if t in _PRIMITIVES:
            raise ValueError(f"null not allowed for primitive {t}")
        return 'null'

    # Boxed wrappers take the same literals as their primitives (autoboxed).
    if t in _BOX_TO_PRIM:
        t = _BOX_TO_PRIM[t]

    if t == 'int':
        return str(int(value))
    if t == 'long':
        return f"{int(value)}L"
    if t == 'short':
        return f"(short){int(value)}"
    if t == 'byte':
        return f"(byte){int(value)}"
    if t == 'double':
        return repr(float(value))
    if t == 'float':
        return f"{repr(float(value))}f"
    if t == 'boolean':
        return 'true' if value else 'false'
    if t == 'char':
        return _java_char_literal(str(value))

    if t == 'String':
        return f'"{_java_str_escape(str(value))}"'

    if t.endswith('[]'):
        base = t[:-2]
        if not isinstance(value, list):
            raise ValueError(f"Expected list for {t}, got {type(value).__name__}")
        elems = [_to_java_literal(v, base) for v in value]
        return f"new {base}[]{{{','.join(elems)}}}"

    m = re.match(r'^(List|ArrayList)<([^<>]+)>$', t)
    if m:
        inner = m.group(2)
        if not isinstance(value, list):
            raise ValueError(f"Expected list for {t}")
        if not value:
            return f"new java.util.ArrayList<{inner}>()"
        # Primitive literal form, so autoboxing yields the right wrapper type.
        if inner in _BOX_TO_PRIM:
            prim = _BOX_TO_PRIM[inner]
            elems = [_to_java_literal(v, prim) for v in value]
        else:  # String
            elems = [_to_java_literal(v, inner) for v in value]
        return f"java.util.Arrays.<{inner}>asList({','.join(elems)})"

    raise ValueError(f"Unsupported type for literal conversion: {t}")


def _cast_expr(idx: int, java_type: str) -> str:
    """Expression that casts a[idx] (an Object) to the method-parameter type."""
    t = _normalize_type(java_type)
    if t in _PRIMITIVES:
        box = _PRIM_TO_BOX[t]
        return f"({t})({box})a[{idx}]"
    if t == 'String':
        return f"(String)a[{idx}]"
    if t.endswith('[]'):
        return f"({t})a[{idx}]"
    # List<T> / ArrayList<T>: unchecked cast, suppressed on the harness class
    return f"({t})a[{idx}]"


_HARNESS_TEMPLATE = textwrap.dedent("""\
__HARNESS_IMPORTS__

    @SuppressWarnings({"unchecked", "rawtypes"})
    public class WMVerify {
    __VARIANT_CLASSES__

        static Object[][] cases = new Object[][] {
    __CASES_INIT__
        };

        public static void main(String[] args) {
            int idx = Integer.parseInt(args[0]);
            StringBuilder out = new StringBuilder("[");
            for (int i = 0; i < cases.length; i++) {
                if (i > 0) out.append(",");
                Object[] a = cases[i];
                try {
                    Object r = invoke(idx, a);
                    out.append("{\\"v\\":").append(jsonSerialize(r)).append(",\\"e\\":null}");
                } catch (Throwable t) {
                    Throwable c = t;
                    while (c.getCause() != null && c != c.getCause()) c = c.getCause();
                    out.append("{\\"v\\":null,\\"e\\":\\"").append(c.getClass().getSimpleName()).append("\\"}");
                }
            }
            out.append("]");
            System.out.print(out.toString());
        }

        static Object invoke(int idx, Object[] a) throws Throwable {
            switch (idx) {
    __DISPATCH_CASES__
                default: throw new IllegalArgumentException("Unknown variant: " + idx);
            }
        }

        static String jsonSerialize(Object o) {
            if (o == null) return "null";
            if (o instanceof String) return "\\"" + escape((String) o) + "\\"";
            if (o instanceof Character) return "\\"" + escape(String.valueOf((char)(Character) o)) + "\\"";
            if (o instanceof Boolean) return o.toString();
            if (o instanceof Double || o instanceof Float) {
                double d = ((Number) o).doubleValue();
                if (Double.isNaN(d)) return "\\"NaN\\"";
                if (Double.isInfinite(d)) return d > 0 ? "\\"Infinity\\"" : "\\"-Infinity\\"";
                return o.toString();
            }
            if (o instanceof Number) return o.toString();
            if (o instanceof int[]) {
                int[] arr = (int[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(arr[i]); }
                return sb.append("]").toString();
            }
            if (o instanceof long[]) {
                long[] arr = (long[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(arr[i]); }
                return sb.append("]").toString();
            }
            if (o instanceof double[]) {
                double[] arr = (double[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(jsonSerialize(arr[i])); }
                return sb.append("]").toString();
            }
            if (o instanceof float[]) {
                float[] arr = (float[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(jsonSerialize((double) arr[i])); }
                return sb.append("]").toString();
            }
            if (o instanceof boolean[]) {
                boolean[] arr = (boolean[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(arr[i]); }
                return sb.append("]").toString();
            }
            if (o instanceof char[]) {
                char[] arr = (char[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append("\\"").append(escape(String.valueOf(arr[i]))).append("\\""); }
                return sb.append("]").toString();
            }
            if (o instanceof short[]) {
                short[] arr = (short[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(arr[i]); }
                return sb.append("]").toString();
            }
            if (o instanceof byte[]) {
                byte[] arr = (byte[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(arr[i]); }
                return sb.append("]").toString();
            }
            if (o instanceof Object[]) {
                Object[] arr = (Object[]) o;
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append(","); sb.append(jsonSerialize(arr[i])); }
                return sb.append("]").toString();
            }
            if (o instanceof java.util.Collection) {
                StringBuilder sb = new StringBuilder("[");
                int i = 0;
                for (Object e : (java.util.Collection<?>) o) {
                    if (i > 0) sb.append(",");
                    sb.append(jsonSerialize(e));
                    i++;
                }
                return sb.append("]").toString();
            }
            if (o instanceof java.util.Map) {
                StringBuilder sb = new StringBuilder("{");
                int i = 0;
                for (java.util.Map.Entry<?,?> e : ((java.util.Map<?,?>) o).entrySet()) {
                    if (i > 0) sb.append(",");
                    sb.append("\\"").append(escape(String.valueOf(e.getKey()))).append("\\":");
                    sb.append(jsonSerialize(e.getValue()));
                    i++;
                }
                return sb.append("}").toString();
            }
            return "\\"" + escape(o.toString()) + "\\"";
        }

        static String escape(String s) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == '"') sb.append("\\\\\\"");
                else if (c == '\\\\') sb.append("\\\\\\\\");
                else if (c == '\\n') sb.append("\\\\n");
                else if (c == '\\r') sb.append("\\\\r");
                else if (c == '\\t') sb.append("\\\\t");
                else if (c < 0x20) sb.append(String.format("\\\\u%04x", (int) c));
                else sb.append(c);
            }
            return sb.toString();
        }
    }
""")


def _build_variant_class(class_idx: int, method_code: str) -> str:
    """Wrap a single variant method in an inner static class."""
    body = _force_static(method_code).rstrip()
    indented = '\n'.join('        ' + line if line else line for line in body.splitlines())
    return f"    static class V{class_idx} {{\n{indented}\n    }}"


def _build_cases_init(test_cases: List[list], param_types: List[str]) -> str:
    """Generate new Object[] { ... } initializers, one per test case."""
    lines = []
    for case in test_cases:
        if not isinstance(case, list) or len(case) != len(param_types):
            raise ValueError(f"Case arity mismatch: got {case!r}, expected {len(param_types)} args")
        arg_lits = []
        for v, pt in zip(case, param_types):
            arg_lits.append(_to_java_literal(v, pt))
        lines.append("            new Object[] { " + ", ".join(arg_lits) + " },")
    return "\n".join(lines)


def _build_dispatch_cases(num_variants: int, sig: Dict[str, Any]) -> str:
    """Generate switch cases that invoke Vi.<method>(casts...)."""
    name = sig['name']
    param_types = [t for t, _ in sig['params']]
    casts = ", ".join(_cast_expr(i, pt) for i, pt in enumerate(param_types))
    ret_t = _normalize_type(sig['return_type'])
    lines = []
    for i in range(num_variants):
        call = f"V{i}.{name}({casts})"
        if ret_t == 'void':
            lines.append(f"                case {i}: {call}; return null;")
        else:
            lines.append(f"                case {i}: return {call};")
    return "\n".join(lines)


def _build_test_harness_java(method_variants: List[str], sig: Dict[str, Any],
                             test_cases: List[list]) -> Tuple[str, List[Tuple[int, int]]]:
    """Return (harness_source, variant_line_ranges).
    method_variants[0] is the original. variant_line_ranges[i] is the 1-based (start, end) of Vi,
    used to attribute compile errors.
    """
    variant_classes = "\n\n".join(
        _build_variant_class(i, m) for i, m in enumerate(method_variants)
    )
    param_types = [t for t, _ in sig['params']]
    cases_init = _build_cases_init(test_cases, param_types)
    dispatch = _build_dispatch_cases(len(method_variants), sig)
    imports = "\n".join(f"import {m};" for m in _HARNESS_IMPORTS)
    harness = (_HARNESS_TEMPLATE
               .replace("__HARNESS_IMPORTS__", imports)
               .replace("__VARIANT_CLASSES__", variant_classes)
               .replace("__CASES_INIT__", cases_init)
               .replace("__DISPATCH_CASES__", dispatch))

    ranges = _find_variant_line_ranges(harness, len(method_variants))
    return harness, ranges


def _find_variant_line_ranges(harness: str, num_variants: int) -> List[Tuple[int, int]]:
    """1-based [start, end] line range of each static class Vi block."""
    lines = harness.split('\n')
    ranges = [(-1, -1)] * num_variants
    for vi in range(num_variants):
        marker = f"static class V{vi} {{"
        start_line = None
        start_col = 0
        for i, line in enumerate(lines, start=1):
            if marker in line:
                start_line = i
                start_col = len(line) - len(line.lstrip())
                break
        if start_line is None:
            continue
        end_line = start_line
        for j in range(start_line, len(lines)):
            line = lines[j]
            stripped = line.rstrip()
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            if indent == start_col and stripped.lstrip().startswith('}'):
                end_line = j + 1
                break
        ranges[vi] = (start_line, end_line)
    return ranges


def _attribute_compile_errors(stderr: str, ranges: List[Tuple[int, int]]) -> set:
    """Variant indices whose line range holds a WMVerify.java:<line>: error in javac stderr."""
    bad = set()
    for m in re.finditer(r'WMVerify\.java:(\d+):', stderr):
        line = int(m.group(1))
        for vi, (s, e) in enumerate(ranges):
            if s <= line <= e:
                bad.add(vi)
                break
    return bad


def _parse_results(stdout: str) -> Optional[List[dict]]:
    try:
        results = json.loads(stdout)
        if isinstance(results, list):
            return results
    except (json.JSONDecodeError, TypeError):
        pass
    return None


class EquivalenceVerifier_Java:
    def __init__(self, provider, timeout: float = 8.0, compile_timeout: float = 30.0,
                 *, verbose: bool = True):
        self.provider = provider
        self.verbose = verbose
        self.base_timeout = timeout
        self.compile_timeout = compile_timeout

        self.tmp_dir = TMP_PATH()

        self.parser = JavaParser()

        self.stats = {
            "total_original": 0, "total_variants": 0,
            "passed": 0,
            "failed_syntax": 0, "failed_compile": 0, "failed_exec": 0,
            "failed_value": 0, "failed_exception": 0, "failed_timeout": 0,
            "failed_signature": 0, "failed_unknown": 0,
            "baseline_failed": 0,
            "time_baseline": 0.0, "time_compile": 0.0, "time_verify": 0.0,
            "total_test_generated": 0, "time_test_gen": 0.0,
        }

    def generate_test_inputs(self, code: str, ninput: int = 10) -> Optional[str]:
        """Test inputs as a JSON string of argument lists, or None on failure."""
        sig = _parse_method_signature(code)
        if sig is None:
            if self.verbose:
                print("[!] Java test input gen: could not parse signature")
            return None

        sig_str = self._format_signature_for_prompt(sig)
        type_hints = "\n".join(
            f"  - {name} ({ptype}): {self._json_hint_for_type(ptype)}"
            for ptype, name in sig['params']
        )

        system_prompt = textwrap.dedent("""
            You are a QA engineer who outputs ONLY valid JSON arrays of test inputs.

            Rules:
            - No comments, no explanations, no markdown fences.
            - Output must be a single JSON array.
            - Each element is an array of positional arguments for one method call.
            - Types MUST match the method signature exactly.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            Generate {ninput} diverse JSON test inputs for this Java method.

            Method signature:
            {sig_str}

            Parameter types (follow these strictly):
            {type_hints}

            FORMAT:
            [ [arg0, arg1, ...], [arg0, arg1, ...], ... ]

            RULES:
            - Use ONLY JSON literals (numbers, strings, booleans, arrays, null).
            - `char` parameters → JSON string of length 1 (e.g. "a").
            - `long`/`short`/`byte`/`int` → JSON integer.
            - `double`/`float` → JSON number (decimal).
            - Arrays of primitives → JSON array of matching literals.
            - List<T> → JSON array of T literals.
            - Include edge cases: empty inputs, single-element inputs, boundary values.
            - No comments, no explanations, no markdown fences.

            <code>
            {code}
            </code>
        """).strip()

        start_t = time.perf_counter()
        raw_text, parse_err = None, None
        for attempt in range(2):
            try:
                raw_text = self.provider.generate(system_prompt, user_prompt)
            except Exception as e:
                if self.verbose:
                    print(f"[!] Java test-gen LLM error: {e}")
                time.sleep(1.0)
                continue
            test_json, parse_err = self._parse_test_inputs(raw_text, sig)
            if test_json is not None:
                self.stats["time_test_gen"] += time.perf_counter() - start_t
                return test_json
            user_prompt = (
                f"Your previous output was invalid: {parse_err}\n"
                f"Return ONLY a valid JSON array of argument arrays for the method below.\n\n"
                f"Method signature:\n{sig_str}\n\n<code>\n{code}\n</code>"
            )

        self.stats["time_test_gen"] += time.perf_counter() - start_t
        if self.verbose and parse_err:
            print(f"[!] Java test input parsing error: {parse_err}")
        return None

    def check_equivalence(self, original_code: str, variants: List[str],
                          test_cases_json: str) -> List[bool]:
        """Return per-variant bool list. True = behaviorally equivalent to original."""
        VERIFIER_LOGGING = True
        log_path = OUT_PATH("equivalence_test_results_java.log")

        self.stats["total_original"] += 1
        self.stats["total_variants"] += len(variants)

        verify_start_t = time.perf_counter()
        final = [False] * len(variants)

        if not variants or not test_cases_json:
            return final

        sig = _parse_method_signature(original_code)
        if sig is None:
            if self.verbose:
                print("[!] Original signature parse failed")
            self.stats["baseline_failed"] += len(variants)
            return final

        try:
            test_cases = json.loads(test_cases_json)
            if not isinstance(test_cases, list) or not test_cases:
                raise ValueError("empty test cases")
        except Exception as e:
            if self.verbose:
                print(f"[!] Test cases JSON parse failed: {e}")
            self.stats["baseline_failed"] += len(variants)
            return final

        # Pre-filter variants by syntax + signature match
        kept: List[Tuple[int, str]] = []  # (original_idx, code)
        detailed: List[Tuple[bool, Optional[str]]] = [(False, None)] * len(variants)
        for i, v in enumerate(variants):
            if not self._syntax_ok(v):
                detailed[i] = (False, "SyntaxError")
                continue
            if not _signature_matches(v, sig):
                detailed[i] = (False, "SignatureMismatch")
                continue
            kept.append((i, v))

        if not self._syntax_ok(original_code):
            if self.verbose:
                print("[!] Original code syntax error")
            self.stats["baseline_failed"] += len(variants)
            return final

        # Build and compile the harness, retrying after dropping variants that javac rejects.
        active = list(kept)  # (orig_idx, code) candidates for the harness
        runs = None
        compile_err = None
        for attempt in range(3):
            method_variants = [original_code] + [v for _, v in active]
            try:
                harness, ranges = _build_test_harness_java(method_variants, sig, test_cases)
            except Exception as e:
                if self.verbose:
                    print(f"[!] Harness build failed: {e}")
                self.stats["baseline_failed"] += len(variants)
                return final

            runs, compile_err, err_stderr = self._compile_and_run_all(
                harness, len(method_variants)
            )
            if runs is not None:
                break
            # Compile failed: identify bad variants by error line
            bad_harness_indices = _attribute_compile_errors(err_stderr or "", ranges)
            if 0 in bad_harness_indices or not bad_harness_indices:
                # Baseline broken or errors not attributable
                break
            # Drop bad variants (harness_idx -> active_idx = harness_idx - 1)
            drop_active = sorted((hi - 1 for hi in bad_harness_indices), reverse=True)
            for ai in drop_active:
                orig_idx, _ = active.pop(ai)
                detailed[orig_idx] = (False, "CompileError")
                self.stats["failed_compile"] += 1
            if not active:
                # Nothing left to check, still run the baseline alone
                continue

        if runs is None:
            if self.verbose:
                print(f"[!] Java harness compile failed (after retries): {(compile_err or '')[:300]}")
            for orig_idx, _ in active:
                detailed[orig_idx] = (False, "CompileError")
                self.stats["failed_compile"] += 1
            self._log_and_finalize(original_code, variants, detailed, None, log_path,
                                   VERIFIER_LOGGING, compile_err=compile_err)
            self.stats["time_verify"] += time.perf_counter() - verify_start_t
            return final

        # After successful compile, active is the list of variants that made it
        # into the harness (harness idx 1..len(active)).
        kept = active

        # Baseline check (run idx 0)
        baseline = runs[0]
        self.stats["time_baseline"] += baseline["duration"]
        if not baseline["success"]:
            if self.verbose:
                print(f"[!] Baseline run failed: {baseline['error']}")
            self.stats["baseline_failed"] += len(variants)
            self._log_and_finalize(original_code, variants, detailed, None, log_path,
                                   VERIFIER_LOGGING, baseline_err=baseline["error"])
            self.stats["time_verify"] += time.perf_counter() - verify_start_t
            return final

        expected = baseline["results"]
        if all(r.get("e") is not None for r in expected):
            if self.verbose:
                print("[!] All baseline cases threw exceptions")
            self.stats["baseline_failed"] += len(variants)
            self._log_and_finalize(original_code, variants, detailed, expected, log_path,
                                   VERIFIER_LOGGING, baseline_err="AllExceptions")
            self.stats["time_verify"] += time.perf_counter() - verify_start_t
            return final

        for harness_idx, (orig_idx, _) in enumerate(kept, start=1):
            vr = runs[harness_idx]
            if not vr["success"]:
                detailed[orig_idx] = (False, vr["error"])
                continue
            actual = vr["results"]
            mismatch = self._compare_results(expected, actual)
            if mismatch is None:
                detailed[orig_idx] = (True, None)
                final[orig_idx] = True
            else:
                detailed[orig_idx] = (False, mismatch)

        for ok, err in detailed:
            if ok:
                self.stats["passed"] += 1
            elif err == "SyntaxError":
                self.stats["failed_syntax"] += 1
            elif err == "SignatureMismatch":
                self.stats["failed_signature"] += 1
            elif err == "CompileError":
                pass  # already counted
            elif err and "Timeout" in err:
                self.stats["failed_timeout"] += 1
            elif err and "ValueMismatch" in err:
                self.stats["failed_value"] += 1
            elif err and "ExceptionMismatch" in err:
                self.stats["failed_exception"] += 1
            elif err and "RuntimeError" in err:
                self.stats["failed_exec"] += 1
            elif err:
                self.stats["failed_unknown"] += 1

        self._log_and_finalize(original_code, variants, detailed, expected, log_path,
                               VERIFIER_LOGGING)
        self.stats["time_verify"] += time.perf_counter() - verify_start_t

        if self.verbose:
            print(f"[*] Java verification: {sum(final)}/{len(variants)} passed.")
        return final

    @staticmethod
    def check_verifiable(code: str, enclosing_class: Optional[str] = None
                         ) -> Tuple[bool, List[str]]:
        return check_verifiable_java(code, enclosing_class)

    def _syntax_ok(self, method_code: str) -> bool:
        wrapped = f"class __W {{\n{method_code}\n}}"
        ok, _ = self.parser.check_syntax(wrapped)
        return ok

    def _compile_and_run_all(self, harness_code: str, num_variants: int):
        """Compile once and run every variant.
        Returns (runs, None, None), or (None, err_summary, stderr) on compile failure.
        """
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmp:
            java_file = os.path.join(tmp, "WMVerify.java")
            with open(java_file, 'w', encoding='utf-8') as f:
                f.write(harness_code)

            compile_t = time.perf_counter()
            try:
                cr = subprocess.run(
                    ["javac", "-encoding", "UTF-8", java_file],
                    capture_output=True, timeout=self.compile_timeout, cwd=tmp
                )
            except subprocess.TimeoutExpired:
                self.stats["time_compile"] += time.perf_counter() - compile_t
                return None, "CompileTimeout", ""
            self.stats["time_compile"] += time.perf_counter() - compile_t

            if cr.returncode != 0:
                err = cr.stderr.decode('utf-8', errors='replace').strip()
                return None, err, err

            runs = []
            for idx in range(num_variants):
                start_t = time.perf_counter()
                try:
                    rr = subprocess.run(
                        ["java", "-Dfile.encoding=UTF-8", "WMVerify", str(idx)],
                        capture_output=True, timeout=self.base_timeout, cwd=tmp
                    )
                    duration = time.perf_counter() - start_t
                except subprocess.TimeoutExpired:
                    runs.append({"success": False, "results": None,
                                 "error": "Timeout", "duration": self.base_timeout})
                    continue

                if rr.returncode != 0:
                    err = rr.stderr.decode('utf-8', errors='replace').strip()
                    runs.append({"success": False, "results": None,
                                 "error": f"RuntimeError: {err[:200]}",
                                 "duration": duration})
                    continue

                stdout = rr.stdout.decode('utf-8', errors='replace').strip()
                parsed = _parse_results(stdout)
                if parsed is None:
                    runs.append({"success": False, "results": None,
                                 "error": f"OutputParseError: {stdout[:200]}",
                                 "duration": duration})
                else:
                    runs.append({"success": True, "results": parsed,
                                 "error": None, "duration": duration})
            return runs, None, None

    def _compare_results(self, expected: List[dict], actual: List[dict]) -> Optional[str]:
        if len(actual) != len(expected):
            return f"ResultCountMismatch: expected {len(expected)}, got {len(actual)}"
        for i, (exp, act) in enumerate(zip(expected, actual)):
            exp_err = exp.get("e")
            act_err = act.get("e")
            if exp_err != act_err:
                return f"ExceptionMismatch at case {i}: expected {exp_err}, got {act_err}"
            if exp_err is None and exp.get("v") != act.get("v"):
                return f"ValueMismatch at case {i}: expected {exp.get('v')}, got {act.get('v')}"
        return None

    def _format_signature_for_prompt(self, sig: Dict[str, Any]) -> str:
        params = ", ".join(f"{t} {n}" for t, n in sig['params'])
        return f"{sig['return_type']} {sig['name']}({params})"

    def _json_hint_for_type(self, java_type: str) -> str:
        t = _normalize_type(java_type)
        if t in _BOX_TO_PRIM:
            t = _BOX_TO_PRIM[t]
        if t == 'char':
            return "JSON string of length 1"
        if t == 'String':
            return "JSON string"
        if t == 'boolean':
            return "JSON boolean"
        if t in ('int', 'long', 'short', 'byte'):
            return "JSON integer"
        if t in ('double', 'float'):
            return "JSON number"
        if t.endswith('[]'):
            return f"JSON array of {self._json_hint_for_type(t[:-2])}"
        m = re.match(r'^(List|ArrayList)<(.+)>$', t)
        if m:
            inner = m.group(2)
            if inner in _BOX_TO_PRIM:
                return f"JSON array of {self._json_hint_for_type(_BOX_TO_PRIM[inner])}"
            return f"JSON array of {self._json_hint_for_type(inner)}"
        return "JSON value matching the type"

    def _parse_test_inputs(self, raw_text: str, sig: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Parse LLM output into a JSON string of argument lists, checking each converts to Java literals."""
        if not raw_text:
            return None, "empty response"
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        start = clean.find('[')
        end = clean.rfind(']')
        if start < 0 or end < 0 or end <= start:
            return None, "no JSON array found"
        snippet = clean[start:end + 1]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"
        if not isinstance(parsed, list) or not parsed:
            return None, "expected non-empty array"
        param_types = [t for t, _ in sig['params']]
        valid = []
        for case in parsed:
            if not isinstance(case, list) or len(case) != len(param_types):
                continue
            try:
                for v, pt in zip(case, param_types):
                    _to_java_literal(v, pt)  # dry-run to validate
                valid.append(case)
            except Exception:
                continue
        if not valid:
            return None, "no cases matched parameter types"
        self.stats["total_test_generated"] += len(valid)
        return json.dumps(valid), None

    def _log_and_finalize(self, original_code, variants, detailed, expected,
                          log_path, enabled, compile_err=None, baseline_err=None):
        if not enabled:
            return
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[Java Verifier log]\n")
            f.write(f"Timestamp     : {time.ctime()}\n")
            if compile_err:
                f.write(f"Compile error : {compile_err[:300]}\n")
            if baseline_err:
                f.write(f"Baseline err  : {baseline_err}\n")
            passed = sum(1 for ok, _ in detailed if ok)
            f.write(f"Passed        : {passed}/{len(variants)}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Original code:\n{original_code.strip()}\n\n")
            for idx, (v_code, (ok, err)) in enumerate(zip(variants, detailed)):
                status = "PASS" if ok else f"FAIL ({err})"
                f.write(f"VARIANT #{idx:02} {status}\n")
                f.write(f"{v_code.strip()}\n")
                f.write("-" * 30 + "\n")
            f.write("=" * 80 + "\n\n")


def _test_verifier():
    provider = get_provider()
    verifier = EquivalenceVerifier_Java(provider, verbose=True)

    # Case 1: simple int add
    code1 = textwrap.dedent("""\
        public int add(int a, int b) {
            return a + b;
        }
    """).strip()

    variant1_ok = textwrap.dedent("""\
        public int add(int a, int b) {
            int result = a + b;
            return result;
        }
    """).strip()

    variant1_bad = textwrap.dedent("""\
        public int add(int a, int b) {
            return a - b;
        }
    """).strip()

    variant1_syntax = "public int add(int a, int b { return a + b; }"

    variant1_loop = textwrap.dedent("""\
        public int add(int a, int b) {
            while (true) {}
            return a + b;
        }
    """).strip()

    print("\n" + "=" * 60)
    print(" Test 1: add(int,int)")
    print("=" * 60)

    tjson = verifier.generate_test_inputs(code1)
    print(f"Test inputs: {tjson}")
    if tjson:
        res = verifier.check_equivalence(code1, [variant1_ok, variant1_bad, variant1_syntax, variant1_loop], tjson)
        for label, ok in zip(["correct", "wrong value", "syntax error", "infinite loop"], res):
            print(f"  {label:20s}: {'PASS' if ok else 'FAIL'}")

    # Case 2: List<Double> input
    code2 = textwrap.dedent("""\
        public boolean hasCloseElements(List<Double> numbers, double threshold) {
            for (int i = 0; i < numbers.size(); i++) {
                for (int j = i + 1; j < numbers.size(); j++) {
                    double distance = Math.abs(numbers.get(i) - numbers.get(j));
                    if (distance < threshold) return true;
                }
            }
            return false;
        }
    """).strip()

    variant2_ok = textwrap.dedent("""\
        public boolean hasCloseElements(List<Double> numbers, double threshold) {
            int n = numbers.size();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    if (Math.abs(numbers.get(i) - numbers.get(j)) < threshold) return true;
                }
            }
            return false;
        }
    """).strip()

    variant2_bad = textwrap.dedent("""\
        public boolean hasCloseElements(List<Double> numbers, double threshold) {
            return false;
        }
    """).strip()

    print("\n" + "=" * 60)
    print(" Test 2: hasCloseElements(List<Double>, double)")
    print("=" * 60)

    tjson2 = verifier.generate_test_inputs(code2)
    print(f"Test inputs: {tjson2}")
    if tjson2:
        res2 = verifier.check_equivalence(code2, [variant2_ok, variant2_bad], tjson2)
        for label, ok in zip(["correct (full scan)", "wrong (always false)"], res2):
            print(f"  {label:25s}: {'PASS' if ok else 'FAIL'}")

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
