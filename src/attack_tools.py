import os
import subprocess
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from paths import CACHE_PATH, _PROJECT_ROOT


# A task the tool refuses comes back untouched and is counted here.
_STATS_LOCK = threading.Lock()

# Per task, shared by MINI and OBFU.
MINIFY_ATTACK_STATS = {"attempted": 0, "failed": 0}

# units are top-level types, not tasks. no_type counts samples with nothing to compile.
DECOMPILE_ATTACK_STATS = {"units": 0, "failed": 0, "no_type": 0}


def _bump(stats: dict, key: str, n: int = 1) -> None:
    with _STATS_LOCK:
        stats[key] += n


def _reset(stats: dict) -> None:
    with _STATS_LOCK:
        for key in stats:
            stats[key] = 0


def _snapshot(stats: dict) -> dict:
    with _STATS_LOCK:
        return dict(stats)


def reset_tool_attack_stats() -> None:
    """Zero every external-tool counter. Call once before a batch."""
    _reset(MINIFY_ATTACK_STATS)
    _reset(DECOMPILE_ATTACK_STATS)


def get_tool_attack_stats() -> dict:
    """Every external-tool counter. Failed tasks came back unchanged, so subtract them
    before reading a detection rate as robustness."""
    return {"minify": _snapshot(MINIFY_ATTACK_STATS),
            "decompile": _snapshot(DECOMPILE_ATTACK_STATS)}


def _bump_minify_stat(key: str) -> None:
    _bump(MINIFY_ATTACK_STATS, key)


def _bump_decompile_stat(key: str, n: int = 1) -> None:
    _bump(DECOMPILE_ATTACK_STATS, key, n)


_npm_lock = threading.Lock()


_OBFUSCATOR_SEED = 0


def _ensure_node_package(package: str) -> None:
    target = os.path.join(_PROJECT_ROOT, "node_modules", package)
    if os.path.exists(target):
        return
    with _npm_lock:
        if os.path.exists(target):
            return
        print(f"[*] npm install {package} at '{_PROJECT_ROOT}'")
        try:
            subprocess.run(["npm", "install", "--no-audit", "--no-fund", package],
                           cwd=_PROJECT_ROOT, capture_output=True, check=True)
        except Exception as e:
            raise RuntimeError(f"npm install {package} failed: {e}") from e


def _run_node_transform(script: str, code: str, package: str, timeout: float = 60.0) -> str:
    """Feed code to a node script on stdin and return its stdout.
    No temp files, so concurrent workers cannot collide on a filename.
    """
    _ensure_node_package(package)
    try:
        proc = subprocess.run(["node", "-e", script], input=code.encode("utf-8"),
                              capture_output=True, cwd=_PROJECT_ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{package} timed out after {timeout}s")
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{package} failed: {err[:300]}")
    return proc.stdout.decode("utf-8", errors="replace")


_TERSER_SCRIPT = """
const {minify} = require('terser');
let src = '';
process.stdin.on('data', d => src += d).on('end', async () => {
  try {
    const out = await minify(src, {
      compress: __COMPRESS__,
      mangle: __MANGLE__,
      // The only non-default. terser keeps license comments otherwise.
      format: {comments: false},
    });
    if (!out || typeof out.code !== 'string') throw new Error('no output');
    process.stdout.write(out.code);
  } catch (e) {
    process.stderr.write(String(e && e.message || e));
    process.exit(1);
  }
});
"""

_OBFUSCATOR_SCRIPT = """
const O = require('javascript-obfuscator');
let src = '';
process.stdin.on('data', d => src += d).on('end', () => {
  try {
    const r = O.obfuscate(src, {
      seed: __SEED__,
      compact: true,
      controlFlowFlattening: true,
      controlFlowFlatteningThreshold: 1,
      numbersToExpressions: true,
      simplify: true,
      stringArray: true,
      stringArrayThreshold: 1,
      transformObjectKeys: true,
      identifierNamesGenerator: 'hexadecimal',
      renameGlobals: true,
      // These break the program when reformatted or run outside a browser.
      selfDefending: false,
      debugProtection: false,
      disableConsoleOutput: false,
    });
    process.stdout.write(r.getObfuscatedCode());
  } catch (e) {
    process.stderr.write(String(e && e.message || e));
    process.exit(1);
  }
});
"""

# python-minifier enables most transforms by default, so every pass starts from all off.
_PY_MINIFY_OFF = {
    "remove_annotations": False,
    "remove_pass": False,
    "remove_literal_statements": False,
    "combine_imports": False,
    "hoist_literals": False,
    "rename_locals": False,
    "rename_globals": False,
    "remove_object_base": False,
    "convert_posargs_to_args": False,
    "remove_asserts": False,
    "remove_debug": False,
    "remove_explicit_return_none": False,
    "remove_builtin_exception_brackets": False,
    "constant_folding": False,
}

# python-minifier has no mangle/compress split, so the pass is defined here.
# Locals only, matching terser's default.
_PY_MINIFY_MANGLE = {
    "rename_locals": True,
}

# The tool's defaults. remove_asserts, remove_debug and remove_literal_statements
# stay off because they change runtime behavior.
_PY_MINIFY_COMPRESS = {
    "remove_annotations": True,
    "remove_pass": True,
    "combine_imports": True,
    "hoist_literals": True,
    "remove_object_base": True,
    "convert_posargs_to_args": True,
    "remove_explicit_return_none": True,
    "remove_builtin_exception_brackets": True,
    "constant_folding": True,
}


def _minify_py(code: str, *, mangle: bool, compress: bool) -> str:
    """Raw python-minifier call. Raises RuntimeError like _run_node_transform."""
    try:
        import python_minifier
    except ImportError as e:
        raise RuntimeError(
            "python-minifier is not installed (pip install python-minifier)"
        ) from e

    options = dict(_PY_MINIFY_OFF)
    if mangle:
        options.update(_PY_MINIFY_MANGLE)
    if compress:
        options.update(_PY_MINIFY_COMPRESS)

    try:
        return python_minifier.minify(code, **options)
    except Exception as e:
        raise RuntimeError(f"python-minifier failed: {str(e)[:300]}") from e


def attack_minify(original_code: str, lang: str = "js", *,
                  mangle: bool = True, compress: bool = True,
                  fmt: bool = False, verbose: bool = False) -> str:
    """Run the language's minifier with any of its three passes enabled.
    F is the AST reprint alone (only comments and layout go).
    M and C both include that reprint, so F is meaningful only on its own.
    """
    if lang not in ("js", "py"):
        raise ValueError(
            f"the minifier attack supports 'js' and 'py', got lang='{lang}'. "
            "Java's minifiers (ProGuard/R8) work on bytecode, not the source "
            "SWIFT watermarks."
        )
    if not (mangle or compress or fmt):
        return original_code

    tool = "python-minifier" if lang == "py" else "terser"
    _bump_minify_stat("attempted")
    try:
        if lang == "py":
            out = _minify_py(original_code, mangle=mangle, compress=compress)
        else:
            script = (_TERSER_SCRIPT
                      .replace("__COMPRESS__", "true" if compress else "false")
                      .replace("__MANGLE__", "true" if mangle else "false"))
            out = _run_node_transform(script, original_code, "terser")
    except RuntimeError as e:
        _bump_minify_stat("failed")
        print(f"[!] {tool} skipped, code returned unchanged: {str(e)[:120]}")
        return original_code
    if verbose:
        print(f"[*] {tool}(mangle={mangle}, compress={compress}, fmt={fmt}): "
              f"{len(original_code)} -> {len(out)} chars")
    return out


def attack_obfuscate(original_code: str, lang: str = "js", *,
                     seed: int = _OBFUSCATOR_SEED, verbose: bool = False) -> str:
    """Run javascript-obfuscator over the code."""
    if lang != "js":
        raise ValueError(f"the obfuscator attack is JavaScript-only, got lang='{lang}'.")
    script = _OBFUSCATOR_SCRIPT.replace("__SEED__", str(seed))
    _bump_minify_stat("attempted")
    try:
        out = _run_node_transform(script, original_code, "javascript-obfuscator")
    except RuntimeError as e:
        _bump_minify_stat("failed")
        print(f"[!] obfuscation skipped, code returned unchanged: {str(e)[:120]}")
        return original_code
    if verbose:
        print(f"[*] obfuscate(seed={seed}): {len(original_code)} -> {len(out)} chars")
    return out


# Decompile round trip. Each top-level type is compiled on its own with the file's imports,
# so a class javac rejects costs only itself. Imports from the decompiler output are merged
# into the file's import block.

# CFR and Vineflower (Fernflower lineage), both pinned jars from Maven Central.
_DECOMPILERS = {
    "cfr": {
        "version": "0.152",
        "url": "https://repo1.maven.org/maven2/org/benf/cfr/{v}/cfr-{v}.jar",
        "jar": "cfr-{v}.jar",
    },
    "vineflower": {
        "version": "1.12.0",
        "url": ("https://repo1.maven.org/maven2/org/vineflower/vineflower/"
                "{v}/vineflower-{v}.jar"),
        "jar": "vineflower-{v}.jar",
    },
}
DEFAULT_DECOMPILER = "cfr"

# Testsets whose samples are complete type declarations.
SUPPORTED_DATASETS = ("humaneval-x", "MBXP")

# -g:none drops local names as in a shipped jar, -g keeps them.
_JAVAC_DEBUG = {False: "-g:none", True: "-g"}

# Output tidying only. Every transform option stays at the tool's default.
_CFR_FLAGS = ["--silent", "true", "--comments", "false"]
_VINEFLOWER_FLAGS = ["--log-level=ERROR"]

_JAVA_LANGUAGE = Language(tsjava.language())
_thread_local = threading.local()
_jar_lock = threading.Lock()


def _parser() -> Parser:
    """One tree-sitter Parser per thread, since Parser is not thread-safe."""
    p = getattr(_thread_local, "parser", None)
    if p is None:
        p = _thread_local.parser = Parser(_JAVA_LANGUAGE)
    return p


def _ensure_jar(decompiler: str) -> str:
    """Download the pinned decompiler jar into cache/ if it is not there yet."""
    spec = _DECOMPILERS[decompiler]
    v = spec["version"]
    path = CACHE_PATH(spec["jar"].format(v=v))
    if os.path.exists(path):
        return path
    with _jar_lock:
        if os.path.exists(path):
            return path
        print(f"[*] downloading {decompiler} {v} to '{path}'")
        try:
            urllib.request.urlretrieve(spec["url"].format(v=v), path)
        except Exception as e:
            raise RuntimeError(f"{decompiler} download failed: {e}") from e
    return path


# Type declarations only. A bare method_declaration also parses at top level, but javac rejects it.
_TYPE_DECLARATIONS = frozenset({
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
})


def _top_level_types(root):
    return [c for c in root.children if c.type in _TYPE_DECLARATIONS]


def _split_preamble(java_source: str) -> tuple[list[str], str]:
    """Split decompiler output into (import lines, the type declarations).
    Everything before the first type (banner, package, imports) is dropped from the body
    so the text can be spliced mid-file.
    """
    root = _parser().parse(bytes(java_source, "utf8")).root_node
    imports = [c.text.decode("utf8").strip() for c in root.children
               if c.type == "import_declaration"]
    types = _top_level_types(root)
    if not types:
        return imports, java_source.strip()
    body = java_source[types[0].start_byte:types[-1].end_byte]
    return imports, body


def _decompile_cfr(jar: str, classes: list, wd: str) -> str | None:
    """CFR prints to stdout. wd is unused and kept for a shared signature."""
    out = subprocess.run(["java", "-jar", jar] + classes + _CFR_FLAGS,
                         capture_output=True, text=True, timeout=120)
    return out.stdout if out.returncode == 0 and out.stdout.strip() else None


def _decompile_vineflower(jar: str, classes: list, wd: str) -> str | None:
    """Vineflower writes .java files into a destination directory instead."""
    dest = os.path.join(wd, "vf")
    os.makedirs(dest, exist_ok=True)
    out = subprocess.run(["java", "-jar", jar] + _VINEFLOWER_FLAGS + classes + [dest],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return None
    produced = sorted(x for x in os.listdir(dest) if x.endswith(".java"))
    if not produced:
        return None
    texts = []
    for name in produced:
        with open(os.path.join(dest, name), encoding="utf-8") as f:
            texts.append(f.read())
    joined = "\n\n".join(texts)
    return joined if joined.strip() else None


_DECOMPILE_FN = {"cfr": _decompile_cfr, "vineflower": _decompile_vineflower}


def _compile_decompile(source: str, class_name: str, jar: str,
                       decompiler: str = DEFAULT_DECOMPILER,
                       debug_info: bool = False) -> str | None:
    """javac -> decompiler for one self-contained type.
    None if either step refuses, and the caller keeps the original.
    """
    with tempfile.TemporaryDirectory() as wd:
        java_file = os.path.join(wd, class_name + ".java")
        with open(java_file, "w", encoding="utf-8") as f:
            f.write(source)
        compiled = subprocess.run(
            ["javac", _JAVAC_DEBUG[debug_info], "-nowarn", "-d", wd, java_file],
            capture_output=True, timeout=120)
        if compiled.returncode != 0:
            return None
        # Inner and anonymous classes have their own .class files, and the decompiler needs all of them.
        classes = [os.path.join(wd, x) for x in sorted(os.listdir(wd))
                   if x.endswith(".class")]
        if not classes:
            return None
        return _DECOMPILE_FN[decompiler](jar, classes, wd)


def _import_insert_offset(root) -> int:
    """Byte offset for new imports: after the last import, else after the package
    declaration, else the top of the file."""
    for kind in ("import_declaration", "package_declaration"):
        nodes = [c for c in root.children if c.type == kind]
        if nodes:
            return nodes[-1].end_byte
    return 0


def attack_decompile(original_code: str, lang: str = "java", *,
                     dataset: str | None = None,
                     decompiler: str = DEFAULT_DECOMPILER,
                     debug_info: bool = False,
                     n_workers: int = 8, verbose: bool = False) -> str:
    """Compile each top-level type on its own, decompile it, splice it back."""
    if lang != "java":
        raise ValueError(f"the decompile attack is Java-only, got lang='{lang}'.")
    if decompiler not in _DECOMPILERS:
        raise ValueError(f"unknown decompiler '{decompiler}'. "
                         f"Available: {', '.join(_DECOMPILERS)}.")
    if dataset is not None and dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"the decompile attack needs a testset of complete types, got "
            f"dataset='{dataset}'. Supported: {', '.join(SUPPORTED_DATASETS)}.")

    src_bytes = bytes(original_code, "utf8")
    root = _parser().parse(src_bytes).root_node
    units = _top_level_types(root)
    if not units:
        # Bare methods, nothing to compile.
        _bump_decompile_stat("no_type")
        return original_code

    own_imports = [c.text.decode("utf8").strip() for c in root.children
                   if c.type == "import_declaration"]
    imports_block = "\n".join(own_imports)
    insert_at = _import_insert_offset(root)
    jar = _ensure_jar(decompiler)

    def one(unit):
        name_node = unit.child_by_field_name("name")
        if name_node is None:
            return None
        source = imports_block + "\n\n" + unit.text.decode("utf8")
        return _compile_decompile(source, name_node.text.decode("utf8"), jar,
                                  decompiler, debug_info)

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        results = list(pool.map(one, units))

    _bump_decompile_stat("units", len(units))
    out = bytearray(src_bytes)
    new_imports = []
    # Back to front so the byte offsets of the units still ahead stay valid.
    for unit, decompiled in zip(reversed(units), reversed(results)):
        if decompiled is None:
            _bump_decompile_stat("failed")
            continue
        imports, body = _split_preamble(decompiled)
        new_imports.extend(i for i in imports if i not in own_imports)
        out[unit.start_byte:unit.end_byte] = bytes(body.strip(), "utf8")

    # The insert point is ahead of every unit, so splicing it last keeps the offsets above valid.
    extra = sorted(set(new_imports))
    if extra:
        text = ("\n" if insert_at else "") + "\n".join(extra)
        out[insert_at:insert_at] = bytes(text, "utf8")

    result = out.decode("utf8")
    if verbose:
        done = sum(r is not None for r in results)
        print(f"[*] {decompiler}: {done}/{len(units)} types, +{len(extra)} imports, "
              f"{len(original_code)} -> {len(result)} chars")
    return result


def _test_mini_obfu():
    from language_parser import JSParser, PythonParser

    js_code = "function add(a, b) {\n  // a comment\n  let c = (a) + b;\n  return c;\n}\n"
    py_code = "def add(a, b):\n    c = a + b\n    return c\n"

    print("=== MINI ===")
    for lang, src, parser in (("js", js_code, JSParser()), ("py", py_code, PythonParser())):
        for passes in ("F", "M", "C", "MC"):
            out = attack_minify(src, lang, mangle="M" in passes,
                                compress="C" in passes, fmt="F" in passes)
            ok, _ = parser.check_syntax(out)
            print(f"  {lang} MINI-{passes:2s} syntax_ok={ok}  {len(src)} -> {len(out)} chars")
            assert ok, out

    print("=== OBFU ===")
    out = attack_obfuscate(js_code)
    ok, _ = JSParser().check_syntax(out)
    print(f"  js OBFU     syntax_ok={ok}  {len(js_code)} -> {len(out)} chars")
    assert ok, out


def _test_dcmp():
    from language_parser import JavaParser
    from testset import load_testset
    from watermarker import Watermarker

    print("\n=== DCMP shape checks ===")
    two_types = ('import java.util.*;\n\n'
                 'class A {\n  public static List<String> f(String s) {\n'
                 '    List<String> r = new ArrayList<>();\n    r.add(s);\n    return r;\n  }\n}\n\n'
                 'class B {\n  public static int g(int x) { return x + 1; }\n}\n')
    for name in _DECOMPILERS:
        out = attack_decompile(two_types, "java", dataset="MBXP", decompiler=name, verbose=True)
        ok, msg = JavaParser().check_syntax(out)
        print(f"  {name}: syntax_ok={ok} {msg}")
        assert ok, out
        body = out[out.index("class A"):]
        assert "import " not in body, f"import spliced between types:\n{out}"

    print("\n=== DCMP detection ===")
    wm = Watermarker("java")
    reset_tool_attack_stats()
    ts = load_testset("MBXP", lang="java", nsamples=80, fold=10, validate=True)

    det = plain_det = n = 0
    for t in ts:
        for tag, code in (("wm", wm.embed(t["code"], True, False)), ("plain", t["code"])):
            out = attack_decompile(code, "java", dataset="MBXP")
            r = wm.detect(out, True, False)
            p = wm.calculate_l1_pvalue(r["l1_n"], r["l1_s"])
            hit = p is not None and p < 0.05
            if tag == "wm":
                det += hit
                n += 1
            else:
                plain_det += hit
    print(f"[decompile] tasks={n}  TPR@.05={100*det/n:.1f}%  FPR@.05={100*plain_det/n:.1f}%")
    print(f"[decompile] stats={get_tool_attack_stats()}")


if __name__ == "__main__":
    _test_mini_obfu()
    _test_dcmp()
