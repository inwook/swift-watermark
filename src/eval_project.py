import os
import sys
import re
import time
import shutil
import subprocess
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=SyntaxWarning)

from watermarker import Watermarker, embed_watermark_files
from stats import is_significant, format_pvalue, p_mark, combine_pvalues
from paths import TMP_PATH, TESTPROJECTS_PATH
from llm_provider import get_usage, diff_usage, resolve_model

LANG_EXTENSIONS = {'py': '.py', 'js': '.js', 'java': '.java'}

# Per test run. A timeout truncates the suite, which _verdict() reads as INCONCLUSIVE.
DEFAULT_TEST_TIMEOUT = 1800

# VCS / dependency / build / cache dirs: not protection targets, and walking
# them would swamp the file list.
EXCLUDE_DIRS = {
    '.git',
    'node_modules', '.venv', 'venv',                      # dependencies
    'build', 'dist', 'target', 'coverage',                # build / coverage output
    '__pycache__', '.tox', '.eggs', '.pytest_cache',      # python tool caches
    '.mypy_cache', '.ruff_cache',
    '.nyc_output', '.gradle',                             # js / jvm tool caches
}


# Matched exactly against a path component, since a substring check would also match networkx's shortest_paths/.
_TEST_DIRS = {'test', 'tests', 'testing', '__tests__', '__mocks__', 'spec'}


def _is_test_dir(part):
    """Exact name, or a test- / test_ prefix (e.g. gson's test-jpms/)."""
    p = part.lower()
    return p in _TEST_DIRS or p.startswith('test-') or p.startswith('test_')


# Per-language filename conventions, case-sensitive so latest.java does not match.
# *IT.java is left out as too close to ordinary names.
_TEST_FILE_RE = re.compile(
    r'^(?:test_.*|conftest)\.py$'
    r'|^.*_test\.py$'
    r'|^.*\.(?:test|spec)\.(?:[cm]?jsx?|[cm]?tsx?)$'
    r'|^.*(?:Test|Tests|TestCase)\.java$'
)


def _is_test_path(filepath):
    """A test-directory component, or a filename following the language's test
    convention. Test files are not the protection target and carry few sites."""
    parts = os.path.normpath(filepath).split(os.sep)
    if any(_is_test_dir(d) for d in parts[:-1]):
        return True
    return bool(_TEST_FILE_RE.match(parts[-1]))


def collect_files(path, lang, include_tests=False):
    """Collect the language's source files under path. Returns (kept, skipped)."""
    ext = LANG_EXTENSIONS[lang]
    kept, skipped = [], []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in sorted(filenames):
            if not f.endswith(ext):
                continue
            full_path = os.path.join(root, f)
            if not include_tests and _is_test_path(os.path.relpath(full_path, path)):
                skipped.append(full_path)
            else:
                kept.append(full_path)
    return kept, skipped


def _count_lines(files):
    total = 0
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                total += sum(1 for _ in fp)
        except (OSError, UnicodeDecodeError):
            pass
    return total


def print_project_stat(project_path, lang):
    """One line of scope: all source past EXCLUDE_DIRS vs. the watermark target
    (tests excluded). N_syn, |F| and p are detected on the unmodified target.
    """
    kept, skipped = collect_files(project_path, lang)
    l_kept, l_skipped = _count_lines(kept), _count_lines(skipped)
    st = aggregate_stats(detect_all(Watermarker(lang), read_files(kept)))
    p = combine_pvalues(st['l1_p'], st['l2_p'])
    print(f"[*] {os.path.basename(project_path):<9} {lang:<5} "
          f"total {len(kept) + len(skipped):>4} files / {l_kept + l_skipped:>8,} lines"
          f"   watermark target {len(kept):>4} / {l_kept:>8,}"
          f"   N_syn {st['l1_n']:>5}  |F| {st['l2_blocks']:>4}  p {format_pvalue(p, '.3f')}")


def read_files(files):
    """Read all files into memory. Returns dict {path: code}."""
    codes = {}
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            codes[f] = fp.read()
    return codes


def detect_all(wm, codes, do_l2=True):
    """Run detect on all codes. Returns list of result dicts."""
    results = []
    for filepath, code in codes.items():
        try:
            det = wm.detect(code, do_l1=True, do_l2=do_l2)
            results.append({'file': filepath, 'ok': True, **det})
        except Exception as e:
            results.append({'file': filepath, 'ok': False, 'error': str(e)})
    return results


def embed_all(files, do_l1, do_l2, lang, secret_key):
    """Embed all files in place. 'codes' maps path to watermarked code, or None on failure."""
    target_files = [(f, f) for f in files]  # overwrite in-place
    t0 = time.perf_counter()
    results = embed_watermark_files(target_files, do_l1=do_l1, do_l2=do_l2, lang=lang, secret_key=secret_key, n_workers=min(len(target_files), 32))
    t_total = time.perf_counter() - t0

    codes = {}
    ok_count = 0
    fail_count = 0
    for r in results:
        path = r['dst']
        if r['ok']:
            with open(path, 'r', encoding='utf-8') as f:
                codes[path] = f.read()
            ok_count += 1
        else:
            codes[path] = None
            fail_count += 1
            print(f"  [FAIL] {os.path.basename(path)}: {r['error']}", file=sys.stderr)

    return {
        'codes': codes,
        'ok': ok_count, 'fail': fail_count, 'l2_fail': 0,
        't_total': t_total,
    }


def aggregate_stats(results):
    """Aggregate L1/L2 stats from detect results."""
    total_n = total_s = total_blocks = total_lo = 0
    file_count = 0
    detected_01 = 0  # files with L1 p < 0.01
    detected_05 = 0  # files with L1 p < 0.05

    for r in results:
        if not r.get('ok'):
            continue
        file_count += 1
        n = r.get('l1_n', 0)
        s = r.get('l1_s', 0)
        total_n += n
        total_s += s
        total_blocks += r.get('l2_blocks', 0)
        total_lo += r.get('l2_lo', 0)

        file_l1_p = Watermarker.calculate_l1_pvalue(n, s)
        if is_significant(file_l1_p, 0.01):
            detected_01 += 1
        if is_significant(file_l1_p, 0.05):
            detected_05 += 1

    l1_p = Watermarker.calculate_l1_pvalue(total_n, total_s)
    l2_p = Watermarker.calculate_l2_pvalue(total_blocks, total_lo)
    return {
        'files': file_count,
        'l1_n': total_n, 'l1_s': total_s, 'l1_p': l1_p,
        'l2_blocks': total_blocks, 'l2_lo': total_lo, 'l2_p': l2_p,
        'detected_01': detected_01,
        'detected_05': detected_05,
    }


def _print_phase_summary(label, stats, t_enc=None, t_dec=None):
    m = p_mark(stats['l1_p'])
    m2 = p_mark(stats['l2_p'])
    d01 = stats['detected_01']
    d05 = stats['detected_05']
    f = stats['files']
    enc_str = f"{t_enc:.1f}s" if t_enc is not None else "-"
    dec_str = f"{t_dec:.1f}s" if t_dec is not None else "-"
    print(f"  {label:<20} files={f:<4}  "
          f"L1: N={stats['l1_n']:<4} S={stats['l1_s']:<4} p={format_pvalue(stats['l1_p'], '.6g'):<12} {m:<4} "
          f"det={d01}/{d05}/{f}  "
          f"L2: R={stats['l2_blocks']:<3} K={stats['l2_lo']:<3} p={format_pvalue(stats['l2_p'], '.6g'):<12} {m2:<4} "
          f"enc={enc_str} dec={dec_str}")


def _print_detail_table(results, base_path):
    print(f"{'File':<55} {'L1 (N/S)':<10} {'L1 p':>10} {'L1':>4}  {'L2 (R/K)':<10} {'L2 p':>10} {'L2':>4}")
    print("-" * 110)
    for r in results:
        fname = os.path.relpath(r['file'], base_path)
        if not r.get('ok'):
            print(f"{fname:<55} {'ERR':>10}")
            continue
        n, s = r.get('l1_n', 0), r.get('l1_s', 0)
        blk, lo = r.get('l2_blocks', 0), r.get('l2_lo', 0)
        l1_p = Watermarker.calculate_l1_pvalue(n, s)
        l2_p = Watermarker.calculate_l2_pvalue(blk, lo)
        print(f"{fname:<55} {n:>3}/{s:<3}     {format_pvalue(l1_p):>10} {p_mark(l1_p):>4}  {blk:>3}/{lo:<3}     {format_pvalue(l2_p):>10} {p_mark(l2_p):>4}")


def _run_tests(cwd, test_cmd, timeout=DEFAULT_TEST_TIMEOUT):
    """Run test command in project directory. Returns result dict."""
    try:
        # Run under the interpreter running this evaluation: shell=True spawns a
        # non-interactive shell that never sourced the caller's venv, so a bare
        # python there is a different interpreter, or none.
        cmd = test_cmd.replace('python -m ', f'{sys.executable} -m ')
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, timeout=timeout
        )
        # mvn on Windows may emit non-UTF-8 bytes, but only ASCII result lines matter.
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        passed, failed, total = _parse_test_result(stdout, stderr, test_cmd)
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'passed': passed, 'failed': failed, 'total': total,
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'returncode': -1,
                'passed': None, 'failed': None, 'total': None}
    except Exception as e:
        return {'success': False, 'returncode': -1,
                'passed': None, 'failed': None, 'total': None}


# CSI sequences (colour, cursor moves), everything a test reporter emits.
_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')


def _test_tool(test_cmd):
    """Which parser to use. The command decides, not the output shape.
    A maven build that dies before surefire prints no `Tests run:` line, and
    pytest's collection-error pattern then reads its `1 error` as one failure.
    """
    c = test_cmd.lower()
    if 'mvn' in c or 'gradle' in c:
        return 'maven'
    if 'vitest' in c or 'jest' in c:
        return 'vitest'
    if 'pytest' in c:
        return 'pytest'
    return None             # unrecognized: try each, pytest last


def _parse_test_result(stdout, stderr, test_cmd=""):
    """Extract (passed, failed, total) from pytest / Maven / Vitest output.
    total is always passed + failed. (None, None, None) means unrecognized, which is also
    how a build failure looks, so _verdict() reads it together with the exit code.
    """
    # Strip ANSI first: vitest colorizes even into a pipe, which hides its
    # summary line from every pattern below.
    combined = _ANSI_RE.sub('', stdout + '\n' + stderr)
    tool = _test_tool(test_cmd)

    # Maven Surefire prints "Tests run: X, Failures: Y, Errors: Z, Skipped: W" per class and per module.
    # Only the per-class lines carry "-- in <Class>", so dropping them leaves one row per module.
    maven_rows = []
    for line in combined.splitlines() if tool in (None, 'maven') else []:
        if 'Tests run:' not in line or '-- in ' in line:
            continue
        m = re.search(
            r'Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)'
            r'(?:,\s*Skipped:\s*(\d+))?',
            line
        )
        if m:
            maven_rows.append(m)
    if maven_rows:
        total_run = total_fail = total_err = total_skip = 0
        for m in maven_rows:
            total_run += int(m.group(1))
            total_fail += int(m.group(2))
            total_err += int(m.group(3))
            total_skip += int(m.group(4)) if m.group(4) else 0
        p = total_run - total_fail - total_err - total_skip
        f = total_fail + total_err
        return p, f, p + f

    # Vitest: "Tests  14 failed | 1048 passed (1062)". Read the two counts independently,
    # from the "Tests" line only ("Test Files" is a file tally).
    if tool in (None, 'vitest'):
        for line in combined.splitlines():
            if not line.strip().startswith('Tests '):
                continue
            m_p = re.search(r'(\d+)\s+passed', line)
            m_f = re.search(r'(\d+)\s+failed', line)
            if m_p or m_f:
                p = int(m_p.group(1)) if m_p else 0
                f = int(m_f.group(1)) if m_f else 0
                return p, f, p + f

    # pytest last: its patterns are substrings of the other tools' output.
    # The errors? arm counts collection errors.
    if tool in (None, 'pytest'):
        m_passed = re.search(r'(\d+) passed', combined)
        m_failed = re.search(r'(\d+) failed', combined)
        m_error = re.search(r'(\d+) errors?\b', combined)
        if m_passed or m_failed or m_error:
            p = int(m_passed.group(1)) if m_passed else 0
            f = (int(m_failed.group(1)) if m_failed else 0) + (int(m_error.group(1)) if m_error else 0)
            return p, f, p + f

    return None, None, None


def _verdict(before, after):
    """Did watermarking break the tests? failed alone cannot answer that.
    A truncated run or a build failure both read as SAFE, so require that tests ran
    and that the same number ran twice before trusting the delta.
    """
    for label, r in (("before", before), ("after", after)):
        # A nonzero exit with no real test population is a build failure. The total <= 1 arm
        # catches a compile error parsed as one failed test, and errs away from SAFE when exactly one test fails.
        if r['total'] == 0 or (not r['success'] and (r['passed'] is None or r['total'] <= 1)):
            return f"BUILD FAILED ({label}: no tests ran, exit={r['returncode']})"
        if r['passed'] is None:
            return f"INCONCLUSIVE ({label}: test output not recognized, exit={r['returncode']})"
    if before['total'] != after['total']:
        return (f"INCONCLUSIVE (test count moved {before['total']} -> {after['total']}; "
                f"a fail-fast flag, a timeout or a collection error truncated a run)")
    delta = after['failed'] - before['failed']
    if delta > 0:
        return (f"{delta:+d} regressions (before: {before['failed']} failed, "
                f"after: {after['failed']} failed)")
    return f"SAFE ({after['passed']}/{after['total']} passed)"


def pre_gson(project_path):
    """Stop gson failing its build on warnings, so Phase 4 reaches the tests.
    gson enforces its style checks as compile errors, and L1 picks the less idiomatic spelling.
    Real errors (@InlineMe, SelfAssignment) still fail the build.
    Edits the pom in place, and git checkout . undoes it with the sources.
    """
    pom = os.path.join(os.path.dirname(project_path), 'pom.xml')
    old, new = '<failOnWarning>true</failOnWarning>', '<failOnWarning>false</failOnWarning>'
    with open(pom, encoding='utf-8') as fp:
        text = fp.read()
    if text.count(old) != 1:
        raise ValueError(f"{old} matched {text.count(old)}x in {pom}")
    with open(pom, 'w', encoding='utf-8', newline='') as fp:
        fp.write(text.replace(old, new, 1))
    print(f"  pretask: {old} -> {new}")


def _restore(git_root):
    """Undo everything Phase 4 wrote into the real project. A silent failure
    would leave the tree watermarked, so report it instead of claiming success."""
    r = subprocess.run(['git', 'checkout', '.'], cwd=git_root,
                       capture_output=True, text=True)
    left = subprocess.run(['git', 'status', '--porcelain'], cwd=git_root,
                          capture_output=True, text=True).stdout.strip()
    if r.returncode != 0 or left:
        print(f"  [!] RESTORE FAILED (exit={r.returncode}) -- project left modified")
        for line in left.splitlines()[:10]:
            print(f"      {line}")
        if len(left.splitlines()) > 10:
            print(f"      ... and {len(left.splitlines()) - 10} more")
    else:
        print(f"  Restored (git checkout .)")


def _test_phase(project_path, work_dir, watermarked_codes, test_cmd,
                timeout=DEFAULT_TEST_TIMEOUT, pretask=None):
    """Phase 4: run the project's tests before/after watermarking, in place on the real tree.
    Needs a clean git tree, and restores with `git checkout .`."""
    print(f"\n[Phase 4] Project Test Verification")

    try:
        r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=project_path,
                           capture_output=True, text=True)
        git_root = r.stdout.strip()
    except Exception:
        print("  [!] Not a git repository, skipping test phase")
        return None

    if not git_root:
        print("  [!] Cannot determine git root, skipping test phase")
        return None

    status = subprocess.run(['git', 'status', '--porcelain'], cwd=git_root,
                            capture_output=True, text=True).stdout.strip()
    if status:
        print(f"  [!] Git working tree not clean, skipping test phase")
        return None

    test_cwd = git_root
    print(f"  cmd: {test_cmd}")
    print(f"  cwd: {test_cwd}")
    if pretask:
        pretask(project_path)

    print(f"  Running tests (before)...", end="", flush=True)
    t0 = time.perf_counter()
    before = _run_tests(test_cwd, test_cmd, timeout)
    t_before = time.perf_counter() - t0
    before['elapsed'] = t_before
    if before['passed'] is not None:
        print(f" {before['passed']} passed, {before['failed']} failed ({t_before:.1f}s)")
    else:
        print(f" exit={before['returncode']} (parse failed, {t_before:.1f}s)")

    # A build failure parses as 0 passed, not None. A baseline that ran no tests cannot be compared against.
    if not before['total']:
        print(f"  [!] Baseline ran no tests (exit={before['returncode']}), skipping after-test")
        # Only the pretask has edited the tree so far. Left modified, the next run
        # would fail in the pretask instead of here.
        if pretask:
            _restore(git_root)
        return {'before': before, 'after': None}

    # Write watermarked files to original project
    written = 0
    for tmp_path, wm_code in watermarked_codes.items():
        rel = os.path.relpath(tmp_path, work_dir)
        orig_path = os.path.join(project_path, rel)
        if os.path.exists(orig_path):
            with open(orig_path, 'w', encoding='utf-8') as f:
                f.write(wm_code)
            written += 1

    print(f"  Running tests (after, {written} files modified)...", end="", flush=True)
    t0 = time.perf_counter()
    after = _run_tests(test_cwd, test_cmd, timeout)
    t_after = time.perf_counter() - t0
    after['elapsed'] = t_after
    if after['passed'] is not None:
        print(f" {after['passed']} passed, {after['failed']} failed ({t_after:.1f}s)")
    else:
        print(f" exit={after['returncode']} (parse failed, {t_after:.1f}s)")

    _restore(git_root)

    print("  => " + _verdict(before, after))

    return {'before': before, 'after': after}


def print_trials(trials):
    """One row per trial, under the metadata header eval_dataset.py also prints.
    L2 columns appear only when L2 ran. Raw numbers only, max / avg are the caller's.
    """
    if not trials:
        return

    def ratio(r):
        if not r or r.get('passed') is None:
            return '-'
        return f"{r['passed']}/{r['failed']}"

    def t_test(t):
        before, after = _test_rows(t)
        if not before:
            return '-'
        return f"{before.get('elapsed', 0) + (after or {}).get('elapsed', 0):.1f}s"

    head = trials[0]
    do_l2 = "L2" in head['layers']

    # (label, width, value) per column, like eval_dataset.METRICS. H0 L1 p is
    # Phase 1's p on the untouched tree: one null draw per key.
    cols = [
        ('#', 2, lambda i, t: i),
        ('key', 10, lambda i, t: t.get('key_tag', '-')),
        ('H0 L1 p', 12, lambda i, t: format_pvalue(t['baseline']['l1_p'], '.6g')),
        ('L1 N', 5, lambda i, t: t['watermarked']['l1_n']),
        ('L1 S', 5, lambda i, t: t['watermarked']['l1_s']),
        ('L1 p', 12, lambda i, t: format_pvalue(t['watermarked']['l1_p'], '.6g')),
        ('det(1%/5%)', 11, lambda i, t: f"{t['watermarked']['detected_01']}/{t['watermarked']['detected_05']}"),
    ]
    if do_l2:
        cols += [
            ('L2 R', 5, lambda i, t: t['watermarked']['l2_blocks']),
            ('L2 K', 5, lambda i, t: t['watermarked']['l2_lo']),
            ('L2 p', 12, lambda i, t: format_pvalue(t['watermarked']['l2_p'], '.6g')),
        ]
    cols += [
        ('fail', 6, lambda i, t: f"{t['embed_fail']}/{t['l2_fail']}" if do_l2 else t['embed_fail']),
        ('Enc', 7, lambda i, t: f"{t['t_embed']:.1f}s"),
        ('Dec', 7, lambda i, t: f"{t['t_detect_wm']:.1f}s"),
    ]
    if do_l2:
        cols += [
            ('Calls', 6, lambda i, t: f"{t['llm_usage']['n_calls']:,}"),
            ('Tok', 10, lambda i, t: f"{t['llm_usage']['total_tokens']:,}"),
        ]
    cols += [
        ('before', 10, lambda i, t: ratio(_test_rows(t)[0])),
        ('after', 10, lambda i, t: ratio(_test_rows(t)[1])),
        ('T_test', 8, lambda i, t: t_test(t)),
    ]

    header = '  ' + ' '.join(f"{lab:>{w}}" for lab, w, _ in cols)
    bar = "=" * len(header)

    print(bar)
    print(f" Trials Summary: {head['project']} ({head['lang']}, {head['layers']})  --  {len(trials)} run{'s' if len(trials) != 1 else ''}")
    if do_l2:
        print(f" Model: {resolve_model(None)}")
    print(f" Scope: {head['files']} files, {head['lines']:,} lines"
          f"  |  Key: {head.get('secret_key', '-')} (trial n appends -n)")
    print(f" Test:  {head.get('test_cmd') or '(Phase 4 skipped)'}")
    t0, t1 = head.get('t_start'), trials[-1].get('t_end')
    if t0 and t1:
        print(f" Start: {t0:%Y-%m-%d %H:%M:%S}  End: {t1:%Y-%m-%d %H:%M:%S}  "
              f"Elapsed: {timedelta(seconds=int((t1 - t0).total_seconds()))}")
    print(bar)
    print(header)
    print('  ' + ' '.join('-' * w for _, w, _ in cols))

    for i, t in enumerate(trials, 1):
        print('  ' + ' '.join(f"{str(fn(i, t)):>{w}}" for _, w, fn in cols))

    print(bar)


def _test_rows(t):
    """(before, after) of a trial's Phase 4, either of which may be absent."""
    test = t.get('test') or {}
    return test.get('before'), test.get('after')


def run_project(project_path, lang, test_cmd=None, pretask=None,
                secret_key=Watermarker.DEFAULT_SECRET_KEY,
                include_tests=False, detail=False, layers="L1"):
    """Run the full project-level watermark evaluation on one project.
    test_cmd runs the project's tests before/after watermarking, in place, and needs a clean git tree.
    None skips Phase 4. layers is 'L1' (default), 'L1L2', or 'L2'.
    """
    do_l1 = "L1" in layers
    do_l2 = "L2" in layers
    t_start = datetime.now()
    project_path = os.path.abspath(project_path)
    project_name = os.path.basename(os.path.normpath(project_path))

    # Work on a temporary copy
    tmp_dir = TMP_PATH(f'eval_prj_{project_name}')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    shutil.copytree(project_path, tmp_dir,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', 'node_modules', '.venv', 'venv'))

    try:
        result = _run_phases(tmp_dir, project_name, lang, layers, do_l1, do_l2,
                             secret_key, include_tests, detail)
        if result is None:  # no source file of this language in the project
            return None

        # Internal handoff for the test phase, not part of the returned report.
        wm_codes = result.pop('_wm_codes', {})
        work_dir = result.pop('_work_dir', tmp_dir)

        if test_cmd:
            test_result = _test_phase(project_path, work_dir, wm_codes, test_cmd, pretask=pretask)
            if test_result:
                result['test'] = test_result

        # Provenance for the report header: what was run, with which key, when.
        result['secret_key'] = secret_key
        result['test_cmd'] = test_cmd
        result['t_start'] = t_start
        result['t_end'] = datetime.now()
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_phases(work_dir, project_name, lang, layers, do_l1, do_l2, secret_key, include_tests, detail):
    """Phases 1-3 on the temporary copy: baseline detect, embed, watermarked detect."""
    print("=" * 110)
    print(f" Project: {project_name}  |  Lang: {lang}  |  Layers: {layers}  |  Tests: {'included' if include_tests else 'excluded'}")
    print("=" * 110)

    files, skipped = collect_files(work_dir, lang, include_tests=include_tests)
    if not files:
        print("[!] No source files found.")
        return None

    total_lines = _count_lines(files)
    dropped = f" ({len(skipped)} test files excluded)" if skipped else ""
    print(f"[*] {len(files)} files, {total_lines:,} lines{dropped}")
    if detail:
        for f in skipped:
            print(f"      - {os.path.relpath(f, work_dir)}")
    print()

    wm = Watermarker(lang, secret_key)
    print(f"[*] Strategies ({len(wm.strategy_names)}): {wm.strategy_names}")
    print(f"[*] L2: {'enabled' if do_l2 else 'disabled'}\n")

    original_codes = read_files(files)

    print("[Phase 1] Detect on ORIGINAL (no watermark expected)")
    t0 = time.perf_counter()
    baseline_results = detect_all(wm, original_codes, do_l2=do_l2)
    t_detect_baseline = time.perf_counter() - t0
    baseline_stats = aggregate_stats(baseline_results)
    _print_phase_summary("Original", baseline_stats, t_dec=t_detect_baseline)
    print()

    print(f"[Phase 2] Embed watermark (layers={layers})")
    usage_before = get_usage()
    embed_result = embed_all(files, do_l1, do_l2, lang, secret_key)
    llm_usage = diff_usage(usage_before, get_usage())
    watermarked_codes = embed_result['codes']
    # Files that failed to embed fall back to their originals.
    for path in watermarked_codes:
        if watermarked_codes[path] is None:
            watermarked_codes[path] = original_codes[path]
    embed_ok, embed_fail = embed_result['ok'], embed_result['fail']
    t_embed = embed_result['t_total']

    print(f"  Embed: {embed_ok} ok, {embed_fail} fail "
          f"(total={t_embed:.1f}s, {t_embed/len(files)*1000:.1f}ms/file)")
    print(f"  LLM:   calls={llm_usage['n_calls']}  "
          f"prompt={llm_usage['prompt_tokens']:,}  "
          f"completion={llm_usage['completion_tokens']:,}  "
          f"total={llm_usage['total_tokens']:,}")
    print()

    print("[Phase 3] Detect on WATERMARKED")
    t0 = time.perf_counter()
    watermarked_results = detect_all(wm, watermarked_codes, do_l2=do_l2)
    t_detect_wm = time.perf_counter() - t0
    wm_stats = aggregate_stats(watermarked_results)
    _print_phase_summary("Watermarked", wm_stats, t_enc=t_embed, t_dec=t_detect_wm)
    print()

    print("=" * 110)
    print(f" SUMMARY: {project_name} ({lang}, {layers})")
    print("=" * 110)
    embed_detail = f"Embed: {embed_ok}/{len(files)} ok"
    if do_l2:
        embed_detail += f" (L2 fail: {embed_result['l2_fail']})"
    print(f"  Files: {len(files)}  |  Lines: {total_lines:,}  |  {embed_detail}")
    print()

    # det=1%/5%/total: files detected at p<0.01 / p<0.05 / total files
    print(f"  {'Phase':<20} {'L1 N':>5} {'L1 S':>5} {'L1 p':>12} {'L1':>4}  {'det(1%/5%/all)':>16}  "
          f"{'L2 R':>4} {'L2 K':>4} {'L2 p':>12} {'L2':>4}  {'Enc':>7} {'Dec':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*12} {'-'*4}  {'-'*16}  "
          f"{'-'*4} {'-'*4} {'-'*12} {'-'*4}  {'-'*7} {'-'*7}")

    for label, stats, t_enc, t_dec in [
        ("Original", baseline_stats, None, t_detect_baseline),
        ("Watermarked", wm_stats, t_embed, t_detect_wm),
    ]:
        n, s = stats['l1_n'], stats['l1_s']
        det_str = f"{stats['detected_01']}/{stats['detected_05']}/{stats['files']}"
        enc_str = f"{t_enc:>6.1f}s" if t_enc is not None else f"{'-':>7}"
        dec_str = f"{t_dec:>6.1f}s"
        print(f"  {label:<20} {n:>5} {s:>5} {format_pvalue(stats['l1_p'], '.6g'):>12} {p_mark(stats['l1_p']):>4}  {det_str:>16}  "
              f"{stats['l2_blocks']:>4} {stats['l2_lo']:>4} {format_pvalue(stats['l2_p'], '.6g'):>12} {p_mark(stats['l2_p']):>4}  "
              f"{enc_str} {dec_str}")

    print("=" * 110)

    if detail:
        print(f"\n[Detail] Per-file detection (watermarked)")
        _print_detail_table(watermarked_results, work_dir)
        agg = aggregate_stats(watermarked_results)
        print("-" * 110)
        print(f"{'TOTAL':<55} {agg['l1_n']:>3}/{agg['l1_s']:<3}     {format_pvalue(agg['l1_p']):>10} {p_mark(agg['l1_p']):>4}  "
              f"{agg['l2_blocks']:>3}/{agg['l2_lo']:<3}     {format_pvalue(agg['l2_p']):>10} {p_mark(agg['l2_p']):>4}")
        print("=" * 110)

    return {
        'project': project_name,
        'lang': lang,
        'layers': layers,
        'files': len(files),
        'lines': total_lines,
        'embed_ok': embed_ok,
        'embed_fail': embed_fail,
        'l2_fail': embed_result['l2_fail'],
        'baseline': baseline_stats,
        'watermarked': wm_stats,
        't_embed': t_embed,
        't_detect_baseline': t_detect_baseline,
        't_detect_wm': t_detect_wm,
        'llm_usage': llm_usage,
        '_wm_codes': watermarked_codes,
        '_work_dir': work_dir,
    }


if __name__ == '__main__':
    # (path, language, test command[, pretask]), under testprojects/.
    # test command: None skips Phase 4. Never use fail-fast flags (-x, --bail).
    PROJECTS = [
        (TESTPROJECTS_PATH("networkx/networkx"),  "py",   "python -m pytest networkx/ -q --tb=line"),
        (TESTPROJECTS_PATH("gson/gson"),          "java", "mvn test -pl gson -Denforcer.skip=true", pre_gson),
        (TESTPROJECTS_PATH("axios"),              "js",   "npx vitest run --project unit"),
    ]

    for path, lang, *_ in PROJECTS:
        print_project_stat(path, lang)
    print()

    # NTRIALS > 1 repeats each project under a different key.
    LAYERS = "L1"
    NTRIALS = 1
    BASE_KEY = Watermarker.DEFAULT_SECRET_KEY

    for path, lang, test_cmd, *pretask in PROJECTS:
        trials = []
        for i in range(NTRIALS):
            tag = f'#{i + 1}' if i else '(base)'
            key = BASE_KEY if not i else f"{BASE_KEY}-{i + 1}"
            if NTRIALS > 1:
                print(f"### Trial {i + 1}/{NTRIALS}   secret_key={key}")
            r = run_project(path, lang, test_cmd, *pretask, secret_key=key, layers=LAYERS)
            if r:
                r['key_tag'] = tag
                trials.append(r)
            print("\n")
        if NTRIALS > 1 and trials:
            print_trials(trials)
            print("\n")
