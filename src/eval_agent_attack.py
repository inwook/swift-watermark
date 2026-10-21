import os
import sys
import json
import traceback
import time
import shutil
import subprocess
from collections import Counter

from watermarker import Watermarker
from paths import TMP_PATH, OUT_PATH, TESTPROJECTS_PATH
from eval_project import (LANG_EXTENSIONS, collect_files, read_files, detect_all, embed_all,
                          aggregate_stats, pre_gson, _run_tests)
from stats import combine_pvalues, format_pvalue, p_mark

# Same model as attacks.W_STRONG_MODEL, so the agent and W-S differ only in the attack.
AGENT_MODEL = "claude-sonnet-4-6"

# The session's first user turn. Claude Code supplies its own system prompt, so _provenance()
# records the CLI version. The two conditions differ only in the requirement bullets.
_TASK_HEAD = """# Goal

This repository's source code carries a watermark. It is embedded in the syntax
and control flow of the code, and changing variable or function names does not
remove it. Remove the watermark.

# Scope

{files}

Do not modify any file outside this scope, and do not modify any test file.

# Requirements

"""

# The only difference between the two conditions.
_REQUIRE_EVERY_FUNCTION = """- Rewrite the syntax and control flow of every function in scope, regardless of
  cost.
- Modify every possible function. Leave no function as-is.
"""

# Always last, so the rewrite demand is not read as subordinate to it.
_PRESERVE = """- Preserve the public API and all observable behavior."""

AGENT_TASKS = {
    "default": _TASK_HEAD + _PRESERVE,
    "full": _TASK_HEAD + _REQUIRE_EVERY_FUNCTION + _PRESERVE,
}

#   repo:     clone root under testprojects/, copied whole so the test suite runs intact.
#   src:      where watermark targets are collected from, relative to the repo root.
#   test_cmd: run from the copy's root, same as eval_project.PROJECTS.
#   prepare:  optional callable(src_root) run on the copy before Phase 1, so its changes are
#             part of the baseline commit.
PROJECTS = {
    "networkx": dict(
        repo="networkx", src="networkx", lang="py",
        test_cmd="python -m pytest networkx/ -q --tb=line",
    ),
    "gson": dict(
        repo="gson", src="gson", lang="java",
        test_cmd="mvn test -pl gson -Denforcer.skip=true",
        prepare=pre_gson,
    ),
    "axios": dict(
        repo="axios", src=".", lang="js",
        test_cmd="npx vitest run --project unit",
    ),
}


# The canonical fingerprint of one chunk, or "" when it cannot be taken.
def _fp(wm, chunk_code):
    try:
        canonical = wm._syntax.normalize(chunk_code)
        return wm.parser.dump_structure(wm.parser.abstract_code(canonical))
    except Exception:
        return ""


def _chunk_fingerprints(wm, code):
    """Canonical fingerprints of a file's function chunks, as a multiset.
    The same string SyntaxEmbedder._get_expected_variation hashes.
    """
    fps = Counter()
    try:
        chunks = wm.parser.get_code_chunks(code)
    except Exception:
        return fps
    for is_func, chunk_code, _ in chunks:
        if not is_func:
            continue
        fingerprint = _fp(wm, chunk_code)
        if fingerprint:
            fps[fingerprint] += 1
    return fps


def canonical_coverage(wm, before_codes, after_codes):
    """Share of function chunks whose canonical fingerprint the agent moved.
    A chunk survives if some chunk in the rewritten file still has the same canonical form.
    """
    before = Counter()
    after = Counter()
    for path, code in before_codes.items():
        before += _chunk_fingerprints(wm, code)
        after += _chunk_fingerprints(wm, after_codes.get(path, ""))

    total = sum(before.values())
    kept = sum((before & after).values())
    return {
        "chunks_before": total,
        "chunks_after": sum(after.values()),
        "chunks_kept": kept,
        "coverage": (1.0 - kept / total) if total else None,
    }


SPLIT_GROUPS = ("touched", "same_file_untouched", "untouched_file", "added", "removed")


def _chunks_by_name(wm, code):
    """name -> sorted list of raw chunk texts. A name can legitimately repeat."""
    out = {}
    try:
        chunks = wm.parser.get_code_chunks(code)
    except Exception:
        return out
    for is_func, chunk_code, name in chunks:
        if is_func:
            out.setdefault(name, []).append(chunk_code)
    return {k: sorted(v) for k, v in out.items()}


def _fps_by_name(wm, code):
    """name -> sorted list of canonical fingerprints, keyed like _chunks_by_name.
    A chunk whose fingerprint cannot be taken contributes an empty string, so it
    matches only another chunk that also failed.
    """
    out = {}
    try:
        chunks = wm.parser.get_code_chunks(code)
    except Exception:
        return out
    for is_func, chunk_code, name in chunks:
        if is_func:
            out.setdefault(name, []).append(_fp(wm, chunk_code))
    return {k: sorted(v) for k, v in out.items()}


def _carriers_by_name(wm, code):
    """name -> [n, s] from the L1 detail rows."""
    out = {}
    try:
        detail = wm.detect(code, do_l1=True, do_l2=False)["l1_detail"]
    except Exception:
        return out
    for r in detail:
        slot = out.setdefault(r["function"], [0, 0])
        slot[0] += 1
        slot[1] += 1 if r["match"] else 0
    return out


def carrier_split(wm, pre_codes, post_codes):
    """N and S before/after per function, split by whether the agent edited it.
    Functions are keyed by name, so a repeated name moves as one unit.
    fp_same counts the functions of a group whose canonical form N(C) is unchanged,
    on the same per-function denominator as fn.
    """
    out = {g: dict(fn=0, fp_same=0, n_pre=0, s_pre=0, n_post=0, s_post=0)
           for g in SPLIT_GROUPS}
    for path, before in pre_codes.items():
        after = post_codes.get(path)
        if after is None:
            after = ""
        file_changed = before != after
        pre_chunks, post_chunks = _chunks_by_name(wm, before), _chunks_by_name(wm, after)
        pre_fps, post_fps = _fps_by_name(wm, before), _fps_by_name(wm, after)
        pre_car, post_car = _carriers_by_name(wm, before), _carriers_by_name(wm, after)
        for name in set(pre_chunks) | set(post_chunks):
            if name not in post_chunks:
                grp = "removed"
            elif name not in pre_chunks:
                grp = "added"
            elif pre_chunks[name] != post_chunks[name]:
                grp = "touched"
            else:
                grp = "same_file_untouched" if file_changed else "untouched_file"
            g = out[grp]
            g["fn"] += 1
            if name in pre_fps and pre_fps[name] == post_fps.get(name):
                g["fp_same"] += 1
            n, s = pre_car.get(name, (0, 0))
            g["n_pre"] += n
            g["s_pre"] += s
            n, s = post_car.get(name, (0, 0))
            g["n_post"] += n
            g["s_post"] += s
    return out


def _print_carrier_split(split, total_n_pre):
    for grp in SPLIT_GROUPS:
        g = split[grp]
        if not g["fn"]:
            continue
        pre = f"{g['s_pre']/g['n_pre']*100:.1f}" if g["n_pre"] else "-"
        post = f"{g['s_post']/g['n_post']*100:.1f}" if g["n_post"] else "-"
        share = g["n_pre"] / total_n_pre * 100 if total_n_pre else 0
        nc = f"{g['fp_same']}/{g['fn']}"
        print(f"    {grp:<20} fn={g['fn']:<4} N(C) kept {nc:<12} "
              f"N {g['n_pre']}->{g['n_post']}  S {g['s_pre']}->{g['s_post']}  "
              f"Match% {pre} -> {post}  ({share:.0f}% of carriers)")

    # The expected bit is 50/50, so an erased carrier set sits at S = N/2, not 0.
    t = split["touched"]
    if t["n_post"]:
        floor = t["n_post"] * 0.5
        available = t["s_pre"] - floor
        if available > 0:
            flipped = t["s_pre"] - t["s_post"]
            print(f"    touched floor: S {t['s_pre']}->{t['s_post']}, chance would be "
                  f"~{floor:.0f}; flipped {flipped:.0f}/{available:.0f} "
                  f"= {flipped/available*100:.0f}% of the way to chance")


# Build output is dropped so the copy compiles from the watermarked sources.
_COPY_SKIP = {".git", "target", "build", "__pycache__",
              ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _copy_ignore(src_dir, names):
    """copytree filter: drop build output, but never inside node_modules
    (some packages ship their code under build/).
    """
    if "node_modules" in src_dir.split(os.sep):
        return set()
    return {n for n in names if n in _COPY_SKIP}


def ensure_java_home():
    """Point maven at the JDK of the javac on PATH.
    Homebrew's maven otherwise uses its own newer openjdk, which may drop the release 8 gson asks for.
    """
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    javac = shutil.which("javac")
    if not javac:
        print("[!] no javac on PATH; maven will pick its own JDK")
        return None
    home = os.path.dirname(os.path.dirname(os.path.realpath(javac)))
    os.environ["JAVA_HOME"] = home
    print(f"[*] JAVA_HOME={home}")
    return home


def _provenance():
    """Repo state and agent CLI version, since results do not reproduce across either."""
    def _run(*args):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=30)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    head = _run("git", "-C", root, "rev-parse", "HEAD")
    dirty = _run("git", "-C", root, "status", "--porcelain")
    return {
        "git_head": head,
        "git_dirty": [l for l in (dirty or "").splitlines() if l.strip()],
        "claude_cli": _run("claude", "--version"),
        "python": sys.version.split()[0],
    }


def _git(work_dir, *args, check=True):
    r = subprocess.run(["git", *args], cwd=work_dir, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def _git_snapshot(work_dir):
    """Commit the watermarked tree so the agent's edits are exactly one diff.
    The clone's .git is not copied, so the agent cannot recover pre-watermark code.
    """
    _git(work_dir, "init", "-q")
    _git(work_dir, "add", "-A")
    _git(work_dir, "-c", "user.email=eval@local", "-c", "user.name=eval",
         "commit", "-q", "-m", "watermarked baseline")


def _changed_files(work_dir):
    out = _git(work_dir, "diff", "--name-only", "HEAD")
    tracked = [p for p in out.splitlines() if p.strip()]
    untracked = [p for p in _git(work_dir, "ls-files", "--others",
                                 "--exclude-standard").splitlines() if p.strip()]
    return tracked, untracked


def run_agent(work_dir, task, *, model=AGENT_MODEL, max_budget_usd=None, timeout=3600):
    """One headless agent invocation against the copy.
    bypassPermissions makes the loop unattended, and cwd confines it to the copy.
    Only user settings are loaded, so the target repo's own CLAUDE.md cannot steer the attacker.
    """
    cmd = [
        "claude", "-p", task,
        "--output-format", "json",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--setting-sources", "user",
    ]
    if max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(max_budget_usd)]

    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True,
                           timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired as e:
        r = e
        timed_out = True
    wall = time.perf_counter() - t0

    raw = (getattr(r, "stdout", None) or "") if timed_out else r.stdout
    payload = {}
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # modelUsage is the run total, the top-level usage is only the last iteration.
    in_tok = out_tok = think_tok = 0
    for per_model in (payload.get("modelUsage") or {}).values():
        in_tok += (per_model.get("inputTokens", 0)
                   + per_model.get("cacheReadInputTokens", 0)
                   + per_model.get("cacheCreationInputTokens", 0))
        out_tok += per_model.get("outputTokens", 0)
        think_tok += per_model.get("thinkingTokens", 0)

    return {
        "model": model,
        "timed_out": timed_out,
        "returncode": getattr(r, "returncode", None),
        "wall_s": wall,
        "duration_ms": payload.get("duration_ms"),
        "num_turns": payload.get("num_turns"),
        "cost_usd": payload.get("total_cost_usd"),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thinking_tokens": think_tok,
        "total_tokens": in_tok + out_tok,
        "is_error": payload.get("is_error"),
        "subtype": payload.get("subtype"),
        # Anything but completed (budget stop, API error) makes the cost a lower bound.
        "stop_reason": payload.get("stop_reason"),
        "terminal_reason": payload.get("terminal_reason"),
        "permission_denials": payload.get("permission_denials"),
        # The field names above are not contractual, so keep the raw payload.
        "raw": payload if payload else {"stdout_head": raw[:2000]},
    }


def _with_comb(stats):
    """Attach the Fisher-combined p-value, which equals l1_p when L2 did not run."""
    stats["comb_p"] = combine_pvalues(stats["l1_p"], stats["l2_p"])
    return stats


def _detect_stats(wm, codes, do_l2):
    return _with_comb(aggregate_stats(detect_all(wm, codes, do_l2=do_l2)))


def _stats_line(label, st):
    """All three p-values on one line, with L2's two inputs."""
    return (f"  {label}: N={st['l1_n']} S={st['l1_s']} "
            f"p_syn={format_pvalue(st['l1_p'], '.4g')} {p_mark(st['l1_p'])}"
            f" | blocks={st['l2_blocks']} LO={st['l2_lo']} "
            f"p_str={format_pvalue(st['l2_p'], '.4g')} {p_mark(st['l2_p'])}"
            f" | p_cmb={format_pvalue(st['comb_p'], '.4g')} {p_mark(st['comb_p'])}")


def run_agent_attack(project, condition, *,
                     watermark=True, layers="L1", model=AGENT_MODEL,
                     max_budget_usd=None, agent_timeout=3600, test_timeout=1800,
                     secret_key=Watermarker.DEFAULT_SECRET_KEY, trial=0, tag=None):
    """One (project, condition) run. watermark=False is the negative control.
    trial only reaches the tag. The caller decides how many to do.
    """
    cfg = PROJECTS[project]
    repo_dir, src_subdir = cfg["repo"], cfg["src"]
    lang, test_cmd = cfg["lang"], cfg["test_cmd"]
    if condition not in AGENT_TASKS:
        raise ValueError(f"condition must be one of {sorted(AGENT_TASKS)}, "
                         f"got {condition!r}")
    if layers not in ("L1", "L1L2"):
        raise ValueError(f"layers must be 'L1' or 'L1L2', got {layers!r}")
    do_l1, do_l2 = "L1" in layers, "L2" in layers
    repo_path = os.path.abspath(TESTPROJECTS_PATH(repo_dir))
    tag = tag or (f"{repo_dir}_{condition}_{layers.lower()}"
                  f"{'' if watermark else '_ctl'}_t{trial}")

    print("=" * 110)
    print(f" Agent attack: {repo_dir} ({lang}, {layers})  |  {condition}"
          f"{'' if watermark else '  [negative control: no watermark]'}  |  model={model}")
    print("=" * 110)

    work_dir = TMP_PATH(f"agent_{tag}")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree(repo_path, work_dir, symlinks=True, ignore=_copy_ignore)
    if lang == "java":
        ensure_java_home()

    src_root = os.path.join(work_dir, src_subdir) if src_subdir != "." else work_dir
    if cfg.get("prepare"):
        cfg["prepare"](src_root)
    files, _ = collect_files(src_root, lang)
    if not files:
        print("[!] No source files found.")
        return None

    scope = (f"Every {LANG_EXTENSIONS[lang]} file in this repository "
             f"({len(files)} files), except test files.")
    task = AGENT_TASKS[condition].format(files=scope)
    print(f"[*] {len(files)} {lang} files under {src_subdir}, "
          f"all handed to the agent\n")

    wm = Watermarker(lang, secret_key)

    if watermark:
        print(f"[Phase 1] Embed (layers={layers})")
        emb = embed_all(files, do_l1, do_l2, lang, secret_key)
        print(f"  {emb['ok']} ok, {emb['fail']} fail  ({emb['t_total']:.1f}s)")
    else:
        print("[Phase 1] Embed SKIPPED (negative control)")
        emb = {"ok": 0, "fail": 0, "t_total": 0.0}
    pre_codes = read_files(files)
    pre_stats = _detect_stats(wm, pre_codes, do_l2)
    print(_stats_line("project", pre_stats) + "\n")

    print("[Phase 2] Snapshot + tests (before)")
    _git_snapshot(work_dir)
    before = _run_tests(work_dir, test_cmd, test_timeout) if test_cmd else None
    if before:
        print(f"  {before['passed']} passed, {before['failed']} failed "
              f"(exit={before['returncode']})\n")

    print(f"[Phase 3] Agent rewrite")
    print(f"  task: {condition}, {len(task)} chars")
    agent = run_agent(work_dir, task, model=model,
                      max_budget_usd=max_budget_usd, timeout=agent_timeout)
    cost = f"${agent['cost_usd']:.2f}" if agent['cost_usd'] is not None else "?"
    print(f"  {agent['wall_s']/60:.1f}min  turns={agent['num_turns']}  "
          f"tokens={agent['total_tokens']:,}  cost={cost}"
          f"{'  [TIMED OUT]' if agent['timed_out'] else ''}")

    tracked, untracked = _changed_files(work_dir)
    touched_tests = [p for p in tracked + untracked if "test" in p.lower()]
    print(f"  changed: {len(tracked)} tracked, {len(untracked)} new"
          f"{f'  [!] {len(touched_tests)} test paths' if touched_tests else ''}\n")

    print("[Phase 4] Detect + tests (after)")
    post_codes = read_files([f for f in files if os.path.exists(f)])
    post_results = detect_all(wm, post_codes, do_l2=do_l2)
    post_stats = _with_comb(aggregate_stats(post_results))
    parse_fail = sum(1 for r in post_results if not r.get("ok"))
    deleted = len(files) - len(post_codes)

    after = _run_tests(work_dir, test_cmd, test_timeout) if test_cmd else None
    if after:
        print(f"  {after['passed']} passed, {after['failed']} failed "
              f"(exit={after['returncode']})")
    cov = canonical_coverage(wm, pre_codes, post_codes)
    cov_str = "-" if cov["coverage"] is None else f"{cov['coverage']:.1%}"
    print(_stats_line("project", post_stats) +
          f"  coverage={cov_str}  parse_fail={parse_fail}  deleted={deleted}")
    split = carrier_split(wm, pre_codes, post_codes)
    _print_carrier_split(split, pre_stats["l1_n"])
    print()

    result = {
        "project": repo_dir, "lang": lang, "layers": layers,
        "condition": condition, "watermark": watermark, "trial": trial,
        # Verbatim: runs are only comparable if they were sent the same text.
        "task": task,
        "files": len(files),
        "embed_ok": emb["ok"], "embed_fail": emb["fail"],
        "t_embed": emb["t_total"],
        "provenance": _provenance(),
        "pre": pre_stats, "post": post_stats,
        "parse_fail": parse_fail, "files_deleted": deleted,
        "changed_tracked": tracked, "changed_untracked": untracked,
        "touched_tests": touched_tests,
        "coverage": cov, "carrier_split": split,
        "agent": agent,
        "test_before": before, "test_after": after,
        "work_dir": work_dir,
    }

    out = OUT_PATH("agent_attack", f"{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[*] {out}")
    print(f"[*] rewritten tree kept at {work_dir}\n")
    return result


def print_report(results):
    """Per-run table plus avg cost, broken-run count and still-detected count."""
    results = [r for r in results if r]
    print("=" * 176)
    print(" AGENT RESTRUCTURING ATTACK")
    print("=" * 176)
    # TxtKept% and NCKept% are the share of the functions present before the attack
    # whose code, and whose canonical form N(C), the agent left intact. They count what
    # survived. NCchg% counts the opposite, what moved, and over the chunk multiset
    # rather than per function, so it is not the complement of NCKept%.
    # Blk is L2's eligible block count before -> after, re-decided on the rewritten code.
    # dFail is relative to the watermarked baseline.
    print(f"{'Project':<10} {'Cnd':<8} {'WM':<3} {'Fn mod':>8} {'TxtKept%':>9} {'NCKept%':>9} "
          f"{'Match%':>7} "
          f"{'S/N':>9} {'dN':>4} {'p_syn':>10} {'Blk':>9} {'p_str':>9} "
          f"{'p_cmb':>10} {'Tests':>14} {'dFail':>6} {'NCchg%':>8} "
          f"{'Turns':>5} {'MTok':>6} {'Min':>6} {'USD':>7}")
    print("-" * 176)
    for r in results:
        pre, tgt = r["pre"], r["post"]
        a, sp = r["agent"], r["carrier_split"]
        tb, ta = r["test_before"], r["test_after"]
        tests = dfail = "-"
        if tb and ta and tb["passed"] is not None and ta["passed"] is not None:
            tests = f"{tb['passed']}->{ta['passed']}"
            dfail = f"{(ta['failed'] or 0) - (tb['failed'] or 0):+d}"
        elif ta:
            tests = f"exit={ta['returncode']}"
        total_fn = sum(g["fn"] for g in sp.values()) - sp["added"]["fn"]
        fn_str = f"{sp['touched']['fn']}/{total_fn}"
        # A removed function is neither edited nor intact, so it is left out of the
        # numerator while staying in total_fn. 100 - Fn mod would count it as intact.
        intact_text = sp["same_file_untouched"]["fn"] + sp["untouched_file"]["fn"]
        intact_nc = sum(g["fp_same"] for g in sp.values())
        txt_str = f"{intact_text / total_fn * 100:.1f}" if total_fn else "-"
        nc_str = f"{intact_nc / total_fn * 100:.1f}" if total_fn else "-"
        # Match% over the functions the agent edited, where 50% is chance.
        t = sp["touched"]
        match_str = f"{t['s_post']/t['n_post']*100:.1f}" if t["n_post"] else "-"
        cv = r["coverage"]["coverage"]
        cov_str = "-" if cv is None else f"{cv * 100:.1f}"
        cost_str = "-" if a["cost_usd"] is None else f"{a['cost_usd']:.2f}"
        wm_str = "Y" if r["watermark"] else "N"
        blk_str = f"{pre['l2_blocks']}->{tgt['l2_blocks']}"
        sn_str = f"{tgt['l1_s']}/{tgt['l1_n']}"
        print(f"{r['project']:<10} {r['condition']:<8} {wm_str:<3} "
              f"{fn_str:>8} {txt_str:>9} {nc_str:>9} {match_str:>7} "
              f"{sn_str:>9} "
              f"{tgt['l1_n'] - pre['l1_n']:>+4} "
              f"{format_pvalue(tgt['l1_p'], '.2g'):>10} {blk_str:>9} "
              f"{format_pvalue(tgt['l2_p'], '.2g'):>9} "
              f"{format_pvalue(tgt['comb_p'], '.2g'):>10} {tests:>14} {dfail:>6} {cov_str:>8} "
              f"{str(a['num_turns'] or '-'):>5} "
              f"{a['total_tokens']/1e6:>6.2f} {a['wall_s']/60:>6.1f} {cost_str:>7}")
    print("-" * 176)

    attacked = [r for r in results if r["watermark"]]
    if not attacked:
        return
    toks = [r["agent"]["total_tokens"] for r in attacked]
    mins = [r["agent"]["wall_s"] / 60 for r in attacked]
    broke = [r for r in attacked if _is_broken(r)]
    passed = [r for r in attacked if r not in broke]
    # Compare against None explicitly: p == 0.0 is falsy.
    still = [r for r in passed
             if r["post"]["comb_p"] is not None
             and r["post"]["comb_p"] < 0.01]
    print(f"  avg tokens {sum(toks)/len(toks):,.0f}   avg minutes {sum(mins)/len(mins):.1f}")
    print(f"  failed to compile or pass tests: {len(broke)}/{len(attacked)}")
    print(f"  watermark still detectable (p_cmb<0.01): "
          f"{len(still)}/{len(passed)} of those that passed")
    censored = [r for r in attacked if r["agent"].get("terminal_reason") != "completed"]
    if censored:
        print(f"  [!] {len(censored)}/{len(attacked)} runs stopped early "
              f"(budget / error), so their cost is a lower bound: "
              f"{', '.join(r['project'] + ':' + str(r['agent'].get('terminal_reason')) for r in censored)}")
    print("=" * 176)


def _is_broken(r):
    """Did the rewrite fail to preserve the project?
    A modified test file counts, since the suite is the oracle.
    """
    if r["parse_fail"] or r["files_deleted"] or r["touched_tests"]:
        return True
    tb, ta = r["test_before"], r["test_after"]
    if not ta:
        return False
    if ta["passed"] is None:
        return True
    return (ta["failed"] or 0) > ((tb["failed"] or 0) if tb else 0)


if __name__ == "__main__":
    # Comment out the cells you don't want for this run.
    RUNS = [
        # (project, condition[, watermark])
        #   condition: "full" demands every function rewritten, "default" leaves the effort to the agent.
        #   watermark: False is the negative control on the pristine tree.
        ("networkx", "full"),
        ("gson",     "full"),
        ("axios",    "full"),
    ]

    LAYERS = "L1L2"

    # NTRIALS > 1 re-runs every cell with the same key, since the agent is stochastic.
    NTRIALS = 1

    # Per-invocation ceiling. Keep it identical across cells being compared.
    BUDGET_USD = {"default": 25.0, "full": 50.0}
    AGENT_TIMEOUT = 7200

    # One failing cell must not cost the whole batch.
    results = []
    for project, condition, *ctl in RUNS:
        for i in range(NTRIALS):
            if NTRIALS > 1:
                print(f"### Trial {i + 1}/{NTRIALS}   {project}/{condition}")
            try:
                results.append(run_agent_attack(
                    project, condition, trial=i, layers=LAYERS,
                    watermark=ctl[0] if ctl else True,
                    max_budget_usd=BUDGET_USD[condition],
                    agent_timeout=AGENT_TIMEOUT))
            except Exception as e:
                print(f"\n[!] {project}/{condition} trial {i} failed: "
                      f"{type(e).__name__}: {e}\n")
                traceback.print_exc()
    print_report(results)
