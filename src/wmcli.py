"""
Watermark source files (embed/detect), single file or entire directory.

Usage:
    python wmcli.py embed <path> [--layers L1|L1L2] [--lang py|js|java] [-k KEY]
    python wmcli.py detect <path> [--lang py|js|java] [-k KEY]

<path> can be a single source file or a directory (recursive).
Supported languages: Python (.py), JavaScript (.js), Java (.java)
Embed overwrites files in-place. Detect prints results per file + summary.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

from watermarker import Watermarker, embed_watermark_files, detect_watermark_files
from stats import format_pvalue, p_mark

DEFAULT_KEY = Watermarker.DEFAULT_SECRET_KEY

LANG_EXTENSIONS = {
    'py': '.py',
    'js': '.js',
    'java': '.java',
}

EXT_TO_LANG = {v: k for k, v in LANG_EXTENSIONS.items()}


def detect_lang(path):
    """Detect language from file extension. Returns lang string or None."""
    _, ext = os.path.splitext(path)
    return EXT_TO_LANG.get(ext)


def collect_source_files(path, lang=None):
    """Collect source files from path (single file or directory).
    If lang is None, auto-detect from extension (single file) or collect all supported (directory).
    Returns list of (filepath, lang) tuples.
    """
    supported_exts = set(LANG_EXTENSIONS.values())

    if os.path.isfile(path):
        file_lang = detect_lang(path)
        if lang and file_lang and lang != file_lang:
            print(f"[!] File extension mismatch: --lang={lang} but file is {os.path.splitext(path)[1]}", file=sys.stderr)
            return []
        effective_lang = lang or file_lang
        if not effective_lang:
            print(f"[!] Unsupported file type: {path}", file=sys.stderr)
            return []
        return [(path, effective_lang)]

    elif os.path.isdir(path):
        if lang:
            target_ext = LANG_EXTENSIONS[lang]
            target_exts = {target_ext}
        else:
            target_exts = supported_exts

        files = []
        for root, dirs, filenames in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
            for f in sorted(filenames):
                _, ext = os.path.splitext(f)
                if ext in target_exts:
                    file_lang = EXT_TO_LANG[ext]
                    files.append((os.path.join(root, f), file_lang))
        return files

    else:
        print(f"[!] Path not found: {path}", file=sys.stderr)
        return []


def group_by_lang(file_lang_pairs):
    """Group (filepath, lang) pairs by language."""
    groups = {}
    for filepath, lang in file_lang_pairs:
        groups.setdefault(lang, []).append(filepath)
    return groups


def cmd_embed(args):
    file_lang_pairs = collect_source_files(args.input, lang=args.lang)
    if not file_lang_pairs:
        return

    do_l1 = "L1" in args.layers
    do_l2 = "L2" in args.layers

    groups = group_by_lang(file_lang_pairs)

    all_results = []
    for lang, files in groups.items():
        pairs = [(f, f) for f in files]  # overwrite in-place
        print(f"[*] Embedding {len(files)} {lang} file(s), layers={args.layers}")
        results = embed_watermark_files(pairs, do_l1=do_l1, do_l2=do_l2, lang=lang, secret_key=args.key)
        all_results.extend(results)

        for r in results:
            if r['ok']:
                print(f"  [OK] {r['dst']}")
            else:
                print(f"  [FAIL] {r['src']}: {r['error']}", file=sys.stderr)

    ok_count = sum(1 for r in all_results if r['ok'])
    total = len(file_lang_pairs)
    print(f"[*] Done. {ok_count}/{total} file(s) embedded.")


def cmd_detect(args):
    file_lang_pairs = collect_source_files(args.input, lang=args.lang)
    if not file_lang_pairs:
        return

    groups = group_by_lang(file_lang_pairs)
    total_files = len(file_lang_pairs)

    print(f"[*] Detecting watermark in {total_files} file(s)\n")

    summary = []
    for lang, files in groups.items():
        results = detect_watermark_files(files, do_l1=True, do_l2=True, lang=lang, secret_key=args.key)

        for r in results:
            filepath = r['file']
            print(f"--- {filepath} ---")

            if not r['ok']:
                print(f"  [ERR] {r['error']}")
                print()
                continue

            l1_p = Watermarker.calculate_l1_pvalue(r['l1_n'], r['l1_s'])
            l2_p = Watermarker.calculate_l2_pvalue(r['l2_blocks'], r['l2_lo'])

            l1_mark = p_mark(l1_p)
            print(f"  L1: N={r['l1_n']}, S={r['l1_s']}, p={format_pvalue(l1_p)} ({l1_mark})")

            for d in r.get('l1_detail', []):
                mark = "O" if d['match'] else "X"
                print(f"    [{mark}] {d['strategy']}: expected={d['expected']} observed={d['observed']}")

            l2_mark = p_mark(l2_p)
            print(f"  L2: blocks={r['l2_blocks']}, LO={r['l2_lo']}, p={format_pvalue(l2_p)} ({l2_mark})")
            print()

            summary.append({
                'file': filepath,
                'l1_n': r['l1_n'], 'l1_s': r['l1_s'], 'l1_p': l1_p, 'l1_mark': l1_mark,
                'l2_blocks': r['l2_blocks'], 'l2_lo': r['l2_lo'], 'l2_p': l2_p, 'l2_mark': l2_mark,
            })

    print("=" * 100)
    print(f"{'File':<50} {'L1 (N/S)':<12} {'L1 p':>10} {'L1':>4}  {'L2 (blk/LO)':<14} {'L2 p':>10} {'L2':>4}")
    print("-" * 100)

    total_n = total_s = total_blocks = total_lo = 0
    for row in summary:
        fname = os.path.relpath(row['file'], args.input) if os.path.isdir(args.input) else row['file']
        print(f"{fname:<50} {row['l1_n']:>3}/{row['l1_s']:<3}     {format_pvalue(row['l1_p']):>10} {row['l1_mark']:>4}  {row['l2_blocks']:>3}/{row['l2_lo']:<3}       {format_pvalue(row['l2_p']):>10} {row['l2_mark']:>4}")
        total_n += row['l1_n']
        total_s += row['l1_s']
        total_blocks += row['l2_blocks']
        total_lo += row['l2_lo']

    print("-" * 100)

    agg_l1_p = Watermarker.calculate_l1_pvalue(total_n, total_s)
    agg_l2_p = Watermarker.calculate_l2_pvalue(total_blocks, total_lo)
    agg_l1_mark = p_mark(agg_l1_p)
    agg_l2_mark = p_mark(agg_l2_p)

    print(f"{'TOTAL':<50} {total_n:>3}/{total_s:<3}     {format_pvalue(agg_l1_p):>10} {agg_l1_mark:>4}  {total_blocks:>3}/{total_lo:<3}       {format_pvalue(agg_l2_p):>10} {agg_l2_mark:>4}")
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description="Watermark source files or detect watermark.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_embed = subparsers.add_parser('embed', help='Embed watermark (overwrites files in-place)')
    p_embed.add_argument('input', help='Source file or directory')
    p_embed.add_argument('--layers', default='L1', choices=['L1', 'L1L2'], help='Watermark layers (default: L1)')
    p_embed.add_argument('--lang', choices=['py', 'js', 'java'], default=None, help='Language (auto-detect from extension if omitted)')
    p_embed.add_argument('-k', '--key', default=DEFAULT_KEY, help='Secret key')

    p_detect = subparsers.add_parser('detect', help='Detect watermark from source file(s)')
    p_detect.add_argument('input', help='Source file or directory')
    p_detect.add_argument('--lang', choices=['py', 'js', 'java'], default=None, help='Language (auto-detect from extension if omitted)')
    p_detect.add_argument('-k', '--key', default=DEFAULT_KEY, help='Secret key')

    args = parser.parse_args()

    if args.command == 'embed':
        cmd_embed(args)
    elif args.command == 'detect':
        cmd_detect(args)


if __name__ == '__main__':
    main()
