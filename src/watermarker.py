from syntax_embedder import SyntaxEmbedder
from stats import (calculate_binomial_pvalue, calculate_negative_binomial_pvalue,
                   combine_pvalues, format_pvalue)


class Watermarker:
    DEFAULT_SECRET_KEY = "SAMPLE_KEY"
    DEFAULT_TAU = 8.0

    def __init__(self, lang="py", secret_key=DEFAULT_SECRET_KEY, *, llm_model=None):
        from language_parser import get_parser

        parser = get_parser(lang)
        if lang == "py":
            from rewriters_py import get_py_strategies
            strategies = get_py_strategies()
        elif lang == "js":
            from rewriters_js import get_js_strategies
            strategies = get_js_strategies()
        elif lang == "java":
            from rewriters_java import get_java_strategies
            strategies = get_java_strategies()
        else:
            raise ValueError(f"Unsupported language: '{lang}'")

        self._syntax = SyntaxEmbedder(strategies, parser, secret_key)
        self.parser = parser

        from structure_embedder import StructureEmbedder

        # The LLM provider and verifier are built lazily, so only L2 embedding needs an API key.
        self._struct = StructureEmbedder(self._syntax, llm_model, lang=lang, tau=self.DEFAULT_TAU)

    @property
    def strategy_names(self):
        return [s.name for s in self._syntax.strategies]

    def embed(self, src: str, do_l1, do_l2, *, verbose=False) -> str:
        result = src
        if do_l2:
            result, _ = self._struct.embed(result, verbose=verbose)
        if do_l1:
            result = self._syntax.embed(result)
        return result

    def detect(self, src: str, do_l1, do_l2) -> dict:
        result = {}

        if do_l1:
            l1 = self._syntax.detect(src)
            result['l1_n'] = l1['N_eff']
            result['l1_s'] = l1['S_eff']
            result['l1_detail'] = l1['results']

        if do_l2:
            l2 = self._struct.detect(src)
            result['l2_blocks'] = l2['num_blocks']
            result['l2_lo'] = l2['total_lo']

        return result

    @staticmethod
    def calculate_l1_pvalue(n, s):
        """Layer 1 p-value, or None when no rewriter site existed (n == 0).
        n > 0 with s == 0 is a performed test and returns a real p-value.
        """
        if n <= 0:
            return None
        return calculate_binomial_pvalue(n, s, p=0.5)

    @staticmethod
    def calculate_l2_pvalue(blocks, lo):
        """Layer 2 p-value, or None when no function was eligible (blocks == 0).
        blocks > 0 with lo == 0 is a performed test and returns a real p-value.
        """
        if blocks <= 0:
            return None
        return calculate_negative_binomial_pvalue(k=lo, r=blocks, p=0.5)

    @staticmethod
    def calculate_combined_pvalue(l1_p, l2_p):
        """Fisher over whichever of the two layers actually ran (None = absent)."""
        return combine_pvalues(l1_p, l2_p)


def embed_watermark_files(pairs, do_l1, do_l2, lang="py", secret_key=Watermarker.DEFAULT_SECRET_KEY, *, n_workers=8, llm_model=None):
    """Embed into each (src_path, dst_path) pair.
    Returns a list of {'src', 'dst', 'ok', 'error'} dicts in input order.
    """
    import concurrent.futures

    def _process(wm, src_path, dst_path):
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                code = f.read()
            watermarked = wm.embed(code, do_l1=do_l1, do_l2=do_l2)
            valid, err = wm.parser.check_syntax(watermarked)
            if not valid:
                return {'src': src_path, 'dst': dst_path, 'ok': False, 'error': f'syntax: {err}'}
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.write(watermarked)
            return {'src': src_path, 'dst': dst_path, 'ok': True, 'error': None}
        except Exception as e:
            return {'src': src_path, 'dst': dst_path, 'ok': False, 'error': str(e)}

    # L1 embed is cheap per file, so threads only help L2's LLM calls.
    if not do_l2:
        n_workers = 1
    else:
        n_workers = min(n_workers, len(pairs))

    if n_workers <= 1:
        wm = Watermarker(lang, secret_key, llm_model=llm_model)
        return [_process(wm, src, dst) for src, dst in pairs]

    watermarkers = [Watermarker(lang, secret_key, llm_model=llm_model) for _ in range(n_workers)]
    results = [None] * len(pairs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i, (src, dst) in enumerate(pairs):
            wm = watermarkers[i % n_workers]
            futures[executor.submit(_process, wm, src, dst)] = i
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()

    return results


def detect_watermark_files(files, do_l1, do_l2, lang="py", secret_key=Watermarker.DEFAULT_SECRET_KEY, *, n_workers=1, llm_model=None):
    """Detect in each file. Returns per-file dicts with Watermarker.detect() fields.
    n_workers defaults to 1 since detection is CPU-bound under the GIL.
    """
    import concurrent.futures

    def _process(wm, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            det = wm.detect(code, do_l1=do_l1, do_l2=do_l2)
            return {'file': filepath, 'ok': True, 'error': None, **det}
        except Exception as e:
            return {'file': filepath, 'ok': False, 'error': str(e)}

    n_workers = min(n_workers, len(files))
    if n_workers <= 1:
        wm = Watermarker(lang, secret_key, llm_model=llm_model)
        return [_process(wm, f) for f in files]

    watermarkers = [Watermarker(lang, secret_key, llm_model=llm_model) for _ in range(n_workers)]
    results = [None] * len(files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i, filepath in enumerate(files):
            wm = watermarkers[i % n_workers]
            futures[executor.submit(_process, wm, filepath)] = i
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()

    return results


def _test_embed_detect():
    from code_snippets import get_snippet

    for lang in ("py", "js", "java"):
        code = get_snippet(lang, "fibonacci")
        print(f"\n{'='*60}")
        print(f" [{lang}] L1 Embed + Detect")
        print(f"{'='*60}")

        wm = Watermarker(lang)
        print(f"Strategies: {wm.strategy_names}")

        # Embed
        watermarked = wm.embed(code, do_l1=True, do_l2=False)
        valid, err = wm.parser.check_syntax(watermarked)
        print(f"Syntax OK: {valid}")

        # Detect watermarked
        det = wm.detect(watermarked, do_l1=True, do_l2=False)
        l1_p = Watermarker.calculate_l1_pvalue(det['l1_n'], det['l1_s'])
        print(f"Watermarked: N={det['l1_n']}, S={det['l1_s']}, p={format_pvalue(l1_p)}")
        for r in det['l1_detail']:
            mark = "O" if r['match'] else "X"
            print(f"  [{mark}] {r['strategy']}: expected={r['expected']} observed={r['observed']}")

        # Detect original (should NOT detect)
        det_orig = wm.detect(code, do_l1=True, do_l2=False)
        l1_p_orig = Watermarker.calculate_l1_pvalue(det_orig['l1_n'], det_orig['l1_s'])
        print(f"Original:    N={det_orig['l1_n']}, S={det_orig['l1_s']}, p={format_pvalue(l1_p_orig)}")


def _test_pvalue():
    # Test static p-value calculations.
    print(f"\n{'='*60}")
    print(f" P-value calculations")
    print(f"{'='*60}")

    # L1
    for n, s in [(10, 10), (10, 5), (20, 15), (0, 0), (66,64)]:
        p = Watermarker.calculate_l1_pvalue(n, s)
        print(f"  L1: N={n}, S={s} => p={format_pvalue(p)}")

    # L2
    for blocks, lo in [(5, 10), (5, 5), (10, 20), (0, 0)]:
        p = Watermarker.calculate_l2_pvalue(blocks, lo)
        print(f"  L2: blocks={blocks}, LO={lo} => p={format_pvalue(p)}")


if __name__ == '__main__':
    _test_embed_detect()
    _test_pvalue()
