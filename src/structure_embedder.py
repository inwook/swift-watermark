import time
from typing import List, Tuple, Dict, Any
import random
import re
import hashlib
import textwrap
import threading
import concurrent.futures

from llm_provider import LLMProvider, get_provider
from syntax_embedder import SyntaxEmbedder
from stats import calculate_negative_binomial_pvalue, count_leading_ones

LANG_CONFIG = {
    "py": {
        "name": "Python",
        "variant_sep": "\n# === VARIANT ===\n",
        "fence_pattern": r"```(?:python)?\s*(.*?)```",
        "fence_strip": r"```(?:python)?",
        "extra_constraints": "- Do NOT modify docstrings or comments.",
    },
    "js": {
        "name": "JavaScript",
        "variant_sep": "\n// === VARIANT ===\n",
        "fence_pattern": r"```(?:javascript|js)?\s*(.*?)```",
        "fence_strip": r"```(?:javascript|js)?",
        "extra_constraints": (
            "- Do NOT modify comments.\n"
            "- Do NOT convert between function declarations and arrow functions."
        ),
    },
    "java": {
        "name": "Java",
        "variant_sep": "\n// === VARIANT ===\n",
        "fence_pattern": r"```(?:java)?\s*(.*?)```",
        "fence_strip": r"```(?:java)?",
        "extra_constraints": (
            "- Do NOT modify comments or Javadoc.\n"
            "- Do NOT change access modifiers or method signatures.\n"
            "- Do NOT change class names or structure."
        ),
    },
}

_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
    You rewrite {lang_name} code to produce behaviorally equivalent structural variants.

    GOAL
    Generate code that has the same observable behavior but a different syntactic structure.

    STRICT CONSTRAINTS
    - Do NOT change function names.
    - Do NOT change function signatures.
    - Do NOT change any variable names (including local variables).
    {extra_constraints}
    - Do NOT add or remove function definitions.
    - Preserve imports exactly.

    SEMANTIC RULE
    - The transformed code must produce identical outputs for all inputs.
    - Transformations must remain natural and idiomatic for human programmers.

    ALLOWED TRANSFORMATIONS
    - Restructure control flow or loops.
    - Replace expressions with equivalent formulations.
    - Introduce temporary variables when needed.
    - Reorganize statements when needed while preserving semantics.

    OUTPUT RULE
    - Output ONLY valid {lang_name} code.
    - Do not include explanations or markdown fences.
""").strip()

_USER_PROMPT_TEMPLATE = textwrap.dedent("""
    Rewrite the following {lang_name} code as {n} behaviorally equivalent structural variants.

    IMPORTANT:
    - All identifiers must remain exactly the same.
    - Do NOT rename any variables.
    - Do NOT change function names or parameters.
    {extra_constraints}
    - Transformations must remain natural and idiomatic for human programmers.

    FORMAT RULE:
    Separate each variant using this marker exactly:
    {sep}

    <code>
    {code}
    </code>
""").strip()

_VARIANT_SEP_RE = re.compile(r"\n?[ \t]*(?:#|//)\s*=+\s*VARIANT\s*=+[ \t]*\n?")

K_CANDIDATES = 15

def _parse_variants(raw_text: str, variant_sep: str, lang: str = "py") -> List[str]:
    """Split the LLM response on the VARIANT marker, falling back to fenced blocks.
    Returns [] rather than the raw blob when neither works, so a malformed response
    cannot leak concatenated variants into the output.
    """
    cfg = LANG_CONFIG[lang]

    parts = _VARIANT_SEP_RE.split(raw_text)
    if len(parts) >= 2:
        raw_candidates = [p for p in parts if p.strip()]
    else:
        blocks = re.findall(cfg["fence_pattern"], raw_text, flags=re.DOTALL)
        raw_candidates = [b for b in blocks if b.strip()] if blocks else []

    candidates = []
    for cand in raw_candidates:
        cand = re.sub(cfg["fence_strip"], "", cand)
        cand = cand.replace("```", "")
        cand = cand.strip()
        if cand:
            candidates.append(cand)

    return candidates


def variation_with_llm(provider: LLMProvider, verifier, original_code: str, \
                       ncandidate: int = K_CANDIDATES, verifier_lock: threading.Lock | None = None,
                       lang: str = "py", *, verbose: bool = True) -> List[str]:
    # _parse_variants needs a separator between candidates.
    assert ncandidate >= 2, f"ncandidate must be >= 2 (got {ncandidate})"

    cfg = LANG_CONFIG[lang]

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"], extra_constraints=cfg["extra_constraints"]
    )
    variant_sep = cfg["variant_sep"]
    sep_lit = variant_sep.strip("\n")

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"], extra_constraints=cfg["extra_constraints"],
        n=ncandidate, sep=sep_lit, code=original_code
    )

    raw_text = ""
    max_retries = 5
    base_delay = 5.0

    for attempt in range(max_retries):
        try:
            raw_text = provider.generate(system_prompt, user_prompt)
            if raw_text: 
                break
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota" in error_msg or "ResourceExhausted" in error_msg:
                # Exponential back-off with jitter for rate-limit errors.
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[!] Rate limit hit. Waiting {wait_time:.2f}s before retry (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"[!] Unexpected Error: {e}")
                time.sleep(2.0)

    candidates = _parse_variants(raw_text, variant_sep, lang)

    if not verifier:
        if verbose:
            print("[!] Skipping equivalence verifier.")
        return candidates

    test_cases = verifier.generate_test_inputs(original_code)
    if not test_cases:
        if verbose:
            print("[!] Failed to generate test cases for validation. Returning all raw candidates.")
        return []

    if verbose:
        print(f"[*] Testing {len(candidates)} generated variants...")

    if verifier_lock:
        # The verifier may refresh or terminate its shared process pool on timeout.
        with verifier_lock:
            r = verifier.check_equivalence(original_code, candidates, test_cases)
    else:
        r = verifier.check_equivalence(original_code, candidates, test_cases)

    valid_variants = [c for c, ok in zip(candidates, r) if ok]

    if verbose:
        print(f"[*] {len(valid_variants)}/{len(candidates)} survived verifier tests.")

    return valid_variants

class StructureEmbedder:
    PARALLELIZE_VARIATION = True
    LLM_VARIATION_THREADS = 32

    def __init__(self, syntax_embedder: SyntaxEmbedder, llm_model: str | None = None, lang: str = "py", tau: float = 0.0):
        self._syntax = syntax_embedder
        self.lang = lang
        self.tau = tau

        # provider and verifier are built on first access, so L1-only and detect-only runs need no API key.
        self._llm_model = llm_model
        self._provider = None
        self._verifier = None
        self._lazy_lock = threading.RLock()   # verifier property reads self.provider

        if self.PARALLELIZE_VARIATION:
            # ThreadPoolExecutor spawns no thread until the first submit().
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.LLM_VARIATION_THREADS)
            self.verifier_lock = threading.Lock()

    @property
    def provider(self) -> LLMProvider:
        """LLM provider, built on first use. Raises if the API key is unset."""
        with self._lazy_lock:
            if self._provider is None:
                self._provider = get_provider(self._llm_model)
            return self._provider

    @property
    def verifier(self):
        """Equivalence verifier, built on first use."""
        with self._lazy_lock:
            if self._verifier is None:
                if self.lang == "py":
                    from equivalence_verifier_py import EquivalenceVerifier_Py
                    self._verifier = EquivalenceVerifier_Py(self.provider, verbose=False)
                elif self.lang == "js":
                    from equivalence_verifier_js import EquivalenceVerifier_JS
                    self._verifier = EquivalenceVerifier_JS(self.provider, verbose=False)
                elif self.lang == "java":
                    from equivalence_verifier_java import EquivalenceVerifier_Java
                    self._verifier = EquivalenceVerifier_Java(self.provider, verbose=False)
                else:
                    raise ValueError(f"Unsupported language: '{self.lang}'")
            return self._verifier

    def __del__(self):
        if self.PARALLELIZE_VARIATION:
            self.executor.shutdown(wait=False)

    def _collect_verifiable_indices(self, chunks, *, verbose: bool = False):
        """Return indices of function chunks that pass check_verifiable."""
        if self.lang == "py":
            from equivalence_verifier_py import EquivalenceVerifier_Py
            _check_fn = EquivalenceVerifier_Py.check_verifiable
        elif self.lang == "js":
            from equivalence_verifier_js import EquivalenceVerifier_JS
            _check_fn = EquivalenceVerifier_JS.check_verifiable
        elif self.lang == "java":
            from equivalence_verifier_java import EquivalenceVerifier_Java
            _check_fn = EquivalenceVerifier_Java.check_verifiable
        else:
            raise ValueError(f"Unsupported language: '{self.lang}'")

        verifiable = []
        for idx, (is_func, chunk_code, func_name) in enumerate(chunks):
            if not is_func:
                continue
            ok, reasons = _check_fn(chunk_code)
            if ok:
                verifiable.append(idx)
            elif verbose:
                print(f"  [skip] {func_name}: not verifiable ({', '.join(reasons[:2])})")
        return verifiable

    def _select_full_embedding_indices(self, chunks, verifiable_indices):
        W_CC, W_STMT, W_NEST = 2.0, 1.0, 2.0

        verifiable_indices = sorted(verifiable_indices)
        if self.tau <= 0 or not verifiable_indices:
            return verifiable_indices

        total_funcs = sum(1 for is_func, _, _ in chunks if is_func)
        if total_funcs == 0:
            return verifiable_indices
        if (len(verifiable_indices) / total_funcs) < 0.50:
            return verifiable_indices

        selected = []
        for i in verifiable_indices:
            cc, stmt, nest = self._syntax.parser.compute_structure_metrics(chunks[i][1])
            if W_CC * cc + W_STMT * stmt + W_NEST * nest >= self.tau:
                selected.append(i)
        return selected

    def get_structure_hash(self, src: str) -> bytes:
        """Layer 2 structure hash: H( Fingerprint(N(C)) || K )."""
        canonical_code = self._syntax.normalize(src)
        canonical_code = self._syntax.parser.abstract_code(canonical_code)
        fingerprint_str = self._syntax.parser.dump_structure(canonical_code)
        fingerprint_bytes = fingerprint_str.encode('utf-8')

        h = hashlib.sha256()
        h.update(fingerprint_bytes)
        h.update(self._syntax.secret_key)
        return h.digest()

    def embed(self, src: str, *, verbose: bool = True) -> Tuple[str, int]:
        chunks = self._syntax.parser.get_code_chunks(src)
        final_pieces = []
        total_lo = 0

        future_to_chunk = {}

        # check_verifiable, then the structure-metric gate. detect() uses the same gate.
        verifiable_idxs = self._collect_verifiable_indices(chunks, verbose=verbose)
        full_embedding_idxs = self._select_full_embedding_indices(chunks, verifiable_idxs)
        full_embedding_set = set(full_embedding_idxs)

        if verbose:
            func_count = sum(1 for is_func, _, _ in chunks if is_func)
            dropped = len(verifiable_idxs) - len(full_embedding_set)
            extra = f", -{dropped} below sigma>={self.tau}" if dropped else ""
            print(f"[*] Layer 2: Processing {len(chunks)} chunks "
                  f"({func_count} functions, {len(full_embedding_set)} full-embedding{extra}).")

        if self.PARALLELIZE_VARIATION:
            for idx, (is_func, chunk_code, func_name) in enumerate(chunks):
                if not is_func or idx not in full_embedding_set:
                    continue

                future = self.executor.submit(
                    variation_with_llm,
                    self.provider, self.verifier, chunk_code,
                    verbose=False, verifier_lock=self.verifier_lock, lang=self.lang,
                )
                future_to_chunk[idx] = future

        for idx, (is_func, chunk_code, func_name) in enumerate(chunks):
            if not is_func:
                # Non-function chunks (imports, gaps) pass through unchanged.
                final_pieces.append(chunk_code)
                continue

            if idx not in full_embedding_set:
                final_pieces.append(chunk_code)
                continue

            if verbose: print(f"[*] Optimizing function: {func_name}...")

            if self.PARALLELIZE_VARIATION:
                future = future_to_chunk[idx]
                try:
                    candidates = future.result()
                except Exception as e:
                    if verbose: print(f"  [!] LLM Generation failed: {e}")
                    candidates = []
            else:
                candidates = variation_with_llm(self.provider, self.verifier, chunk_code, verbose=verbose, lang=self.lang)
            
            # Always include the original chunk among the candidates.
            candidates.insert(0, chunk_code)
            best_chunk_code = chunk_code
            max_chunk_lo = -1

            for candidate_idx, cand_code in enumerate(candidates):
                is_valid, err_msg = self._syntax.parser.check_syntax(cand_code)
                if not is_valid:
                    if verbose: print(f"  [Candidate {candidate_idx}] Syntax Error (Skip): {err_msg}")
                    continue

                digest = self.get_structure_hash(cand_code)
                lo = count_leading_ones(digest)

                if lo > max_chunk_lo:
                    max_chunk_lo = lo
                    best_chunk_code = cand_code

            # variation_with_llm() strips its candidates, so restore the chunk's own trailing newline
            # (python line slices end with one, js/java byte slices end at }).
            if chunk_code.endswith('\n') and not best_chunk_code.endswith('\n'):
                best_chunk_code += '\n'

            final_pieces.append(best_chunk_code)
            total_lo += max_chunk_lo
            
            if verbose:
                print(f"    - [{func_name}] Selected Max LO: {max_chunk_lo}")

        final_source = "".join(final_pieces)
        
        if verbose:
            print("="*40)
            print(f"[*] Layer 2 Finished. Total LO Score: {total_lo}")

            pvalue = self.calculate_pvalue(total_lo, len(full_embedding_set))
            print(f"[*] Aggregated P-value: {pvalue:.2e}")
            print("="*40)

        return final_source, total_lo

    def detect(self, src: str) -> Dict[str, Any]:
        chunks = self._syntax.parser.get_code_chunks(src)

        results = []
        total_lo = 0
        num_blocks = 0

        # Same gate as embed(), so detection scans the chunks that received L2.
        verifiable_idxs = self._collect_verifiable_indices(chunks, verbose=False)
        full_embedding_set = set(self._select_full_embedding_indices(chunks, verifiable_idxs))

        for idx, (is_func, chunk_code, func_name) in enumerate(chunks):
            if not is_func:
                continue
            if idx not in full_embedding_set:
                continue

            is_valid, err_msg = self._syntax.parser.check_syntax(chunk_code)
            if is_valid:
                digest = self.get_structure_hash(chunk_code)
                lo = count_leading_ones(digest)

                total_lo += lo
                num_blocks += 1
                
                results.append({
                    "function": func_name,
                    "lo": lo,
                    "hash": digest.hex()[:8],  # short prefix for monitoring
                    "status": "success"
                })
            else:
                results.append({
                    "function": func_name,
                    "lo": 0,
                    "status": f"error: {err_msg}"
                })

        return {
            "results": results,
            "total_lo": total_lo,
            "num_blocks": num_blocks
        }

    def calculate_pvalue(self, total_lo: int, num_blocks: int) -> float:
        return calculate_negative_binomial_pvalue(k=total_lo, r=num_blocks, p=0.5)

def _test_embed():
    from code_snippets import get_snippet
    from language_parser import get_parser

    CASES = [
        ("py", "find_max_num_sorted"),
        ("py", "calculate_grade"),
        ("py", "score_segments"),
        ("java", "fibonacci"),
        ("java", "score_segments"),
    ]
    key = "abcd"

    for lang, name in CASES:
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

        syntax_embedder = SyntaxEmbedder(strategies, parser, key)
        struct_embedder = StructureEmbedder(syntax_embedder, lang=lang)

        print("=" * 80)
        print(f" {lang}/{name}")
        print("=" * 80)
        best_variant, best_lo = struct_embedder.embed(get_snippet(lang, name))
        print("\n[Selected Best Variant Code]")
        print(best_variant)
        print(struct_embedder.detect(best_variant))
        print()

if __name__ == "__main__":
    _test_embed()
