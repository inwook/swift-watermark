import hashlib
from typing import List, Dict, Any

from language_parser import LanguageParser
from rewriters_base import RewriteStrategy
from stats import calculate_binomial_pvalue

class SyntaxEmbedder:
    def __init__(self, strategies: List[RewriteStrategy], parser:LanguageParser, secret_key: str):
        self.strategies = strategies
        self.secret_key = secret_key.encode('utf-8')
        self.parser = parser

    def normalize(self, src: str) -> str:
        current_code = src
        for strategy in self.strategies:
            if strategy.is_target(current_code):
                current_code = strategy.normalize(current_code)
        return current_code

    def _get_expected_variation(self, canonical_code: str, strategy_name: str) -> int:
        # i* = H(C* || K || ID(R)) mod |V|

        canonical_code = self.parser.abstract_code(canonical_code)
        fingerprint_str = self.parser.dump_structure(canonical_code)

        h = hashlib.sha256()
        h.update(fingerprint_str.encode('utf-8'))   # Fingerprint(C*)
        h.update(self.secret_key)                   # K
        h.update(strategy_name.encode('utf-8'))     # ID(R)
        
        digest_int = int(h.hexdigest(), 16)
        # assume |V_j| == 2
        return digest_int % 2

    def embed(self, src: str) -> str:
        chunks = self.parser.get_code_chunks(src)
        result_pieces = []

        for is_func, chunk_code, func_name in chunks:
            if is_func:
                canonical_code = self.normalize(chunk_code)
                current_code = canonical_code

                for strategy in self.strategies:
                    if not strategy.is_target(canonical_code):
                        continue
                    
                    variation_index = self._get_expected_variation(canonical_code, strategy.name)
                    
                    if variation_index != 0:
                        current_code = strategy.rewrite(current_code, variation_index)
                
                # Restore the trailing newline that ast.unparse drops. js/java chunks end at } and must not gain one.
                if chunk_code.endswith('\n') and not current_code.endswith('\n'):
                    current_code += '\n'
                
                result_pieces.append(current_code)
            else:
                result_pieces.append(chunk_code)

        return "".join(result_pieces)

    def detect(self, src: str) -> Dict[str, Any]:
        chunks = self.parser.get_code_chunks(src)
        
        total_results = []
        total_n = 0
        total_s = 0

        for is_func, chunk_code, func_name in chunks:
            if is_func:
                canonical_code = self.normalize(chunk_code)
                
                for strategy in self.strategies:
                    if not strategy.is_target(canonical_code):
                        continue

                    total_n += 1
                    expected_val = self._get_expected_variation(canonical_code, strategy.name)
                    observed_val = strategy.get_current_state(chunk_code)

                    is_match = (expected_val == observed_val)
                    if is_match:
                        total_s += 1

                    total_results.append({
                        "function": func_name,
                        "strategy": strategy.name,
                        "expected": expected_val,
                        "observed": observed_val,
                        "match": is_match
                    })
            else:
                pass

        return {
            "results": total_results,
            "N_eff": total_n,
            "S_eff": total_s
        }

    def calculate_pvalue(self, n_eff: int, s_eff: int) -> float:
        """
        Calculates the P-value using the binomial distribution.
        Null Hypothesis (H0): The code is not watermarked (randomly generated, p=0.5).
        P-value = P(X >= s_eff) under B(n_eff, 0.5)
                = sum_{k=s_eff}^{n_eff} comb(n, k) * 0.5^n
        """
        assert s_eff <= n_eff

        return calculate_binomial_pvalue(n_eff, s_eff, p=0.5)

def _test_syntax_embed():
    import rewriters_py
    from language_parser import PythonParser
    from code_snippets import get_snippets
    strategies = rewriters_py.get_py_strategies()
    k = "abcd"
    syntax_embedder = SyntaxEmbedder(strategies, PythonParser(), k)

    for name, code in get_snippets("py").items():
        print(f"=== Testing src {name} ===")

        print("[Embedded Code]")
        embedded = syntax_embedder.embed(code)
        print(embedded)

        print("[Detection Result]")
        detection_res = syntax_embedder.detect(embedded)
        print(f"N_eff: {detection_res['N_eff']}, S_eff: {detection_res['S_eff']}")
        for r in detection_res['results']:
            print(f"  - {r['function']} | {r['strategy']}: expected {r['expected']}, observed {r['observed']} -> Match: {r['match']}")

if __name__ == "__main__":
    _test_syntax_embed()