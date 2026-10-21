import math
import random
import threading

def count_leading_ones(digest: bytes) -> int:
    count = 0
    for byte in digest:
        if byte == 0xFF: # 11111111
            count += 8
            continue

        mask = 0x80 # 10000000
        while byte & mask:
            count += 1
            mask >>= 1
        break

    return count

def calculate_binomial_pvalue(n: int, s: int, p: float = 0.5) -> float:
    """
    H0: X ~ B(n, p)
    P(X >= s) = sum_{k=s}^{n} comb(n, k) * p^k * (1-p)^{n-k}
    """
    if n <= 0:
        return 1.0
    if s > n:
        return 0.0
    if s <= 0:
        return 1.0

    if n > 1000:
        mu = n * p
        var = n * p * (1 - p)
        # Continuity correction (s - 0.5) for discrete -> continuous approx.
        return calculate_normal_approx_pvalue(mu, var, s - 0.5)

    # Shortcut: p == 0.5
    if p == 0.5:
        prob_factor = 0.5 ** n
        return sum(math.comb(n, k) for k in range(s, n + 1)) * prob_factor

    pvalue = 0.0
    for k in range(s, n + 1):
        term = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        pvalue += term
    return min(1.0, pvalue)

def calculate_negative_binomial_pvalue(k: int, r: int, p: float = 0.5) -> float:
    """
    H0: X ~ NB(r, p) - The number of failures X until r successes occur.
        - p: Probability of success in each independent Bernoulli trial.

    P(X >= k): The probability of observing k or more failures (P-value).
    """

    if r <= 0:
        return 1.0
    if k < 0:
        return 1.0

    if r > 200 or k > 200:
        mu = r * (1 - p) / p
        var = r * (1 - p) / (p ** 2)
        # Continuity correction (k - 0.5) for discrete -> continuous approx.
        return calculate_normal_approx_pvalue(mu, var, k - 0.5)

    if p == 0.5:
        cdf_sum = 0.0
        prob_base = 0.5 ** r
        for i in range(k):
            # PMF: comb(i + r - 1, i) * 0.5^i * 0.5^r
            pmf_i = math.comb(i + r - 1, i) * (0.5 ** i)
            cdf_sum += pmf_i
        
        pvalue = 1.0 - (cdf_sum * prob_base)
        return max(0.0, min(1.0, pvalue))

    cdf_sum = 0.0
    for i in range(k):
        pmf_i = math.comb(i + r - 1, i) * (p ** r) * ((1 - p) ** i)
        cdf_sum += pmf_i
        
    return max(0.0, min(1.0, 1.0 - cdf_sum))

def calculate_normal_approx_pvalue(mu: float, var: float, x: float) -> float:
    """Upper-tail probability via normal approximation."""
    std_dev = math.sqrt(var)
    if std_dev == 0:
        return 1.0 if x <= mu else 0.0
    
    z = (x - mu) / std_dev
    
    # P(X >= x) = 1 - Phi(z)
    phi_z = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return max(0.0, min(1.0, 1.0 - phi_z))

_COMBINE_EPS = 1e-300


def _chi2_sf_even_df(x: float, df: int) -> float:
    """Exact survival function for chi-square with even df.

    P(X > x) = exp(-x/2) * sum_{i=0}^{m-1} (x/2)^i / i!   where df = 2m.
    The series is accumulated iteratively so large x does not overflow i!.
    """
    if x <= 0:
        return 1.0
    half = x / 2.0
    term = 1.0
    total = 1.0
    for i in range(1, df // 2):
        term *= half / i
        total += term
    return total * math.exp(-half)


def combine_pvalues(*pvalues) -> float:
    """Fisher's method: Stat = -2 * sum(ln p_i) ~ chi-square(df = 2k) under H0.
    None marks a test that was not performed and is dropped. A test that ran and found nothing passes its real p-value."""
    ps = [max(min(p, 1.0), _COMBINE_EPS) for p in pvalues if p is not None]
    if not ps:
        return 1.0
    if len(ps) == 1:
        return ps[0]
    stat = -2.0 * sum(math.log(p) for p in ps)
    return _chi2_sf_even_df(stat, df=2 * len(ps))


def is_significant(p, alpha: float) -> bool:
    return p is not None and p < alpha


def format_pvalue(p, spec: str = ".6f", absent: str = "n/a") -> str:
    return absent if p is None else format(p, spec)


# "-" means tested but not significant, absent means not performed.
def p_mark(p, absent: str = "n/a") -> str:
    if p is None:
        return absent
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    return "-"


# Bonferroni correction for m simultaneous tests: min(1, p * m).
def bonferroni_correct(p: float, m: int) -> float:
    if m <= 1:
        return min(1.0, max(0.0, p))
    return min(1.0, max(0.0, p) * m)


# One RNG per worker thread, so a parallel batch reproduces a sequential one.
_tls = threading.local()

def get_rng() -> random.Random:
    r = getattr(_tls, "rng", None)
    if r is None:
        r = _tls.rng = random.Random()
    return r


def stochastic_round(x):
    floor_val = int(x)
    frac = x - floor_val
    return floor_val + (1 if get_rng().random() < frac else 0)


def _test_combine_pvalues():
    print("\n[Combined P-value] Fisher's method")
    cases = [
        (1.0,   1.0,   "no signal"),
        (0.5,   0.5,   "both moderate"),
        (0.05,  0.05,  "both marginal"),
        (0.01,  0.01,  "both strong"),
        (0.5,   0.01,  "moderate + strong"),
        (0.001, 0.001, "both very strong"),
    ]
    for p1, p2, label in cases:
        p = combine_pvalues(p1, p2)
        print(f"  > ({p1:>6}, {p2:>6})  {label:<22}  combined={p:.4e}")

    print("\n[Combined P-value] Absent test (None) vs. performed-but-empty (1.0)")
    for p1 in (0.05, 0.01, 0.001):
        absent = combine_pvalues(p1, None)
        empty = combine_pvalues(p1, 1.0)
        print(f"  > L1 p={p1:<7g} L2 absent -> {absent:.4e}   L2 ran, no signal -> {empty:.4e}")

    print("\n[Combined P-value] Edge cases (clamped, must not crash)")
    for p1, p2 in [(0.0, 0.0), (0.0, 0.5), (1.0001, 0.5)]:
        p = combine_pvalues(p1, p2)
        print(f"  > ({p1!r}, {p2!r})  combined={p:.4e}")

    # sanity asserts
    assert combine_pvalues(1.0, 1.0) == 1.0
    assert combine_pvalues(0.0, 0.0) < 1e-299
    assert 0.59 < combine_pvalues(0.5, 0.5) < 0.60
    assert combine_pvalues(0.01, 0.01) < combine_pvalues(0.05, 0.05)

    # A dropped test must leave the surviving p-value untouched (df=2 identity),
    # and must never be worse than feeding it 1.0.
    for p1 in (0.5, 0.05, 0.01, 1e-8):
        assert combine_pvalues(p1, None) == p1
        assert combine_pvalues(None, p1) == p1
        assert combine_pvalues(p1, None) <= combine_pvalues(p1, 1.0)
    assert combine_pvalues(None, None) == 1.0
    assert combine_pvalues() == 1.0

    # df must grow with the number of surviving tests.
    assert abs(combine_pvalues(0.1, 0.1, 0.1) - _chi2_sf_even_df(-6 * math.log(0.1), 6)) < 1e-12

    # is_significant / format_pvalue treat None as "not performed".
    assert is_significant(0.01, 0.05) and not is_significant(0.06, 0.05)
    assert not is_significant(None, 0.05)
    assert format_pvalue(None) == "n/a" and format_pvalue(0.5, ".2f") == "0.50"
    print("  [OK] all asserts passed")

def _test_binomial():
    """Layer 1: s matches out of n trials. n > 1000 routes through the normal approx."""
    print("\n[Layer 1] Binomial Distribution (n=20, p=0.5)")
    for s in [5, 10, 15, 20]:
        pvalue = calculate_binomial_pvalue(n=20, s=s, p=0.5)
        print(f"  > Match(s)={s:2d} / N=20 | P-value={pvalue:.6f}]")

    print("\n[Layer 1] Large Sample -- Normal Approximation (n=2000, p=0.5)")
    s_large = 1100  # well above the expected 1000
    pvalue_large = calculate_binomial_pvalue(n=2000, s=s_large, p=0.5)
    print(f"  > Match(s)={s_large} / N=2000 | P-value={pvalue_large:.4e}")


def _test_negative_binomial():
    """Layer 2: k leading-one bits across r function blocks."""
    print("\n[Layer 2] Negative Binomial (r=5, p=0.5)")
    for k in [5, 10, 15, 20]:
        pvalue = calculate_negative_binomial_pvalue(k=k, r=5, p=0.5)
        print(f"  > Total LO(k)={k:2d} / Blocks(r)=5 | P-value={pvalue:.6f}]")

    print("\n[Layer 2] Large Sample (k=r, p=0.5)")
    k_large = r_large = 2500
    pvalue = calculate_negative_binomial_pvalue(k=k_large, r=r_large, p=0.5)
    print(f"  > Total LO(k)={k_large:2d} / Blocks(r)={r_large} | P-value={pvalue:.6f}]")


def _test_stochastic_round():
    """The mean over many draws should converge to the un-rounded value."""
    print("\n[Stochastic Round] mean over 1,000,000 draws of 3.1415")
    sr = [stochastic_round(3.1415) for _ in range(1000000)]
    print(f"  > mean={sum(sr)/len(sr)}  values={set(sr)}")


if __name__ == "__main__":
    _test_binomial()
    _test_negative_binomial()
    _test_combine_pvalues()
    _test_stochastic_round()
