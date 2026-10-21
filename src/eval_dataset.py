import time
from datetime import datetime, timedelta

from experiment import run_experiment_batch
from llm_provider import resolve_model
from testset import get_testset_stats, load_testset

METRICS = {
    "pass_rate":  ("Pass%",     "{:>7.1f}%", lambda s, r: None if r.get("passed_tasks") is None else r["passed_tasks"] / r["ntasks"] * 100),
    "l1_det_05":  ("L1 Det@.05","{:>7.1f}%", lambda s, r: r["detected_l1_cnt"][0] / r["ntasks"] * 100),
    "l1_det_01":  ("L1 Det@.01","{:>7.1f}%", lambda s, r: r["detected_l1_cnt"][1] / r["ntasks"] * 100),
    "l2_det_05":  ("L2 Det@.05","{:>7.1f}%", lambda s, r: r["detected_l2_cnt"][0] / r["ntasks"] * 100),
    "l2_det_01":  ("L2 Det@.01","{:>7.1f}%", lambda s, r: r["detected_l2_cnt"][1] / r["ntasks"] * 100),
    "l2_cov":     ("L2 Cov",    "{:>7.1f}%", lambda s, r: r.get("l2_tested_cnt", r["ntasks"]) / r["ntasks"] * 100),
    "l2_detc_05": ("L2 DetC@.05","{:>7.1f}%", lambda s, r: (lambda t: r["detected_l2_cnt"][0] / t * 100 if t else 0)(r.get("l2_tested_cnt", r["ntasks"]))),
    "l2_detc_01": ("L2 DetC@.01","{:>7.1f}%", lambda s, r: (lambda t: r["detected_l2_cnt"][1] / t * 100 if t else 0)(r.get("l2_tested_cnt", r["ntasks"]))),
    "cb_det_05":  ("Cb Det@.05","{:>7.1f}%", lambda s, r: r.get("detected_combined_cnt", [0,0])[0] / r["ntasks"] * 100),
    "cb_det_01":  ("Cb Det@.01","{:>7.1f}%", lambda s, r: r.get("detected_combined_cnt", [0,0])[1] / r["ntasks"] * 100),
    "avg_n":      ("Avg N",     "{:>7.2f}",  lambda s, r: sum(v["n"] for v in r["strategy_stats"].values()) / r["ntasks"] if r["ntasks"] else 0),
    "match_rate": ("Match%",    "{:>7.1f}%", lambda s, r: (lambda n, sc: sc / n * 100 if n else 0)(sum(v["n"] for v in r["strategy_stats"].values()), sum(v["s"] for v in r["strategy_stats"].values()))),
    "avg_enc":    ("Enc(ms)",   "{:>8.2f}",  lambda s, r: r["total_enc_time"] / r["ntasks"] if r["ntasks"] else 0),
    "avg_dec":    ("Dec(ms)",   "{:>8.2f}",  lambda s, r: r["total_dec_time"] / r["ntasks"] if r["ntasks"] else 0),
    "llm_calls":  ("Calls",     "{:>8,}",    lambda s, r: r.get("llm_usage", {}).get("n_calls", 0)),
    "llm_prompt": ("PromptTok", "{:>10,}",   lambda s, r: r.get("llm_usage", {}).get("prompt_tokens", 0)),
    "llm_comp":   ("ComplTok",  "{:>10,}",   lambda s, r: r.get("llm_usage", {}).get("completion_tokens", 0)),
}

SETTING_COLUMNS = {
    "lang":    ("Lang",    "{:<6}"),
    "dataset": ("Dataset", "{:<14}"),
    "layers":  ("Layers",  "{:<8}"),
    "attack":  ("Attack",  "{:<12}"),
    "pass_check": ("Pass by", "{:<8}"),
    "fold":    ("Fold",    "{:>4}"),
    "ntasks":  ("Tasks",   "{:>5}"),
}

ALL_LANGS = ["py", "java", "js"]
ALL_DATASETS = [
    "MBXP",
    "csn",
    #"humaneval-x"
]


def evaluation(eval_config, metrics=None, columns=None, *, n_workers=1, llm_model=None):
    """Run a batch of experiments and print a summary table. eval_config is (title, experiments).
    In-flight LLM calls are roughly n_workers x LLM_VARIATION_THREADS, so keep n_workers small under rate limits.
    """
    title, experiments = eval_config
    t_start = datetime.now()
    t0 = time.perf_counter()
    results = run_experiment_batch(experiments, n_workers=n_workers, llm_model=llm_model)
    elapsed = time.perf_counter() - t0
    t_end = datetime.now()
    assert len(experiments) == len(results)

    if metrics is None:
        metrics = list(METRICS.keys())
    if columns is None:
        columns = [k for k in SETTING_COLUMNS if any(k in s for s in experiments) or any(k in r for r in results)]

    # Column width: max(label length, formatted sample length)
    metric_widths = {}
    for key in metrics:
        label, fmt, _ = METRICS[key]
        sample = fmt.format(100.0)
        metric_widths[key] = max(len(label), len(sample))

    header_parts = []
    for col in columns:
        label, fmt = SETTING_COLUMNS[col]
        header_parts.append(fmt.format(label))

    metric_header_parts = []
    for key in metrics:
        label = METRICS[key][0]
        w = metric_widths[key]
        metric_header_parts.append(f"{label:>{w}}")

    header = "  ".join(header_parts) + " | " + "  ".join(metric_header_parts)
    sep = "-" * len(header)

    par_str = f"parallel n_workers={n_workers}" if n_workers > 1 else "sequential"
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f"\n{'='*len(header)}")
    print(f" Evaluation Summary: {title}")
    print(f" Model: {resolve_model(llm_model)}")
    print(f" Mode: {par_str}  |  {len(experiments)} experiments")
    print(f" Start: {t_start:%Y-%m-%d %H:%M:%S}  End: {t_end:%Y-%m-%d %H:%M:%S}  Elapsed: {elapsed_str}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for setting, result in zip(experiments, results):
        row_parts = []
        for col in columns:
            _, fmt = SETTING_COLUMNS[col]
            val = setting.get(col, result.get(col, ""))
            if val == "" and col == "attack":
                val = "-"
            if val == "" and col == "layers":
                val = "noWM"
            row_parts.append(fmt.format(val))

        metric_parts = []
        for key in metrics:
            _, fmt, extractor = METRICS[key]
            val = extractor(setting, result)
            raw = "-" if val is None else fmt.format(val)
            w = metric_widths[key]
            metric_parts.append(f"{raw:>{w}}")

        print("  ".join(row_parts) + " | " + "  ".join(metric_parts))

    print(sep)
    print()


# Eval configs return (title, [setting, ...]).
# nsamples caps the samples drawn before folding, so a fold=10 row yields nsamples/10 tasks.
# The values below are sized for a quick first run. nsamples=0 uses the full testset.

def stat_datasets():
    """Print a table of (samples, NCLOC, avg functions) per (dataset, lang)."""
    header = f"{'Dataset':<14}  {'Lang':<6}  {'Samples':>8}  {'NCLOC':>10}  {'AvgFuncs':>10}"
    print(header)
    print("-" * len(header))
    for dataset in ALL_DATASETS:
        for lang in ALL_LANGS:
            try:
                testset = load_testset(dataset, lang=lang, nsamples=0, fold=1, validate=False)
            except Exception as e:
                print(f"{dataset:<14}  {lang:<6}  {'ERR':>8}  load failed: {e}")
                continue
            ncloc, avg_funcs = get_testset_stats(testset, lang)
            print(f"{dataset:<14}  {lang:<6}  {len(testset):>8}  {ncloc:>10.2f}  {avg_funcs:>10.2f}")
        print()

def eval_fidelity():
    experiments = []
    fold = 1
    for dataset in ALL_DATASETS:
        for lang in ALL_LANGS:
            pass_check = "syntax" if dataset == "csn" else "tests"

            # L1 watermarking
            experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 100, "pass_check": pass_check})

            # L2 watermarking
            experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L2", "nsamples": 100, "pass_check": pass_check})

    return "fidelity", experiments


def eval_detection():
    experiments = []
    for dataset in ALL_DATASETS:
        for fold in [10]:
            for lang in ALL_LANGS:
                #FPR
                experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "", "nsamples": 200})

                #TPR L1
                experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 200})

                #TPR L2 only - not used
                #experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L2", "nsamples": 200})

                #TPR L1L2
                experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1L2", "nsamples": 200})

    return "detection", experiments


def eval_disclosure():
    experiments = []
    dataset, fold, attack_str = "MBXP", 10, "ST-100"
    for lang in ALL_LANGS:
        experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 0, "attack": attack_str})
        experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1L2", "nsamples": 0, "attack": attack_str})

    return "robustness-full-disclosure", experiments


def eval_folding():
    experiments = []
    for fold in [2, 4, 6, 8, 10]:
        for dataset in ALL_DATASETS:
            for lang in ALL_LANGS:
                experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 200})
                experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1L2", "nsamples": 200})

    return "folding", experiments


def eval_robustness():
    experiments = []

    # Comment out the attacks you don't want for this run.
    # T stops at 50%: at a full budget targeted renaming degenerates into L-100.
    ATTACKS = [
        ("R", [20, 50, 100]),      # random rename
        ("L", [20, 50, 100]),      # blind LLM rename
        ("T", [20, 50]),           # targeted LLM rename
        ("ST", [10, 30, 50]),      # syntactic transform
    ]

    for dataset in ALL_DATASETS:
        for fold in [10]:
            for lang in ALL_LANGS:
                # before attack
                experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 200})

                for attack_type, rates in ATTACKS:
                    for rate in rates:
                        attack_str = attack_type if rate is None else f"{attack_type}-{rate}"

                        experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 200, "attack": attack_str})
                        experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1L2", "nsamples": 200, "attack": attack_str})

    return "robustness", experiments


def eval_tools():
    ATTACKS = {
        "py":   ["MINI-F", "MINI-M", "MINI-C", "MINI"],
        "js":   ["MINI-F", "MINI-M", "MINI-C", "MINI", "OBFU"],
    }

    dataset, fold = "MBXP", 10
    experiments = []
    for lang in ATTACKS:
        # before attack
        experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 200})
        experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1L2", "nsamples": 200})

        for attack_str in ATTACKS[lang]:
            experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1", "nsamples": 200, "attack": attack_str})
            experiments.append({"lang": lang, "dataset": dataset, "fold": fold, "layers": "L1L2", "nsamples": 200, "attack": attack_str})

    return "robustness-tools", experiments


if __name__ == "__main__":
    # L2 rows and LLM attacks need a provider key in .env (see llm_provider.get_provider).

    # Not an eval config, prints a dataset stats table.
    #stat_datasets()

    # Comment out the configs you don't want for this run.
    RUN = [
        eval_fidelity,
        eval_detection,
        eval_robustness,
        eval_tools,

        # Auxiliary evaluations
        # eval_folding,
        # eval_disclosure,
    ]

    # n_workers>1 runs the batch in threads. Results stay in input order.
    for eval_config in RUN:
        evaluation(eval_config(), n_workers=1)
