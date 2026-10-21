import time
import warnings

from testset import load_testset
from paths import OUT_PATH
from watermarker import Watermarker
from runners import get_runner
from attacks import apply_attacks
from llm_provider import get_usage, diff_usage, get_provider, model_tag
from stats import is_significant, format_pvalue, get_rng


def experiment(testset, log_filename, eval_language: str, layers: str = "", attack: str = "", pass_check: str = "", *, llm_model: str | None = None):
    warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
    get_rng().seed(0)  # every random draw below comes from this RNG

    do_l1_embed = "L1" in layers
    do_l2_embed = "L2" in layers

    # Fail fast on a missing API key, since wm.embed() would swallow it per task.
    if do_l2_embed:
        get_provider(llm_model)

    wm = Watermarker(eval_language, llm_model=llm_model)
    test_runner = get_runner(eval_language)

    ntasks = len(testset)
    passed_tasks = 0

    detected_l1_cnt = [0, 0]  # [p<0.05, p<0.01]
    detected_l2_cnt = [0, 0]  # [p<0.05, p<0.01]
    detected_combined_cnt = [0, 0]  # [p<0.05, p<0.01]

    # Tasks where the layer had something to test (p is not None).
    l1_tested_cnt = 0
    l2_tested_cnt = 0

    total_enc_time = 0.0
    total_dec_time = 0.0

    strategy_stats = {name: {"n": 0, "s": 0} for name in wm.strategy_names}

    print(f"[*] Starting Experiment on {ntasks} tasks...")
    print(f"[*] Logging to: {log_filename}")
    print("-" * 140)
    print(f"{'Dataset':<10} | {'ID':<15} | {'Pass':<5} | {'Enc(ms)':<8} | {'Dec(ms)':<8} | {'L1 Summary (Syntax)':<25} | {'L2 Summary (Structure)':<25} | {'Active Strategies'}")
    print("-" * 140)

    flog = open(log_filename, "w", encoding="utf-8")

    usage_before = get_usage()

    for i, task in enumerate(testset):
        d_type = task['dataset']
        task_id = str(task['task_id'])
        original_code = task['code']
        test_payload = task['test_payload']

        watermarked_code = ""

        # Encode
        start_enc = time.perf_counter()

        try:
            watermarked_code = wm.embed(original_code, do_l1=do_l1_embed, do_l2=do_l2_embed)
        except Exception as e:
            print(f"{d_type:<10} | {task_id:<15} | ERR   | Encoding Failed: {e}")
            flog.write(f"[{d_type}] Task {task_id}: Encoding Error - {e}\n\n")
            continue

        end_enc = time.perf_counter()
        enc_duration = (end_enc - start_enc) * 1000
        total_enc_time += enc_duration

        if pass_check == "tests":
            test_code = "\n".join(test_payload)
            full_program = f"{watermarked_code}\n\n{test_code}"
            res = test_runner.run_code(full_program)
            if res["success"]:
                pass_status = "PASS"
                passed_tasks += 1
            else:
                pass_status = "FAIL"
        elif pass_check == "syntax":
            is_valid, _ = wm.parser.check_syntax(watermarked_code)
            if is_valid:
                pass_status = "PASS"
                passed_tasks += 1
            else:
                pass_status = "FAIL"
        elif pass_check == "":
            pass_status = "-"
        else:
            raise ValueError(f"Unknown pass_check: '{pass_check}'")

        # Attack
        if attack:
            watermarked_code = apply_attacks(watermarked_code, attack, eval_language, llm_model=llm_model)

        # Detect
        start_dec = time.perf_counter()

        result_det = wm.detect(watermarked_code, do_l1=True, do_l2=True)

        n_eff = result_det['l1_n']
        s_eff = result_det['l1_s']
        l1_pvalue = Watermarker.calculate_l1_pvalue(n_eff, s_eff)
        num_blocks = result_det['l2_blocks']
        l2_lo = result_det['l2_lo']
        l2_pvalue = Watermarker.calculate_l2_pvalue(num_blocks, l2_lo)
        combined_pvalue = Watermarker.calculate_combined_pvalue(l1_pvalue, l2_pvalue)

        end_dec = time.perf_counter()
        dec_duration = (end_dec - start_dec) * 1000
        total_dec_time += dec_duration

        if l1_pvalue is not None: l1_tested_cnt += 1
        if l2_pvalue is not None: l2_tested_cnt += 1

        # A layer that never ran (p is None) is never a detection.
        is_l1 = [is_significant(l1_pvalue, alpha) for alpha in (0.05, 0.01)]
        is_l2 = [is_significant(l2_pvalue, alpha) for alpha in (0.05, 0.01)]
        is_combined = [is_significant(combined_pvalue, alpha) for alpha in (0.05, 0.01)]

        # Per-strategy L1 breakdown.
        active_strategies_str = []
        strategy_breakdown_log = []
        current_stats = {}

        for item in result_det.get('l1_detail', []):
            if item.get('ignored', False):
                continue
            strat_name = item.get('strategy', 'Unknown')
            is_match = item.get('match', False)
            if strat_name not in current_stats:
                current_stats[strat_name] = {'n': 0, 's': 0}
            current_stats[strat_name]['n'] += 1
            if is_match:
                current_stats[strat_name]['s'] += 1

        for strat_name, stats in current_stats.items():
            sn, ss = stats['n'], stats['s']
            active_strategies_str.append(f"{strat_name}:{ss}/{sn}")
            strategy_breakdown_log.append(f"    - {strat_name:<20}: N={sn}, S={ss}")
            if strat_name in strategy_stats:
                strategy_stats[strat_name]['n'] += sn
                strategy_stats[strat_name]['s'] += ss

        if not current_stats and n_eff > 0:
            strategy_breakdown_log.append("    (No details found in results)")

        if is_l1[0]: detected_l1_cnt[0] += 1
        if is_l1[1]: detected_l1_cnt[1] += 1
        if is_l2[0]: detected_l2_cnt[0] += 1
        if is_l2[1]: detected_l2_cnt[1] += 1
        if is_combined[0]: detected_combined_cnt[0] += 1
        if is_combined[1]: detected_combined_cnt[1] += 1

        l1_mark = "***" if is_l1[1] else ("**" if is_l1[0] else "-")
        l2_mark = "***" if is_l2[1] else ("**" if is_l2[0] else "-")

        l1_str = f"N={n_eff} S={s_eff} p={format_pvalue(l1_pvalue, '.3f')} ({l1_mark})"
        if l2_pvalue is None:
            l2_p_disp = "n/a"
        else:
            l2_p_disp = f"{l2_pvalue:.3f}" if l2_pvalue > 0.001 else f"{l2_pvalue:.1e}"
        l2_str = f"N={num_blocks} SumLO={l2_lo} p={l2_p_disp} ({l2_mark})"

        strats_console = ", ".join(active_strategies_str)
        if len(strats_console) > 60:
            strats_console = strats_console[:57] + "..."
        if not strats_console: strats_console = "None"

        print(f"{d_type:<10} | {task_id:<15} | {pass_status:<5} | {enc_duration:>8.2f} | {dec_duration:>8.2f} | {l1_str:<25} | {l2_str:<25} | {strats_console}")

        flog.write("=" * 80 + "\n")
        flog.write(f"Dataset: {d_type} | Task ID: {task_id} | Pass: {pass_status}\n")
        flog.write(f"[Detection Summary]\n")
        flog.write(f"  > Layer 1: N={n_eff}, S={s_eff}, P={format_pvalue(l1_pvalue)} ({l1_mark})\n")
        flog.write(f"  > Layer 2: S(Total LO)={l2_lo}, M(Blocks)={num_blocks}, P={format_pvalue(l2_pvalue, '.4e')} ({l2_mark})\n")
        flog.write(f"  > Combined (Fisher): P={format_pvalue(combined_pvalue, '.4e')} ({'***' if is_combined[1] else ('**' if is_combined[0] else '-')})\n")
        flog.write(f"[Strategy Breakdown]\n")
        if strategy_breakdown_log:
            for log_line in strategy_breakdown_log:
                flog.write(log_line + "\n")
        else:
            flog.write("    (No strategies were applicable)\n")
        flog.write("-" * 30 + " [Original] " + "-" * 30 + "\n")
        flog.write(original_code + "\n")
        flog.write("-" * 30 + " [Encoded] " + "-" * 30 + "\n")
        flog.write(watermarked_code + "\n\n")

    flog.close()

    usage_diff = diff_usage(usage_before, get_usage())

    return {
        "log_filename": log_filename,
        "ntasks": ntasks,
        # None marks "no fidelity check ran", which is not the same as 0 passed.
        "passed_tasks": passed_tasks if pass_check else None,
        "detected_l1_cnt": detected_l1_cnt,
        "detected_l2_cnt": detected_l2_cnt,
        "detected_combined_cnt": detected_combined_cnt,
        "l1_tested_cnt": l1_tested_cnt,
        "l2_tested_cnt": l2_tested_cnt,
        "strategy_stats": strategy_stats,
        "total_enc_time": total_enc_time,
        "total_dec_time": total_dec_time,
        "llm_usage": usage_diff,
    }


def print_report(result: dict):
    ntasks = result["ntasks"]
    passed_tasks = result["passed_tasks"]
    detected_l1_cnt = result["detected_l1_cnt"]
    detected_l2_cnt = result["detected_l2_cnt"]
    detected_combined_cnt = result.get("detected_combined_cnt", [0, 0])
    l1_tested_cnt = result.get("l1_tested_cnt", ntasks)
    l2_tested_cnt = result.get("l2_tested_cnt", ntasks)
    strategy_stats = result["strategy_stats"]
    total_enc_time = result["total_enc_time"]
    total_dec_time = result["total_dec_time"]

    print("\n" + "="*80)
    print(f" [Final Report] Log saved to: {result['log_filename']}")
    print("="*80)
    print(f" Total Tasks        : {ntasks}")
    pass_str = ("-" if passed_tasks is None
                else f"{passed_tasks} ({passed_tasks / ntasks * 100:.1f}%)")
    print(f" Task Pass          : {pass_str}")
    print(f" L1 Applicable      : {l1_tested_cnt} ({l1_tested_cnt/ntasks*100:.1f}%)   "
          f"L2 Applicable: {l2_tested_cnt} ({l2_tested_cnt/ntasks*100:.1f}%)")
    print("-" * 80)
    print(f" Layer 1 Detected (p<0.05): {detected_l1_cnt[0]:<4} ({detected_l1_cnt[0]/ntasks*100:.1f}%)")
    print(f" Layer 1 Detected (p<0.01): {detected_l1_cnt[1]:<4} ({detected_l1_cnt[1]/ntasks*100:.1f}%)")
    print("-" * 40)
    print(f" Layer 2 Detected (p<0.05): {detected_l2_cnt[0]:<4} ({detected_l2_cnt[0]/ntasks*100:.1f}%)")
    print(f" Layer 2 Detected (p<0.01): {detected_l2_cnt[1]:<4} ({detected_l2_cnt[1]/ntasks*100:.1f}%)")
    print("-" * 40)
    print(f" Combined Detected (p<0.05): {detected_combined_cnt[0]:<4} ({detected_combined_cnt[0]/ntasks*100:.1f}%)")
    print(f" Combined Detected (p<0.01): {detected_combined_cnt[1]:<4} ({detected_combined_cnt[1]/ntasks*100:.1f}%)")
    print("-" * 80)
    print(" [Top Contributing Strategies (N/S)]")

    sorted_stats = sorted(strategy_stats.items(), key=lambda x: x[1]['n'], reverse=True)
    for name, stat in sorted_stats:
        if stat['n'] > 0:
            print(f"  - {name:<20}: Total N={stat['n']}, Total S={stat['s']} (Match Rate: {stat['s']/stat['n']*100:.1f}%)")

    avg_enc = total_enc_time / ntasks if ntasks > 0 else 0
    avg_dec = total_dec_time / ntasks if ntasks > 0 else 0
    print(f" [Performance Report] Avg Latency")
    print(f"  - Average Encoding Time: {avg_enc:.2f} ms")
    print(f"  - Average Decoding Time: {avg_dec:.2f} ms")

    usage = result.get("llm_usage")
    if usage and usage.get("n_calls", 0) > 0:
        n = usage["n_calls"]
        pt, ct, tt = usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]
        print(f" [LLM Usage] calls={n}  prompt={pt:,}  completion={ct:,}  total={tt:,}  "
              f"(avg {tt/n:.0f} tok/call, {tt/ntasks:.0f} tok/task)")


def run_experiment(setting: dict, *, llm_model: str | None = None):
    lang = setting["lang"]
    dataset = setting["dataset"]
    fold = setting.get("fold", 1)
    layers = setting.get("layers", "")
    nsamples = setting.get("nsamples", 0)
    attack = setting.get("attack", "")
    pass_check = setting.get("pass_check", "")

    testset = load_testset(dataset, lang=lang, nsamples=nsamples, fold=fold, validate=True)
    fold_suffix = f"_agg{fold}" if fold > 1 else ""
    attack_suffix = f"_{attack}" if attack else ""
    model_suffix = f"_{model_tag(llm_model)}"

    layers_tag = layers if layers else "noWM"
    log_filename = OUT_PATH(f"{dataset}{fold_suffix}_{lang}_{layers_tag}{attack_suffix}{model_suffix}_logs.txt")
    print(f"[*] Log file: {log_filename}")

    print("\n" + "=" * 40)
    print(f" TESTSET STATISTICS: {dataset}({fold})")
    print("=" * 40)
    print(f" Total Samples : {len(testset)}")
    print("=" * 40 + "\n")

    result = experiment(testset, log_filename, lang, layers=layers, attack=attack, pass_check=pass_check, llm_model=llm_model)
    print_report(result)
    return result


def _run_one_in_batch(i: int, n: int, setting: dict, *, llm_model: str | None = None):
    print(f"\n{'#'*80}")
    print(f" EXPERIMENT {i+1}/{n}: {setting}")
    print(f"{'#'*80}\n")
    result = run_experiment(setting, llm_model=llm_model)
    print(f" EXPERIMENT {i+1}/{n} finished.")
    return result


def run_experiment_batch(experiments: list, *, n_workers: int = 1, llm_model: str | None = None):
    """Run experiments, threaded when n_workers > 1. Results stay in input order.
    llm_model is used by every L2 embed and LLM attack (None = provider default).
    """
    n = len(experiments)
    if n_workers <= 1:
        return [_run_one_in_batch(i, n, s, llm_model=llm_model) for i, s in enumerate(experiments)]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_run_one_in_batch, i, n, s, llm_model=llm_model) for i, s in enumerate(experiments)]
        return [f.result() for f in futures]


if __name__ == "__main__":
    EXPERIMENTS = [
        {"lang": "py",   "dataset": "humaneval-x", "fold": 8, "layers": "L1L2", "nsamples": 0},
    ]

    run_experiment_batch(EXPERIMENTS)
