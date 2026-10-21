import textwrap
from typing import Dict


def _clean(code: str) -> str:
    """Strip common leading whitespace and trim surrounding whitespace."""
    return textwrap.dedent(code).strip()


SNIPPETS: Dict[str, Dict[str, str]] = {
    "py": {
        "calculate_grade": _clean("""
            def calculate_grade(scores):
                if not scores:
                    return 0
                total = sum(scores)
                average = total / len(scores)
                return average * 1.1
        """),
        "truncate_number": _clean("""
            def truncate_number(number: float) -> float:
                return number % 1.0
        """),
        # find_max_num / find_max_num_sorted: an equivalent pair.
        "find_max_num": _clean("""
            def find_Max_Num(arr) : 
                n = len(arr)
                if n == 0:
                    return 0
                arr.sort(reverse = True) 
                num = arr[0] 
                for i in range(1,n) : 
                    num = num * 10 + arr[i] 
                return num
        """),
        "find_max_num_sorted": _clean("""
            def find_Max_Num(arr):
                if not arr:
                    return 0
                sorted_digits = sorted(arr, reverse=True)
                result_str = "".join(map(str, sorted_digits))
                return int(result_str)        
        """),
        "magic_square_test": _clean("""
            def magic_square_test(my_matrix):
                iSize = len(my_matrix[0])
                sum_list = []
                sum_list.extend([sum (lines) for lines in my_matrix])   
                for col in range(iSize):
                    sum_list.append(sum(row[col] for row in my_matrix))
                result1 = 0
                for i in range(0,iSize):
                    result1 +=my_matrix[i][i]
                sum_list.append(result1)      
                result2 = 0
                for i in range(iSize-1,-1,-1):
                    result2 +=my_matrix[i][i]
                sum_list.append(result2)
                if len(set(sum_list))>1:
                    return False
                return True
        """),
        "calculate_metrics": _clean("""
            def calculate_metrics(data, factor):
                total = 0
                count = 0
                for x in data:
                    if x > 0:
                        total = total + x
                        count = count + 1
                
                if count == 0:
                    return 0.0
                
                avg = total / count
                return avg * factor
        """),
        "complex_accumulator": _clean("""
            def complex_accumulator(numbers, threshold):
                sum_val = 0
                prod_val = 1
                
                for n in range(numbers):
                    sum_val = sum_val + n
                    if n > threshold:
                        prod_val = prod_val * n
                    else:
                        prod_val = prod_val * 1
                
                result = sum_val + prod_val
                return result
        """),
        "fibonacci": _clean("""
            def fibonacci(n):
                result = []
                a, b = 0, 1
                for i in range(n):
                    result.append(a)
                    a, b = b, a + b
                return result
        """),
        "score_segments": _clean("""
            def score_segments(values, threshold):
                if not values:
                    return [0, 0, 0]

                best_len = 0
                best_sum = 0
                run_len = 0
                run_sum = 0
                peaks = 0

                for i in range(len(values)):
                    v = values[i]
                    if v > threshold:
                        run_len += 1
                        run_sum += v
                        if i > 0 and values[i - 1] <= threshold:
                            peaks += 1
                    else:
                        if run_len > best_len or (run_len == best_len and run_sum > best_sum):
                            best_len = run_len
                            best_sum = run_sum
                        run_len = 0
                        run_sum = 0

                if run_len > best_len or (run_len == best_len and run_sum > best_sum):
                    best_len = run_len
                    best_sum = run_sum

                return [best_len, best_sum, peaks]
        """),
    },
    "js": {
        "fibonacci": _clean("""
            function fibonacci(n) {
                let result = [];
                let a = 0, b = 1;
                for (let i = 0; i < n; i++) {
                    result.push(a);
                    [a, b] = [b, a + b];
                }
                return result;
            }
        """),
    },
    "java": {
        "fibonacci": _clean("""
            public class Fib {
                public static int[] fibonacci(int n) {
                    int[] result = new int[n];
                    int a = 0, b = 1;
                    for (int i = 0; i < n; i++) {
                        result[i] = a;
                        int temp = a;
                        a = b;
                        b = temp + b;
                    }
                    return result;
                }
            }
        """),
        "calculator": _clean("""
            public class Calculator {
                public static void main(String[] args) {
                    int result = add(5, 3);
                    System.out.println(result);
                }

                public static int add(int a, int b) {
                    return a + b;
                }
            }
        """),
        # Java counterpart of py/score_segments.
        "score_segments": _clean("""
            public class Segments {
                public static int[] scoreSegments(int[] values, int threshold) {
                    if (values.length == 0) {
                        return new int[] {0, 0, 0};
                    }

                    int bestLen = 0;
                    int bestSum = 0;
                    int runLen = 0;
                    int runSum = 0;
                    int peaks = 0;

                    for (int i = 0; i < values.length; i++) {
                        int v = values[i];
                        if (v > threshold) {
                            runLen = runLen + 1;
                            runSum = runSum + v;
                            if (i > 0 && values[i - 1] <= threshold) {
                                peaks = peaks + 1;
                            }
                        } else {
                            if (runLen > bestLen || (runLen == bestLen && runSum > bestSum)) {
                                bestLen = runLen;
                                bestSum = runSum;
                            }
                            runLen = 0;
                            runSum = 0;
                        }
                    }

                    if (runLen > bestLen || (runLen == bestLen && runSum > bestSum)) {
                        bestLen = runLen;
                        bestSum = runSum;
                    }

                    return new int[] {bestLen, bestSum, peaks};
                }
            }
        """),
    },
}


def get_snippets(lang: str) -> Dict[str, str]:
    """All snippets for a language, as an ordered {name: code} dict."""
    if lang not in SNIPPETS:
        raise KeyError(f"no snippets for lang '{lang}'; available: {list(SNIPPETS)}")
    return SNIPPETS[lang]


def get_snippet(lang: str, name: str) -> str:
    """One snippet, addressed by (lang, name). Raises on an unknown name."""
    snippets = get_snippets(lang)
    if name not in snippets:
        raise KeyError(f"no snippet '{lang}/{name}'; available: {list(snippets)}")
    return snippets[name]
