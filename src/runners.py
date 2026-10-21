import os
import io
import subprocess
import tempfile
import contextlib
import warnings
import re
from abc import ABC, abstractmethod
from paths import TMP_PATH, _PROJECT_ROOT

class BaseRunner(ABC):
    def __init__(self, timeout, suffix=".tmp", tmp_dir=None):
        self.timeout = timeout
        self.suffix = suffix

        self.tmp_dir = os.path.abspath(tmp_dir) if tmp_dir else TMP_PATH()

    @abstractmethod
    def get_run_command(self, file_path: str) -> list:
        pass

    def run_code(self, code: str) -> dict:
        with tempfile.NamedTemporaryFile(dir=self.tmp_dir, suffix=self.suffix, delete=False, mode='w', encoding='utf-8') as tf:
            tf.write(code)
            tmp_path = tf.name

        try:
            cmd = self.get_run_command(tmp_path)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=self.timeout
            )
            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stderr": "Timeout", "exit_code": -1, "stdout": ""}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class PythonRunner(BaseRunner):
    def __init__(self):
        # Runs in-process via exec(), so timeout and tmp_dir are unused.
        super().__init__(timeout=0, suffix=".py")

        import math, re, string, collections, datetime
        self.base_globals = {
            'math': math,
            're': re,
            'string': string,
            'collections': collections,
            'datetime': datetime,
            '__builtins__': __builtins__
        }

    def get_run_command(self, file_path: str) -> list:
        # not used
        return []

    def run_code(self, code: str) -> dict:
        success = False
        error_msg = ""
        stdout_buffer = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(stdout_buffer):
                    exec_globals = self.base_globals.copy()
                    exec(code, exec_globals)

            success = True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"

        return {
            "stdout": stdout_buffer.getvalue().strip(),
            "stderr": error_msg,
            "exit_code": 0 if success else 1,
            "success": success,
        }


class JSRunner(BaseRunner):
    def __init__(self, timeout=5):
        super().__init__(timeout, suffix=".js")

        self.node_dir = _PROJECT_ROOT
        if not os.path.exists(os.path.join(self.node_dir, "node_modules", "lodash")):
            print(f"[*] npm install at '{self.node_dir}'")
            try:
                subprocess.run(
                    ["npm", "install", "lodash"],
                    cwd=self.node_dir,
                    capture_output=True,
                    check=True
                )
                print("[*] lodash install finished.")
            except Exception as e:
                print(f"[!] npm install error: {e}")
                raise

    def get_run_command(self, file_path: str) -> list:
        return ["node", file_path]


class JavaRunner(BaseRunner):
    def __init__(self, timeout=10):
        super().__init__(timeout, suffix=".java")

        try:
            result = subprocess.run(
                ["javac", "-version"],
                capture_output=True
            )
            if result.returncode == 0:
                pass
            else:
                err_msg = result.stderr.decode('utf-8', errors='ignore').strip() or "Unknown error"
                print(f"[!] Error: Failed to run javac. {err_msg}")
                raise RuntimeError(err_msg)
        except FileNotFoundError:
            print("[!] Error: javac not found in PATH. Please install JDK.")
            raise
        except Exception as e:
            print(f"[!] Unexpected error while executing javac: {e}")
            raise

    def get_run_command(self, file_path: str) -> list:
        class_name = os.path.splitext(os.path.basename(file_path))[0]
        return ["java", class_name]

    def run_code(self, code: str) -> dict:
        file_class = None
        main_class = None

        # The public class names the file (LLM output may add public).
        pub_match = re.search(r'public\s+class\s+(\w+)', code)
        if pub_match:
            file_class = pub_match.group(1)

        # The class that owns main is the one to run.
        main_match = re.search(r'public\s+static\s+void\s+main\s*\(', code)
        if main_match:
            before_main = code[:main_match.start()]
            class_matches = list(re.finditer(r'\bclass\s+(\w+)', before_main))
            if class_matches:
                main_class = class_matches[-1].group(1)

        if not file_class:
            file_class = main_class or "Main"
        if not main_class:
            main_class = file_class

        if not file_class or not main_class:
            return {"success": False, "stderr": "Cannot determine class names", "exit_code": -1, "stdout": ""}

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as temp_dir:
            java_file = os.path.join(temp_dir, f"{file_class}.java")

            try:
                with open(java_file, 'w', encoding='utf-8') as f:
                    f.write(code)

                compile_result = subprocess.run(
                    ["javac", "-encoding", "UTF-8", java_file],
                    capture_output=True, timeout=self.timeout, cwd=temp_dir
                )

                if compile_result.returncode != 0:
                    compile_stderr = compile_result.stderr.decode('utf-8', errors='replace').strip()
                    return {
                        "stdout": compile_result.stdout.decode('utf-8', errors='replace').strip(),
                        "stderr": f"Compilation Error:\n{compile_stderr}",
                        "exit_code": compile_result.returncode,
                        "success": False
                    }

                run_result = subprocess.run(
                    ["java", "-Dfile.encoding=UTF-8", main_class],
                    capture_output=True, timeout=self.timeout, cwd=temp_dir
                )

                return {
                    "stdout": run_result.stdout.decode('utf-8', errors='replace').strip(),
                    "stderr": run_result.stderr.decode('utf-8', errors='replace').strip(),
                    "exit_code": run_result.returncode,
                    "success": run_result.returncode == 0
                }

            except subprocess.TimeoutExpired:
                return {"success": False, "stderr": "Timeout", "exit_code": -1, "stdout": ""}
            except Exception as e:
                return {"success": False, "stderr": f"Error: {str(e)}", "exit_code": -1, "stdout": ""}

def get_runner(lang):
    if lang == "py":
        return PythonRunner()
    elif lang == "js":
        return JSRunner()
    elif lang == "java":
        return JavaRunner()
    else:
        raise ValueError(f"Unsupported language: '{lang}'")

def _test_py_runner():
    import testset
    dataset = testset.load_mbpp_testset()
    results = {"pass": 0, "fail": 0}

    py_runner = PythonRunner()

    print("[*] Starting Evaluation...\n" + "="*40)
    for i, data in enumerate(dataset):
        test_code = "\n".join(data["test_payload"])
        full_program = f"{data['code']}\n\n{test_code}"

        res = py_runner.run_code(full_program)

        if res["success"]:
            results["pass"] += 1
        else:
            results["fail"] += 1
            print(res["stderr"])

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(dataset)}] Pass: {results['pass']} | Fail: {results['fail']}")

    print(f"[*] Pass: {results['pass']} | Fail: {results['fail']}")

def _test_js_runner():
    import testset
    dataset = testset.load_mbjsp_testset(nsamples=100)
    js_runner = JSRunner()

    results = {"pass": 0, "fail": 0, "timeout": 0}
    failed_cases = []

    print("\n[*] Starting Evaluation...\n" + "="*40)
    for i, data in enumerate(dataset):
        task_id = data["task_id"]

        test_code = "\n".join(data["test_payload"])
        full_program = f"{data['code']}\n\n{test_code}"
        res = js_runner.run_code(full_program)

        if res["success"]:
            results["pass"] += 1
            status = "PASS"
        elif res["exit_code"] == -1:
            results["timeout"] += 1
            status = "TIMEOUT"
            failed_cases.append((task_id, "Timeout"))
        else:
            results["fail"] += 1
            status = "FAIL"
            error_msg = res["stderr"] if res["stderr"] else res["stdout"]
            failed_cases.append((task_id, error_msg))

        print(f"[{i+1}/{len(dataset)}] {task_id} : {status}")

    print("\n" + "="*40)
    print(f"[*] Evaluation Summary")
    print(f"  Total  : {len(dataset)}")
    print(f"  Pass   : {results['pass']}")
    print(f"  Fail   : {results['fail']}")
    print(f"  Timeout: {results['timeout']}")

    if failed_cases:
        print("\n[*] Failed cases:")
        for i in range(min(2, len(failed_cases))):
            print(f"    Task ID : {failed_cases[i][0]}")
            sample_err = "\n".join(failed_cases[i][1].splitlines()[:5])
            print(f"    Error   : \n{sample_err}\n\n")

def _test_java_runner():
    import testset
    from code_snippets import get_snippet

    src = get_snippet("java", "calculator")

    java_runner = JavaRunner()
    res = java_runner.run_code(src)

    if res["success"] and res["stdout"] == "8":
        print(f"SUCCESS: {res['stdout']}")
    else:
        if res["stderr"]:
            print(f"Error: {res['stderr']}")

    dataset = testset.load_mbjp_testset(nsamples = 100) 
    results = {"pass": 0, "fail": 0}

    print("[*] Starting Evaluation...\n" + "="*40)
    for i, data in enumerate(dataset):
        test_code = "\n".join(data["test_payload"])
        full_program = f"{data['code']}\n\n{test_code}"

        res = java_runner.run_code(full_program)
        if res["success"]:
            results["pass"] += 1
        else:
            results["fail"] += 1
            print(res["stderr"])

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(dataset)}] Pass: {results['pass']} | Fail: {results['fail']}")

    print(f"[*] Pass: {results['pass']} | Fail: {results['fail']}")

if __name__ == "__main__":
    _test_py_runner()
    _test_js_runner()
    _test_java_runner()
