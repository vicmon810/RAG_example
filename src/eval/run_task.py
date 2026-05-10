from pathlib import Path
import subprocess
import json
import time

from src.rag.retriever import retrieve_context
from src.agent.code_agent import generate_code
from src.agent.repair_agent import repaired_code
from src.eval.classifier import classify_failure
from src.eval.correctness import check_correctness
from src.eval.trajectory_exporter import append_trajectory

def execute_python_file(code_path: Path, timeout: int = 30) -> dict:
    start = time.time()

    try:
        result = subprocess.run(
            ["python", str(code_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode

    except subprocess.TimeoutExpired as e:
        success = False
        stdout = e.stdout or ""
        stderr = f"TimeoutExpired: {e}"
        returncode = None

    end = time.time()

    return {
        "success": success,
        "runtime_seconds": end - start,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def run_task(task_path: str, max_repairs: int = 1):
    task_text = Path(task_path).read_text(encoding="utf-8")

    context = retrieve_context(task_text, top_k=3)

    run_id = int(time.time())
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    # First attempt
    generation = generate_code(task_text, context)
    code = generation["generated_code"]
    raw_model_output = generation["raw_model_output"]
    model = generation["model"]

    attempt_1_code_path = runs_dir / f"{run_id}_attempt_1.py"
    attempt_1_code_path.write_text(code, encoding="utf-8")

    attempt_1_result = execute_python_file(attempt_1_code_path)

    attempt_1_failure_type = classify_failure(
        attempt_1_result["success"],
        attempt_1_result["stderr"],
        code,
    )

    repair_attempts = []

    final_success = attempt_1_result["success"]
    final_code_path = str(attempt_1_code_path)
    final_stdout = attempt_1_result["stdout"]
    final_stderr = attempt_1_result["stderr"]
    final_failure_type = attempt_1_failure_type

    # Repair if failed
    if not attempt_1_result["success"] and max_repairs > 0:
        repair = repaired_code(
            task=task_text,
            context=context,
            broken_code=code,
            stderr=attempt_1_result["stderr"],
            model=model,
        )

        repair_code = repair["repaired_code"]
        raw_repair_output = repair["raw_repair_output"]

        attempt_2_code_path = runs_dir / f"{run_id}_attempt_2_repaired.py"
        attempt_2_code_path.write_text(repair_code, encoding="utf-8")

        attempt_2_result = execute_python_file(attempt_2_code_path)

        attempt_2_failure_type = classify_failure(
            attempt_2_result["success"],
            attempt_2_result["stderr"],
            repair_code,
        )

        repair_attempts.append(
            {
                "attempt": 2,
                "code_path": str(attempt_2_code_path),
                "success": attempt_2_result["success"],
                "failure_type": attempt_2_failure_type,
                "stdout": attempt_2_result["stdout"],
                "stderr": attempt_2_result["stderr"],
                "runtime_seconds": attempt_2_result["runtime_seconds"],
                "raw_repair_output": raw_repair_output,
                "repaired_code": repair_code,
            }
        )

        final_success = attempt_2_result["success"]
        final_code_path = str(attempt_2_code_path)
        final_stdout = attempt_2_result["stdout"]
        final_stderr = attempt_2_result["stderr"]
        final_failure_type = attempt_2_failure_type
    try:    
        correctness = check_correctness(task_path, final_stdout)
    except Exception as e:
        raise(f"Correctness check failed : {e}")
        correctness = False
    log = {
        "run_id": run_id,
        "task_path": task_path,
        "model": model,
        "retrieved_context": context,
        "correctness": correctness,
        "initial_attempt": {
            "attempt": 1,
            "code_path": str(attempt_1_code_path),
            "success": attempt_1_result["success"],
            "failure_type": attempt_1_failure_type,
            "stdout": attempt_1_result["stdout"],
            "stderr": attempt_1_result["stderr"],
            "runtime_seconds": attempt_1_result["runtime_seconds"],
            "raw_model_output": raw_model_output,
            "generated_code": code,
        },

        "repair_attempts": repair_attempts,

        "final_result": {
            "success": final_success,
            "failure_type": final_failure_type,
            "code_path": final_code_path,
            "stdout": final_stdout,
            "stderr": final_stderr,
            "repaired": len(repair_attempts) > 0,

        },
    }

    log_path = runs_dir / f"{run_id}_log.json"
    log_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    append_trajectory(log)

    return log


if __name__ == "__main__":
    log = run_task("tasks/task_001.md", max_repairs=1)
    print(json.dumps(log, indent=2, ensure_ascii=False))