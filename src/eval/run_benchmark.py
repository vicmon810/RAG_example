from pathlib import Path 
import json 
import time 
from collections import Counter

from src.eval.run_task import run_task

def run_benchmark(task_dir: str = "tasks", max_repairs: int = 1):
    tasks = sorted(Path(task_dir).glob("*.md"))

    if not tasks:
        raise FileExistsError(f"No .md file found in {task_dir}")
    

    benchmark_id = int(time.time())

    summary_path = Path("runs") / f"{benchmark_id}_benchmark_summary.json"

    results = []

    for task_path in tasks:
        print(f"\n=== Running {task_path}")

        log = run_task(str(task_path), max_repairs=max_repairs)

        final = log["final_result"]
        inital = log["initial_attempt"]

        item = {
            "task_path": str(task_path),
            "model": log["model"],
            "initial_success": inital["success"],
            "initial_failure_type": inital["failure_type"],
            "fianl_success": final["success"],
            "final_failure_type": final["failure_type"],
            "repaired": final["repaired"],
            "final_stdout": final["stdout"],
            "final_stderr": final["stderr"],
        }

        results.append(item)

        print(
            f"initial={item['initial_success']}",
            f"final={item['fianl_success']}",
            f"failuer={item['final_failure_type']}",
            f"repaired={item['repaired']}"
        )

    total = len(results)

    initial_success_count = sum(r["initial_success"] for r in results)
    final_success_count  = sum(r["final_success"] for r in results)
    repaired_success_count = sum(
        (not r["initial_success"]) and r["final_success"]
        for r in results
    )

    failure_counter = Counter(r["final_failure_type"] for r in results)

    summary = {
        "benchmark_id": benchmark_id,
        "total_tasks": total,
        "initial_success_count": initial_success_count,
        "initial_success_rate": initial_success_count / total,
        "final_success_count": final_success_count,
        "final_success_rate": final_success_count / total,
        "repaired_success_count": repaired_success_count,
        "final_failure_types": dict(failure_counter),
        "results": results,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n == Benchmark Summary ==")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {summary_path}")

    return summary

if __name__ == "__main__":
    run_benchmark(task_dir="tasks", max_repairs=1)