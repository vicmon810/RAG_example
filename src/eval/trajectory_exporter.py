import json 
from pathlib import Path 
from typing import Any 

def append_trajectory(log: dict[str, Any], output_path:str="data/trajectories.json") -> dict:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    task_path = log.get("task_path")
    model = log.get("model")
    context = log.get("retrived_context", "")

    initial = log.get("initial_attempt", {})
    final = log.get("final_result", {})
    repairs = log.get("repair_attempts", [])

    correctness = final.get("correctness") or log.get("correctness")

    record = {
        "run_id": log.get("run_id"),
        "task_path": task_path,
        "model": model, 
        "retrieved_context": context,

        "initial_success": initial.get("success"),
        "initial_failure_type": initial.get("failure_type"),
        "initial_stdout": initial.get("stdout"),
        "initial_stderr": initial.get("stderr"),
        "raw_model_output": initial.get("raw_model_output"),
        "generated_code": initial.get("generated_code"),

        "num_repairs": len(repairs),
        "repair_attempts": repairs,

        "final_success": final.get("success"),
        "final_failure_type": final.get("failure_type"),
        "final_stdout": final.get("stdout"),
        "final_stderr": final.get("stderr"),
        "final_code_path": final.get("code_path"),
        "repaired": final.get("repaired"),

        "correctness": correctness,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False)+ "\n")