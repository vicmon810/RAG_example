from pathlib import Path 
import subprocess
import json 
import time 
from config import settings
from src.rag.retriever import retrieve_context
from src.agent.code_agent import generate_code

def run_task(task_path: str):
    task_text = Path(task_path).read_text(encoding="utf-8")

    context = retrieve_context(task_text, top_k=3)

    code = generate_code(task_text, context)

    run_id = int(time.time())
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    code_path = runs_dir / f"{run_id}_solution.py"
    log_path = runs_dir / f"{run_id}_log.json"

    code_path.write_text(code, encoding="utf-8")

    start = time.time()

    try:
        result = subprocess.run(
            ["python", str(code_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        print("="*30)
        print(result)
        print("="*30)
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

    log = {
        "task_path": task_path,
        "success": success,
        "runtime_second": end - start,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "code_path": str(code_path),
        "retrieved_context": context,
        "generated_code": code,
    }

    log_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return log 

if __name__ == "__main__":
    log = run_task("tasks/task_001.md")
    print(json.dumps(log, indent=2, ensure_ascii=False))