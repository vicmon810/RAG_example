import ast 
import json 
from pathlib import Path 
from typing import Any 

def normalize(obj: Any) -> Any:
    if isinstance(obj, dict) :
        return {str(k):normalize(v) for k,v in sorted(obj.items())}
    
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    
    if isinstance(obj, float):
        return round(obj,6)
    
    return obj 


def parse_stdout(stdout:str) -> Any:
    text = stdout.strip()

    if not text: return None 

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass 

    try:
        return ast.literal_eval(text)
    except Exception:
        pass 

    return text 


def expected_path_for_task(task_path: str) -> Path:
    path = Path(task_path)
    return path.with_name(path.stem + "_expected.json")

def check_correctness(task_path: str, stdout:str) -> dict:
    expect_path = expected_path_for_task(task_path)

    if not expect_path.exists():
        return{
            "has_expected_output": False,
            "correct": None,
            "reason": "No expected output file found"
        }
    
    expected = json.load(expect_path.read_text(encoding='utf-8'))

    actual = parse_stdout(stdout)

    correct = normalize(actual) == normalize(expected)

    return {
        "has_expected_output": True,
        "correct": correct,
        "actual": actual,
    }