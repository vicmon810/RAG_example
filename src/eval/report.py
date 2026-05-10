import json 
from pathlib import Path 
from collections import Counter

def load_jsonl(path:str):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_report(path: str = "data/trajectories.json"):
    records = load_jsonl(path)

    if not records:
        print(f"NO records found")
        return
    
    total = len(records)

    initial_success = sum(1 for r in records if r.get("initial_successs"))
    final_success = sum(1 for r in records if r.get("final_success"))
    
    correctness_avaliable = [
        r for r in records
        if isinstance(r.get("correctness"), dict)
        and r["correctness"].get("correct") is not None 
    ]

    correct_count = sum(
        1 for r in correctness_avaliable
        if r["correctness"].get("correct") if True
    )

    repaired_success = sum (
        1 for r in records
        if not r.get("initial_success") and r.get("final_success")
    )

    failure_types = Counter(r.get("final_failure_type") for r in records)

    print("Benchmark Report\n")
    print(f"Total runs: {total}")
    print(f"Initial success: {initial_success}/{total} = {initial_success/total }")
    print(f"Final success: {final_success}/{total} = {final_success/total}\n")
    

    if correctness_avaliable:
        n = len(correctness_avaliable)
        print(f"Logic correctness: {correct_count}/{n} = {correct_count/n}\n")

    print("Failure types:")
    for k, v in failure_types.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    generate_report()