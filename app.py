import json
from src.rag.core import build_index, answer_query

from src.eval.run_task import run_task
from src.eval.run_benchmark import run_benchmark


def print_separator():
    print("\n" + "=" * 60)


def build_index_cli():
    try:
        build_index()
        print("Index built successfully.")
    except Exception as e:
        print(f"Error building index: {e}")


def ask_question_cli():
    query = input("Enter your question: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    try:
        answer, results = answer_query(query, top_k=3)

        print_separator()
        print("Retrieved chunks:")

        for i, r in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Score : {r['score']:.4f}")
            print(f"Source: {r['doc_name']} | Chunk: {r['chunk_id']}")
            print(f"Text  : {r['text']}")

        print_separator()
        print("Model response:\n")
        print(answer)
        print_separator()

    except Exception as e:
        print(f"Error during query: {e}")


def run_task_cli():
    task_path = input("Enter task path [default: tasks/task_001.md]: ").strip()

    if not task_path:
        task_path = "tasks/task_001.md"

    try:
        log = run_task(task_path, max_repairs=1)
        print_separator()
        print("Task run completed.")
        print(json.dumps(log["final_result"], indent=2, ensure_ascii=False))
        print_separator()

    except Exception as e:
        print(f"Error running task: {e}")


def run_benchmark_cli():
    tasks_dir = input("Enter tasks directory [default: tasks]: ").strip()

    if not tasks_dir:
        tasks_dir = "tasks"

    try:
        summary = run_benchmark(tasks_dir=tasks_dir, max_repairs=1)
        print_separator()
        print("Benchmark completed.")
        print(json.dumps(
            {
                "total_tasks": summary["total_tasks"],
                "initial_success_rate": summary["initial_success_rate"],
                "final_success_rate": summary["final_success_rate"],
                "repaired_success_count": summary["repaired_success_count"],
                "final_failure_types": summary["final_failure_types"],
            },
            indent=2,
            ensure_ascii=False,
        ))
        print_separator()

    except Exception as e:
        print(f"Error running benchmark: {e}")


def main():
    while True:
        print("\n====== Turning LLM CLI =======")
        print("1. Build RAG index")
        print("2. Ask RAG question")
        print("3. Run one CodeAct task")
        print("4. Run benchmark")
        print("5. Exit")

        choice = input("Choose: ").strip()

        if choice == "1":
            build_index_cli()

        elif choice == "2":
            ask_question_cli()

        elif choice == "3":
            run_task_cli()

        elif choice == "4":
            run_benchmark_cli()

        elif choice == "5":
            print("Bye.")
            break

        else:
            print("Invalid choice. Please choose 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()