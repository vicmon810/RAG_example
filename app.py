from re import L
import rag


def main():
    while True:
        print("\n======Simple RAG=======")
        print("1. Build index")
        print("2. Ask question")
        print("3. Exit")

        choice = input("Choose: ").strip()

        if choice == "1":
            try:
                rag.build_index()
            except Exception as e:
                print(f"Error building index: {e}")
        elif choice == "2":
            query = input("Entry your question:").strip()
            if not query:
                print("Query cannot be empty")
                continue
            try:
                answer, results = rag.answer_query(query, top_k=3)

                print("\n", +"=" * 60)
                print("Retrieved chunks:")
                for i, r in enumerate(results, start=1):
                    print(f"\nResult {i}")
                    print(f"Score\t:{r['score']}:.4f")
                    print(f"Text\t:{r['text']}")
                print("\n" + "=" * 60)
                print("Model response:\n")
                print(answer)
                print("=" * 60)
            except Exception as e:
                print(f"Error during query: {e}")
        elif choice == "3":
            print("Bye.")
            break
        else:
            print("Bye.")


if __name__ == "__main__":
    main()
