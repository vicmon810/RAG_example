# Turning LLM: Local RAG-assisted CodeAct Agent Evaluation

This project builds a local Retrieval-Augmented Generation (RAG) system and extends it into a lightweight CodeAct-style agent evaluation framework.

The system uses FAISS for vector search, SentenceTransformers for embedding, and Ollama for local LLM inference. Instead of using RAG only for question answering, this project uses retrieved context to help an LLM generate executable Python code, run the code, classify failures, repair broken outputs, and evaluate correctness.

## Project Goals

The goal of this project is to explore how local LLM agents can solve small data science and Python programming tasks in a controlled environment.

The system is designed to answer questions such as:

- Can a local LLM generate executable Python code for practical tasks?
- Does RAG context improve code generation quality?
- What types of failures occur during code generation?
- Can an agent repair its own failed code using error messages?
- How often does the final output match the expected answer?

## How to run 

Create and activate a python virtual environment:
```{bash}
python -m venv llm
source llm/bin/activate
```

install dependencies:
```{bash}
pip install -r requirement.txt
#uv install -r requirement.txt
```

install and start ollama: 
```{bash}
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:1.5b
```

Configuration: 
Create a *.env* file 
```{bash}
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=deepseek-r1:1.5b
EMBED_MODEL=all-MiniLM-L6-v2
TOP_K=3
```
 
Build the RAG index
put *.txt* documents inside the */node* directory, then run:
```{bash}
python -c "from rag import build_index; build_index()"
```

Test retrieval
```{bash}
python -c "from rag import search; print(search('what is RAG', top_k=3))
```

Test RAG quesiton answering:
```{bash}
python -c "from rag import answer_query; ans, refs = answer_query('what is RAG?'); print(ans)"
```

Run a single Agent Task:

```{bash}
python -m src.eval.run_task
```

This will generate a log under:

``` *runs/*```

each run stores:
- retrieved context
- raw model output
- extracted Python code
- execution result 
- stdout and stderr
- failure type
- repair attempt
- correctness result


## Run the benchmark
Run all taskin the *tasks/* directory
```{bash}
python -m src.eval.run_benchmark
```

the benchmark reports:
- inital success rate
- final success after repair
- repair imporovement count
- failure type distribution
- logic correctness rate


## Features

- Multi-document ingestion
- Text chunking for local documents
- FAISS-based vector search
- Persistent local vector index
- Local LLM inference through Ollama
- RAG-assisted prompt construction
- Python code generation
- Universal Python code extraction from LLM outputs
- Safe execution of generated Python scripts
- Runtime error capture
- Failure classification
- One-step code repair using traceback feedback
- Logic correctness checking against expected outputs
- JSON-based run logging
- Benchmark trajectory export for future fine-tuning or RL-style optimization

## System Pipeline

```text
Task file
  ↓
RAG retrieval
  ↓
Prompt construction
  ↓
Local LLM code generation
  ↓
Python code cleaning
  ↓
Code execution
  ↓
Failure classification
  ↓
Optional repair attempt
  ↓
Correctness checking
  ↓
Run log / trajectory export
```

### Tech Stack 
 
- python 
- FAISS 
- Sentence Transformers
- Ollama 
- Local LLM : Deepseek-R1 / Qwen-coder
- JSON/JSONL logging
- MArkdown task definitions 

## Project structure

```
turning_llm/
├── app.py
├── rag.py
├── config.py
├── requirement.txt
├── note/
│   └── note.txt
├── index/
│   ├── docs.index
│   ├── chunks.json
│   └── meta.json
├── tasks/
│   ├── task_001.md
│   └── task_001_expected.json
├── runs/
│   └── run logs
├── data/
│   └── trajectories.jsonl
└── src/
    ├── rag/
    │   └── retriever.py
    ├── agent/
    │   ├── code_agent.py
    │   ├── repair_agent.py
    │   └── code_cleaner.py
    └── eval/
        ├── run_task.py
        ├── run_benchmark.py
        ├── classifier.py
        ├── correctness.py
        └── trajectory_exporter.py
```

## Example Task Format

each task is stored as Markdown file:

```{markdown}
# Task 001: Summarise Transactions

Given a list of transactions with user IDs and amounts, write Python code to produce a summary for each user.

For each user, calculate:
- total amount
- number of transactions
- average transaction amount
```

Expected output is stored separately as JSON:


{
  "A": {"total": 25, "count": 2, "average": 12.5},
  "B": {"total": 20, "count": 1, "average": 20.0}
}