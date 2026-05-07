import requests
from config import settings
from src.agent.code_cleaning import extract_python_code

def clean_code(raw: str)-> str:
    text = raw.strip()

    if text.startswith("```python"):
        text = text.removeprefix("```python").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text 


def generate_code(task:str, context:str, model:str | None=None)-> str:
    model = model or settings.LLM_MODEL

    prompt = f"""
your are a Python data science coding specialist

your job is to write a complete executabel Python script.

Task:
{task}

Retrieved context:
{context}

Rules:
- Return only Python code
- Do nt use Markdown
- Do not explain 
- If not input file is provided, create the example data inside the script.
- print the final result 
- if you did not know the answer, admit it and tell the user you do not know the anser.
"""
    host = settings.OLLAMA_HOST.rstrip("/")
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1
        },
        timeout=(100,6000)
    )

    response.raise_for_status()

    raw = response.json()["response"]
    return extract_python_code(raw)


