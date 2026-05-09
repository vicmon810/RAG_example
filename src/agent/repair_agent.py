import requests
from config import settings
from src.agent.code_cleaning import extract_python_code

def repaired_code(task: str,
                context: str, 
                broken_code: str,
                stderr: str,
                model:str | None = None) -> dict:
    model = model or settings.LLM_MODEL

    prompt = f"""
    You are a  python code repair agent.

    The pervious Python script failed.

    you have to : 
    - fix the code
    - return only executable python code 
    - Do not include explanations 
    - Do not include markdown 
    - Do not include code fence 
    
    Task:
    {task}

    Retrieved context:
    {context}

    Broken code:
    {broken_code}

    Error_message:
    {stderr}

    Requirements:
    - return a complete executable Python script
    - Preserve the original task objective
    - print the final result.
    
    """
    host = settings.OLLAMA_HOST

    response = requests.post(
        f"{host}/api/generate",
        json = {
            "model" : model, 
            "prompt" : prompt,
            "stream": False,
            "keep_alive": -1
        }
        , timeout=(10,600)

    )
    response.raise_for_status()

    raw = response.json()["response"]
    code = extract_python_code(raw)

    return {
        "raw_repair_output": raw,
        "repaired_code": code,
        "model": model
    }