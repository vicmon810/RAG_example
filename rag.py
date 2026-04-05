import requests
import os 

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def ask_ollama(prompt, model="qwen3.5:0.8b"):
    print(prompt)
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1
        },
        timeout=(10,600)
    )
    response.raise_for_status()
    return response.json()["response"]