import requests
import os
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


def ask_ollama(prompt, model="deepseek-r1:1.5b"):
    print(prompt)
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "keep_alive": -1},
        timeout=(10, 600),
    )
    # print("status: ", response.status_code)
    # print("body: ", response.text[:500])
    response.raise_for_status()

    return response.json()["response"]
