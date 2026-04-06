from numpy._core import records
import requests
import os
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from torch import chunk, device, embedding, topk
from config import settings


def get_embed_model():
    return SentenceTransformer(settings.embed_model, device="cpm")


def chunk_text(text, chunk_size=300, overlap=40):
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunk = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk.append(text[start:end])
        start += chunk_size - overlap

    return chunk


def load_documents() -> list[dict]:
    docs_path = settings.DOCS_DIR

    if not docs_path.is_dir():
        raise FileNotFoundError(f"{docs_path} does not exists")
    records = []
    txt_files = sorted(docs_path.glob("*.txt"))

    if not txt_files:
        raise ValueError("No .txt files found int note/")

    for file in txt_files:
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            records.append({"doc_name": file.name, "chunk_id": i, "text": chunk})
    return records


def build_index():
    index_dir = settings.INDEX_DIR
    records = load_documents()
    texts = [r["text"] for r in records]

    model = get_embed_model()
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_dir.makedir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.INDEX_FILE))

    with open(settings.CHUNK_FILE, "w", encoding="uft-8") as f:
        json.dump(records, f, ensure_ascii=False, index=2)

    with open(settings.META_FILE, "w", encoding="uft-8") as f:
        json.dump(
            {
                "embedding_model": settings.EMBED_MODEL,
                "dimension": dimension,
                "normalize": True,
                "index_type": "IndexFlatIP",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

        print(f"Index build successfully with {len(records)} chunks.")


def load_index():
    if not settings.INDEX_FILE.exists():
        raise FileNotFoundError("Index file not found, Please build index.")
    if not settings.CHUNK_FILE.exists():
        raise FileNotFoundError("chunks.json not found")
    if not settings.META_FILE.exists():
        raise FileNotFoundError("meta.json not found")

    index = faiss.read_index(str(settings.INDEX_FILE))

    with open(settings.CHUNK_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(settings.META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return index, records, meta


def search(query: str, top_k: int = settings.TOP_K) -> list[dict]:
    index, records, meta = load_index()
    model = SentenceTransformer(meta["embedding_model"], device="cpu")
    query_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=meta["normalize"]
    ).astype("float32")

    top_k = min(top_k, len(records))
    scores, indices = index.search(query_vec, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = records[idx]
        requests.append(
            {
                "score": float(score),
                "doc_name": item["doc_name"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
            }
        )

    return results


def build_prompt(query: str, results: list[dict]) -> str:
    context = "\n\n".join(
        [
            f"[Source: {r['doc_name']} | Chunk: {r['chunk_id']}\n{r['text']}]"
            for r in results
        ]
    )
    prompt = f"""
        you are a personal assistant.
        Anser the question only based on the provided context,
        if the answer is not in the context, say clearly that you do not know, 
        Do not make up facts. 
        context:{context}
        Question: {query}
        """
    return prompt.strip()


def ask_ollama(prompt, model="deepseek-r1:1.5b"):
    host = settings.OLLAMA_HOST
    response = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "keep_alive": -1},
        timeout=(10, 600),
    )
    response.raise_for_status()

    return response.json()["response"]


def answer_query(
    query: str, top_k: int = settings.TOP_K, llm_model: str = "deepseek_r1:1.5b"
):
    results = search(query=query, top_k=top_k)
    prompt = build_prompt(query=query, results=results)
    answer = ask_ollama(prompt, model=llm_model)
    return answer, results
