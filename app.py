from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import rag
# read doc
with open("note/note.txt", "r", encoding="utf-8") as f:
    text = f.read()


# switch


def chunk_text(text, chunk_size=300, overlap=40):
    chunk = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk.append(text[start:end])
        start += chunk_size - overlap

    return chunk


chunks = chunk_text(text)
if len(chunks) == 0:
    raise ValueError("No chunk were created, please check input text")


model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embeddings = model.encode(chunks, convert_to_numpy=True)

# print(embeddings.shape)
# # FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# quary
query = "What is the context in this documents"
query_vec = model.encode([query], convert_to_numpy=True)

k = 3
distances, indices = index.search(query_vec, k)

# print("Top results:\n")
# for i, idx in enumerate(indices[0]):
#     print(f"Results {i + 1}:")
#     print(chunks[idx])
#     print("-" * 50)

# RAG logic
# query -> embedding -> retrieve top chunk -> build prompt -> AKL LLM
#

retrieve_chunks = [chunks[idx] for idx in indices[0]]

prompt = f"""
    you are a personal assistent, anwser all questin base on context provided,
    if you can't find anwser in the context, report the truth do not make up any story.

    context: {chr(10).join(retrieve_chunks)}
    questin: {query} 
    """


ans = rag.ask_ollama(prompt)
print("="*50)
print("model response: \t",ans)
print("-"*50)