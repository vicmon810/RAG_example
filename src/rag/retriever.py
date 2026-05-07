from rag import search 
from config import settings
def retrieve_context(query:str, top_k: int = 3) -> str:
    results = search(query=query, top_k=top_k)

    if not results:
        raise ValueError("search query not exists")
    
    blocks = []

    for r in results:
        blocks.append(
            f"[Souces: {r['doc_name']} |Chunk: {r['chunk_id']}| Score: {r['score']:.4f}\n]"
            f"{r['text']}"
        )
    
    return "\n\n".join(blocks)