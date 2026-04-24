import truststore
truststore.inject_into_ssl()

import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Allow `import` of sibling script
sys.path.append(str(Path(__file__).parent))
from ingest_and_search import (  # noqa: E402
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    get_qdrant_client,
)

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "papers provided in their message. Every claim must end with a citation "
    "like [1], [2] referring to the numbered papers. If the papers don't "
    "contain the answer, say so honestly — do not invent papers, authors, "
    "or facts."
)

def retrieve(
    client: QdrantClient,
    embed_model: SentenceTransformer,
    query: str,
    k: int = TOP_K,
) -> list[dict]:
    # Embed the query, fetch top-k papers from Qdrant
    print(f"[retrieve] query={query!r} k={k}")
    query_vector = embed_model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    )
    papers = []
    for point in response.points:
        papers.append({
            "score": point.score,
            "title": point.payload["title"],
            "abstract": point.payload["abstract"],
            "url": point.payload["url"],
        })
    print(f"[retrieve] got {len(papers)} papers (top score={papers[0]['score']:.3f})")
    return papers

def build_prompt(query: str, papers: list[dict]) -> str:
    # Format the user message: numbered papers + the question
    paper_blocks = []
    for i, p in enumerate(papers, start=1):
        paper_blocks.append(
            f"[{i}] Title: {p['title']}\n"
            f"    Abstract: {p['abstract']}\n"
            f"    URL: {p['url']}"
        )
    context = "\n\n".join(paper_blocks)
    return (
        f"Papers:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the papers above. Cite each claim with [N]."
    )

def rag_answer(
    client: QdrantClient,
    embed_model: SentenceTransformer,
    llm: Groq,
    query: str,
    k: int = TOP_K,
) -> str:
    papers = retrieve(client, embed_model, query, k)
    user_prompt = build_prompt(query, papers)

    print("[llm] generating answer...")
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content or ""

if __name__ == "__main__":
    qdrant = get_qdrant_client()
    print(f"[rag] loading embedding model {EMBEDDING_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    llm = Groq()

    queries = [
        "What recent work has been done on multi-agent systems?",
        "Are there papers about retrieval-augmented generation?",
        "Tell me about quantum computing breakthroughs.",
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        answer = rag_answer(qdrant, embed_model, llm, q)
        print(answer)