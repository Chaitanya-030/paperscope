import truststore
truststore.inject_into_ssl()

import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Allow `import` of sibling script week1_arxiv.py
sys.path.append(str(Path(__file__).parent))
from arxiv_tool import search_arxiv  # noqa: E402

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

def get_qdrant_client() -> QdrantClient:
    print(f"[qdrant] connecting to {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"[qdrant] collection {COLLECTION_NAME!r} already exists — skipping create")
        return
    print(f"[qdrant] creating collection {COLLECTION_NAME!r} (dim={EMBEDDING_DIM}, metric=cosine)")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

def arxiv_id_from_url(url: str) -> str:
    # 'http://arxiv.org/abs/2604.19748v1' --> '2604.19748v1'
    return url.rstrip("/").split("/")[-1]


def stable_uuid(name: str) -> str:
    # Deterministic UUID from a name string. Same name → same UUID, always
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

def ingest_papers(client: QdrantClient, model: SentenceTransformer, papers: list[dict]) -> None:
    # Embed title+abstract of each paper and upsert into Qdrant
    print(f"[ingest] embedding {len(papers)} papers...")
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    vectors = model.encode(texts, show_progress_bar=False)

    points = []
    for paper, vector in zip(papers, vectors):
        arxiv_id = arxiv_id_from_url(paper["url"])
        points.append(
            PointStruct(
                id=stable_uuid(f"arxiv:{arxiv_id}"),
                vector=vector.tolist(),
                payload={
                    "arxiv_id": arxiv_id,
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "abstract": paper["abstract"],
                    "url": paper["url"],
                    "published": paper["published"],
                },
            )
        )

    print(f"[ingest] upserting {len(points)} points to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[ingest] done")

def search(client: QdrantClient, model: SentenceTransformer, query: str, k: int = 5) -> list[dict]:
    # Return the top-k most similar papers to query
    print(f"\n[search] query: {query!r}  (top {k})")
    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    results = []
    for point in response.points:
        results.append(
            {
                "score": point.score,
                "title": point.payload["title"],
                "url": point.payload["url"],
                "published": point.payload["published"],
            }
        )
    return results

def print_results(results: list[dict]) -> None:
    """Pretty-print search results."""
    for i, r in enumerate(results, start=1):
        print(f"  {i}. score={r['score']:.3f}  {r['title'][:80]}")
        print(f"     {r['url']}  ({r['published']})")

if __name__ == "__main__":
    client = get_qdrant_client()
    ensure_collection(client)

    print(f"[embed] loading model {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Ingest only if collection is empty (cache-first)
    info = client.get_collection(COLLECTION_NAME)
    if info.points_count == 0:
        papers = search_arxiv("LLM agents", max_results=30)
        ingest_papers(client, model, papers)
    else:
        print(f"[ search and ingest] skipping — {info.points_count} papers already in collection")

    queries = [
        "agents that use tools to solve tasks",
        "evaluation of large language models",
        "retrieval augmented generation",
    ]

    for q in queries:
        results = search(client, model, q, k=5)
        print_results(results)