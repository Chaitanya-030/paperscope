import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "groq:openai/gpt-oss-20b"
TOP_K = 5

print(f"[setup] embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print(f"[setup] connecting to Qdrant at {QDRANT_URL}")
qdrant_client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

print(f"[setup] LLM: {LLM_MODEL}")
llm = init_chat_model(LLM_MODEL)

def format_papers(docs: list) -> str:
    """Turn retrieved Documents into a numbered string for the prompt."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Untitled")
        url = doc.metadata.get("url", "")
        abstract = doc.page_content
        blocks.append(f"[{i}] Title: {title}\n    URL: {url}\n    Abstract: {abstract}")
    return "\n\n".join(blocks)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant. Answer the user's question using ONLY the "
     "papers provided below. Cite each claim with [N] referring to the paper "
     "number. If the papers do not contain the answer, say so honestly — do "
     "not invent citations."),
    ("user",
     "Papers:\n{context}\n\nQuestion: {question}"),
])

chain = (
    {
        "context": retriever | format_papers,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    queries = [
        "What recent work has been done on multi-agent systems?",
        "Are there papers about retrieval-augmented generation?",
        "Tell me about quantum computing breakthroughs.",
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        answer = chain.invoke(q)
        print(answer)