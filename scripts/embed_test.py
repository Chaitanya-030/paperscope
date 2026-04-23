"""
Goal: load the model once, embed two sentences, see that similar sentences
produce similar vectors and dissimilar sentences produce different ones.
"""

import truststore
truststore.inject_into_ssl()

from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# cosine(alpha) = dot product / norm(a) * norm(b)
# cosine(0) = 1 shows similarity, cosine(90) = 0 shows not unrelated
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors. 1.0 = identical, 0.0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    print(f"[embed] loading model {MODEL_NAME} (first run downloads ~80 MB)...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[embed] model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    sentences = [
        "LLM agents for automated data quality monitoring",
        "Using language models to detect anomalies in datasets",
        "Banana bread recipe with walnuts",
    ]

    vectors = model.encode(sentences)
    print(f"[embed] encoded {len(sentences)} sentences. Vector shape: {vectors.shape}")

    # will compare similarity between each pair of sentences
    print("\nSimilarities:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  {sim:.3f}  | {sentences[i][:40]!r}  vs  {sentences[j][:40]!r}")