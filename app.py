"""
NHS Navigation RAG Agent - Minimal Working Prototype
======================================================

Loads the knowledge_base docs and will let the user ask a question
in the terminal. Retrieval + answer generation to be added.

HOW TO RUN:
  python app.py
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_FOLDER = os.path.join(os.path.dirname(__file__), "knowledge_base")


def load_documents(folder):
    """Load every .txt file in the knowledge_base folder as one document."""
    docs = []
    filenames = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
            filenames.append(fname)
    return docs, filenames


def build_index(docs):
    """Turn documents into TF-IDF vectors."""
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)
    return vectorizer, doc_vectors


def retrieve(query, vectorizer, doc_vectors, docs, filenames, top_k=1):
    """Find the most relevant document(s) for a query."""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1][:top_k]
    results = []
    for idx in ranked_indices:
        results.append({
            "filename": filenames[idx],
            "text": docs[idx],
            "score": similarities[idx],
        })
    return results


def main():
    docs, filenames = load_documents(KB_FOLDER)
    print(f"Loaded {len(docs)} knowledge base documents: {filenames}")
    vectorizer, doc_vectors = build_index(docs)

    query = input("Your question: ").strip()
    results = retrieve(query, vectorizer, doc_vectors, docs, filenames, top_k=1)
    print(results[0])
    # TODO: answer generation, proper loop, out-of-scope handling


if __name__ == "__main__":
    main()
