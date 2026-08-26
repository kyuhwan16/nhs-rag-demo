"""
NHS Navigation RAG Agent - CLI prototype

Small RAG-style demo: a handful of NHS-related docs (GP reg, 111,
referrals, IHS, emergency vs urgent) get TF-IDF'd, the question gets
compared against them with cosine similarity, and the top match's
most relevant sentences are pulled out as the "answer". No LLM call -
extractive only, so it's fully offline and doesn't need an API key.
Source doc + similarity score is always shown so it's clear where the
answer came from (and if the score's too low it just says so instead
of guessing).

Run: python app.py, then type a question. 'quit' to exit.
"""

import os
import re
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


def split_sentences(text):
    """Very simple sentence splitter (good enough for this demo)."""
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


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


def generate_answer(query, retrieved_doc):
    """
    Extractive 'generation' step: pick the sentences from the retrieved
    document most relevant to the query, using the same TF-IDF idea at
    sentence level. This keeps the whole pipeline dependency-free and
    offline, while still demonstrating retrieval-grounded answering.
    """
    sentences = split_sentences(retrieved_doc["text"])
    if not sentences:
        return "No relevant information found in the knowledge base."

    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences + [query])
    query_vec = sentence_vectors[-1]
    sentence_vecs = sentence_vectors[:-1]
    sims = cosine_similarity(query_vec, sentence_vecs).flatten()

    top_n = min(2, len(sentences))
    top_indices = sims.argsort()[::-1][:top_n]
    top_indices = sorted(top_indices)  # keep original order for readability

    answer = " ".join(sentences[i] for i in top_indices)
    return answer


def main():
    print("=" * 60)
    print("NHS Navigation RAG Agent - Prototype Demo")
    print("=" * 60)

    docs, filenames = load_documents(KB_FOLDER)
    if not docs:
        print("ERROR: No documents found in knowledge_base/. "
              "Check the folder path.")
        return

    print(f"Loaded {len(docs)} knowledge base documents: {filenames}")
    vectorizer, doc_vectors = build_index(docs)

    print("\nType a question about NHS GP registration, referrals, "
          "111, IHS, or emergency vs urgent care.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            break
        if not query:
            continue

        results = retrieve(query, vectorizer, doc_vectors, docs, filenames, top_k=1)
        top_result = results[0]

        if top_result["score"] < 0.05:
            print("\n[No sufficiently relevant document found in the "
                  "knowledge base - this question is likely outside "
                  "the system's current scope.]\n")
            continue

        answer = generate_answer(query, top_result)

        print(f"\n[Retrieved source: {top_result['filename']} "
              f"(similarity score: {top_result['score']:.2f})]")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
