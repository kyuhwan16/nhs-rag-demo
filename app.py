"""
NHS Navigation RAG Agent - Minimal Working Prototype
======================================================

WHAT THIS DOES (in plain English, for explaining in your viva):
1. We have a small "knowledge base" - a handful of text documents about
   how international students navigate the NHS (GP registration, 111,
   referrals, IHS, emergency vs urgent care).
2. When a user types a question, we convert both the question and every
   document into TF-IDF vectors (a way of representing text as numbers
   based on important/distinctive words).
3. We compute cosine similarity between the question and every document
   to find which document(s) are most relevant - this is the
   "Retrieval" step of RAG.
4. We then generate a grounded answer using ONLY the retrieved
   document's content (extracting the most relevant sentences) - this
   is a lightweight, fully-offline stand-in for the "Generation" step.
   (In a full version, this step would call an LLM API such as GPT-4
   to paraphrase the retrieved content fluently - the retrieval logic
   here is identical either way, only the generation step changes.)
5. We show which document the answer came from - this is the
   "grounding" / explainability part that directly relates to RQ1
   (does RAG improve factual accuracy vs a plain chatbot that just
   makes things up).

WHY THIS DESIGN (for Q&A defence):
- TF-IDF + cosine similarity was chosen over a full embedding model
  (e.g. OpenAI embeddings, sentence-transformers) because it requires
  no API key and no internet connection, making the live demo more
  reliable. The trade-off is that it captures keyword overlap rather
  than deep semantic meaning - a known limitation, documented as such.
- Extractive answer generation (returning the most relevant sentence(s)
  verbatim from the source) was used instead of calling an LLM for
  generation, again for demo reliability within the project timeframe.
  This is disclosed as a scope decision / limitation, not hidden.

HOW TO RUN:
  python app.py
Then type a question and press Enter. Type 'quit' to exit.
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
