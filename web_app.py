"""
NHS Navigation RAG Agent - Streamlit Web Version
==================================================
Same retrieval logic as app.py (TF-IDF + cosine similarity), wrapped in a
simple web interface so it can be opened from ANY computer's browser via
a URL, with no local Python installation required at demo time.

Run locally with: streamlit run web_app.py
"""

import os
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_FOLDER = os.path.join(os.path.dirname(__file__), "knowledge_base")


@st.cache_data
def load_documents(folder):
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
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


@st.cache_resource
def build_index(docs):
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)
    return vectorizer, doc_vectors


def retrieve(query, vectorizer, doc_vectors, docs, filenames, top_k=1):
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
    top_indices = sorted(top_indices)

    return " ".join(sentences[i] for i in top_indices)


st.set_page_config(page_title="NHS Navigation RAG Agent", page_icon="🩺", layout="centered")

st.title("🩺 NHS Navigation RAG Agent")
st.caption("A retrieval-augmented prototype helping international students navigate NHS GP registration, referrals, NHS 111, and healthcare access.")

docs, filenames = load_documents(KB_FOLDER)
vectorizer, doc_vectors = build_index(docs)

with st.expander("Knowledge base loaded (click to view sources)"):
    st.write(filenames)

query = st.text_input("Ask a question about registering with a GP, NHS 111, referrals, IHS, or emergency vs urgent care:")

example_cols = st.columns(3)
examples = [
    "How do I register with a GP?",
    "What is the Immigration Health Surcharge?",
    "When should I use NHS 111?",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex):
        query = ex

if query:
    results = retrieve(query, vectorizer, doc_vectors, docs, filenames, top_k=1)
    top_result = results[0]

    if top_result["score"] < 0.05:
        st.warning("No sufficiently relevant document found in the knowledge base — this question is likely outside the system's current scope.")
    else:
        answer = generate_answer(query, top_result)
        st.success(answer)
        st.caption(f"Retrieved source: `{top_result['filename']}`  |  Similarity score: {top_result['score']:.2f}")
