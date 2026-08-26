# nhs-rag-demo

A retrieval-augmented generation (RAG) prototype that helps international
students navigate NHS GP registration, referrals, NHS 111, the Immigration
Health Surcharge, and emergency vs urgent care.

Same retrieval + extractive-answer logic (TF-IDF + cosine similarity),
implemented three times:

- **`app.py`** - Python CLI prototype (scikit-learn `TfidfVectorizer` +
  `cosine_similarity`). Run with `python app.py`.
- **`web_app.py`** - Streamlit web version of the exact same logic. Run
  with `streamlit run web_app.py`.
- **`index.html`** - fully offline, dependency-free JavaScript
  reimplementation used for the live demo. No server, no API key, no
  Python runtime needed — just open the file in a browser.

`knowledge_base/` holds the five source documents shared by all three
versions (loaded from disk in the Python versions, embedded as string
constants in `index.html`).

## How it works

1. The question and all 5 documents are converted into TF-IDF vectors
2. Cosine similarity picks the single closest-matching document
3. Within that document, the same TF-IDF + cosine similarity method
   scores individual sentences and pulls out the top 1-2 sentences
   verbatim (no paraphrasing / no LLM call)
4. The source filename + similarity score is always shown next to the
   answer
5. If the best score is too low (under 0.05), it reports that no
   relevant document was found instead of guessing

## Usage

```
pip install -r requirements.txt

python app.py              # CLI version
streamlit run web_app.py    # web version
```

Or just double-click `index.html` for the offline JS version — no
install step at all.

## Note

Academic dissertation project prototype, not a substitute for real NHS
advice. Content summarised from NHS.uk / UKCISA guidance.
