# nhs-rag-demo

RAG (retrieval-augmented generation) prototype for international students
navigating NHS GP registration, referrals, NHS 111, the Immigration Health
Surcharge, and emergency vs urgent care.

Three implementations of the same retrieval + extractive-answer logic:
- `app.py` - Python CLI prototype (scikit-learn TF-IDF + cosine similarity)
- `web_app.py` - Streamlit web version of the same logic
- `index.html` - fully offline, dependency-free JS reimplementation used for
  the live demo (no server, no API key, works from a double-clicked file)

## todo
- knowledge_base docs
- app.py (retrieval + answer gen)
- web_app.py
- JS/HTML version
- styling
