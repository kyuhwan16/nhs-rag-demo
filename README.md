# nhs-rag-demo

Small RAG prototype for helping international students navigate the NHS
(GP registration, referrals, 111, IHS, emergency vs urgent care).

Same retrieval logic (TF-IDF + cosine similarity), done three ways so
the demo doesn't depend on anything being installed on the day:

- `app.py` - CLI version, sklearn's `TfidfVectorizer` + `cosine_similarity`
- `web_app.py` - same logic, Streamlit UI (`streamlit run web_app.py`)
- `index.html` - JS reimplementation, no server/API key/Python needed,
  just open it in a browser

`knowledge_base/` has the 5 source docs (read from disk in the python
versions, embedded as strings in index.html since local files can't be
fetch()'d from a browser).

## how it works

Question + all 5 docs -> TF-IDF vectors -> cosine similarity picks the
closest doc -> same trick again at sentence level to pull out the 1-2
most relevant sentences from that doc as the answer (no LLM, extractive
only). Source doc + score always shown next to the answer. If the score's
too low (<0.05) it just says it can't find anything relevant instead of
making something up.

## running it

```
pip install -r requirements.txt
python app.py              # cli
streamlit run web_app.py   # web
```

or just open `index.html` directly, no install needed.

## note

Dissertation prototype, not real medical/NHS advice. Doc content
summarised from NHS.uk / UKCISA.
