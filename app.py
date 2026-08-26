"""
NHS Navigation RAG Agent - Minimal Working Prototype
======================================================

Loads the knowledge_base docs and will let the user ask a question
in the terminal. Retrieval + answer generation to be added.

HOW TO RUN:
  python app.py
"""

import os

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


def main():
    docs, filenames = load_documents(KB_FOLDER)
    print(f"Loaded {len(docs)} knowledge base documents: {filenames}")
    # TODO: retrieval + answer generation


if __name__ == "__main__":
    main()
