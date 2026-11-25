"""Source-backed retrieval and ranking evaluation."""

from .retrieval import (
    ALIASES, COLLECTION, CORPUS, DEFAULT_RAG_DB, Passage, Retriever,
    evaluate, hashed_embedding, ingest_pdf, main, read_corpus, term_id, terms,
)
