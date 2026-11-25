"""Local Qdrant RAG with auditable citations and an offline ranking baseline."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import log, sqrt
from pathlib import Path
import re
from statistics import mean
from uuid import NAMESPACE_URL, uuid5

from evidence.storage import RUNTIME, write_json


CORPUS = Path(__file__).with_name("corpus.json")
DEFAULT_RAG_DB = RUNTIME / "qdrant"
COLLECTION = "evidence_sources"
ALIASES = (("磁共振", " MRI "), ("超声", " ultrasound "), ("胸片", " chest x-ray "),
           ("射线", " x-ray "), ("成像", " imaging "), ("原理", " principle "), ("对比剂", " contrast "))


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    source: str
    locator: str
    text: str


def terms(text):
    text = text.casefold()
    for pattern, replacement in ALIASES:
        text = text.replace(pattern, replacement.casefold())
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)


def term_id(word):
    return int.from_bytes(sha256(word.encode()).digest()[:4], "big")


def hashed_embedding(text, dimensions=384):
    vec = [0.0] * dimensions
    for word, count in Counter(terms(text)).items():
        index = term_id(word)
        vec[index % dimensions] += (1 if index & 0x80000000 else -1) * (1 + log(count))
    length = sqrt(sum(value * value for value in vec)) or 1.0
    return [value / length for value in vec]


def read_corpus(corpus):
    rows = [Passage(**entry) for entry in json.loads(Path(corpus).read_text(encoding="utf-8"))]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Corpus needs non-empty, unique passage IDs.")
    return rows


def ingest_pdf(path, source, title, out):
    from pypdf import PdfReader

    passages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        words = (page.extract_text() or "").split()
        for start in range(0, len(words), 150):
            text = " ".join(words[max(0, start - 25): start + 150])
            key = sha256(f"{source}|{number}|{start}".encode()).hexdigest()[:16]
            passages.append(asdict(Passage(key, title, source, f"p. {number}", text)))
    if not passages:
        raise ValueError("No text was extracted. Supply a text PDF or perform OCR first.")
    if out.exists():
        raise FileExistsError(f"Choose a new output path: {out}")
    write_json(out, passages)


class Retriever:
    def __init__(self, corpus=CORPUS, *, db=DEFAULT_RAG_DB, embedding_model=None, reranker=None):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.passages = read_corpus(corpus)
        documents = [row.title + " " + row.text for row in self.passages]
        self.counts = [Counter(terms(doc)) for doc in documents]
        self.df = Counter(word for counts in self.counts for word in counts)
        self.average = mean(sum(counts.values()) for counts in self.counts)
        if not self.average:
            raise ValueError("Corpus passages contain no indexable text.")
        self.embed = hashed_embedding
        self.semantic_embeddings = bool(embedding_model)
        self.reranker = None
        if embedding_model:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(embedding_model, device="cpu")
            self.embed = lambda text: embedder.encode(text, normalize_embeddings=True).tolist()
        if reranker:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker, device="cpu")
        self.client = QdrantClient(":memory:" if db is None else None, path=str(db) if db else None)
        descriptor = json.dumps([embedding_model, [asdict(row) for row in self.passages]], sort_keys=True)
        self.collection = COLLECTION + "_" + sha256(descriptor.encode()).hexdigest()[:16]
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection,
                vectors_config={"dense": models.VectorParams(size=len(self.embed(documents[0])), distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams()})
        ids = [str(uuid5(NAMESPACE_URL, row.id)) for row in self.passages]
        for start in range(0, len(ids), 64):
            stop = start + 64
            points = [models.PointStruct(id=id_, payload=asdict(row),
                      vector={"dense": self.embed(doc), "bm25": self.sparse(counts)})
                      for id_, row, doc, counts in zip(ids[start:stop], self.passages[start:stop],
                                                       documents[start:stop], self.counts[start:stop])]
            existing = self.client.retrieve(self.collection, [point.id for point in points])
            if len(existing) < len(points):
                self.client.upsert(self.collection, points, wait=True)

    def sparse(self, counts, query=False):
        weights = Counter()
        length = sum(counts.values())
        for word, freq in counts.items():
            df = self.df.get(word, 0)
            if query and not df:
                continue
            if query:
                weight = log(1 + (len(self.passages) - df + 0.5) / (df + 0.5))
            else:
                weight = freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / self.average))
            weights[term_id(word)] += weight
        pairs = sorted(weights.items())
        return self.models.SparseVector(indices=[idx for idx, _ in pairs], values=[value for _, value in pairs])

    def retrieve(self, question, *, limit=3, method="hybrid"):
        if not question.strip() or limit < 1:
            return []
        if method not in {"dense", "sparse", "hybrid"}:
            raise ValueError(f"Unknown retrieval method: {method}")
        q = self.models
        sparse = self.sparse(Counter(terms(question)), query=True)
        if not sparse.indices and (method == "sparse" or not self.semantic_embeddings):
            return []
        dense = self.embed(question)
        if not any(dense):
            return []
        if method == "dense" or not sparse.indices:
            kwargs = {"query": dense, "using": "dense", "score_threshold": 0.01}
        elif method == "sparse":
            kwargs = {"query": sparse, "using": "bm25"}
        else:
            kwargs = {"query": q.FusionQuery(fusion=q.Fusion.RRF),
                      "prefetch": [q.Prefetch(query=dense, using="dense", limit=max(12, limit), score_threshold=0.01),
                                   q.Prefetch(query=sparse, using="bm25", limit=max(12, limit))]}
        hits = self.client.query_points(self.collection, limit=max(10, limit) if self.reranker else limit,
                                        with_payload=True, **kwargs).points
        results = [dict(hit.payload) for hit in hits]
        if self.reranker and results:
            scores = self.reranker.predict([(question, hit["title"] + " " + hit["text"]) for hit in results])
            results = [row for _, row in sorted(zip(scores, results), key=lambda item: -item[0])]
        return results[:limit]

    def close(self):
        self.client.close()


def evaluate(retriever, cases, k=3):
    if not cases or any(not row["relevant"] for row in cases):
        raise ValueError("RAG eval needs questions with at least one relevant passage ID each.")
    known = {passage.id for passage in retriever.passages}
    if any(set(row["relevant"]) - known for row in cases):
        raise ValueError("RAG relevance labels reference passages absent from the corpus.")
    metrics = {}
    for method in ("dense", "sparse", "hybrid"):
        recalls, ranks = [], []
        for row in cases:
            relevant = set(row["relevant"])
            results = [hit["id"] for hit in retriever.retrieve(row["question"], limit=k, method=method)]
            recalls.append(len(relevant & set(results)) / len(relevant))
            ranks.append(next((1 / rank for rank, key in enumerate(results, 1) if key in relevant), 0))
        metrics[method] = {f"recall@{k}": mean(recalls), f"mrr@{k}": mean(ranks)}
    return {"questions": len(cases), "passages": len(retriever.passages), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--db", type=Path, default=DEFAULT_RAG_DB)
    parser.add_argument("--embedding-model")
    parser.add_argument("--reranker")
    sub = parser.add_subparsers(dest="action", required=True)
    fetch = sub.add_parser("query")
    fetch.add_argument("question")
    test = sub.add_parser("evaluate")
    test.add_argument("--cases", type=Path, default=Path(__file__).with_name("rag_eval.json"))
    test.add_argument("--output", type=Path, default=RUNTIME / "rag_summary.json")
    pdf = sub.add_parser("ingest-pdf")
    pdf.add_argument("path", type=Path)
    pdf.add_argument("--source", required=True)
    pdf.add_argument("--title", required=True)
    pdf.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "ingest-pdf":
        ingest_pdf(args.path, args.source, args.title, args.output)
        return
    model = Retriever(args.corpus, db=args.db, embedding_model=args.embedding_model, reranker=args.reranker)
    try:
        if args.action == "query":
            result = model.retrieve(args.question)
        else:
            cases = json.loads(args.cases.read_text(encoding="utf-8"))
            result = {"embedding": args.embedding_model or "hashed_tokens", "reranker": args.reranker, **evaluate(model, cases)}
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
