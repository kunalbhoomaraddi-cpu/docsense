"""Retrievers: BM25 (keyword), TF-IDF (vector) and a hybrid that fuses both.

All retrievers share one interface: ``fit(chunks)`` then ``search(query, k)``,
which returns ``(chunk, score)`` pairs, best first. That makes them easy to swap
and to compare in the eval suite.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from .ingest import Chunk

log = logging.getLogger("docsense")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its
    may must my of on or our per should so than that the their there these this to
    up was we what when where which who will with within you your""".split()
)

_TOKEN = re.compile(r"[a-z0-9₹$]+(?:[.,][0-9]+)?")


def _stem(token: str) -> str:
    """Very light suffix stripping: 'passwords' -> 'password', 'booked' -> 'book'."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "ies":
                return token[: -len(suffix)] + "y"
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower().replace("-", " "))
    return [_stem(t) for t in tokens if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def fit(self, chunks: list[Chunk]) -> Retriever: ...

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]: ...


class BM25Retriever:
    """Okapi BM25, implemented directly so every step is visible and testable."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []

    def fit(self, chunks: list[Chunk]) -> BM25Retriever:
        self.chunks = chunks
        self.docs = [Counter(tokenize(_index_text(c))) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(chunks) else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.chunks))
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            s = 0.0
            for t in q:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out[i] = s
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return _top_k(self.chunks, self.scores(query), k)


class TfidfRetriever:
    """Cosine similarity over TF-IDF vectors (word unigrams + bigrams)."""

    name = "tfidf"

    def fit(self, chunks: list[Chunk]) -> TfidfRetriever:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), sublinear_tf=True, token_pattern=None
        )
        self.matrix = self.vectorizer.fit_transform([_index_text(c) for c in chunks])
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        return _top_k(self.chunks, scores, k)


class GeminiEmbeddingRetriever:
    """Dense retrieval with Gemini embeddings. Needs GEMINI_API_KEY; used when available."""

    name = "gemini-embed"

    def __init__(self, client):
        self.client = client

    def fit(self, chunks: list[Chunk]) -> GeminiEmbeddingRetriever:
        self.chunks = chunks
        vectors = self.client.embed([_index_text(c) for c in chunks], task="RETRIEVAL_DOCUMENT")
        self.matrix = _normalize(np.array(vectors))
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = _normalize(np.array(self.client.embed([query], task="RETRIEVAL_QUERY")))[0]
        return _top_k(self.chunks, self.matrix @ q, k)


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) of several retrievers.

    RRF only uses ranks, not raw scores, so it can combine retrievers whose scores
    are on different scales (BM25 is unbounded, cosine is 0..1).
    The returned score is the top component's normalised score for that chunk, so
    it stays meaningful for the "do I know enough to answer?" threshold.
    """

    def __init__(self, retrievers: list, rrf_k: int = 60):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def fit(self, chunks: list[Chunk]) -> HybridRetriever:
        self.chunks = chunks
        working = []
        for r in self.retrievers:
            try:
                r.fit(chunks)
                working.append(r)
            except Exception as e:  # e.g. the embedding API is down: keep the keyword retrievers
                log.warning("retriever %s disabled: %s", r.name, e)
        if not working:
            raise RuntimeError("no retriever could be built")
        self.retrievers = working
        self.name = "hybrid(" + "+".join(r.name for r in working) + ")"
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        confidence: dict[str, float] = {}
        by_id = {c.id: c for c in self.chunks}
        depth = max(k * 4, 20)
        for r in list(self.retrievers):
            try:
                results = r.search(query, depth)
            except Exception:  # a remote retriever failing mid-query must not break the answer
                continue
            top = results[0][1] if results and results[0][1] > 0 else 1.0
            for rank, (chunk, score) in enumerate(results):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                confidence[chunk.id] = max(confidence.get(chunk.id, 0.0), relevance(score, r.name, top))
        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(by_id[i], confidence[i]) for i in ranked]


def relevance(score: float, retriever_name: str, top: float = 1.0) -> float:
    """Map a raw retriever score to roughly 0..1 so one threshold works for all retrievers."""
    if retriever_name == "bm25":
        return 1 - math.exp(-score / 6)  # BM25 ~6 ≈ 0.63, ~12 ≈ 0.86
    return max(0.0, min(1.0, score))


def _index_text(chunk: Chunk) -> str:
    # Section headings carry a lot of meaning ("Sick leave"), so index them with the body.
    return f"{chunk.section or ''}. {chunk.text}"


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
    if not len(chunks):
        return []
    idx = np.argsort(-scores, kind="stable")[:k]
    return [(chunks[i], float(scores[i])) for i in idx]


def build_retriever(name: str, gemini_client=None):
    if name == "bm25":
        return BM25Retriever()
    if name == "tfidf":
        return TfidfRetriever()
    if name == "hybrid":
        parts = [BM25Retriever(), TfidfRetriever()]
        if gemini_client is not None:
            parts.append(GeminiEmbeddingRetriever(gemini_client))
        return HybridRetriever(parts)
    if name == "gemini-embed":
        if gemini_client is None:
            raise ValueError("gemini-embed needs GEMINI_API_KEY")
        return GeminiEmbeddingRetriever(gemini_client)
    raise ValueError(f"Unknown retriever: {name}")
