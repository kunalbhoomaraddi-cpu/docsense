"""DocSense pipeline: ingest -> chunk -> retrieve -> decide -> answer with citations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .answer import NOT_FOUND, Answer, ExtractiveAnswerer, LLMAnswerer
from .ingest import Block, Chunk, chunk_blocks, load_bytes, load_path
from .llm import GeminiClient
from .retrieval import build_retriever, relevance, tokenize


@dataclass
class Config:
    retriever: str = "hybrid"
    chunk_chars: int = 600
    top_k: int = 4
    # Below this confidence the system refuses to answer. Tuned on the eval set
    # (see evals/results.md): it best separates answerable from unanswerable
    # questions while still answering ~91% of real ones.
    min_confidence: float = 0.38


class DocSense:
    def __init__(self, config: Config | None = None, gemini: GeminiClient | None = None, use_llm: bool | None = None):
        self.config = config or Config()
        self.gemini = gemini
        llm_on = gemini is not None if use_llm is None else use_llm
        self.answerer = LLMAnswerer(gemini) if (llm_on and gemini) else ExtractiveAnswerer()
        self.blocks: list[Block] = []
        self.chunks: list[Chunk] = []
        self.retriever = None

    # ---------- indexing ----------
    def add_bytes(self, data: bytes, filename: str) -> int:
        new = load_bytes(data, filename)
        self.blocks = [b for b in self.blocks if b.source != Path(filename).name] + new
        self._reindex()
        return len(new)

    def add_paths(self, paths) -> None:
        for p in paths:
            self.blocks.extend(load_path(p))
        self._reindex()

    def add_folder(self, folder: str | Path) -> None:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
        self.add_paths(files)

    def clear(self) -> None:
        self.blocks, self.chunks, self.retriever = [], [], None

    def _reindex(self) -> None:
        self.chunks = chunk_blocks(self.blocks, max_chars=self.config.chunk_chars)
        self.retriever = build_retriever(self.config.retriever, self.gemini).fit(self.chunks) if self.chunks else None
        # Inverse document frequency: rare words ("phishing") matter more than common ones ("leave").
        df: dict[str, int] = {}
        for c in self.chunks:
            for t in set(tokenize(f"{c.section or ''}. {c.text}")):
                df[t] = df.get(t, 0) + 1
        n = max(len(self.chunks), 1)
        self.idf = {t: math.log(1 + n / f) for t, f in df.items()}
        self.max_idf = math.log(1 + n)
        if isinstance(self.answerer, ExtractiveAnswerer):
            self.answerer.idf, self.answerer.max_idf = self.idf, self.max_idf

    def coverage(self, question: str, chunk: Chunk) -> float:
        """Share of the question's meaning (IDF-weighted words) that appears in the chunk.

        "What is the dress code?" retrieves *something*, but "dress" and "code" appear
        nowhere in it, so coverage is low. Words never seen in any document get the
        maximum weight, because they are the strongest sign the answer is not there.
        """
        q = set(tokenize(question))
        if not q:
            return 0.0
        c = set(tokenize(f"{chunk.section or ''}. {chunk.text}"))
        weight = lambda t: self.idf.get(t, self.max_idf)  # noqa: E731
        return sum(weight(t) for t in q if t in c) / sum(weight(t) for t in q)

    def confidence(self, question: str, results: list[tuple[Chunk, float]]) -> float:
        if not results:
            return 0.0
        chunk, rel = results[0]
        return 0.5 * rel + 0.5 * self.coverage(question, chunk)

    @property
    def sources(self) -> list[str]:
        return sorted({b.source for b in self.blocks})

    # ---------- querying ----------
    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        if not self.retriever:
            return []
        results = self.retriever.search(question, k or self.config.top_k)
        if self.config.retriever in ("bm25", "tfidf", "gemini-embed"):
            top = results[0][1] if results else 1.0
            results = [(c, relevance(s, self.config.retriever, top)) for c, s in results]
        return results

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            return Answer("Please ask a question.", found=False)
        results = self.retrieve(question)
        confidence = self.confidence(question, results)
        if confidence < self.config.min_confidence:
            # Decide *before* calling the LLM: cheaper, and the model never sees weak context.
            return Answer(NOT_FOUND, found=False, confidence=confidence, answerer="threshold")
        return self.answerer.answer(question, [c for c, _ in results], confidence)
