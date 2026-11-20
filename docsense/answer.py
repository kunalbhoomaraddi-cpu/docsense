"""Turn retrieved chunks into a cited answer — or an honest "not found".

Two answerers share one interface:
- ``LLMAnswerer`` asks Gemini to answer only from the numbered sources and cite them as [1], [2].
  Citations are then checked: any [n] that does not point at a real source is removed.
- ``ExtractiveAnswerer`` needs no API key. It picks the sentences that best match the
  question and cites where each came from. It can never invent text, which makes it a
  useful baseline in the evals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Chunk, split_sentences
from .retrieval import tokenize

NOT_FOUND = "I couldn't find this in the documents."


@dataclass
class Citation:
    n: int
    source: str
    location: str
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    found: bool = True
    confidence: float = 0.0
    answerer: str = ""


SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Rules:
- Every sentence of your answer must end with at least one citation like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly: NOT_FOUND
- Do not use outside knowledge. Do not guess numbers, dates or names.
- Be concise: at most 4 sentences."""


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.citation()})\n{c.text}" for i, c in enumerate(chunks, start=1))


_CITE = re.compile(r"\[(\d+)\]")


def _citations_for(chunks: list[Chunk], numbers: list[int]) -> list[Citation]:
    out = []
    for n in sorted(set(numbers)):
        c = chunks[n - 1]
        out.append(Citation(n=n, source=c.source, location=c.citation(), snippet=c.text[:240]))
    return out


class LLMAnswerer:
    name = "gemini"

    def __init__(self, client):
        self.client = client

    def answer(self, question: str, chunks: list[Chunk], confidence: float) -> Answer:
        prompt = f"Sources:\n{format_sources(chunks)}\n\nQuestion: {question}"
        raw = self.client.generate(prompt, system=SYSTEM_PROMPT)
        if not raw or "NOT_FOUND" in raw:
            return Answer(NOT_FOUND, found=False, confidence=confidence, answerer=self.name)
        # Guardrail: drop citations that point at sources that don't exist.
        valid = [int(n) for n in _CITE.findall(raw) if 1 <= int(n) <= len(chunks)]
        cleaned = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(chunks) else "", raw)
        if not valid:
            # An answer with no valid citation is not grounded; refuse rather than mislead.
            return Answer(NOT_FOUND, found=False, confidence=confidence, answerer=self.name)
        return Answer(cleaned.strip(), _citations_for(chunks, valid), True, confidence, self.name)


class ExtractiveAnswerer:
    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
        self.idf: dict[str, float] = {}
        self.max_idf = 1.0

    def answer(self, question: str, chunks: list[Chunk], confidence: float) -> Answer:
        q = set(tokenize(question))
        weight = lambda t: self.idf.get(t, self.max_idf)  # noqa: E731
        total = sum(weight(t) for t in q) or 1.0
        scored: list[tuple[float, int, str]] = []
        for i, chunk in enumerate(chunks, start=1):
            # The section heading counts towards every sentence in it ("Sick leave" → "cannot be carried forward").
            heading = set(tokenize(chunk.section or ""))
            rank_bonus = 0.25 if i == 1 else 0.0  # trust the retriever's top pick a little more
            for sentence in split_sentences(chunk.text):
                s = set(tokenize(sentence)) | heading
                overlap = sum(weight(t) for t in q & s) / total
                scored.append((overlap + rank_bonus, i, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked: list[tuple[int, str]] = []
        if scored and scored[0][0] > 0.2:
            best = scored[0][0]
            for score, n, sentence in scored:
                # A second sentence must be nearly as relevant as the first, or it is noise.
                if len(picked) == self.max_sentences or score < 0.75 * best:
                    break
                picked.append((n, sentence))
        if not picked:
            return Answer(NOT_FOUND, found=False, confidence=confidence, answerer=self.name)
        text = " ".join(f"{s} [{n}]" for n, s in picked)
        return Answer(text, _citations_for(chunks, [n for n, _ in picked]), True, confidence, self.name)
