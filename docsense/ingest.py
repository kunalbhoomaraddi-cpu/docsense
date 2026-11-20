"""Load documents and split them into citable chunks.

Every chunk remembers where it came from (file, page, section heading) so the
answer can cite it precisely. Supported inputs: PDF, Markdown and plain text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    meta: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Human-readable location, e.g. 'leave-policy.md › Sick leave' or 'handbook.pdf p.4'."""
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(f"› {self.section}")
        return " ".join(parts)


@dataclass
class Block:
    """A unit of source text before chunking: one section of Markdown or one PDF page."""

    text: str
    source: str
    page: int | None = None
    section: str | None = None


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def load_markdown(text: str, source: str) -> list[Block]:
    blocks: list[Block] = []
    section: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append(Block(text=body, source=source, section=section))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            flush()
            # Level-1 headings are document titles; keep them as a section name only
            # until the first level-2 heading arrives.
            section = m.group(2).strip()
            continue
        if line.strip().startswith(">"):
            continue  # skip block quotes (disclaimers / notes)
        buf.append(line)
    flush()
    return blocks


def load_pdf(data: bytes, source: str) -> list[Block]:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    blocks = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append(Block(text=text, source=source, page=i))
    return blocks


def load_bytes(data: bytes, filename: str) -> list[Block]:
    name = Path(filename).name
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(data, name)
    text = data.decode("utf-8", errors="replace")
    if suffix in {".md", ".markdown"}:
        return load_markdown(text, name)
    if suffix in {".txt", ""}:
        return [Block(text=text.strip(), source=name)] if text.strip() else []
    raise ValueError(f"Unsupported file type: {suffix} (use .pdf, .md or .txt)")


def load_path(path: str | Path) -> list[Block]:
    p = Path(path)
    return load_bytes(p.read_bytes(), p.name)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENTENCE_END.split(text) if s]


def chunk_blocks(blocks: list[Block], max_chars: int = 600, overlap_sentences: int = 1) -> list[Chunk]:
    """Pack sentences into chunks of at most ``max_chars``, never crossing a section or page.

    Chunks never cross a block boundary, so each one has exactly one citation.
    ``overlap_sentences`` repeats the last sentence(s) of a chunk at the start of the next,
    so a fact split across a boundary is still retrievable.
    """
    chunks: list[Chunk] = []
    for block in blocks:
        sentences = split_sentences(block.text)
        current: list[str] = []
        for sentence in sentences:
            candidate = " ".join(current + [sentence])
            if current and len(candidate) > max_chars:
                chunks.append(_make_chunk(current, block, len(chunks)))
                current = current[-overlap_sentences:] if overlap_sentences else []
            current.append(sentence)
        if current:
            chunks.append(_make_chunk(current, block, len(chunks)))
    return chunks


def _make_chunk(sentences: list[str], block: Block, index: int) -> Chunk:
    return Chunk(
        id=f"c{index}",
        text=" ".join(sentences),
        source=block.source,
        page=block.page,
        section=block.section,
    )
