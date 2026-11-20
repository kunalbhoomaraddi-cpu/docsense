"""FastAPI app: upload documents, ask questions, get cited answers.

Run locally:  uvicorn docsense.api:app --reload
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .llm import GeminiClient
from .pipeline import DocSense

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT.parent / "data" / "sample"
MAX_UPLOAD_BYTES = int(os.getenv("DOCSENSE_MAX_UPLOAD_MB", "10")) * 1024 * 1024

app = FastAPI(title="DocSense", description="Ask questions about your documents. Every answer is cited.")
engine = DocSense(gemini=GeminiClient.from_env())
if SAMPLE.exists():
    engine.add_folder(SAMPLE)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "answerer": engine.answerer.name,
        "retriever": engine.config.retriever,
        "documents": engine.sources,
        "chunks": len(engine.chunks),
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    answer = engine.ask(req.question)
    return asdict(answer)


@app.post("/documents")
async def upload(file: UploadFile) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    try:
        blocks = engine.add_bytes(data, file.filename or "upload.txt")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # unreadable / corrupt PDF etc.
        raise HTTPException(400, f"Could not read {file.filename}: {e}") from e
    if not blocks:
        raise HTTPException(400, "No text found in that file (scanned PDFs need OCR first).")
    return {"added": file.filename, "documents": engine.sources, "chunks": len(engine.chunks)}


@app.delete("/documents")
def reset() -> dict:
    """Remove uploaded documents and go back to the sample set."""
    engine.clear()
    if SAMPLE.exists():
        engine.add_folder(SAMPLE)
    return {"documents": engine.sources}
