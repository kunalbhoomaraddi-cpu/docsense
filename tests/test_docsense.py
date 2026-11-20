import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from docsense.answer import NOT_FOUND, LLMAnswerer
from docsense.ingest import Block, chunk_blocks, load_markdown
from docsense.llm import GeminiClient
from docsense.pipeline import Config, DocSense
from docsense.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever, tokenize

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def ds():
    d = DocSense(Config())
    d.add_folder(SAMPLE)
    return d


# ---------- ingest ----------
def test_markdown_sections_and_disclaimers():
    blocks = load_markdown("# Title\n> note\n## Sick leave\nTwelve days.\n## Holidays\nTwelve holidays.", "x.md")
    assert [b.section for b in blocks] == ["Sick leave", "Holidays"]
    assert all("note" not in b.text for b in blocks)


def test_chunks_respect_size_and_never_cross_sections():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_blocks([Block(text, "a.md", section="A"), Block("Short.", "a.md", section="B")], max_chars=200)
    assert all(len(c.text) <= 200 + 40 for c in chunks)  # a single sentence may overflow slightly
    assert chunks[-1].section == "B" and chunks[-1].text == "Short."
    assert len({c.id for c in chunks}) == len(chunks)


def test_overlap_repeats_last_sentence():
    text = "One is first. Two is second. Three is third. Four is fourth."
    chunks = chunk_blocks([Block(text, "a.md")], max_chars=30, overlap_sentences=1)
    assert chunks[1].text.startswith(chunks[0].text.split(". ")[-1].rstrip("."))


def test_citation_format():
    c = chunk_blocks([Block("Hello there.", "doc.pdf", page=3)])[0]
    assert c.citation() == "doc.pdf p.3"


# ---------- retrieval ----------
def test_tokenizer_stems_and_drops_stopwords():
    assert tokenize("The passwords are booked") == ["password", "book"]


@pytest.mark.parametrize("cls", [BM25Retriever, TfidfRetriever])
def test_retrievers_find_the_right_section(ds, cls):
    r = cls().fit(ds.chunks)
    top = r.search("minimum password length", 1)[0][0]
    assert top.section == "Passwords and sign-in"


def test_hybrid_fuses_and_returns_k(ds):
    r = HybridRetriever([BM25Retriever(), TfidfRetriever()]).fit(ds.chunks)
    results = r.search("hotel limit in metro cities", 3)
    assert len(results) == 3 and results[0][0].section == "Hotel limits"
    assert all(0 <= s <= 1 for _, s in results)


# ---------- answering ----------
def test_answers_with_citation(ds):
    a = ds.ask("How many days of sick leave do I get?")
    assert a.found and "12 days" in a.text
    assert any(c.location.endswith("Sick leave") for c in a.citations)


def test_refuses_when_not_in_documents(ds):
    for q in ["What is the dress code?", "What is the notice period for resignation?"]:
        a = ds.ask(q)
        assert not a.found and a.text == NOT_FOUND and a.citations == []


def test_empty_question(ds):
    assert ds.ask("   ").found is False


def test_coverage_is_low_for_unrelated_words(ds):
    chunk = ds.retrieve("dress code")[0][0]
    assert ds.coverage("dress code", chunk) < 0.2


# ---------- LLM path (Gemini mocked, no network) ----------
class FakeGemini:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, system="", temperature=0.1):
        assert "ONLY the numbered sources" in system
        return self.reply


def test_llm_answer_keeps_valid_citations(ds):
    chunks = [c for c, _ in ds.retrieve("sick leave days")]
    a = LLMAnswerer(FakeGemini("You get 12 days of sick leave [1].")).answer("q", chunks, 0.9)
    assert a.found and a.citations[0].n == 1


def test_llm_invalid_citations_are_removed(ds):
    chunks = [c for c, _ in ds.retrieve("sick leave days")]
    a = LLMAnswerer(FakeGemini("12 days [1][9].")).answer("q", chunks, 0.9)
    assert "[9]" not in a.text and [c.n for c in a.citations] == [1]


def test_llm_uncited_answer_is_refused(ds):
    chunks = [c for c, _ in ds.retrieve("sick leave days")]
    a = LLMAnswerer(FakeGemini("Probably about 15 days.")).answer("q", chunks, 0.9)
    assert not a.found


def test_llm_not_found(ds):
    chunks = [c for c, _ in ds.retrieve("sick leave days")]
    assert not LLMAnswerer(FakeGemini("NOT_FOUND")).answer("q", chunks, 0.9).found


def test_gemini_client_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-goog-api-key"]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "hi [1]"}]}}]})

    client = GeminiClient("k123", model="m1", transport=httpx.MockTransport(handler))
    assert client.generate("hello", system="sys") == "hi [1]"
    assert seen["url"].endswith("/models/m1:generateContent") and seen["key"] == "k123"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"


# ---------- API ----------
@pytest.fixture()
def api():
    from docsense import api as module

    module.engine.clear()
    module.engine.add_folder(SAMPLE)
    return TestClient(module.app)


def test_api_health_and_ask(api):
    h = api.get("/health").json()
    assert h["status"] == "ok" and len(h["documents"]) == 4
    r = api.post("/ask", json={"question": "What is the hotel limit in Mumbai?"}).json()
    assert r["found"] and "8,000" in r["text"]


def test_api_upload_and_reset(api):
    r = api.post("/documents", files={"file": ("pets.md", b"## Pets\nDogs are welcome on Fridays.", "text/markdown")})
    assert r.status_code == 200 and "pets.md" in r.json()["documents"]
    a = api.post("/ask", json={"question": "Can I bring my dog on Friday?"}).json()
    assert a["found"] and "Fridays" in a["text"]
    assert "pets.md" not in api.delete("/documents").json()["documents"]


def test_api_rejects_bad_files(api):
    assert api.post("/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert api.post("/ask", json={"question": ""}).status_code == 422


# ---------------- resilience: slow or failing Gemini ----------------
def test_generate_switches_to_fallback_model(monkeypatch):
    monkeypatch.setattr("docsense.llm.time.sleep", lambda s: None)
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if "primary" in request.url.path:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok [1]"}]}}]})

    client = GeminiClient("k", model="primary", fallback_model="lite", transport=httpx.MockTransport(handler))
    assert client.generate("hi") == "ok [1]" and client.model == "lite"


def test_app_keeps_working_when_gemini_is_down(monkeypatch):
    monkeypatch.setattr("docsense.llm.time.sleep", lambda s: None)
    down = GeminiClient("k", fallback_model=None, transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    ds = DocSense(Config(), gemini=down)  # embeddings fail at startup -> keyword retrievers only
    ds.add_folder(SAMPLE)
    assert "gemini-embed" not in ds.retriever.name
    a = ds.ask("How many days of sick leave do I get?")
    assert a.found and "12 days" in a.text and a.answerer == "extractive (AI unavailable)"
