"""Minimal Gemini REST client (generate + embed). No SDK, so it is easy to read and mock."""

from __future__ import annotations

import os

import httpx

API = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        embed_model: str = "gemini-embedding-001",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.embed_model = embed_model
        self.http = httpx.Client(
            base_url=API, headers={"x-goog-api-key": api_key}, transport=transport, timeout=timeout
        )

    @classmethod
    def from_env(cls) -> GeminiClient | None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            return None
        return cls(key, model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))

    def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = self.http.post(f"/models/{self.model}:generateContent", json=body)
        r.raise_for_status()
        candidates = r.json().get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), 100):  # API batch limit
            batch = texts[start : start + 100]
            body = {
                "requests": [
                    {"model": f"models/{self.embed_model}", "content": {"parts": [{"text": t}]}, "taskType": task}
                    for t in batch
                ]
            }
            r = self.http.post(f"/models/{self.embed_model}:batchEmbedContents", json=body)
            r.raise_for_status()
            out.extend(e["values"] for e in r.json()["embeddings"])
        return out
