"""Minimal Gemini REST client (generate + embed). No SDK, so it is easy to read and mock."""

from __future__ import annotations

import logging
import os
import time

import httpx

log = logging.getLogger("docsense")
RETRYABLE = {429, 500, 502, 503, 504}

API = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        embed_model: str = "gemini-embedding-001",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        fallback_model: str | None = "gemini-3.5-flash-lite",
        retries: int = 1,
    ):
        self.model = model
        self.embed_model = embed_model
        self.fallback_model = fallback_model
        self.retries = retries
        self.http = httpx.Client(
            base_url=API, headers={"x-goog-api-key": api_key}, transport=transport, timeout=timeout
        )

    @classmethod
    def from_env(cls) -> GeminiClient | None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            return None
        return cls(
            key,
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite") or None,
        )

    def _post(self, path: str, body: dict) -> dict:
        """POST with retries on network errors, rate limits and server errors."""
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post(path, json=body)
                if r.status_code not in RETRYABLE:
                    r.raise_for_status()
                    return r.json()
                last = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            except httpx.TransportError as e:
                last = e
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise last  # type: ignore[misc]

    def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            data = self._post(f"/models/{self.model}:generateContent", body)
        except httpx.HTTPError as e:
            if not self.fallback_model or self.fallback_model == self.model:
                raise
            log.warning("Gemini %s failed (%s); switching to %s", self.model, e, self.fallback_model)
            self.model = self.fallback_model
            data = self._post(f"/models/{self.model}:generateContent", body)
        candidates = data.get("candidates") or []
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
            data = self._post(f"/models/{self.embed_model}:batchEmbedContents", body)
            out.extend(e["values"] for e in data["embeddings"])
        return out
