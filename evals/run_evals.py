"""Evaluate DocSense on a labelled question set and write evals/results.md.

What is measured
- Retrieval (answerable questions): Hit@1, Hit@3 and MRR — is the chunk from the
  right document section ranked first / in the top 3?
- Refusal: for each relevance threshold, how many answerable questions are still
  answered and how many unanswerable ones are correctly refused.
- End to end: does the final answer contain the expected fact, cite the right
  section, and refuse when it should?

Run:  python -m evals.run_evals            (offline, extractive answerer)
      GEMINI_API_KEY=... python -m evals.run_evals --llm   (also scores Gemini answers)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from docsense.llm import GeminiClient
from docsense.pipeline import Config, DocSense

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample"
DATASET = ROOT / "evals" / "dataset.jsonl"


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


def norm(s: str) -> str:
    return re.sub(r"[\s,]+", " ", s.lower()).strip()


def is_hit(chunk, item) -> bool:
    return chunk.source == item["source"] and (chunk.section or "") == item["section"]


def retrieval_metrics(ds: DocSense, items: list[dict]) -> dict:
    hits1 = hits3 = 0
    rr = []
    for item in items:
        results = ds.retrieve(item["question"], k=10)
        rank = next((i for i, (c, _) in enumerate(results, start=1) if is_hit(c, item)), None)
        hits1 += rank == 1
        hits3 += rank is not None and rank <= 3
        rr.append(1 / rank if rank else 0.0)
    n = len(items)
    return {"hit@1": hits1 / n, "hit@3": hits3 / n, "mrr": float(np.mean(rr))}


def threshold_sweep(ds: DocSense, answerable: list[dict], unanswerable: list[dict]) -> list[dict]:
    conf = lambda q: ds.confidence(q, ds.retrieve(q))  # noqa: E731
    top_ans = [conf(i["question"]) for i in answerable]
    top_un = [conf(i["question"]) for i in unanswerable]
    rows = []
    for t in np.round(np.arange(0.10, 0.62, 0.02), 2):
        kept = float(np.mean([s >= t for s in top_ans]))
        refused = float(np.mean([s < t for s in top_un]))
        rows.append({"threshold": float(t), "answered": kept, "refused": refused, "balanced": (kept + refused) / 2})
    return rows


def end_to_end(ds: DocSense, items: list[dict]) -> dict:
    correct = cited = refused_ok = answered_bad = 0
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]
    failures = []
    for item in answerable:
        a = ds.ask(item["question"])
        ok = a.found and norm(item["answer_contains"]) in norm(a.text)
        correct += ok
        cited += a.found and any(
            c.source == item["source"] and c.location.endswith(item["section"]) for c in a.citations
        )
        if not ok:
            failures.append({"question": item["question"], "expected": item["answer_contains"], "got": a.text[:160]})
    for item in unanswerable:
        a = ds.ask(item["question"])
        refused_ok += not a.found
        if a.found:
            answered_bad += 1
            failures.append({"question": item["question"], "expected": "refuse", "got": a.text[:160]})
    return {
        "answer_accuracy": correct / len(answerable),
        "citation_accuracy": cited / len(answerable),
        "refusal_accuracy": refused_ok / len(unanswerable),
        "hallucinated_answers": answered_bad,
        "failures": failures,
    }


def build(retriever: str, chunk_chars: int, gemini=None, use_llm=False, min_confidence=None) -> DocSense:
    cfg = Config(retriever=retriever, chunk_chars=chunk_chars)
    if min_confidence is not None:
        cfg.min_confidence = min_confidence
    ds = DocSense(cfg, gemini=gemini, use_llm=use_llm)
    ds.add_folder(DATA)
    return ds


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also evaluate Gemini answers (needs GEMINI_API_KEY)")
    args = parser.parse_args()

    items = load_dataset()
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]

    # 1) Retrieval comparison across retrievers and chunk sizes
    grid = []
    for retriever in ("bm25", "tfidf", "hybrid"):
        for size in (300, 600, 1000):
            m = retrieval_metrics(build(retriever, size), answerable)
            grid.append({"retriever": retriever, "chunk_chars": size, **m})

    # 2) Refusal threshold for the default configuration
    default = Config()
    sweep = threshold_sweep(build(default.retriever, default.chunk_chars), answerable, unanswerable)
    best = max(sweep, key=lambda r: (round(r["balanced"], 4), r["answered"]))

    # 3) End-to-end with the extractive (no-LLM) answerer, and optionally Gemini
    e2e = {"extractive": end_to_end(build(default.retriever, default.chunk_chars), items)}
    if args.llm:
        gemini = GeminiClient.from_env()
        if gemini is None:
            raise SystemExit("--llm needs GEMINI_API_KEY")
        e2e["gemini"] = end_to_end(build(default.retriever, default.chunk_chars, gemini, True), items)

    write_report(items, grid, sweep, best, default, e2e)
    print((ROOT / "evals" / "results.md").read_text())


def write_report(items, grid, sweep, best, default, e2e) -> None:
    n_ans = sum(i["answerable"] for i in items)
    lines = [
        "# DocSense evaluation results",
        "",
        f"Dataset: {len(items)} questions over 4 sample policy documents — "
        f"{n_ans} answerable, {len(items) - n_ans} deliberately unanswerable. "
        "Generated by `python -m evals.run_evals`.",
        "",
        "## Retrieval: which setup finds the right section?",
        "",
        "| Retriever | Chunk size (chars) | Hit@1 | Hit@3 | MRR |",
        "|---|---|---|---|---|",
    ]
    best_mrr = max(r["mrr"] for r in grid)
    for r in grid:
        mark = " **(best)**" if r["mrr"] == best_mrr else ""
        lines.append(
            f"| {r['retriever']}{mark} | {r['chunk_chars']} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {r['mrr']:.3f} |"
        )
    lines += [
        "",
        f"## Refusal threshold ({default.retriever}, {default.chunk_chars} chars)",
        "",
        "Confidence = average of (a) the retriever's relevance score and (b) query coverage: how much of the "
        "question's meaning appears in the best passage. Below the threshold, DocSense says it couldn't find "
        "the answer instead of guessing. Higher threshold = fewer made-up answers, but more real questions refused.",
        "",
        "| Threshold | Answerable questions answered | Unanswerable questions refused |",
        "|---|---|---|",
    ]
    for r in sweep[::2]:
        mark = " ← chosen" if abs(r["threshold"] - default.min_confidence) < 1e-9 else ""
        lines.append(f"| {r['threshold']:.2f}{mark} | {pct(r['answered'])} | {pct(r['refused'])} |")
    lines += [
        "",
        f"Best balanced threshold on this set: **{best['threshold']:.2f}** "
        f"(answers {pct(best['answered'])}, refuses {pct(best['refused'])}). "
        f"The app uses `min_confidence = {default.min_confidence}`.",
        "",
        "## End to end",
        "",
        "| Answerer | Answer contains the right fact | Cites the right section | Correctly refuses | Made-up answers |",
        "|---|---|---|---|---|",
    ]
    for name, m in e2e.items():
        lines.append(
            f"| {name} | {pct(m['answer_accuracy'])} | {pct(m['citation_accuracy'])} | "
            f"{pct(m['refusal_accuracy'])} | {m['hallucinated_answers']} |"
        )
    lines += ["", "## Failure cases (extractive answerer)", ""]
    fails = e2e["extractive"]["failures"]
    if not fails:
        lines.append("None.")
    for f in fails:
        lines.append(f"- **{f['question']}** — expected `{f['expected']}`, got: _{f['got']}_")
    lines.append("")
    (ROOT / "evals" / "results.md").write_text("\n".join(lines))
    (ROOT / "evals" / "results.json").write_text(
        json.dumps({"retrieval": grid, "threshold_sweep": sweep, "end_to_end": e2e}, indent=2)
    )


if __name__ == "__main__":
    main()
