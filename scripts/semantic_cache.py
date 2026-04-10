"""
Semantic Cache — Session 5: Token Economics & Caching.

Avoids redundant LLM calls by caching answers to semantically similar queries.
Uses cosine similarity on query embeddings to detect near-duplicates — so
"What is the return policy?" and "How long can I return items?" resolve to
the same cached answer.

Key concepts taught:
  - Embedding-based lookup (not exact string match)
  - Similarity threshold as a quality/hit-rate tradeoff
  - TTL to prevent stale answers from persisting forever
  - Cache hit rate as a cost reduction metric

Usage:
    cache = SemanticCache(threshold=0.92, ttl_hours=24)
    hit, answer = cache.check(query_embedding)
    if not hit:
        answer = run_pipeline(query)
        cache.store(query_embedding, query, answer)

Run: python -m scripts.semantic_cache   (runs a demo)
"""
import time
import json
import os
import math
from typing import Optional

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "semantic_cache.json")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity — no numpy dependency."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    """
    In-memory semantic cache with optional JSON persistence across restarts.

    threshold : cosine similarity above which we return a cached answer (default 0.92)
                Higher → fewer hits but more accurate matches
                Lower  → more hits but risk of wrong answers
    ttl_hours : entries older than this are ignored (default 24h)
    persist   : if True, saves/loads cache from disk (CACHE_FILE)
    """

    def __init__(
        self,
        threshold: float = 0.92,
        ttl_hours: float = 24,
        persist: bool = False,
    ):
        self.threshold = threshold
        self.ttl_seconds = ttl_hours * 3600
        self.persist = persist
        self._store: list[dict] = []
        self._hits = 0
        self._misses = 0

        if persist and os.path.exists(CACHE_FILE):
            self._load()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def check(self, query_embedding: list[float]) -> tuple[bool, Optional[str]]:
        """
        Look up a semantically similar query in the cache.

        Returns:
            (True, answer)  — cache hit, skip the pipeline
            (False, None)   — cache miss, run the pipeline
        """
        now = time.time()
        best_sim = 0.0
        best_entry = None

        for entry in self._store:
            if now - entry["timestamp"] > self.ttl_seconds:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self.threshold:
            self._hits += 1
            return True, best_entry["answer"]

        self._misses += 1
        return False, None

    def store(self, query_embedding: list[float], query: str, answer: str) -> None:
        """Store a new query/answer pair in the cache."""
        self._store.append({
            "embedding": query_embedding,
            "query": query,
            "answer": answer,
            "timestamp": time.time(),
        })
        if self.persist:
            self._save()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        before = len(self._store)
        self._store = [
            e for e in self._store
            if now - e["timestamp"] <= self.ttl_seconds
        ]
        removed = before - len(self._store)
        if self.persist and removed:
            self._save()
        return removed

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "total_queries": total,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "cache_size": len(self._store),
            "threshold": self.threshold,
            "ttl_hours": self.ttl_seconds / 3600,
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
        if self.persist and os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        with open(CACHE_FILE, "w") as f:
            json.dump(self._store, f)

    def _load(self) -> None:
        try:
            with open(CACHE_FILE) as f:
                self._store = json.load(f)
        except (json.JSONDecodeError, KeyError):
            self._store = []


# ---------------------------------------------------------------------------
# Module-level default instance — shared across all pipeline calls in a process
# ---------------------------------------------------------------------------
_default_cache = SemanticCache(threshold=0.92, ttl_hours=24)


def get_cache() -> SemanticCache:
    return _default_cache


# ---------------------------------------------------------------------------
# Demo / standalone run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from openai import OpenAI
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI()
    console = Console()

    def _embed(text: str) -> list[float]:
        return client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding

    console.print(Panel(
        "[bold]Semantic Cache Demo[/]\n"
        "[dim]Threshold: 0.92 | First query populates cache, similar queries hit it[/]",
        title="[bold cyan]Session 5 — Semantic Cache[/]",
        border_style="cyan",
    ))

    cache = SemanticCache(threshold=0.92)

    queries = [
        ("What is the return window for electronics?",    "SEED — populates cache"),
        ("How long can I return electronics?",            "PARAPHRASE — should hit"),
        ("What's the electronics return policy?",         "PARAPHRASE — should hit"),
        ("How do I track my order?",                      "DIFFERENT — should miss"),
        ("Can I return my laptop if it's broken?",        "RELATED but distinct"),
    ]

    # Seed the cache with the first answer
    seed_query, _ = queries[0]
    seed_emb = _embed(seed_query)
    seed_answer = "Electronics can be returned within 30 days of delivery (60 days for Premium Gold members)."
    cache.store(seed_emb, seed_query, seed_answer)

    table = Table(title="Cache Lookup Results", box=box.ROUNDED)
    table.add_column("Query", width=48)
    table.add_column("Note", width=28)
    table.add_column("Hit?", justify="center", width=6)
    table.add_column("Similarity", justify="right", width=10)

    for query, note in queries[1:]:
        emb = _embed(query)
        # Compute similarity manually just for display
        sim = cosine_similarity(emb, seed_emb)
        hit, answer = cache.check(emb)
        hit_str = "[green]YES[/]" if hit else "[red]no[/]"
        table.add_row(query[:48], note, hit_str, f"{sim:.4f}")

    console.print(table)
    console.print()
    console.print("[bold]Cache Stats:[/]", cache.stats())
