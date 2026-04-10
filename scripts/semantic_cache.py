"""
Semantic Cache — Session 5: Token Economics & Caching.

Supports two backends:
  In-memory  — default for local development (no external deps)
  Redis      — for production on AWS (shared across ECS tasks, survives restarts)

The core idea: instead of matching queries by exact string, we embed the query
and compare the embedding against cached embeddings using cosine similarity.
"What is the return window?" and "How long can I return items?" will both hit
the same cache entry because their embeddings are close in vector space.

Usage (local):
    cache = SemanticCache(threshold=0.92, ttl_hours=24)
    hit, answer = cache.check(query_embedding)
    if not hit:
        answer = run_pipeline(query)
        cache.store(query_embedding, query, answer)

Usage (production with Redis):
    import redis
    r = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    cache = SemanticCache(threshold=0.92, ttl_hours=24, redis_client=r)
    # Same API from here on

Run: python -m scripts.semantic_cache
"""
import time
import json
import os
import math
import uuid
from typing import Optional

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "semantic_cache.json")
REDIS_KEY_PREFIX = "semantic_cache:"


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
    Semantic cache with pluggable backend (in-memory or Redis).

    Parameters
    ----------
    threshold   : cosine similarity above which we return a cached answer.
                  0.92 is a good default — high enough to avoid wrong answers,
                  low enough to catch paraphrases.
    ttl_hours   : entries older than this are ignored / expired.
    persist     : (in-memory only) save/load cache to disk across restarts.
    redis_client: if provided, use Redis backend instead of in-memory.
                  Pass a redis.Redis instance (decode_responses=True).
    """

    def __init__(
        self,
        threshold: float = 0.92,
        ttl_hours: float = 24,
        persist: bool = False,
        redis_client=None,
    ):
        self.threshold = threshold
        self.ttl_seconds = int(ttl_hours * 3600)
        self._redis = redis_client
        self._hits = 0
        self._misses = 0

        # In-memory store (used when redis_client is None)
        self._store: list[dict] = []
        self._persist = persist and redis_client is None

        if self._persist and os.path.exists(CACHE_FILE):
            self._load()

    # ------------------------------------------------------------------
    # Public API — identical regardless of backend
    # ------------------------------------------------------------------

    def check(self, query_embedding: list[float]) -> tuple[bool, Optional[str]]:
        """
        Look for a semantically similar cached answer.
        Returns (True, answer) on hit, (False, None) on miss.
        """
        if self._redis is not None:
            return self._check_redis(query_embedding)
        return self._check_memory(query_embedding)

    def store(self, query_embedding: list[float], query: str, answer: str) -> None:
        """Store a query/answer pair in the cache."""
        if self._redis is not None:
            self._store_redis(query_embedding, query, answer)
        else:
            self._store_memory(query_embedding, query, answer)

    def evict_expired(self) -> int:
        """Remove expired entries (in-memory only; Redis TTL handles expiry automatically)."""
        if self._redis is not None:
            return 0
        now = time.time()
        before = len(self._store)
        self._store = [e for e in self._store if now - e["timestamp"] <= self.ttl_seconds]
        removed = before - len(self._store)
        if self._persist and removed:
            self._save()
        return removed

    def stats(self) -> dict:
        total = self._hits + self._misses
        backend = "redis" if self._redis is not None else "in-memory"
        cache_size = self._redis_size() if self._redis else len(self._store)
        return {
            "backend": backend,
            "total_queries": total,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "cache_size": cache_size,
            "threshold": self.threshold,
            "ttl_hours": self.ttl_seconds / 3600,
        }

    def clear(self) -> None:
        if self._redis is not None:
            keys = list(self._redis.scan_iter(f"{REDIS_KEY_PREFIX}*"))
            if keys:
                self._redis.delete(*keys)
        else:
            self._store.clear()
            if self._persist and os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Redis backend
    # ------------------------------------------------------------------

    def _check_redis(self, query_embedding: list[float]) -> tuple[bool, Optional[str]]:
        best_sim = 0.0
        best_answer = None

        for key in self._redis.scan_iter(f"{REDIS_KEY_PREFIX}*"):
            raw = self._redis.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_answer = entry["answer"]

        if best_answer and best_sim >= self.threshold:
            self._hits += 1
            return True, best_answer

        self._misses += 1
        return False, None

    def _store_redis(self, query_embedding: list[float], query: str, answer: str) -> None:
        key = f"{REDIS_KEY_PREFIX}{uuid.uuid4()}"
        entry = {"query": query, "embedding": query_embedding, "answer": answer}
        self._redis.setex(key, self.ttl_seconds, json.dumps(entry))

    def _redis_size(self) -> int:
        return sum(1 for _ in self._redis.scan_iter(f"{REDIS_KEY_PREFIX}*"))

    # ------------------------------------------------------------------
    # In-memory backend
    # ------------------------------------------------------------------

    def _check_memory(self, query_embedding: list[float]) -> tuple[bool, Optional[str]]:
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

    def _store_memory(self, query_embedding: list[float], query: str, answer: str) -> None:
        self._store.append({
            "embedding": query_embedding,
            "query": query,
            "answer": answer,
            "timestamp": time.time(),
        })
        if self._persist:
            self._save()

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
# Module-level default instance (in-memory, for local dev and CLI usage)
# ---------------------------------------------------------------------------
_default_cache = SemanticCache(threshold=0.92, ttl_hours=24)


def get_cache() -> SemanticCache:
    """Returns the module-level in-memory cache. Use this for local dev / CLI."""
    return _default_cache


# ---------------------------------------------------------------------------
# Demo
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
        "[bold]Semantic Cache Demo — In-Memory Backend[/]\n"
        "[dim]Threshold 0.92 | First query seeds cache, similar queries hit it[/]",
        title="[bold cyan]Session 5 — Semantic Cache[/]",
        border_style="cyan",
    ))

    cache = SemanticCache(threshold=0.92)

    seed_query = "What is the return window for electronics?"
    seed_emb = _embed(seed_query)
    seed_answer = "Electronics can be returned within 30 days (60 days for Premium Gold)."
    cache.store(seed_emb, seed_query, seed_answer)

    test_queries = [
        ("How long can I return electronics?",         "paraphrase  → HIT expected"),
        ("What's the electronics return policy?",      "paraphrase  → HIT expected"),
        ("Can I return a broken laptop?",              "related     → borderline"),
        ("How do I track my order?",                   "unrelated   → MISS expected"),
    ]

    table = Table(title="Cache Lookup Results", box=box.ROUNDED)
    table.add_column("Query", width=44)
    table.add_column("Expectation", width=26)
    table.add_column("Result", justify="center", width=8)
    table.add_column("Similarity", justify="right", width=10)

    for query, expectation in test_queries:
        emb = _embed(query)
        sim = cosine_similarity(emb, seed_emb)
        hit, _ = cache.check(emb)
        result_str = "[green]HIT[/]" if hit else "[red]MISS[/]"
        table.add_row(query, expectation, result_str, f"{sim:.4f}")

    console.print(table)
    console.print()
    stats = cache.stats()
    console.print(f"[bold]Stats:[/] {stats['cache_hits']} hits / "
                  f"{stats['cache_misses']} misses / "
                  f"hit rate {stats['hit_rate']:.0%}")
