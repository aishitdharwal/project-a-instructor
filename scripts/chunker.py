"""
Three chunking strategies for Week 2 A/B testing.

Usage — via ingest:
    python scripts/ingest.py --strategy sentence_aware
    python scripts/ingest.py --strategy sliding_window
    python scripts/ingest.py --strategy fixed_size  (default, Week 1 baseline)

Usage — direct:
    from chunker import chunk_document
    chunks = chunk_document(text, strategy="sentence_aware")

The goal: understand how chunking strategy affects retrieval quality.
Measure each strategy with Week 1's eval harness — don't guess which is better.
"""

CHUNK_SIZE = 500
OVERLAP = 100


def fixed_size_chunk(text, chunk_size=CHUNK_SIZE):
    """
    Week 1 baseline — naive character splitting.
    Fast and simple. Splits mid-sentence, which hurts retrieval quality.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def sentence_aware_chunk(text, chunk_size=CHUNK_SIZE):
    """
    Split on paragraph boundaries (double newlines), merge small paragraphs.
    Keeps logical units together — good for structured policy docs.
    Chunks may vary in size but sentences are never split.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_parts = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 2 > chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(para)
        current_len += len(para) + 2

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def sliding_window_chunk(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Fixed-size chunks with overlap between adjacent chunks.
    The overlap window ensures boundary context isn't lost.
    Trade-off: more chunks, higher embedding cost, better boundary coverage.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


STRATEGIES = {
    "fixed_size": fixed_size_chunk,
    "sentence_aware": sentence_aware_chunk,
    "sliding_window": sliding_window_chunk,
}


def chunk_document(text, strategy="fixed_size", **kwargs):
    """
    Chunk a document using the specified strategy.

    Args:
        text: Document text
        strategy: One of 'fixed_size', 'sentence_aware', 'sliding_window'
        **kwargs: Strategy-specific args (chunk_size, overlap)

    Returns:
        List of chunk strings
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, **kwargs)


def compare_strategies(text):
    """Compare all strategies on a document — useful for analysis."""
    results = {}
    for name in STRATEGIES:
        chunks = chunk_document(text, strategy=name)
        sizes = [len(c) for c in chunks]
        results[name] = {
            "count": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes)) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
        }
    return results


if __name__ == "__main__":
    sample = """
## Return Policy

Items must be returned within 30 days of delivery. The item must be in its original condition.

Premium Silver members have a 45-day return window. Premium Gold members have 60 days.

### What Cannot Be Returned

Opened software, digital downloads, and gift cards cannot be returned.
Personalized items are also excluded from returns.

### Refund Processing

Refunds are processed within 5-7 business days after receipt and inspection.
Premium members receive refunds within 2 business days.
    """.strip()

    print("Comparing chunking strategies:\n")
    stats = compare_strategies(sample)
    for strategy, s in stats.items():
        print(f"  {strategy}: {s['count']} chunks, avg {s['avg_size']} chars")
