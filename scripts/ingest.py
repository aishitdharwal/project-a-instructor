"""
Ingest documents into pgvector — instructor version.

Session 3: supports multiple chunking strategies.
A/B test them against each other using the eval harness.

Run:
    python -m scripts.ingest                           # fixed_size (Week 1 default)
    python -m scripts.ingest --strategy sentence_aware
    python -m scripts.ingest --strategy sliding_window
    python -m scripts.ingest --compare               # show chunk stats for all strategies
"""
import os
import glob
import json
import argparse
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

from scripts.chunker import chunk_document, compare_strategies, STRATEGIES

load_dotenv()

client = OpenAI()
console = Console()

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5433"),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )
    register_vector(conn)
    return conn


def embed_texts(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def ingest(strategy="fixed_size"):
    console.print(f"\n[bold cyan]Ingesting corpus[/] with strategy: [bold yellow]{strategy}[/]\n")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks;")

    doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    total_chunks = 0

    for filepath in doc_files:
        doc_name = os.path.basename(filepath)
        with open(filepath, "r") as f:
            content = f.read()

        chunks = chunk_document(content, strategy=strategy)
        console.print(f"  [dim]{doc_name}[/]: [green]{len(chunks)}[/] chunks")

        for batch_start in range(0, len(chunks), 20):
            batch = chunks[batch_start:batch_start + 20]
            embeddings = embed_texts(batch)

            for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_index = batch_start + i
                metadata = json.dumps({
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                    "strategy": strategy,
                })
                cur.execute(
                    """INSERT INTO chunks (doc_name, chunk_index, content, embedding, metadata)
                       VALUES (%s, %s, %s, %s::vector, %s)""",
                    (doc_name, chunk_index, chunk, embedding, metadata),
                )

        total_chunks += len(chunks)

    conn.commit()
    cur.close()
    conn.close()
    console.print(
        f"\n[bold green]Done:[/] {len(doc_files)} documents, {total_chunks} chunks "
        f"(strategy: [yellow]{strategy}[/])"
    )


def show_comparison():
    doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    totals = {s: {"count": 0, "chars": 0} for s in STRATEGIES}

    for filepath in doc_files:
        with open(filepath) as f:
            text = f.read()
        stats = compare_strategies(text)
        for strategy, s in stats.items():
            totals[strategy]["count"] += s["count"]
            totals[strategy]["chars"] += s["count"] * s["avg_size"]

    table = Table(
        title="Chunking Strategy Comparison (full corpus)",
        box=box.ROUNDED, title_style="bold cyan",
    )
    table.add_column("Strategy", style="bold", width=20)
    table.add_column("Total Chunks", justify="center")
    table.add_column("Avg Chars", justify="center")
    table.add_column("Note", style="dim")

    for strategy, data in totals.items():
        avg = data["chars"] // data["count"] if data["count"] else 0
        note = "← Week 1 baseline" if strategy == "fixed_size" else ""
        table.add_row(strategy, str(data["count"]), str(avg), note)

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into pgvector")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="fixed_size")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        show_comparison()
    else:
        ingest(strategy=args.strategy)
