"""
Corpus Ingestion Script — Run after setup_rds.py.

Reads all .md files from the corpus/ directory, chunks them, embeds each chunk
with OpenAI text-embedding-3-small, and inserts into the chunks table on Aurora.

Safe to re-run: uses INSERT ... ON CONFLICT DO NOTHING (idempotent).

Run locally:
    python -m scripts.ingest_to_rds
    python -m scripts.ingest_to_rds --strategy sentence_aware
    python -m scripts.ingest_to_rds --clear   # wipe and re-ingest

Run via ECS one-off task (deploy.sh does this automatically).

Required environment variables:
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
    OPENAI_API_KEY
"""
import os
import sys
import glob
import json
import argparse
import time

import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

from scripts.chunker import chunk_document, STRATEGIES

load_dotenv()

console = Console()
client = OpenAI()

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 20   # OpenAI embedding API batch size


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
        connect_timeout=10,
    )


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Retries once on rate-limit errors."""
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]
    except Exception as e:
        console.print(f"  [yellow]Embedding error (retrying in 5s): {e}[/]")
        time.sleep(5)
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]


def ingest(strategy: str = "fixed_size", clear: bool = False):
    doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    if not doc_files:
        console.print(f"[bold red]No .md files found in {CORPUS_DIR}[/]")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Ingesting {len(doc_files)} documents[/]\n"
        f"[dim]Strategy: {strategy} | Corpus: {CORPUS_DIR}[/]",
        title="[bold cyan]Corpus Ingestion[/]",
        border_style="cyan",
    ))

    try:
        conn = get_connection()
        register_vector(conn)
        cur = conn.cursor()
    except Exception as e:
        console.print(f"[bold red]Connection failed:[/] {e}")
        sys.exit(1)

    if clear:
        console.print("[yellow]Clearing existing chunks...[/]", end=" ")
        cur.execute("DELETE FROM chunks;")
        conn.commit()
        console.print("[green]done[/]")

    total_chunks = 0
    stats_rows = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=len(doc_files))

        for filepath in doc_files:
            doc_name = os.path.basename(filepath)
            progress.update(task, description=f"[bold blue]{doc_name}")

            with open(filepath, "r") as f:
                content = f.read()

            chunks = chunk_document(content, strategy=strategy)
            inserted = 0

            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch_texts = chunks[batch_start: batch_start + BATCH_SIZE]
                embeddings = embed_batch(batch_texts)

                for i, (text, embedding) in enumerate(zip(batch_texts, embeddings)):
                    chunk_index = batch_start + i
                    metadata = json.dumps({
                        "doc_name": doc_name,
                        "chunk_index": chunk_index,
                        "strategy": strategy,
                    })
                    cur.execute(
                        """
                        INSERT INTO chunks (doc_name, chunk_index, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s::vector, %s)
                        ON CONFLICT (doc_name, chunk_index) DO NOTHING
                        """,
                        (doc_name, chunk_index, text, embedding, metadata),
                    )
                    inserted += 1

            conn.commit()
            total_chunks += inserted
            stats_rows.append((doc_name, len(chunks), inserted))
            progress.advance(task)

    cur.close()
    conn.close()

    # Summary table
    table = Table(title="Ingestion Summary", box=box.SIMPLE, title_style="bold green")
    table.add_column("Document", width=40)
    table.add_column("Chunks", justify="right", width=8)
    table.add_column("Inserted", justify="right", width=10)
    for doc_name, n_chunks, n_inserted in stats_rows:
        skipped = n_chunks - n_inserted
        inserted_str = str(n_inserted)
        if skipped:
            inserted_str += f" [dim]({skipped} skipped)[/]"
        table.add_row(doc_name, str(n_chunks), inserted_str)

    console.print(table)
    console.print(f"\n[bold green]Done.[/] {total_chunks} chunks inserted across {len(doc_files)} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into Aurora PostgreSQL")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="fixed_size",
        help="Chunking strategy (default: fixed_size)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing chunks before ingesting",
    )
    args = parser.parse_args()
    ingest(strategy=args.strategy, clear=args.clear)
