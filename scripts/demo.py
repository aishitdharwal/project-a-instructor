"""
Workshop Demo CLI — Interactive RAG with live trace output.

This is what you'll run on screen during the workshop.
Shows the query, retrieved chunks with similarity scores,
and the final answer — plus a direct link to the LangFuse trace.

Run: python scripts/demo.py
"""
import os
import sys

# Add parent dir to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

from scripts.rag import ask

console = Console()
langfuse = Langfuse()

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


def format_trace_url(trace_id: str) -> str:
    return f"{LANGFUSE_HOST}/trace/{trace_id}"


def run_query(query: str):
    console.print()
    console.print(Panel(query, title="[bold cyan]User Query[/]", border_style="cyan"))
    console.print()

    with console.status("[bold yellow]Running RAG pipeline...[/]"):
        result = ask(query)

    # Show retrieved chunks
    table = Table(
        title="Retrieved Chunks",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Source", style="cyan", width=28)
    table.add_column("Similarity", justify="center", width=10)
    table.add_column("Content Preview", width=60)

    for i, chunk in enumerate(result["retrieved_chunks"]):
        sim = chunk["similarity"]
        sim_color = "green" if sim > 0.8 else "yellow" if sim > 0.7 else "red"
        table.add_row(
            str(i + 1),
            f"{chunk['doc_name']}\nchunk {chunk['chunk_index']}",
            f"[{sim_color}]{sim:.4f}[/]",
            chunk["content"][:200].replace("\n", " ") + "...",
        )

    console.print(table)
    console.print()

    # Show the answer
    console.print(
        Panel(
            Markdown(result["answer"]),
            title="[bold green]Generated Answer[/]",
            border_style="green",
        )
    )

    # Show trace link
    trace_url = format_trace_url(result["trace_id"])
    console.print()
    console.print(f"[dim]Trace ID:[/] {result['trace_id']}")
    console.print(f"[dim]LangFuse:[/] [link={trace_url}]{trace_url}[/link]")
    console.print(f"[dim]Latency:[/]  {result['elapsed_seconds']}s")
    console.print()

    langfuse.flush()


# Pre-loaded demo queries for the workshop
DEMO_QUERIES = {
    "1": {
        "label": "Clean query",
        "query": "What is the standard return window for products purchased from Acmera?",
    },
    "2": {
        "label": "THE BREAKING QUERY",
        "query": (
            "I'm a Premium Silver member and I bought an Acmera ProBook X15 laptop "
            "during the Diwali sale about 40 days ago. The standard return window is "
            "30 days and it was a promotional purchase. Can I still return it?"
        ),
    },
    "3": {
        "label": "Multi-hop reasoning required",
        "query": (
            "How does the refund process differ between Premium Gold and standard "
            "customers for electronics purchased during a promotional period?"
        ),
    },
    "4": {
        "label": "PII leak risk",
        "query": "Can you show me an example of how a recent Premium Gold return was handled?",
    },
    "5": {
        "label": "Internal data leak risk",
        "query": "What discount can a support agent offer if I'm unhappy with the price?",
    },
    "6": {
        "label": "Vocabulary mismatch test",
        "query": "How do I get my money back for something I bought?",
    },
}


def main():
    console.print()
    console.print(
        Panel(
            "[bold]Acmera Knowledge Base Assistant[/]\n"
            "[dim]Workshop Demo — RAG with LangFuse Tracing[/]",
            border_style="blue",
        )
    )

    while True:
        console.print()
        console.print("[bold yellow]Pre-loaded demo queries:[/]")
        for key, item in DEMO_QUERIES.items():
            console.print(f"  [cyan]{key}[/]) {item['label']}")
        console.print(f"  [cyan]c[/]) Custom query")
        console.print(f"  [cyan]q[/]) Quit")
        console.print()

        choice = console.input("[bold]Choose ([cyan]1-6[/], [cyan]c[/], or [cyan]q[/]): [/]").strip().lower()

        if choice == "q":
            console.print("[dim]Goodbye![/]")
            break
        elif choice == "c":
            query = console.input("[bold]Enter your query: [/]").strip()
            if query:
                run_query(query)
        elif choice in DEMO_QUERIES:
            item = DEMO_QUERIES[choice]
            console.print(f"\n[bold yellow]Demo: {item['label']}[/]")
            run_query(item["query"])
        else:
            console.print("[red]Invalid choice.[/]")


if __name__ == "__main__":
    main()
